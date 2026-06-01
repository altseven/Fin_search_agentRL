from __future__ import annotations

import json
import math
from typing import Any

CLASSES = ["up", "neutral", "down"]


REWARD_DEFAULTS: dict[str, Any] = {
    "score": 0.0,
    "parse_error": 0.0,
    "missing_probability": 0.0,
    "invalid_probability": 0.0,
    "direction_reward": 0.0,
    "prob_reward": 0.0,
    "brier_reward": 0.0,
    "pnl_reward": 0.0,
    "format_reward": 0.0,
    "turn_cost": 0.0,
    "pred": "",
    "pred_label": "",
    "true_label": "",
    "future_relative_return": 0.0,
}


def _with_reward_defaults(values: dict[str, Any]) -> dict[str, Any]:
    out = dict(REWARD_DEFAULTS)
    out.update(values)
    return out


def parse_final_answer(text: str) -> dict | None:
    if not text:
        return None
    candidates: list[str] = []
    stack = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and ("p_up" in obj or "prediction" in obj):
            return obj
    return None


def compute_stock_reward(answer: dict | None, extra_info: dict[str, Any]) -> dict[str, Any]:
    true_label = str(extra_info.get("label", "neutral"))
    if true_label not in CLASSES:
        true_label = "neutral"
    try:
        future_rel = float(extra_info.get("future_relative_return", 0.0))
    except Exception:
        future_rel = 0.0

    if answer is None:
        return _with_reward_defaults(
            {
                "score": -1.0,
                "parse_error": 1.0,
                "true_label": true_label,
                "future_relative_return": future_rel,
            }
        )

    try:
        probs = [float(answer["p_up"]), float(answer["p_neutral"]), float(answer["p_down"])]
    except Exception:
        return _with_reward_defaults(
            {
                "score": -1.0,
                "missing_probability": 1.0,
                "true_label": true_label,
                "future_relative_return": future_rel,
            }
        )

    if any(math.isnan(p) or p < 0.0 or p > 1.0 for p in probs) or abs(sum(probs) - 1.0) > 0.08:
        return _with_reward_defaults(
            {
                "score": -0.8,
                "invalid_probability": 1.0,
                "true_label": true_label,
                "future_relative_return": future_rel,
            }
        )

    s = sum(probs)
    probs = [p / s for p in probs]
    true_idx = CLASSES.index(true_label)
    pred_idx = max(range(3), key=lambda i: probs[i])
    pred_label = CLASSES[pred_idx]

    direction_reward = 1.0 if pred_label == true_label else -0.5
    prob_reward = probs[true_idx]
    brier_error = sum((probs[i] - (1.0 if i == true_idx else 0.0)) ** 2 for i in range(3))
    brier_reward = 1.0 - brier_error / 2.0
    pnl_scale = max(float(extra_info.get("pnl_scale", 0.03)), 1e-6)
    position = probs[0] - probs[2]
    pnl_reward = max(-1.0, min(1.0, position * future_rel / pnl_scale))

    format_reward = 0.04
    if answer.get("prediction") == pred_label:
        format_reward += 0.03
    if isinstance(answer.get("evidence_summary"), list) and answer["evidence_summary"]:
        format_reward += 0.03

    num_turns = extra_info.get("num_turns")
    try:
        turn_cost = -0.01 * max(0, int(num_turns or 0) - 1)
    except Exception:
        turn_cost = 0.0

    total = (
        0.18 * direction_reward
        + 0.27 * prob_reward
        + 0.27 * brier_reward
        + 0.20 * pnl_reward
        + format_reward
        + turn_cost
    )
    total = max(-1.0, min(1.5, total))
    return _with_reward_defaults(
        {
            "score": float(total),
            "direction_reward": float(direction_reward),
            "prob_reward": float(prob_reward),
            "brier_reward": float(brier_reward),
            "pnl_reward": float(pnl_reward),
            "format_reward": float(format_reward),
            "turn_cost": float(turn_cost),
            "pred": pred_label,
            "pred_label": pred_label,
            "true_label": true_label,
            "future_relative_return": float(future_rel),
        }
    )


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {"label": ground_truth}
    return {}


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> dict[str, Any]:
    """verl custom reward entrypoint.

    Keep this module intentionally small and free of dataclasses/Tushare/pandas so
    Ray reward workers can dynamically import it without pulling in the training
    CLI.
    """
    try:
        gt = _parse_ground_truth(ground_truth)
        info = {}
        info.update(gt)
        if extra_info:
            info.update(dict(extra_info))
        answer = parse_final_answer(solution_str)
        return compute_stock_reward(answer, info)
    except Exception:
        return _with_reward_defaults({"score": -1.0, "parse_error": 1.0})
