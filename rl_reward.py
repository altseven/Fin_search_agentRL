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
    "prediction_mismatch": 0.0,
    "alpha_mismatch": 0.0,
    "direction_reward": 0.0,
    "prob_reward": 0.0,
    "brier_reward": 0.0,
    "pnl_reward": 0.0,
    "evidence_reward": 0.0,
    "format_reward": 0.0,
    "search_cost": 0.0,
    "turn_cost": 0.0,
    "invalid_tool_penalty": 0.0,
    "future_time_penalty": 0.0,
    "num_tool_calls": 0.0,
    "num_invalid_tool_calls": 0.0,
    "num_future_time_violations": 0.0,
    "position": 0.0,
    "probability_sum": 0.0,
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

    probability_sum = sum(probs)
    if any(math.isnan(p) or p < 0.0 or p > 1.0 for p in probs) or abs(probability_sum - 1.0) > 0.08:
        return _with_reward_defaults(
            {
                "score": -0.8,
                "invalid_probability": 1.0,
                "true_label": true_label,
                "future_relative_return": future_rel,
                "probability_sum": probability_sum,
            }
        )

    s = probability_sum
    probs = [p / s for p in probs]
    true_idx = CLASSES.index(true_label)
    pred_idx = max(range(3), key=lambda i: probs[i])
    pred_label = CLASSES[pred_idx]
    reported_pred = str(answer.get("prediction", "")).strip().lower()
    prediction_mismatch = 0.0 if reported_pred == pred_label else 1.0
    try:
        alpha_score = float(answer.get("alpha_score", probs[0] - probs[2]))
    except Exception:
        alpha_score = probs[0] - probs[2]
    alpha_mismatch = 1.0 if abs(alpha_score - (probs[0] - probs[2])) > 0.08 else 0.0

    direction_reward = 1.0 if pred_label == true_label else -0.5
    prob_reward = probs[true_idx]
    brier_error = sum((probs[i] - (1.0 if i == true_idx else 0.0)) ** 2 for i in range(3))
    brier_reward = 1.0 - brier_error / 2.0
    pnl_scale = max(float(extra_info.get("pnl_scale", 0.03)), 1e-6)
    position = probs[0] - probs[2]
    pnl_reward = max(-1.0, min(1.0, position * future_rel / pnl_scale))

    format_reward = 0.03
    format_reward += 0.03
    if prediction_mismatch == 0.0:
        format_reward += 0.02
    if alpha_mismatch == 0.0:
        format_reward += 0.01

    evidence = answer.get("evidence_summary")
    evidence_reward = 0.0
    if isinstance(evidence, list) and evidence:
        format_reward += 0.01
        directions = {str(x.get("direction", "")).lower() for x in evidence if isinstance(x, dict)}
        source_types = {str(x.get("source_type", "")).lower() for x in evidence if isinstance(x, dict)}
        if directions & {"positive", "neutral", "negative"}:
            evidence_reward += 0.02
        if source_types & {"price_factor", "market_context", "industry", "peer", "fundamental", "announcement", "news"}:
            evidence_reward += 0.03

    num_turns = extra_info.get("num_turns")
    try:
        num_turns_i = int(num_turns or 0)
    except Exception:
        num_turns_i = 0
    try:
        raw_tool_scores = extra_info.get("rollout_reward_scores", {})
        if isinstance(raw_tool_scores, dict):
            num_tool_calls = int(raw_tool_scores.get("num_tool_calls", 0) or 0)
        else:
            num_tool_calls = 0
    except Exception:
        num_tool_calls = 0
    if num_tool_calls <= 0 and num_turns_i > 0:
        num_tool_calls = max(0, (num_turns_i - 2) // 2)
    max_tool_calls = max(0, int(extra_info.get("max_tool_calls", 4) or 4))
    search_cost = -0.02 * num_tool_calls
    if num_tool_calls > max_tool_calls:
        search_cost -= 0.2 * (num_tool_calls - max_tool_calls)
    invalid_tool_calls = int(extra_info.get("num_invalid_tool_calls", 0) or 0)
    future_time_violations = int(extra_info.get("num_future_time_violations", 0) or 0)
    invalid_tool_penalty = -0.10 * invalid_tool_calls
    future_time_penalty = -0.30 * future_time_violations
    turn_cost = search_cost

    total = (
        0.20 * direction_reward
        + 0.25 * prob_reward
        + 0.25 * brier_reward
        + 0.20 * pnl_reward
        + 0.05 * evidence_reward
        + format_reward
        + search_cost
        + invalid_tool_penalty
        + future_time_penalty
    )
    total = max(-1.0, min(1.5, total))
    return _with_reward_defaults(
        {
            "score": float(total),
            "prediction_mismatch": float(prediction_mismatch),
            "alpha_mismatch": float(alpha_mismatch),
            "direction_reward": float(direction_reward),
            "prob_reward": float(prob_reward),
            "brier_reward": float(brier_reward),
            "pnl_reward": float(pnl_reward),
            "evidence_reward": float(evidence_reward),
            "format_reward": float(format_reward),
            "search_cost": float(search_cost),
            "turn_cost": float(turn_cost),
            "invalid_tool_penalty": float(invalid_tool_penalty),
            "future_time_penalty": float(future_time_penalty),
            "num_tool_calls": float(num_tool_calls),
            "num_invalid_tool_calls": float(invalid_tool_calls),
            "num_future_time_violations": float(future_time_violations),
            "position": float(position),
            "probability_sum": float(probability_sum),
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
