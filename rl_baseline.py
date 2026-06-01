from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from rl_common import CLASSES, ensure_dirs, log, pd, read_table, require_pandas, save_table
from rl_config import MVPConfig
from rl_reward import compute_stock_reward
from rl_tools import _safe_float, get_market_context, get_price_factors


def rule_agent_predict(task: dict[str, Any]) -> dict[str, Any]:
    pf = get_price_factors(str(task["ts_code"]), str(task["trade_date"]))
    mc = get_market_context(str(task["trade_date"]))
    factors = pf.get("factors", {}) if pf.get("status") == "ok" else {}

    momentum = _safe_float(factors.get("momentum_rank"), 0.5) or 0.5
    rs_rank = _safe_float(factors.get("rs_rank"), 0.5) or 0.5
    liq = _safe_float(factors.get("liquidity_rank"), 0.5) or 0.5
    vol = _safe_float(factors.get("vol_rank"), 0.5) or 0.5
    ret_5d = _safe_float(factors.get("ret_5d"), 0.0) or 0.0
    score = 1.2 * (momentum - 0.5) + 1.0 * (rs_rank - 0.5) + 0.3 * (liq - 0.5) - 0.35 * (vol - 0.5)
    score += max(-0.2, min(0.2, ret_5d)) * 1.5
    edge = math.tanh(score)
    p_up = 0.33 + 0.24 * edge
    p_down = 0.33 - 0.24 * edge
    p_neutral = 1.0 - p_up - p_down
    probs = [max(0.02, p_up), max(0.02, p_neutral), max(0.02, p_down)]
    s = sum(probs)
    probs = [p / s for p in probs]
    pred = CLASSES[max(range(3), key=lambda i: probs[i])]
    return {
        "prediction": pred,
        "p_up": round(probs[0], 4),
        "p_neutral": round(probs[1], 4),
        "p_down": round(probs[2], 4),
        "alpha_score": round(probs[0] - probs[2], 4),
        "confidence": round(max(probs), 4),
        "evidence_summary": [
            {
                "direction": "positive" if edge >= 0 else "negative",
                "source_type": "price_factor",
                "source_id": f"{task['ts_code']}_{task['trade_date']}",
                "summary": pf.get("summary", ""),
            },
            {
                "direction": "neutral",
                "source_type": "market_context",
                "source_id": str(task["trade_date"]),
                "summary": mc.get("summary", ""),
            },
        ],
        "risk_factors": ["This MVP uses price/market factors only; no verified announcement/news table is loaded."],
        "search_steps_used": 2,
    }


def run_rule_rollout(cfg: MVPConfig) -> Path:
    require_pandas()
    dirs = ensure_dirs(cfg)
    os.environ["STOCK_AGENT_DATA_DIR"] = str(Path(cfg.data_dir).expanduser().resolve())
    tasks = read_table(dirs["processed"] / "tasks")
    labels = read_table(dirs["processed"] / "labels")
    df = tasks.merge(labels, on=["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"], how="inner")

    rows = []
    for _, row in df.iterrows():
        task = row.to_dict()
        ans = rule_agent_predict(task)
        reward = compute_stock_reward(ans, task)
        rows.append(
            {
                "sample_id": task["sample_id"],
                "split": task["split"],
                "ts_code": task["ts_code"],
                "trade_date": task["trade_date"],
                "label": task["label"],
                "future_relative_return": task["future_relative_return"],
                "prediction": ans["prediction"],
                "p_up": ans["p_up"],
                "p_neutral": ans["p_neutral"],
                "p_down": ans["p_down"],
                "alpha_score": ans["alpha_score"],
                "reward": reward["score"],
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, dirs["rollouts"] / "rule_baseline_predictions")
    metrics = summarize_predictions(out)
    metrics_path = dirs["rollouts"] / "rule_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Rule rollout metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics_path


def summarize_predictions(df: "pd.DataFrame") -> dict[str, Any]:
    require_pandas()
    out: dict[str, Any] = {}
    for split, sub in df.groupby("split"):
        if sub.empty:
            continue
        acc = float((sub["prediction"] == sub["label"]).mean())
        mean_reward = float(sub["reward"].mean())
        brier = 0.0
        for _, row in sub.iterrows():
            y = [1.0 if row["label"] == c else 0.0 for c in CLASSES]
            p = [float(row["p_up"]), float(row["p_neutral"]), float(row["p_down"])]
            brier += sum((p[i] - y[i]) ** 2 for i in range(3)) / len(sub)
        rank_ics = []
        for _, g in sub.groupby("trade_date"):
            if len(g) >= 5:
                rank_ics.append(float(g["alpha_score"].rank().corr(g["future_relative_return"].rank())))
        out[split] = {
            "n": int(len(sub)),
            "mean_reward": mean_reward,
            "accuracy": acc,
            "brier": float(brier),
            "mean_rank_ic": float(sum(rank_ics) / len(rank_ics)) if rank_ics else None,
        }
    return out
