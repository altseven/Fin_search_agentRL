from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from rl_common import CLASSES, ensure_dirs, log, pd, read_table, require_pandas, save_table
from rl_config import MVPConfig
from rl_reward import compute_stock_reward
from rl_tools import (
    _safe_float,
    get_fundamental_snapshot,
    get_industry_context,
    get_market_context,
    get_peer_context,
    get_price_factors,
    search_announcements,
    search_news,
)


def rule_agent_predict(task: dict[str, Any]) -> dict[str, Any]:
    pf = get_price_factors(str(task["ts_code"]), str(task["trade_date"]))
    mc = get_market_context(str(task["trade_date"]))
    ic = get_industry_context(str(task.get("industry_id", "")), str(task["trade_date"]))
    pc = get_peer_context(str(task["ts_code"]), str(task["trade_date"]), top_k=5)
    fs = get_fundamental_snapshot(str(task["ts_code"]), str(task["trade_date"]))
    anns = search_announcements(str(task["ts_code"]), str(task["trade_date"]), query=str(task.get("company_name", "")), top_k=3)
    news = search_news(
        str(task["ts_code"]),
        str(task["trade_date"]),
        query=f"{task.get('company_name', '')} {task.get('industry_name', task.get('industry', ''))}",
        top_k=3,
    )
    factors = pf.get("factors", {}) if pf.get("status") == "ok" else {}
    market = mc.get("metrics", {}) if mc.get("status") == "ok" else {}
    industry = ic.get("metrics", {}) if ic.get("status") == "ok" else {}
    fundamentals = fs.get("fundamentals", {}) if fs.get("status") == "ok" else {}

    momentum = _safe_float(factors.get("momentum_rank"), 0.5) or 0.5
    ind_momentum = _safe_float(factors.get("momentum_rank_in_industry"), momentum) or momentum
    rs_rank = _safe_float(factors.get("rs_rank"), 0.5) or 0.5
    liq = _safe_float(factors.get("liquidity_rank"), 0.5) or 0.5
    vol = _safe_float(factors.get("vol_rank"), 0.5) or 0.5
    ret_5d = _safe_float(factors.get("ret_5d"), 0.0) or 0.0
    market_ret = _safe_float(market.get("market_ret_20d"), 0.0) or 0.0
    industry_ret = _safe_float(industry.get("industry_ret_20d"), 0.0) or 0.0
    pe_ttm = _safe_float(fundamentals.get("pe_ttm"), 35.0) or 35.0
    news_sent = 0.0
    news_results = news.get("results", []) if news.get("status") == "ok" else []
    if news_results:
        vals = [_safe_float(x.get("sentiment_score"), 0.0) or 0.0 for x in news_results]
        news_sent = sum(vals) / max(1, len(vals))
    ann_count = int(anns.get("result_count", 0) or 0)

    score = (
        1.0 * (momentum - 0.5)
        + 0.9 * (ind_momentum - 0.5)
        + 0.9 * (rs_rank - 0.5)
        + 0.25 * (liq - 0.5)
        - 0.30 * (vol - 0.5)
    )
    score += max(-0.2, min(0.2, ret_5d)) * 1.5
    score += max(-0.12, min(0.12, industry_ret - market_ret)) * 1.2
    score += max(-0.4, min(0.4, news_sent)) * 0.25
    score += min(ann_count, 3) * 0.015
    if pe_ttm > 80:
        score -= 0.10
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
            {
                "direction": "positive" if industry_ret >= market_ret else "negative",
                "source_type": "industry",
                "source_id": str(task.get("industry_id", "")),
                "summary": ic.get("summary", ""),
            },
            {
                "direction": "neutral",
                "source_type": "fundamental",
                "source_id": f"{task['ts_code']}_{task['trade_date']}",
                "summary": fs.get("summary", ""),
            },
        ],
        "risk_factors": [
            "Rule baseline uses deterministic factors and derived point-in-time event documents; it is not an oracle.",
            f"peer_results={pc.get('result_count', 0)}, announcement_results={ann_count}, news_results={len(news_results)}",
        ],
        "search_steps_used": 6,
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
            "macro_f1": macro_f1(sub["label"].tolist(), sub["prediction"].tolist()),
            "brier": float(brier),
            "mean_rank_ic": float(sum(rank_ics) / len(rank_ics)) if rank_ics else None,
            "top_bottom_return": top_bottom_return(sub),
        }
    return out


def macro_f1(labels: list[str], preds: list[str]) -> float:
    scores = []
    for cls in CLASSES:
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(sum(scores) / len(scores))


def top_bottom_return(df: "pd.DataFrame") -> float | None:
    spreads = []
    for _, g in df.groupby("trade_date"):
        if len(g) < 5:
            continue
        ranked = g.sort_values("alpha_score")
        k = max(1, int(len(ranked) * 0.3))
        bottom = ranked.head(k)["future_relative_return"].mean()
        top = ranked.tail(k)["future_relative_return"].mean()
        spreads.append(float(top - bottom))
    return float(sum(spreads) / len(spreads)) if spreads else None
