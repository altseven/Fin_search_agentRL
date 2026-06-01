from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rl_common import DATA_SOURCE, ensure_dirs, log, pd, read_table, require_pandas
from rl_config import MVPConfig


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a financial search prediction agent. Your task is to predict the "
        "future relative return direction of one A-share stock using only point-in-time "
        "information visible at the given as_of date. You may call function tools for "
        "price factors, market context, industry context, peer comparison, fundamentals, "
        "announcements, and news/events. Do not invent tool observations or future "
        "information. After searching, output only strict JSON with prediction, p_up, "
        "p_neutral, p_down, alpha_score, confidence, evidence_summary, risk_factors, "
        "and search_steps_used."
    )
    tools = row.get("tools") or [
        "get_price_factors",
        "get_market_context",
        "get_industry_context",
        "get_peer_context",
        "get_fundamental_snapshot",
        "search_announcements",
        "search_news",
    ]
    tool_lines = "\n".join(f"- {name}" for name in tools)
    user = (
        f"Predict the relative return direction for stock {row.get('ts_code')} "
        f"({row.get('company_name', row.get('name', ''))}) over the next {row.get('horizon')} trading days.\n"
        f"Market: {row.get('market', 'A-share')}\n"
        f"Industry: {row.get('industry_name', row.get('industry', ''))}\n"
        f"Industry ID: {row.get('industry_id', '')}\n"
        f"As-of timestamp: {row.get('as_of', row.get('trade_date'))}\n"
        f"Base trade date: {row.get('trade_date')}\n"
        f"Max tool calls: {row.get('max_tool_calls', 4)}\n"
        f"Available tools:\n{tool_lines}\n\n"
        "Use the tools actively but stay within the budget. A good trajectory usually checks price factors, "
        "industry/market context, and at least one document or fundamental source when available. "
        "Labels and future returns are hidden from you.\n\n"
        "Final JSON schema:\n"
        "{\n"
        '  "prediction": "up|neutral|down",\n'
        '  "p_up": float,\n'
        '  "p_neutral": float,\n'
        '  "p_down": float,\n'
        '  "alpha_score": float,\n'
        '  "confidence": float,\n'
        '  "evidence_summary": [{"direction": "positive|neutral|negative", "source_type": str, "source_id": str, "summary": str}],\n'
        '  "risk_factors": [str],\n'
        '  "search_steps_used": int\n'
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def export_verl_dataset(cfg: MVPConfig) -> tuple[Path, Path]:
    require_pandas()
    dirs = ensure_dirs(cfg)
    tasks = read_table(dirs["processed"] / "tasks")
    labels = read_table(dirs["processed"] / "labels")
    df = tasks.merge(labels, on=["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"], how="inner")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        gt = {
            "label": item["label"],
            "future_relative_return": float(item["future_relative_return"]),
        }
        tools = item.get("tools") or [
            "get_price_factors",
            "get_market_context",
            "get_industry_context",
            "get_peer_context",
            "get_fundamental_snapshot",
            "search_announcements",
            "search_news",
        ]
        rows.append(
            {
                "uid": str(item["sample_id"]),
                "data_source": DATA_SOURCE,
                "agent_name": "tool_agent",
                "prompt": make_prompt(item),
                "ability": "stock_prediction_agent",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
                "extra_info": {
                    "split": item["split"],
                    "sample_id": item["sample_id"],
                    "ts_code": item["ts_code"],
                    "trade_date": item["trade_date"],
                    "as_of": item["as_of"],
                    "horizon": int(item["horizon"]),
                    "entry_date": item["entry_date"],
                    "exit_date": item["exit_date"],
                    "industry_id": item.get("industry_id", ""),
                    "industry_name": item.get("industry_name", item.get("industry", "")),
                    "label": item["label"],
                    "label_id": int(item["label_id"]),
                    "future_relative_return": float(item["future_relative_return"]),
                    "future_volatility": float(item.get("future_volatility", 0.0) or 0.0),
                    "cross_section_rank": float(item.get("cross_section_rank", 0.0) or 0.0),
                    "stock_return": float(item.get("stock_return", 0.0) or 0.0),
                    "market_return": float(item.get("market_future_return", 0.0) or 0.0),
                    "industry_return": float(item.get("industry_future_return", 0.0) or 0.0),
                    "relative_benchmark": item.get("relative_benchmark", "market"),
                    "max_tool_calls": int(item.get("max_tool_calls", cfg.max_tool_calls)),
                    "pnl_scale": float(cfg.pnl_scale),
                    "tool_selection": tools,
                },
            }
        )

    random.Random(cfg.seed).shuffle(rows)
    train_rows = [r for r in rows if r["extra_info"]["split"] == "train"]
    valid_rows = [r for r in rows if r["extra_info"]["split"] == "valid"]
    if not valid_rows:
        valid_rows = [r for r in rows if r["extra_info"]["split"] == "test"][: max(1, len(rows) // 10)]
    if not valid_rows and train_rows:
        valid_rows = train_rows[: max(1, min(len(train_rows), len(rows) // 10 or 1))]
    if not train_rows:
        train_rows = rows

    train_path = dirs["verl"] / "train.parquet"
    valid_path = dirs["verl"] / "valid.parquet"
    write_records_parquet(train_rows, train_path)
    write_records_parquet(valid_rows, valid_path)
    log(f"verl dataset exported: train={len(train_rows)} valid={len(valid_rows)}")
    return train_path, valid_path


def write_records_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import datasets

        datasets.Dataset.from_list(rows).to_parquet(str(path))
        return
    except Exception as exc:
        log(f"datasets parquet writer failed: {exc}")

    try:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return
    except Exception as exc:
        jsonl = path.with_suffix(".jsonl")
        with jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"Could not write parquet to {path}. Wrote JSONL fallback at {jsonl}. "
            "Install pyarrow or datasets on the training server."
        ) from exc
