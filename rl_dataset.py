from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rl_common import DATA_SOURCE, ensure_dirs, log, pd, read_table, require_pandas
from rl_config import MVPConfig


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a stock prediction search agent. You must use only point-in-time "
        "information available at the given as_of date. You can call tools to get "
        "price factors, market context, and optional announcements. Finally output "
        "strict JSON with prediction, p_up, p_neutral, p_down, alpha_score, "
        "confidence, evidence_summary, risk_factors, and search_steps_used."
    )
    user = (
        f"Predict the relative return direction for stock {row.get('ts_code')} "
        f"({row.get('name', '')}) over the next {row.get('horizon')} trading days.\n"
        f"Industry: {row.get('industry', '')}\n"
        f"As-of date: {row.get('trade_date')}\n"
        f"Max tool calls: {row.get('max_tool_calls', 3)}\n"
        "Available tools: get_price_factors, get_market_context, search_announcements.\n"
        "Labels are hidden from you. Do not invent future information."
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
                    "label": item["label"],
                    "label_id": int(item["label_id"]),
                    "future_relative_return": float(item["future_relative_return"]),
                    "pnl_scale": 0.03,
                },
            }
        )

    random.Random(cfg.seed).shuffle(rows)
    train_rows = [r for r in rows if r["extra_info"]["split"] == "train"]
    valid_rows = [r for r in rows if r["extra_info"]["split"] == "valid"]
    if not valid_rows:
        valid_rows = [r for r in rows if r["extra_info"]["split"] == "test"][: max(1, len(rows) // 10)]

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
