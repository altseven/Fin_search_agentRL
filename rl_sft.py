from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rl_baseline import rule_agent_predict
from rl_common import ensure_dirs, log, read_table, require_pandas
from rl_config import MVPConfig
from rl_dataset import make_prompt, write_records_parquet
from rl_tools import get_fundamental_snapshot, get_industry_context, get_market_context, get_price_factors, search_news
from rl_verl import resolve_model_path_for_command


def _assistant_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _tool_response(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False)[:2000],
    }


def _load_tool_schemas() -> list[dict[str, Any]]:
    try:
        import rl_tools  # noqa: F401
        from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY

        names = [
            "get_price_factors",
            "get_market_context",
            "get_industry_context",
            "get_peer_context",
            "get_fundamental_snapshot",
            "search_announcements",
            "search_news",
        ]
        out = []
        for name in names:
            tool = FUNCTION_TOOL_REGISTRY.get(name)
            if tool is not None:
                out.append(tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True))
        return out
    except Exception:
        return []


def make_rule_sft_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    ts_code = str(task["ts_code"])
    trade_date = str(task["trade_date"])
    industry_id = str(task.get("industry_id", ""))
    company_name = str(task.get("company_name", task.get("name", "")))
    industry_name = str(task.get("industry_name", task.get("industry", "")))

    messages: list[dict[str, Any]] = make_prompt(task)

    calls = [
        ("get_price_factors", {"ts_code": ts_code, "as_of_date": trade_date, "lookback_days": 20}),
        ("get_market_context", {"as_of_date": trade_date}),
        ("get_industry_context", {"industry_id": industry_id, "as_of_date": trade_date, "lookback_days": 20}),
        ("get_fundamental_snapshot", {"ts_code": ts_code, "as_of_date": trade_date}),
        (
            "search_news",
            {
                "ts_code": ts_code,
                "as_of_date": trade_date,
                "query": f"{company_name} {industry_name} 量价 估值 行业",
                "top_k": 3,
            },
        ),
    ][: int(task.get("max_tool_calls", 4))]

    for name, args in calls:
        messages.append(_assistant_tool_call(name, args))
        if name == "get_price_factors":
            payload = get_price_factors(**args)
        elif name == "get_market_context":
            payload = get_market_context(**args)
        elif name == "get_industry_context":
            payload = get_industry_context(**args)
        elif name == "get_fundamental_snapshot":
            payload = get_fundamental_snapshot(**args)
        elif name == "search_news":
            payload = search_news(**args)
        else:
            payload = {"status": "error", "message": f"Unsupported rule SFT tool: {name}"}
        messages.append(_tool_response(name, payload))

    answer = rule_agent_predict(task)
    messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return messages


def export_sft_dataset(cfg: MVPConfig) -> tuple[Path, Path]:
    require_pandas()
    dirs = ensure_dirs(cfg)
    os.environ["STOCK_AGENT_DATA_DIR"] = str(Path(cfg.data_dir).expanduser().resolve())
    tasks = read_table(dirs["processed"] / "tasks")
    labels = read_table(dirs["processed"] / "labels")
    df = tasks.merge(labels, on=["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"], how="inner")

    schemas = _load_tool_schemas()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        rows.append(
            {
                "uid": str(item["sample_id"]),
                "messages": make_rule_sft_messages(item),
                "tools": schemas,
                "enable_thinking": False,
                "split": item["split"],
            }
        )

    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    if not valid_rows:
        valid_rows = [r for r in rows if r["split"] == "test"][: max(1, len(rows) // 10)]
    if not valid_rows and train_rows:
        valid_rows = train_rows[: max(1, min(len(train_rows), len(rows) // 10 or 1))]
    if not train_rows:
        train_rows = rows

    train_path = dirs["verl"] / "sft_train.parquet"
    valid_path = dirs["verl"] / "sft_valid.parquet"
    write_records_parquet(train_rows, train_path)
    write_records_parquet(valid_rows, valid_path)
    log(f"SFT dataset exported: train={len(train_rows)} valid={len(valid_rows)}")
    return train_path, valid_path


def make_sft_command(cfg: MVPConfig, train_path: Path, valid_path: Path) -> str:
    project_root = Path(__file__).resolve().parent
    repo = Path(cfg.verl_dir).expanduser().resolve()
    model_path = resolve_model_path_for_command(cfg.model_path)
    ckpt_dir = ensure_dirs(cfg)["run"] / "sft_checkpoints"
    max_token_len = int(cfg.max_prompt_length) + int(cfg.max_response_length)
    return f"""#!/usr/bin/env bash
set -xeuo pipefail

export PYTHONPATH="{project_root}:{repo}:${{PYTHONPATH:-}}"
export STOCK_AGENT_DATA_DIR="{Path(cfg.data_dir).expanduser().resolve()}"

cd "{repo}"
torchrun --standalone --nnodes=1 --nproc-per-node={cfg.n_gpus_per_node} -m verl.trainer.sft_trainer \\
  data.train_files="{train_path.resolve()}" \\
  data.val_files="{valid_path.resolve()}" \\
  data.train_batch_size={cfg.train_batch_size} \\
  data.micro_batch_size_per_gpu={cfg.ppo_micro_batch_size_per_gpu} \\
  data.use_dynamic_bsz=True \\
  data.max_token_len_per_gpu={max_token_len} \\
  data.messages_key=messages \\
  data.tools_key=tools \\
  data.enable_thinking_key=enable_thinking \\
  data.ignore_input_ids_mismatch=True \\
  data.truncation=error \\
  model.path="{model_path}" \\
  model.use_remove_padding={str(cfg.use_remove_padding).lower()} \\
  optim.lr=1e-5 \\
  optim.weight_decay=0.1 \\
  trainer.logger='["console"]' \\
  trainer.project_name=stock_agent_rl_mvp \\
  trainer.experiment_name=qwen3_4b_sse50_sft_tool_coldstart \\
  trainer.total_epochs=1 \\
  trainer.save_freq=1 \\
  trainer.test_freq=1 \\
  trainer.default_local_dir="{ckpt_dir.resolve()}" \\
  "$@"
"""


def write_sft_command_script(cfg: MVPConfig) -> Path:
    dirs = ensure_dirs(cfg)
    train_path = dirs["verl"] / "sft_train.parquet"
    valid_path = dirs["verl"] / "sft_valid.parquet"
    if not train_path.exists() or not valid_path.exists():
        train_path, valid_path = export_sft_dataset(cfg)
    out = dirs["run"] / "run_verl_stock_sft.sh"
    out.write_text(make_sft_command(cfg, train_path, valid_path), encoding="utf-8")
    out.chmod(0o755)
    log(f"Wrote SFT command script: {out}")
    return out
