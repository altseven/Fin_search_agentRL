#!/usr/bin/env python3
"""CLI entrypoint for the stock search-agent RL MVP.

The implementation is split into focused modules:

- rl_data.py: Tushare download, factor snapshots, labels.
- rl_tools.py: verl function-call tools.
- rl_reward.py: standalone reward function for verl reward workers.
- rl_dataset.py: verl parquet export.
- rl_baseline.py: local rule-agent baseline.
- rl_verl.py: generated verl training command.

This file keeps the original command-line and main() interface stable.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rl_baseline import run_rule_rollout
from rl_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_CODE,
    DEFAULT_MARKET_INDEX,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULT_DIR,
    DEFAULT_TUSHARE_HTTP_URL,
    ensure_dirs,
    log,
)
from rl_config import MVPConfig
from rl_data import build_data
from rl_dataset import export_verl_dataset
from rl_reward import compute_score
from rl_tools import get_market_context, get_price_factors, search_announcements
from rl_verl import (
    ensure_model_available,
    find_latest_verl_command,
    make_verl_command,
    print_download_hints,
    run_verl_command_script,
    write_verl_command_script,
)

__all__ = [
    "MVPConfig",
    "compute_score",
    "get_price_factors",
    "get_market_context",
    "search_announcements",
    "main",
    "parse_args",
]


def parse_args(argv: list[str] | None = None) -> MVPConfig:
    p = argparse.ArgumentParser(description="MVP for stock search-agent RL with Tushare + verl.")
    p.add_argument(
        "--mode",
        default="all",
        choices=[
            "all",
            "all-train",
            "build-data",
            "export-verl",
            "rule-rollout",
            "print-verl-command",
            "train-latest",
            "download-hints",
        ],
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--verl-dir", default="verl-main")
    p.add_argument("--tushare-token", default=None)
    p.add_argument("--tushare-http-url", default=DEFAULT_TUSHARE_HTTP_URL)
    p.add_argument("--index-code", default=DEFAULT_INDEX_CODE)
    p.add_argument("--market-index-code", default=DEFAULT_MARKET_INDEX)
    p.add_argument("--index-date", default="20260101")
    p.add_argument("--start-date", default="20230101")
    p.add_argument("--end-date", default="20260531")
    p.add_argument("--train-end-date", default="20241231")
    p.add_argument("--valid-end-date", default="20250930")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--sample-stride", type=int, default=5)
    p.add_argument("--max-stocks", type=int, default=50)
    p.add_argument("--max-tasks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sleep-seconds", type=float, default=0.12)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--rollout-n", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--ppo-mini-batch-size", type=int, default=8)
    p.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    p.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-response-length", type=int, default=1024)
    p.add_argument("--total-epochs", type=int, default=1)
    p.add_argument("--n-gpus-per-node", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--rollout-tp", type=int, default=2)
    p.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--rollout-max-model-len", type=int, default=None)
    p.add_argument("--rollout-max-num-batched-tokens", type=int, default=None)
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="HF attention backend. sdpa avoids the flash-attn package and is easiest on 3090 boxes.",
    )
    p.add_argument("--use-remove-padding", dest="use_remove_padding", action="store_true")
    p.add_argument("--no-use-remove-padding", dest="use_remove_padding", action="store_false")
    p.add_argument("--command-file", default="run_verl_stock_grpo.sh")
    p.add_argument("--no-auto-download-model", dest="auto_download_model", action="store_false")
    p.add_argument("--no-write-verl-command", dest="write_verl_command", action="store_false")
    p.set_defaults(write_verl_command=True, auto_download_model=True, use_remove_padding=False)
    args = p.parse_args(argv)
    return MVPConfig(**vars(args))


def main(
    mode: str = "all",
    data_dir: str = DEFAULT_DATA_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    run_dir: str | None = None,
    verl_dir: str = "verl-main",
    tushare_token: str | None = None,
    tushare_http_url: str = DEFAULT_TUSHARE_HTTP_URL,
    index_code: str = DEFAULT_INDEX_CODE,
    market_index_code: str = DEFAULT_MARKET_INDEX,
    index_date: str = "20260101",
    start_date: str = "20230101",
    end_date: str = "20260531",
    train_end_date: str = "20241231",
    valid_end_date: str = "20250930",
    horizon: int = 5,
    sample_stride: int = 5,
    max_stocks: int = 50,
    max_tasks: int = 3000,
    seed: int = 7,
    sleep_seconds: float = 0.12,
    force_refresh: bool = False,
    write_verl_command: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    rollout_n: int = 2,
    train_batch_size: int = 8,
    ppo_mini_batch_size: int = 8,
    ppo_micro_batch_size_per_gpu: int = 1,
    log_prob_micro_batch_size_per_gpu: int = 1,
    max_prompt_length: int = 2048,
    max_response_length: int = 1024,
    total_epochs: int = 1,
    n_gpus_per_node: int = 2,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    rollout_tp: int = 2,
    rollout_gpu_memory_utilization: float = 0.45,
    rollout_max_model_len: int | None = None,
    rollout_max_num_batched_tokens: int | None = None,
    attn_implementation: str = "sdpa",
    use_remove_padding: bool = False,
    command_file: str = "run_verl_stock_grpo.sh",
    auto_download_model: bool = True,
) -> dict[str, Any]:
    cfg = MVPConfig(
        mode=mode,
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        run_dir=run_dir,
        verl_dir=verl_dir,
        tushare_token=tushare_token,
        tushare_http_url=tushare_http_url,
        index_code=index_code,
        market_index_code=market_index_code,
        index_date=index_date,
        start_date=start_date,
        end_date=end_date,
        train_end_date=train_end_date,
        valid_end_date=valid_end_date,
        horizon=horizon,
        sample_stride=sample_stride,
        max_stocks=max_stocks,
        max_tasks=max_tasks,
        seed=seed,
        sleep_seconds=sleep_seconds,
        force_refresh=force_refresh,
        write_verl_command=write_verl_command,
        model_path=model_path,
        rollout_n=rollout_n,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        rollout_tp=rollout_tp,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        rollout_max_model_len=rollout_max_model_len,
        rollout_max_num_batched_tokens=rollout_max_num_batched_tokens,
        attn_implementation=attn_implementation,
        use_remove_padding=use_remove_padding,
        command_file=command_file,
        auto_download_model=auto_download_model,
    )
    random.seed(cfg.seed)
    dirs = ensure_dirs(cfg)
    config_for_log = {k: ("***" if k == "tushare_token" and v else v) for k, v in asdict(cfg).items()}
    (dirs["run"] / "run_config.json").write_text(
        json.dumps(config_for_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Config: {json.dumps(config_for_log, ensure_ascii=False)}")
    log(f"Run output dir: {dirs['run']}")

    results: dict[str, Any] = {"config": config_for_log}
    if cfg.mode in ("all", "all-train", "download-hints"):
        print_download_hints(cfg.model_path, cfg.model_dir)
        if cfg.mode == "download-hints":
            return results

    if cfg.mode == "all-train":
        ensure_model_available(cfg)

    if cfg.mode in ("all", "all-train", "build-data"):
        build_data(cfg)
        results["data_built"] = True

    if cfg.mode in ("all", "all-train", "export-verl"):
        train_path, valid_path = export_verl_dataset(cfg)
        results["train_parquet"] = str(train_path)
        results["valid_parquet"] = str(valid_path)

    if cfg.mode in ("all", "all-train", "rule-rollout"):
        metrics_path = run_rule_rollout(cfg)
        results["rule_metrics"] = str(metrics_path)

    if cfg.mode in ("all", "all-train", "print-verl-command"):
        if cfg.write_verl_command:
            path = write_verl_command_script(cfg)
            results["verl_command_file"] = str(path)
        else:
            dirs = ensure_dirs(cfg)
            train_path = (dirs["verl"] / "train.parquet").resolve()
            valid_path = (dirs["verl"] / "valid.parquet").resolve()
            print(make_verl_command(cfg, train_path, valid_path))

    if cfg.mode == "all-train":
        script = Path(str(results["verl_command_file"]))
        run_verl_command_script(script)
        results["verl_launched"] = True

    if cfg.mode == "train-latest":
        script = find_latest_verl_command(cfg.result_dir, cfg.command_file)
        results["verl_command_file"] = str(script)
        run_verl_command_script(script)
        results["verl_launched"] = True

    return results


if __name__ == "__main__":
    cli_cfg = parse_args()
    main(**asdict(cli_cfg))
