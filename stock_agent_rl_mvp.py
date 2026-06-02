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
from rl_report import generate_report_for_latest, generate_report_for_run
from rl_sft import export_sft_dataset, write_sft_command_script
from rl_tools import (
    get_fundamental_snapshot,
    get_industry_context,
    get_market_context,
    get_peer_context,
    get_price_factors,
    search_announcements,
    search_news,
)
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
    "get_industry_context",
    "get_fundamental_snapshot",
    "get_peer_context",
    "search_announcements",
    "search_news",
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
            "export-sft",
            "rule-rollout",
            "print-verl-command",
            "print-sft-command",
            "train-latest",
            "download-hints",
            "report-latest",
            "report-run",
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
    p.add_argument("--sample-stride", type=int, default=3)
    p.add_argument("--max-stocks", type=int, default=50)
    p.add_argument("--max-tasks", type=int, default=8000)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--up-quantile", type=float, default=0.70)
    p.add_argument("--down-quantile", type=float, default=0.30)
    p.add_argument("--pnl-scale", type=float, default=0.03)
    p.add_argument("--fetch-optional-docs", dest="fetch_optional_docs", action="store_true")
    p.add_argument("--no-fetch-optional-docs", dest="fetch_optional_docs", action="store_false")
    p.add_argument("--fetch-fundamentals", dest="fetch_fundamentals", action="store_true")
    p.add_argument("--no-fetch-fundamentals", dest="fetch_fundamentals", action="store_false")
    p.add_argument("--doc-lookback-days", type=int, default=180)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sleep-seconds", type=float, default=0.12)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--rollout-n", type=int, default=4)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--ppo-mini-batch-size", type=int, default=32)
    p.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=2)
    p.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=2)
    p.add_argument("--max-prompt-length", type=int, default=3072)
    p.add_argument("--max-response-length", type=int, default=1024)
    p.add_argument("--total-epochs", type=int, default=2)
    p.add_argument("--n-gpus-per-node", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=0)
    p.add_argument("--actor-lr", type=float, default=1e-6)
    p.add_argument("--actor-weight-decay", type=float, default=0.1)
    p.add_argument("--actor-lr-warmup-steps", type=int, default=10)
    p.add_argument("--actor-ppo-max-token-len-per-gpu", type=int, default=None)
    p.add_argument("--ref-log-prob-max-token-len-per-gpu", type=int, default=None)
    p.add_argument("--rollout-log-prob-max-token-len-per-gpu", type=int, default=None)
    p.add_argument("--actor-fsdp-size", type=int, default=-1)
    p.add_argument("--ref-fsdp-size", type=int, default=-1)
    p.add_argument("--actor-ulysses-sequence-parallel-size", type=int, default=1)
    p.add_argument("--ref-ulysses-sequence-parallel-size", type=int, default=1)
    p.add_argument("--actor-fsdp-param-offload", dest="actor_fsdp_param_offload", action="store_true")
    p.add_argument("--no-actor-fsdp-param-offload", dest="actor_fsdp_param_offload", action="store_false")
    p.add_argument("--actor-fsdp-optimizer-offload", dest="actor_fsdp_optimizer_offload", action="store_true")
    p.add_argument("--no-actor-fsdp-optimizer-offload", dest="actor_fsdp_optimizer_offload", action="store_false")
    p.add_argument("--ref-fsdp-param-offload", dest="ref_fsdp_param_offload", action="store_true")
    p.add_argument("--no-ref-fsdp-param-offload", dest="ref_fsdp_param_offload", action="store_false")
    p.add_argument("--rollout-tp", type=int, default=4)
    p.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.70)
    p.add_argument("--rollout-max-model-len", type=int, default=None)
    p.add_argument("--rollout-max-num-batched-tokens", type=int, default=None)
    p.add_argument("--rollout-agent-num-workers", type=int, default=8)
    p.add_argument("--reward-num-workers", type=int, default=8)
    p.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="HF attention backend. sdpa is the most portable; flash_attention_2 needs flash-attn installed.",
    )
    p.add_argument("--use-remove-padding", dest="use_remove_padding", action="store_true")
    p.add_argument("--no-use-remove-padding", dest="use_remove_padding", action="store_false")
    p.add_argument("--command-file", default="run_verl_stock_grpo.sh")
    p.add_argument("--no-auto-download-model", dest="auto_download_model", action="store_false")
    p.add_argument("--no-write-verl-command", dest="write_verl_command", action="store_false")
    p.add_argument("--write-report", dest="write_report", action="store_true")
    p.add_argument("--no-write-report", dest="write_report", action="store_false")
    p.set_defaults(
        write_verl_command=True,
        auto_download_model=True,
        write_report=True,
        use_remove_padding=True,
        actor_fsdp_param_offload=False,
        actor_fsdp_optimizer_offload=False,
        ref_fsdp_param_offload=False,
        fetch_optional_docs=True,
        fetch_fundamentals=True,
    )
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
    sample_stride: int = 3,
    max_stocks: int = 50,
    max_tasks: int = 8000,
    max_tool_calls: int = 4,
    up_quantile: float = 0.70,
    down_quantile: float = 0.30,
    pnl_scale: float = 0.03,
    fetch_optional_docs: bool = True,
    fetch_fundamentals: bool = True,
    doc_lookback_days: int = 180,
    seed: int = 7,
    sleep_seconds: float = 0.12,
    force_refresh: bool = False,
    write_verl_command: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    rollout_n: int = 4,
    train_batch_size: int = 64,
    ppo_mini_batch_size: int = 32,
    ppo_micro_batch_size_per_gpu: int = 2,
    log_prob_micro_batch_size_per_gpu: int = 2,
    max_prompt_length: int = 3072,
    max_response_length: int = 1024,
    total_epochs: int = 2,
    n_gpus_per_node: int = 8,
    lora_rank: int = 0,
    lora_alpha: int = 0,
    actor_lr: float = 1e-6,
    actor_weight_decay: float = 0.1,
    actor_lr_warmup_steps: int = 10,
    actor_ppo_max_token_len_per_gpu: int | None = None,
    ref_log_prob_max_token_len_per_gpu: int | None = None,
    rollout_log_prob_max_token_len_per_gpu: int | None = None,
    actor_fsdp_size: int = -1,
    ref_fsdp_size: int = -1,
    actor_ulysses_sequence_parallel_size: int = 1,
    ref_ulysses_sequence_parallel_size: int = 1,
    actor_fsdp_param_offload: bool = False,
    actor_fsdp_optimizer_offload: bool = False,
    ref_fsdp_param_offload: bool = False,
    rollout_tp: int = 4,
    rollout_gpu_memory_utilization: float = 0.70,
    rollout_max_model_len: int | None = None,
    rollout_max_num_batched_tokens: int | None = None,
    rollout_agent_num_workers: int = 8,
    reward_num_workers: int = 8,
    attn_implementation: str = "sdpa",
    use_remove_padding: bool = True,
    command_file: str = "run_verl_stock_grpo.sh",
    auto_download_model: bool = True,
    write_report: bool = True,
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
        max_tool_calls=max_tool_calls,
        up_quantile=up_quantile,
        down_quantile=down_quantile,
        pnl_scale=pnl_scale,
        fetch_optional_docs=fetch_optional_docs,
        fetch_fundamentals=fetch_fundamentals,
        doc_lookback_days=doc_lookback_days,
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
        actor_lr=actor_lr,
        actor_weight_decay=actor_weight_decay,
        actor_lr_warmup_steps=actor_lr_warmup_steps,
        actor_ppo_max_token_len_per_gpu=actor_ppo_max_token_len_per_gpu,
        ref_log_prob_max_token_len_per_gpu=ref_log_prob_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=rollout_log_prob_max_token_len_per_gpu,
        actor_fsdp_size=actor_fsdp_size,
        ref_fsdp_size=ref_fsdp_size,
        actor_ulysses_sequence_parallel_size=actor_ulysses_sequence_parallel_size,
        ref_ulysses_sequence_parallel_size=ref_ulysses_sequence_parallel_size,
        actor_fsdp_param_offload=actor_fsdp_param_offload,
        actor_fsdp_optimizer_offload=actor_fsdp_optimizer_offload,
        ref_fsdp_param_offload=ref_fsdp_param_offload,
        rollout_tp=rollout_tp,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        rollout_max_model_len=rollout_max_model_len,
        rollout_max_num_batched_tokens=rollout_max_num_batched_tokens,
        rollout_agent_num_workers=rollout_agent_num_workers,
        reward_num_workers=reward_num_workers,
        attn_implementation=attn_implementation,
        use_remove_padding=use_remove_padding,
        command_file=command_file,
        auto_download_model=auto_download_model,
        write_report=write_report,
    )
    random.seed(cfg.seed)

    if cfg.mode == "report-latest":
        report_dir = generate_report_for_latest(cfg.result_dir, cfg.data_dir)
        return {"report_dir": str(report_dir)}

    if cfg.mode == "report-run":
        if not cfg.run_dir:
            raise ValueError("--mode report-run requires --run-dir")
        report_dir = generate_report_for_run(cfg.run_dir, cfg.data_dir)
        return {"report_dir": str(report_dir)}

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

    if cfg.mode in ("all", "all-train", "export-sft", "print-sft-command"):
        sft_train_path, sft_valid_path = export_sft_dataset(cfg)
        results["sft_train_parquet"] = str(sft_train_path)
        results["sft_valid_parquet"] = str(sft_valid_path)

    if cfg.mode in ("all", "all-train", "rule-rollout"):
        metrics_path = run_rule_rollout(cfg)
        results["rule_metrics"] = str(metrics_path)
        _maybe_write_report(cfg, dirs, results)

    if cfg.mode in ("all", "all-train", "print-sft-command"):
        sft_script = write_sft_command_script(cfg)
        results["sft_command_file"] = str(sft_script)

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
        try:
            run_verl_command_script(script)
            results["verl_launched"] = True
        finally:
            _maybe_write_report(cfg, dirs, results)

    if cfg.mode == "train-latest":
        script = find_latest_verl_command(cfg.result_dir, cfg.command_file)
        results["verl_command_file"] = str(script)
        run_verl_command_script(script)
        results["verl_launched"] = True

    return results


def _maybe_write_report(cfg: MVPConfig, dirs: dict[str, Path], results: dict[str, Any]) -> None:
    if not cfg.write_report:
        return
    try:
        report_dir = generate_report_for_run(dirs["run"], cfg.data_dir)
        results["report_dir"] = str(report_dir)
    except Exception as exc:
        log(f"Report generation skipped: {exc}")


if __name__ == "__main__":
    cli_cfg = parse_args()
    main(**asdict(cli_cfg))
