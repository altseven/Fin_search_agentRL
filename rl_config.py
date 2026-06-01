from __future__ import annotations

from dataclasses import dataclass

from rl_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_CODE,
    DEFAULT_MARKET_INDEX,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULT_DIR,
    DEFAULT_TUSHARE_HTTP_URL,
)


@dataclass
class MVPConfig:
    mode: str = "all"
    data_dir: str = DEFAULT_DATA_DIR
    model_dir: str = DEFAULT_MODEL_DIR
    result_dir: str = DEFAULT_RESULT_DIR
    run_dir: str | None = None
    verl_dir: str = "verl-main"
    tushare_token: str | None = None
    tushare_http_url: str = DEFAULT_TUSHARE_HTTP_URL
    index_code: str = DEFAULT_INDEX_CODE
    market_index_code: str = DEFAULT_MARKET_INDEX
    index_date: str = "20260101"
    start_date: str = "20230101"
    end_date: str = "20260531"
    train_end_date: str = "20241231"
    valid_end_date: str = "20250930"
    horizon: int = 5
    sample_stride: int = 5
    max_stocks: int = 50
    max_tasks: int = 3000
    seed: int = 7
    sleep_seconds: float = 0.12
    force_refresh: bool = False
    write_verl_command: bool = True
    model_path: str = DEFAULT_MODEL_PATH
    rollout_n: int = 2
    train_batch_size: int = 8
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 1
    log_prob_micro_batch_size_per_gpu: int = 1
    max_prompt_length: int = 2048
    max_response_length: int = 1024
    total_epochs: int = 1
    n_gpus_per_node: int = 2
    lora_rank: int = 32
    lora_alpha: int = 32
    rollout_tp: int = 2
    rollout_gpu_memory_utilization: float = 0.45
    attn_implementation: str = "sdpa"
    use_remove_padding: bool = False
    command_file: str = "run_verl_stock_grpo.sh"
    auto_download_model: bool = True
