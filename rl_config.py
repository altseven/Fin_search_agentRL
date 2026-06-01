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
    sample_stride: int = 3
    max_stocks: int = 50
    max_tasks: int = 8000
    seed: int = 7
    sleep_seconds: float = 0.12
    force_refresh: bool = False
    write_verl_command: bool = True
    model_path: str = DEFAULT_MODEL_PATH
    rollout_n: int = 4
    train_batch_size: int = 64
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size_per_gpu: int = 2
    log_prob_micro_batch_size_per_gpu: int = 2
    max_prompt_length: int = 3072
    max_response_length: int = 1024
    total_epochs: int = 2
    n_gpus_per_node: int = 8
    lora_rank: int = 0
    lora_alpha: int = 0
    actor_lr: float = 1e-6
    actor_weight_decay: float = 0.1
    actor_lr_warmup_steps: int = 10
    actor_ppo_max_token_len_per_gpu: int | None = None
    ref_log_prob_max_token_len_per_gpu: int | None = None
    rollout_log_prob_max_token_len_per_gpu: int | None = None
    actor_fsdp_size: int = -1
    ref_fsdp_size: int = -1
    actor_ulysses_sequence_parallel_size: int = 1
    ref_ulysses_sequence_parallel_size: int = 1
    actor_fsdp_param_offload: bool = False
    actor_fsdp_optimizer_offload: bool = False
    ref_fsdp_param_offload: bool = False
    rollout_tp: int = 4
    rollout_gpu_memory_utilization: float = 0.70
    rollout_max_model_len: int | None = None
    rollout_max_num_batched_tokens: int | None = None
    rollout_agent_num_workers: int = 8
    reward_num_workers: int = 8
    attn_implementation: str = "sdpa"
    use_remove_padding: bool = True
    command_file: str = "run_verl_stock_grpo.sh"
    auto_download_model: bool = True
