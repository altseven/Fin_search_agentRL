#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_stock_a800_1gpu.sh [TUSHARE_TOKEN] [options] [-- extra stock_agent_rl_mvp.py args]

Single-A800 entry for Qwen3-4B GRPO full-flow training.

It reuses run_stock_a800_4gpu.sh's persistent venv/cache/model logic, but
uses a conservative 1-GPU profile:

  n_gpus_per_node=1
  rollout_tp=1
  train_batch_size=8
  ppo_mini_batch_size=4
  rollout_n=2
  max_prompt_length=3072
  max_response_length=768
  total_epochs=1
  max_tasks=3000

The default torch wheel flavor is cu124 because this entry is intended for
ubuntu22.04-pytorch2.3.0-py3.10-gpu-cuda12.4.1 style images.

Examples:
  bash run_stock_a800_1gpu.sh "your_tushare_token"
  bash run_stock_a800_1gpu.sh "your_tushare_token" --python-bin "$(which python)" --no-setup --no-download-model
  MODEL_ID=Qwen/Qwen3-1.7B MODEL_DIR=/kunlun_data/temp_ag_rl/model/Qwen3-1.7B bash run_stock_a800_1gpu.sh "your_tushare_token"
  bash run_stock_a800_1gpu.sh "your_tushare_token" -- --total-epochs 1 --max-tasks 1500
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STOCK_AGENT_A800_PROFILE_LABEL="${STOCK_AGENT_A800_PROFILE_LABEL:-A800 1-GPU}"
export STOCK_AGENT_A800_LOG_NAME="${STOCK_AGENT_A800_LOG_NAME:-output_a800_1gpu.log}"
export STOCK_AGENT_A800_REQUIRED_GPUS="${STOCK_AGENT_A800_REQUIRED_GPUS:-1}"
export STOCK_AGENT_A800_N_GPUS="${STOCK_AGENT_A800_N_GPUS:-1}"
export STOCK_AGENT_A800_ROLLOUT_TP="${STOCK_AGENT_A800_ROLLOUT_TP:-1}"
export STOCK_AGENT_A800_TRAIN_BATCH_SIZE="${STOCK_AGENT_A800_TRAIN_BATCH_SIZE:-8}"
export STOCK_AGENT_A800_PPO_MINI_BATCH_SIZE="${STOCK_AGENT_A800_PPO_MINI_BATCH_SIZE:-4}"
export STOCK_AGENT_A800_PPO_MICRO_BATCH_SIZE_PER_GPU="${STOCK_AGENT_A800_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export STOCK_AGENT_A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${STOCK_AGENT_A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export STOCK_AGENT_A800_ROLLOUT_N="${STOCK_AGENT_A800_ROLLOUT_N:-2}"
export STOCK_AGENT_A800_MAX_PROMPT_LENGTH="${STOCK_AGENT_A800_MAX_PROMPT_LENGTH:-3072}"
export STOCK_AGENT_A800_MAX_RESPONSE_LENGTH="${STOCK_AGENT_A800_MAX_RESPONSE_LENGTH:-768}"
export STOCK_AGENT_A800_ROLLOUT_GPU_MEMORY_UTILIZATION="${STOCK_AGENT_A800_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export STOCK_AGENT_A800_ROLLOUT_MAX_MODEL_LEN="${STOCK_AGENT_A800_ROLLOUT_MAX_MODEL_LEN:-4096}"
export STOCK_AGENT_A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${STOCK_AGENT_A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export STOCK_AGENT_A800_ROLLOUT_AGENT_NUM_WORKERS="${STOCK_AGENT_A800_ROLLOUT_AGENT_NUM_WORKERS:-2}"
export STOCK_AGENT_A800_REWARD_NUM_WORKERS="${STOCK_AGENT_A800_REWARD_NUM_WORKERS:-2}"
export STOCK_AGENT_A800_TOTAL_EPOCHS="${STOCK_AGENT_A800_TOTAL_EPOCHS:-1}"
export STOCK_AGENT_A800_SAMPLE_STRIDE="${STOCK_AGENT_A800_SAMPLE_STRIDE:-4}"
export STOCK_AGENT_A800_MAX_TASKS="${STOCK_AGENT_A800_MAX_TASKS:-3000}"
export TORCH_CUDA="${TORCH_CUDA:-cu124}"

exec bash "$SCRIPT_DIR/run_stock_a800_4gpu.sh" "$@"
