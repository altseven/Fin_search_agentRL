#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_a100_1gpu.sh [TUSHARE_TOKEN] [extra stock_agent_rl_mvp.py args...]

Examples:
  bash run_a100_1gpu.sh "your_tushare_token"
  bash run_a100_1gpu.sh
  bash run_a100_1gpu.sh "your_tushare_token" --total-epochs 2

Notes:
  - This script is a one-command A100 single-GPU LoRA run profile.
  - If TUSHARE_TOKEN is omitted, stock_agent_rl_mvp.py will try local_config.py.
  - Run setup once on a new server first:
      bash setup_stockverl_env.sh --cn-mirror --download-model
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-stockverl}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  else
    echo "Conda env '$ENV_NAME' not found. Run setup first:"
    echo "  bash setup_stockverl_env.sh --cn-mirror --download-model"
    exit 1
  fi
else
  echo "conda not found. Run setup first or activate your Python env manually."
  exit 1
fi

TOKEN_ARGS=()
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  TOKEN_ARGS=(--tushare-token "$1")
  shift
elif [[ -n "${TUSHARE_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--tushare-token "$TUSHARE_TOKEN")
fi

echo "== A100 single-GPU stock agent RL run =="
echo "Repo: $REPO_ROOT"
echo "Env: $ENV_NAME"
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

ray stop --force || true

python stock_agent_rl_mvp.py \
  --mode all-train \
  "${TOKEN_ARGS[@]}" \
  --n-gpus-per-node 1 \
  --rollout-tp 1 \
  --train-batch-size 4 \
  --ppo-mini-batch-size 4 \
  --ppo-micro-batch-size-per-gpu 1 \
  --log-prob-micro-batch-size-per-gpu 1 \
  --rollout-n 2 \
  --max-prompt-length 2048 \
  --max-response-length 1024 \
  --rollout-gpu-memory-utilization 0.45 \
  "$@"
