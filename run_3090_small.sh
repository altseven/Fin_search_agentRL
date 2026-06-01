#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_3090_small.sh [TUSHARE_TOKEN] [options] [-- extra stock_agent_rl_mvp.py args]

Single-RTX-3090 smoke test for the full-flow stock search-agent RL pipeline.

Default behavior:
  - use setup_stockverl_env.sh in compatible mode
  - prefer China mirrors for pip/PyTorch/model downloads
  - download Qwen/Qwen3-0.6B to model/Qwen3-0.6B
  - run the existing full-flow pipeline through run_stock_agent_rl.sh --profile debug
  - keep LoRA disabled and use smaller data/batch/sequence settings

Options:
  --env-mode MODE          auto|conda|system. Default: auto
  --env-name NAME          Conda env name. Default: stockverl
  --python-bin PATH        Python executable for system mode
  --requirements PATH      Requirements file. Default: requirements-stockverl.txt
  --dependency-policy P    compatible|strict. Default: compatible
  --torch-cuda FLAVOR      cu121|cu124|cu126|cu128|cpu. Default: cu128
  --model-id ID            ModelScope/HuggingFace model id. Default: Qwen/Qwen3-0.6B
  --model-dir DIR          Local model dir. Default: model/Qwen3-0.6B
  --max-stocks N           Number of SSE50 stocks to use. Default: 20
  --max-tasks N            Max RL tasks. Default: 800
  --sample-stride N        Sampling stride. Default: 8
  --no-setup               Skip environment setup
  --no-download-model      Skip model download in setup
  --no-cn-mirror           Do not prefer China mirrors
  -h, --help               Show this help

Examples:
  bash run_3090_small.sh "your_tushare_token"
  MODEL_ID=Qwen/Qwen3-1.7B MODEL_DIR=model/Qwen3-1.7B bash run_3090_small.sh "your_tushare_token"
  bash run_3090_small.sh "your_tushare_token" --no-setup --no-download-model
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/output_3090_small.log"
if [[ -z "${STOCK_AGENT_3090_LOGGING_STARTED:-}" ]]; then
  : > "$LOG_FILE"
  export STOCK_AGENT_3090_LOGGING_STARTED=1
  bash "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
echo "== Logging to $LOG_FILE =="

TOKEN="${TUSHARE_TOKEN:-}"
ENV_MODE="${ENV_MODE:-auto}"
ENV_NAME="${ENV_NAME:-stockverl}"
PYTHON_BIN="${PYTHON_BIN:-}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-stockverl.txt}"
DEPENDENCY_POLICY="${DEPENDENCY_POLICY:-compatible}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
MODEL_DIR="${MODEL_DIR:-model/Qwen3-0.6B}"
MAX_STOCKS="${MAX_STOCKS:-20}"
MAX_TASKS="${MAX_TASKS:-800}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-8}"
RUN_SETUP=1
DOWNLOAD_MODEL=1
USE_CN_MIRROR=1
EXTRA_ARGS=()
PASSTHROUGH_STARTED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --env-mode)
      ENV_MODE="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="$2"
      shift 2
      ;;
    --dependency-policy)
      DEPENDENCY_POLICY="$2"
      shift 2
      ;;
    --torch-cuda)
      TORCH_CUDA="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --max-stocks)
      MAX_STOCKS="$2"
      shift 2
      ;;
    --max-tasks)
      MAX_TASKS="$2"
      shift 2
      ;;
    --sample-stride)
      SAMPLE_STRIDE="$2"
      shift 2
      ;;
    --no-setup)
      RUN_SETUP=0
      shift
      ;;
    --no-download-model)
      DOWNLOAD_MODEL=0
      shift
      ;;
    --no-cn-mirror)
      USE_CN_MIRROR=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    --*)
      PASSTHROUGH_STARTED=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ "$PASSTHROUGH_STARTED" == "0" && -z "$TOKEN" ]]; then
        TOKEN="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

echo "== Single-3090 full-flow smoke test =="
echo "Repo: $REPO_ROOT"
echo "Env mode: $ENV_MODE"
echo "Model id: $MODEL_ID"
echo "Model dir: $MODEL_DIR"
echo "Data smoke size: max_stocks=$MAX_STOCKS, max_tasks=$MAX_TASKS, sample_stride=$SAMPLE_STRIDE"
if [[ -z "$TOKEN" ]]; then
  echo "Tushare token: not passed; stock_agent_rl_mvp.py will try local_config.py"
else
  echo "Tushare token: passed by argument/env"
fi
echo

if [[ "$RUN_SETUP" == "1" ]]; then
  SETUP_ARGS=(
    --env-mode "$ENV_MODE"
    --env-name "$ENV_NAME"
    --requirements "$REQUIREMENTS_FILE"
    --dependency-policy "$DEPENDENCY_POLICY"
    --torch-cuda "$TORCH_CUDA"
  )
  if [[ -n "$PYTHON_BIN" ]]; then
    SETUP_ARGS+=(--python-bin "$PYTHON_BIN")
  fi
  if [[ "$USE_CN_MIRROR" == "1" ]]; then
    SETUP_ARGS+=(--cn-mirror)
  fi
  if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
    SETUP_ARGS+=(--download-model --model-id "$MODEL_ID" --model-dir "$MODEL_DIR")
  fi
  bash setup_stockverl_env.sh "${SETUP_ARGS[@]}"
fi

RUN_ARGS=(
  --env-mode "$ENV_MODE"
  --env-name "$ENV_NAME"
  --requirements "$REQUIREMENTS_FILE"
  --profile debug
  --no-setup
  --no-download-model
)
if [[ -n "$TOKEN" ]]; then
  RUN_ARGS=("$TOKEN" "${RUN_ARGS[@]}")
fi
if [[ -n "$PYTHON_BIN" ]]; then
  RUN_ARGS+=(--python-bin "$PYTHON_BIN")
fi
if [[ "$USE_CN_MIRROR" == "0" ]]; then
  RUN_ARGS+=(--no-cn-mirror)
fi

bash run_stock_agent_rl.sh "${RUN_ARGS[@]}" -- \
  --model-path "$MODEL_DIR" \
  --max-stocks "$MAX_STOCKS" \
  --max-tasks "$MAX_TASKS" \
  --sample-stride "$SAMPLE_STRIDE" \
  --no-fetch-fundamentals \
  --train-batch-size 4 \
  --ppo-mini-batch-size 4 \
  --ppo-micro-batch-size-per-gpu 1 \
  --log-prob-micro-batch-size-per-gpu 1 \
  --rollout-n 2 \
  --max-prompt-length 1536 \
  --max-response-length 384 \
  --rollout-gpu-memory-utilization 0.45 \
  --rollout-max-model-len 2048 \
  --rollout-max-num-batched-tokens 4096 \
  --rollout-agent-num-workers 1 \
  --reward-num-workers 1 \
  --total-epochs 1 \
  --actor-fsdp-param-offload \
  --actor-fsdp-optimizer-offload \
  --ref-fsdp-param-offload \
  "${EXTRA_ARGS[@]}"
