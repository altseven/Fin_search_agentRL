#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_stock_a800_4gpu.sh [TUSHARE_TOKEN] [options] [-- extra stock_agent_rl_mvp.py args]

Four-A800 smoke/full-flow entry for Qwen3-4B GRPO training.

Default behavior:
  - use setup_stockverl_env.sh in compatible mode
  - prefer China mirrors for pip/PyTorch/model downloads
  - install flash-attn, because current verl training imports flash_attn.bert_padding
  - download Qwen/Qwen3-4B to model/Qwen3-4B
  - require at least 4 GPUs with 70GB+ memory each
  - run full-flow data build, rule baseline, verl GRPO training, and report generation

Options:
  --env-mode MODE          auto|conda|system. Default: auto
  --env-name NAME          Conda env name. Default: stockverl
  --python-bin PATH        Python executable for system mode
  --requirements PATH      Requirements file. Default: requirements-stockverl.txt
  --dependency-policy P    compatible|strict. Default: compatible
  --torch-cuda FLAVOR      cu121|cu124|cu126|cu128|cpu. Default: cu128
  --model-id ID            ModelScope/HuggingFace model id. Default: Qwen/Qwen3-4B
  --model-dir DIR          Local model dir. Default: model/Qwen3-4B
  --no-setup               Skip environment setup and model download
  --no-cn-mirror           Do not prefer China mirrors
  --no-download-model      Do not download Qwen3-4B during setup
  --no-install-flash-attn  Do not install flash-attn during setup
  -h, --help               Show this help

Examples:
  bash run_stock_a800_4gpu.sh "your_tushare_token"
  bash run_stock_a800_4gpu.sh "your_tushare_token" --no-setup --no-download-model
  bash run_stock_a800_4gpu.sh "your_tushare_token" -- --total-epochs 1 --max-tasks 3000
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/output_a800_4gpu.log"
if [[ -z "${STOCK_AGENT_A800_4GPU_LOGGING_STARTED:-}" ]]; then
  : > "$LOG_FILE"
  export STOCK_AGENT_A800_4GPU_LOGGING_STARTED=1
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
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-model/Qwen3-4B}"
RUN_SETUP=1
USE_CN_MIRROR=1
DOWNLOAD_MODEL=1
INSTALL_FLASH_ATTN=1
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
    --no-setup)
      RUN_SETUP=0
      shift
      ;;
    --no-cn-mirror)
      USE_CN_MIRROR=0
      shift
      ;;
    --no-download-model)
      DOWNLOAD_MODEL=0
      shift
      ;;
    --no-install-flash-attn)
      INSTALL_FLASH_ATTN=0
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

case "$ENV_MODE" in
  auto|conda|system) ;;
  *)
    echo "--env-mode must be one of: auto, conda, system" >&2
    exit 2
    ;;
esac

case "$DEPENDENCY_POLICY" in
  compatible|strict) ;;
  *)
    echo "--dependency-policy must be one of: compatible, strict" >&2
    exit 2
    ;;
esac

case "$TORCH_CUDA" in
  cu121|cu124|cu126|cu128|cpu) ;;
  *)
    echo "--torch-cuda must be one of: cu121, cu124, cu126, cu128, cpu" >&2
    exit 2
    ;;
esac

echo "== A800 4-GPU stock agent RL run =="
echo "Repo: $REPO_ROOT"
echo "Env mode: $ENV_MODE"
echo "Conda env: $ENV_NAME"
echo "Model id: $MODEL_ID"
echo "Model dir: $MODEL_DIR"
echo "Install flash-attn: $INSTALL_FLASH_ATTN"
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
  if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
    SETUP_ARGS+=(--install-flash-attn)
  fi
  if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
    SETUP_ARGS+=(--download-model --model-id "$MODEL_ID" --model-dir "$MODEL_DIR")
  fi
  bash setup_stockverl_env.sh "${SETUP_ARGS[@]}"
fi

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      set +u
      source "$conda_sh"
      set -u
      return 0
    fi
  done
  return 1
}

find_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi
    if [[ -x "$PYTHON_BIN" ]]; then
      echo "$PYTHON_BIN"
      return 0
    fi
    echo "--python-bin not found or not executable: $PYTHON_BIN" >&2
    return 1
  fi
  for py in python3.10 python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      command -v "$py"
      return 0
    fi
  done
  echo "No usable Python found. Expected python3.10, python3, or python." >&2
  return 1
}

USE_CONDA=0
if [[ "$ENV_MODE" == "conda" || "$ENV_MODE" == "auto" ]]; then
  if load_conda && command -v conda >/dev/null 2>&1; then
    USE_CONDA=1
  elif [[ "$ENV_MODE" == "conda" ]]; then
    echo "conda not found, but --env-mode conda was requested." >&2
    exit 1
  fi
fi

if [[ "$USE_CONDA" == "1" ]]; then
  set +u
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  set -u
  PYTHON_CMD="python"
else
  PYTHON_CMD="$(find_python_bin)"
fi

PYTHON_EXE="$("$PYTHON_CMD" -c 'import sys; print(sys.executable)')"

readarray -t PROFILE_LINES < <("$PYTHON_CMD" - <<'PY'
import torch

count = torch.cuda.device_count()
if count < 4:
    raise SystemExit(f"A800 4-GPU profile expects at least 4 CUDA GPUs; detected {count}.")

mem_gb = [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(count)]
min_mem = min(mem_gb[:4])
if min_mem < 70:
    raise SystemExit(f"A800 4-GPU profile expects first 4 GPUs to have at least 70GB; min memory is {min_mem:.1f}GB.")

print(f"GPU_COUNT={count}")
print(f"MIN_GPU_MEM_GB={min_mem:.1f}")
print("N_GPUS=4")
print("ROLLOUT_TP=2")
print("TRAIN_BATCH_SIZE=32")
print("PPO_MINI_BATCH_SIZE=16")
print("PPO_MICRO_BATCH_SIZE_PER_GPU=2")
print("LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2")
print("ROLLOUT_N=4")
print("MAX_PROMPT_LENGTH=4096")
print("MAX_RESPONSE_LENGTH=1024")
print("ROLLOUT_GPU_MEMORY_UTILIZATION=0.72")
print("ROLLOUT_MAX_MODEL_LEN=5120")
print("ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384")
print("ROLLOUT_AGENT_NUM_WORKERS=8")
print("REWARD_NUM_WORKERS=8")
print("TOTAL_EPOCHS=2")
PY
)

for line in "${PROFILE_LINES[@]}"; do
  eval "$line"
done

if ! "$PYTHON_CMD" - <<'PY'
import flash_attn
print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
PY
then
  cat >&2 <<'EOF'
flash-attn is not importable in the active environment.

This verl version imports flash_attn.bert_padding during training even when
stock_agent_rl_mvp.py uses --attn-implementation sdpa.

Fix:
  bash run_stock_a800_4gpu.sh "YOUR_TOKEN"

Do not use --no-setup unless flash-attn is already installed.
EOF
  exit 1
fi

TOKEN_ARGS=()
if [[ -n "$TOKEN" ]]; then
  TOKEN_ARGS=(--tushare-token "$TOKEN")
fi

AUTO_MODEL_ARGS=()
if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  AUTO_MODEL_ARGS=(--no-auto-download-model)
fi

echo "== Stock agent RL training: A800 4-GPU profile =="
if [[ "$USE_CONDA" == "1" ]]; then
  echo "Env: conda:$ENV_NAME"
else
  echo "Env: system-python"
fi
echo "Python: $PYTHON_EXE"
echo "Detected GPUs: $GPU_COUNT, first-four min memory: ${MIN_GPU_MEM_GB}GB"
echo "Training GPUs: $N_GPUS, rollout TP: $ROLLOUT_TP"
echo "Batch: train=$TRAIN_BATCH_SIZE, ppo_mini=$PPO_MINI_BATCH_SIZE, ppo_micro_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU, rollout_n=$ROLLOUT_N"
echo "Prompt/response: $MAX_PROMPT_LENGTH/$MAX_RESPONSE_LENGTH, rollout max model len=$ROLLOUT_MAX_MODEL_LEN"
echo "Workers: agent=$ROLLOUT_AGENT_NUM_WORKERS, reward=$REWARD_NUM_WORKERS"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

"$PYTHON_CMD" -m ray.scripts.scripts stop --force || true

"$PYTHON_CMD" stock_agent_rl_mvp.py \
  --mode all-train \
  "${TOKEN_ARGS[@]}" \
  "${AUTO_MODEL_ARGS[@]}" \
  --model-path "$MODEL_DIR" \
  --n-gpus-per-node "$N_GPUS" \
  --rollout-tp "$ROLLOUT_TP" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
  --ppo-micro-batch-size-per-gpu "$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  --log-prob-micro-batch-size-per-gpu "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  --rollout-n "$ROLLOUT_N" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-response-length "$MAX_RESPONSE_LENGTH" \
  --rollout-gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  --rollout-max-model-len "$ROLLOUT_MAX_MODEL_LEN" \
  --rollout-max-num-batched-tokens "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  --rollout-agent-num-workers "$ROLLOUT_AGENT_NUM_WORKERS" \
  --reward-num-workers "$REWARD_NUM_WORKERS" \
  --total-epochs "$TOTAL_EPOCHS" \
  --sample-stride 2 \
  --max-tasks 8000 \
  --max-tool-calls 4 \
  --up-quantile 0.70 \
  --down-quantile 0.30 \
  --pnl-scale 0.03 \
  --lora-rank 0 \
  --lora-alpha 0 \
  --actor-lr 1e-6 \
  --no-use-remove-padding \
  "${EXTRA_ARGS[@]}"
