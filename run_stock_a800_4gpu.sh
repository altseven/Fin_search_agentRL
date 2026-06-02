#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_stock_a800_4gpu.sh [TUSHARE_TOKEN] [options] [-- extra stock_agent_rl_mvp.py args]

Four-A800 smoke/full-flow entry for Qwen3-4B GRPO training.

Default behavior:
  - use setup_stockverl_env.sh in compatible mode
  - keep env/cache/model under the persistent parent directory of this repo
  - prefer China mirrors for pip/PyTorch/model downloads
  - install flash-attn, because current verl training imports flash_attn.bert_padding
  - download Qwen/Qwen3-4B to <persist-root>/model/Qwen3-4B
  - require at least 4 GPUs with 70GB+ memory each
  - run full-flow data build, rule baseline, verl GRPO training, and report generation

Options:
  --env-mode MODE          system only. Kept for compatibility. Default: system
  --env-name NAME          Persistent venv name. Default: stockverl
  --python-bin PATH        Python executable. Default: <persist-root>/venvs/<env-name>/bin/python
  --requirements PATH      Requirements file. Default: requirements-stockverl.txt
  --dependency-policy P    compatible|strict. Default: compatible
  --torch-cuda FLAVOR      cu121|cu124|cu126|cu128|cpu. Default: cu128
  --model-id ID            ModelScope/HuggingFace model id. Default: Qwen/Qwen3-4B
  --model-dir DIR          Local model dir. Default: <persist-root>/model/Qwen3-4B
  --persist-root DIR       Persistent root. Default: parent of repo, e.g. /kunlun_data/temp_ag_rl
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
DEFAULT_PERSIST_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
A800_PROFILE_LABEL="${STOCK_AGENT_A800_PROFILE_LABEL:-A800 4-GPU}"
A800_LOG_NAME="${STOCK_AGENT_A800_LOG_NAME:-output_a800_4gpu.log}"
A800_REQUIRED_GPUS="${STOCK_AGENT_A800_REQUIRED_GPUS:-4}"
A800_N_GPUS="${STOCK_AGENT_A800_N_GPUS:-4}"
A800_ROLLOUT_TP="${STOCK_AGENT_A800_ROLLOUT_TP:-2}"
A800_TRAIN_BATCH_SIZE="${STOCK_AGENT_A800_TRAIN_BATCH_SIZE:-32}"
A800_PPO_MINI_BATCH_SIZE="${STOCK_AGENT_A800_PPO_MINI_BATCH_SIZE:-16}"
A800_PPO_MICRO_BATCH_SIZE_PER_GPU="${STOCK_AGENT_A800_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${STOCK_AGENT_A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
A800_ROLLOUT_N="${STOCK_AGENT_A800_ROLLOUT_N:-4}"
A800_MAX_PROMPT_LENGTH="${STOCK_AGENT_A800_MAX_PROMPT_LENGTH:-4096}"
A800_MAX_RESPONSE_LENGTH="${STOCK_AGENT_A800_MAX_RESPONSE_LENGTH:-1024}"
A800_ROLLOUT_GPU_MEMORY_UTILIZATION="${STOCK_AGENT_A800_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.72}"
A800_ROLLOUT_MAX_MODEL_LEN="${STOCK_AGENT_A800_ROLLOUT_MAX_MODEL_LEN:-5120}"
A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${STOCK_AGENT_A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
A800_ROLLOUT_AGENT_NUM_WORKERS="${STOCK_AGENT_A800_ROLLOUT_AGENT_NUM_WORKERS:-8}"
A800_REWARD_NUM_WORKERS="${STOCK_AGENT_A800_REWARD_NUM_WORKERS:-8}"
A800_TOTAL_EPOCHS="${STOCK_AGENT_A800_TOTAL_EPOCHS:-2}"
A800_SAMPLE_STRIDE="${STOCK_AGENT_A800_SAMPLE_STRIDE:-2}"
A800_MAX_TASKS="${STOCK_AGENT_A800_MAX_TASKS:-8000}"

LOG_FILE="$REPO_ROOT/$A800_LOG_NAME"
if [[ -z "${STOCK_AGENT_A800_4GPU_LOGGING_STARTED:-}" ]]; then
  : > "$LOG_FILE"
  export STOCK_AGENT_A800_4GPU_LOGGING_STARTED=1
  bash "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
echo "== Logging to $LOG_FILE =="

TOKEN="${TUSHARE_TOKEN:-}"
PERSIST_ROOT="${STOCK_AGENT_PERSIST_ROOT:-$DEFAULT_PERSIST_ROOT}"
ENV_MODE="${ENV_MODE:-system}"
ENV_NAME="${ENV_NAME:-stockverl}"
PYTHON_BIN="${PYTHON_BIN:-}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-stockverl.txt}"
DEPENDENCY_POLICY="${DEPENDENCY_POLICY:-compatible}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-$PERSIST_ROOT/model/Qwen3-4B}"
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
      MODEL_DIR_SET_BY_USER=1
      shift 2
      ;;
    --persist-root)
      PERSIST_ROOT="$2"
      if [[ -z "${MODEL_DIR_SET_BY_USER:-}" ]]; then
        MODEL_DIR="$PERSIST_ROOT/model/Qwen3-4B"
      fi
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

PERSIST_ROOT="$(mkdir -p "$PERSIST_ROOT" && cd "$PERSIST_ROOT" && pwd)"
PERSISTENT_VENV_DIR="${STOCK_AGENT_VENV_DIR:-$PERSIST_ROOT/venvs/$ENV_NAME}"
export CACHE_ROOT="${CACHE_ROOT:-$PERSIST_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
mkdir -p "$CACHE_ROOT" "$PIP_CACHE_DIR" "$HF_HOME" "$MODELSCOPE_CACHE" "$PERSIST_ROOT/model" "$PERSIST_ROOT/venvs"

case "$ENV_MODE" in
  system|auto) ;;
  *)
    echo "--env-mode must be system for run_stock_a800_4gpu.sh. This server workflow does not use conda." >&2
    exit 2
    ;;
esac
ENV_MODE="system"

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

echo "== $A800_PROFILE_LABEL stock agent RL run =="
echo "Repo: $REPO_ROOT"
echo "Persistent root: $PERSIST_ROOT"
echo "Env mode: $ENV_MODE"
echo "Persistent venv name: $ENV_NAME"
echo "Persistent venv: $PERSISTENT_VENV_DIR"
echo "Cache root: $CACHE_ROOT"
echo "Model id: $MODEL_ID"
echo "Model dir: $MODEL_DIR"
echo "Install flash-attn: $INSTALL_FLASH_ATTN"
if [[ -z "$TOKEN" ]]; then
  echo "Tushare token: not passed; stock_agent_rl_mvp.py will try local_config.py"
else
  echo "Tushare token: passed by argument/env"
fi
echo

find_base_python_bin() {
  for py in python3.10 python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      command -v "$py"
      return 0
    fi
  done
  return 1
}

install_python_venv_support() {
  local base_python="$1"
  if ! command -v apt-get >/dev/null 2>&1 || [[ "$(id -u)" != "0" ]]; then
    return 1
  fi

  local py_mm
  py_mm="$("$base_python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  echo "== Installing Python venv/pip support via apt-get =="
  echo "Python version for venv: $py_mm"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  if apt-get install -y "python${py_mm}-venv" python3-venv python3-pip; then
    return 0
  fi
  echo "WARNING: version-specific venv package install failed; trying generic packages." >&2
  apt-get install -y python3-venv python3-pip
}

venv_is_usable() {
  [[ -x "$PERSISTENT_VENV_DIR/bin/python" ]] && "$PERSISTENT_VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1
}

create_persistent_venv() {
  local base_python="$1"
  echo "== Creating persistent venv: $PERSISTENT_VENV_DIR =="
  if "$base_python" -m venv "$PERSISTENT_VENV_DIR" && venv_is_usable; then
    return 0
  fi

  echo "WARNING: Python venv creation failed or pip is missing; trying to install venv support." >&2
  rm -rf "$PERSISTENT_VENV_DIR"
  if install_python_venv_support "$base_python"; then
    "$base_python" -m venv "$PERSISTENT_VENV_DIR"
  else
    cat >&2 <<EOF
Could not create a persistent venv because this image lacks ensurepip/python-venv support.

Automatic apt-get installation was not available. Use an image with python venv support, or install:
  apt-get update && apt-get install -y python3-venv python3-pip

Then rerun:
  bash run_stock_a800_4gpu.sh "YOUR_TOKEN"
EOF
    return 1
  fi

  if ! venv_is_usable; then
    echo "Persistent venv was created but pip is still unavailable: $PERSISTENT_VENV_DIR" >&2
    return 1
  fi
}

prepare_persistent_python_mode() {
  if [[ -z "$PYTHON_BIN" ]]; then
    if venv_is_usable; then
      echo "== Reusing persistent venv: $PERSISTENT_VENV_DIR =="
    else
      local base_python
      base_python="$(find_base_python_bin)" || {
        echo "No usable Python found for creating persistent venv." >&2
        return 1
      }
      if [[ -d "$PERSISTENT_VENV_DIR" ]]; then
        echo "== Removing broken persistent venv: $PERSISTENT_VENV_DIR =="
        rm -rf "$PERSISTENT_VENV_DIR"
      fi
      create_persistent_venv "$base_python"
    fi
    PYTHON_BIN="$PERSISTENT_VENV_DIR/bin/python"
  fi
  ENV_MODE="system"
}

prepare_persistent_python_mode

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

PYTHON_CMD="$(find_python_bin)"

PYTHON_EXE="$("$PYTHON_CMD" -c 'import sys; print(sys.executable)')"

if [[ -n "${STOCK_AGENT_REQUIRE_PYTHON_MINOR:-}" ]]; then
  PYTHON_MINOR="$("$PYTHON_CMD" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "$PYTHON_MINOR" != "$STOCK_AGENT_REQUIRE_PYTHON_MINOR" ]]; then
    cat >&2 <<EOF
This profile requires Python $STOCK_AGENT_REQUIRE_PYTHON_MINOR, but the selected Python is $PYTHON_MINOR:
  $PYTHON_EXE

The A800 single-GPU cloud failures have been coming from the Python 3.12 +
vLLM/Ray stack. Use the script-managed persistent venv, or explicitly pass a
Python 3.10 executable:

  bash run_stock_a800_1gpu.sh "YOUR_TOKEN"
  bash run_stock_a800_1gpu.sh "YOUR_TOKEN" --python-bin "\$(command -v python3.10)"

Do not pass --python-bin "\$(which python)" unless it resolves to Python 3.10.
EOF
    exit 1
  fi
fi

readarray -t PROFILE_LINES < <(
STOCK_AGENT_A800_REQUIRED_GPUS="$A800_REQUIRED_GPUS" \
STOCK_AGENT_A800_N_GPUS="$A800_N_GPUS" \
STOCK_AGENT_A800_ROLLOUT_TP="$A800_ROLLOUT_TP" \
STOCK_AGENT_A800_TRAIN_BATCH_SIZE="$A800_TRAIN_BATCH_SIZE" \
STOCK_AGENT_A800_PPO_MINI_BATCH_SIZE="$A800_PPO_MINI_BATCH_SIZE" \
STOCK_AGENT_A800_PPO_MICRO_BATCH_SIZE_PER_GPU="$A800_PPO_MICRO_BATCH_SIZE_PER_GPU" \
STOCK_AGENT_A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
STOCK_AGENT_A800_ROLLOUT_N="$A800_ROLLOUT_N" \
STOCK_AGENT_A800_MAX_PROMPT_LENGTH="$A800_MAX_PROMPT_LENGTH" \
STOCK_AGENT_A800_MAX_RESPONSE_LENGTH="$A800_MAX_RESPONSE_LENGTH" \
STOCK_AGENT_A800_ROLLOUT_GPU_MEMORY_UTILIZATION="$A800_ROLLOUT_GPU_MEMORY_UTILIZATION" \
STOCK_AGENT_A800_ROLLOUT_MAX_MODEL_LEN="$A800_ROLLOUT_MAX_MODEL_LEN" \
STOCK_AGENT_A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS="$A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
STOCK_AGENT_A800_ROLLOUT_AGENT_NUM_WORKERS="$A800_ROLLOUT_AGENT_NUM_WORKERS" \
STOCK_AGENT_A800_REWARD_NUM_WORKERS="$A800_REWARD_NUM_WORKERS" \
STOCK_AGENT_A800_TOTAL_EPOCHS="$A800_TOTAL_EPOCHS" \
"$PYTHON_CMD" - <<'PY'
import os
import torch

required = int(os.environ["STOCK_AGENT_A800_REQUIRED_GPUS"])
count = torch.cuda.device_count()
if count < required:
    raise SystemExit(f"A800 profile expects at least {required} CUDA GPUs; detected {count}.")

mem_gb = [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(count)]
min_mem = min(mem_gb[:required])
if min_mem < 70:
    raise SystemExit(f"A800 profile expects first {required} GPUs to have at least 70GB; min memory is {min_mem:.1f}GB.")

print(f"GPU_COUNT={count}")
print(f"MIN_GPU_MEM_GB={min_mem:.1f}")
mapping = {
    "N_GPUS": "STOCK_AGENT_A800_N_GPUS",
    "ROLLOUT_TP": "STOCK_AGENT_A800_ROLLOUT_TP",
    "TRAIN_BATCH_SIZE": "STOCK_AGENT_A800_TRAIN_BATCH_SIZE",
    "PPO_MINI_BATCH_SIZE": "STOCK_AGENT_A800_PPO_MINI_BATCH_SIZE",
    "PPO_MICRO_BATCH_SIZE_PER_GPU": "STOCK_AGENT_A800_PPO_MICRO_BATCH_SIZE_PER_GPU",
    "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": "STOCK_AGENT_A800_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
    "ROLLOUT_N": "STOCK_AGENT_A800_ROLLOUT_N",
    "MAX_PROMPT_LENGTH": "STOCK_AGENT_A800_MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH": "STOCK_AGENT_A800_MAX_RESPONSE_LENGTH",
    "ROLLOUT_GPU_MEMORY_UTILIZATION": "STOCK_AGENT_A800_ROLLOUT_GPU_MEMORY_UTILIZATION",
    "ROLLOUT_MAX_MODEL_LEN": "STOCK_AGENT_A800_ROLLOUT_MAX_MODEL_LEN",
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "STOCK_AGENT_A800_ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    "ROLLOUT_AGENT_NUM_WORKERS": "STOCK_AGENT_A800_ROLLOUT_AGENT_NUM_WORKERS",
    "REWARD_NUM_WORKERS": "STOCK_AGENT_A800_REWARD_NUM_WORKERS",
    "TOTAL_EPOCHS": "STOCK_AGENT_A800_TOTAL_EPOCHS",
}
for shell_name, env_name in mapping.items():
    print(f"{shell_name}={os.environ[env_name]}")
PY
)

for line in "${PROFILE_LINES[@]}"; do
  eval "$line"
done

if ! "$PYTHON_CMD" - <<'PY'
import flash_attn
from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
print("flash_attn.bert_padding: ok")
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

if ! "$PYTHON_CMD" - <<'PY'
import vllm
from vllm import LLM  # noqa: F401

print("vllm:", getattr(vllm, "__version__", "unknown"))
print("vllm.LLM import: ok")
PY
then
  cat >&2 <<EOF
vLLM is installed but its compiled CUDA extension cannot be loaded.

Your latest log showed:
  ImportError: libcudart.so.13: cannot open shared object file

That means the active vLLM wheel expects a CUDA 13 runtime, while this
environment is using a CUDA 12.x/PyTorch cu12x stack. The 3090-small run worked
because it used a different conda environment with matching vLLM/Torch/CUDA.

Fix:
  bash run_stock_a800_4gpu.sh "YOUR_TOKEN" --python-bin "$PYTHON_CMD" --no-install-flash-attn

Do not use --no-setup until this preflight prints "vllm.LLM import: ok".
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

echo "== Stock agent RL training: $A800_PROFILE_LABEL profile =="
echo "Env: persistent-venv"
echo "Python: $PYTHON_EXE"
echo "Detected GPUs: $GPU_COUNT, first-${A800_REQUIRED_GPUS} min memory: ${MIN_GPU_MEM_GB}GB"
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
  --sample-stride "$A800_SAMPLE_STRIDE" \
  --max-tasks "$A800_MAX_TASKS" \
  --max-tool-calls 4 \
  --up-quantile 0.70 \
  --down-quantile 0.30 \
  --pnl-scale 0.03 \
  --lora-rank 0 \
  --lora-alpha 0 \
  --actor-lr 1e-6 \
  --no-use-remove-padding \
  "${EXTRA_ARGS[@]}"
