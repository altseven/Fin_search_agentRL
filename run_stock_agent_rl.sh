#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_stock_agent_rl.sh [TUSHARE_TOKEN] [options] [-- extra stock_agent_rl_mvp.py args]

One-command run after git clone:
  bash run_stock_agent_rl.sh "your_tushare_token"

Token alternatives:
  TUSHARE_TOKEN="your_token" bash run_stock_agent_rl.sh
  bash run_stock_agent_rl.sh
    The last form requires local_config.py to contain tushare_token.

Options:
  --env-name NAME       Conda env name. Default: stockverl
  --profile PROFILE     auto|tiny|conservative|large. Default: auto
  --no-setup            Skip environment setup and model download
  --no-cn-mirror        Do not use the Tsinghua PyPI mirror during setup
  --no-download-model   Do not download Qwen3-4B during setup
  -h, --help            Show this help

Examples:
  bash run_stock_agent_rl.sh "your_token"
  bash run_stock_agent_rl.sh "your_token" --profile conservative
  bash run_stock_agent_rl.sh "your_token" --no-setup
  bash run_stock_agent_rl.sh "your_token" -- --total-epochs 2
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-stockverl}"
PROFILE="auto"
RUN_SETUP=1
USE_CN_MIRROR=1
DOWNLOAD_MODEL=1
TOKEN="${TUSHARE_TOKEN:-}"
EXTRA_ARGS=()
PASSTHROUGH_STARTED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
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

case "$PROFILE" in
  auto|tiny|conservative|large) ;;
  *)
    echo "--profile must be one of: auto, tiny, conservative, large" >&2
    exit 2
    ;;
esac

if [[ "$RUN_SETUP" == "1" ]]; then
  SETUP_ARGS=(--env-name "$ENV_NAME")
  if [[ "$USE_CN_MIRROR" == "1" ]]; then
    SETUP_ARGS+=(--cn-mirror)
  fi
  if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
    SETUP_ARGS+=(--download-model)
  fi
  bash setup_stockverl_env.sh "${SETUP_ARGS[@]}"
fi

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      set +u
      source "$conda_sh"
      set -u
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda or use a GPU image with conda." >&2
  exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

readarray -t PROFILE_LINES < <(PROFILE="$PROFILE" python - <<'PY'
import os

profile = os.environ["PROFILE"]

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Could not import torch after setup: {exc}")

count = torch.cuda.device_count()
if count <= 0:
    raise SystemExit("No CUDA GPU detected by torch.")

mem_gb = [
    torch.cuda.get_device_properties(i).total_memory / 1024**3
    for i in range(count)
]
min_mem = min(mem_gb)

def emit(
    name,
    n_gpus,
    rollout_tp,
    train_batch,
    ppo_mini,
    rollout_n,
    max_prompt,
    max_response,
    gpu_mem_util,
):
    print(f"PROFILE_NAME={name}")
    print(f"GPU_COUNT={count}")
    print(f"MIN_GPU_MEM_GB={min_mem:.1f}")
    print(f"N_GPUS={n_gpus}")
    print(f"ROLLOUT_TP={rollout_tp}")
    print(f"TRAIN_BATCH_SIZE={train_batch}")
    print(f"PPO_MINI_BATCH_SIZE={ppo_mini}")
    print(f"ROLLOUT_N={rollout_n}")
    print(f"MAX_PROMPT_LENGTH={max_prompt}")
    print(f"MAX_RESPONSE_LENGTH={max_response}")
    print(f"ROLLOUT_GPU_MEMORY_UTILIZATION={gpu_mem_util}")

if profile == "tiny":
    emit("tiny", 1, 1, 1, 1, 1, 1024, 256, 0.20)
elif profile == "conservative":
    emit("conservative", 1, 1, 2, 2, 1, 1536, 512, 0.35)
elif profile == "large":
    n = count
    train = min(max(n * 4, 4), 16)
    emit("large", n, min(n, 2), train, min(train, 8), 2, 2048, 1024, 0.45)
elif min_mem >= 70:
    n = count
    train = min(max(n * 4, 4), 16)
    emit("auto_large_70gb", n, min(n, 2), train, min(train, 8), 2, 2048, 1024, 0.45)
elif min_mem >= 39:
    emit("auto_conservative_40gb", 1, 1, 2, 2, 1, 1536, 512, 0.35)
else:
    emit("auto_tiny_lt40gb", 1, 1, 1, 1, 1, 1024, 256, 0.20)
PY
)

for line in "${PROFILE_LINES[@]}"; do
  eval "$line"
done

TOKEN_ARGS=()
if [[ -n "$TOKEN" ]]; then
  TOKEN_ARGS=(--tushare-token "$TOKEN")
fi

AUTO_MODEL_ARGS=()
if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  AUTO_MODEL_ARGS=(--no-auto-download-model)
fi

echo "== Stock agent RL training =="
echo "Repo: $REPO_ROOT"
echo "Env: $ENV_NAME"
echo "Selected profile: $PROFILE_NAME"
echo "Detected GPUs: $GPU_COUNT, min memory: ${MIN_GPU_MEM_GB}GB"
echo "Training GPUs: $N_GPUS, rollout TP: $ROLLOUT_TP"
echo "Batch: train=$TRAIN_BATCH_SIZE, ppo_mini=$PPO_MINI_BATCH_SIZE, rollout_n=$ROLLOUT_N"
echo "Prompt/response: $MAX_PROMPT_LENGTH/$MAX_RESPONSE_LENGTH"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

ray stop --force || true

python stock_agent_rl_mvp.py \
  --mode all-train \
  "${TOKEN_ARGS[@]}" \
  "${AUTO_MODEL_ARGS[@]}" \
  --n-gpus-per-node "$N_GPUS" \
  --rollout-tp "$ROLLOUT_TP" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" \
  --ppo-micro-batch-size-per-gpu 1 \
  --log-prob-micro-batch-size-per-gpu 1 \
  --rollout-n "$ROLLOUT_N" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-response-length "$MAX_RESPONSE_LENGTH" \
  --rollout-gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  "${EXTRA_ARGS[@]}"
