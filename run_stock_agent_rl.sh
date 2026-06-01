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
  --env-mode MODE      auto|conda|system. Default: auto
  --env-name NAME       Conda env name. Default: stockverl
  --python-bin PATH     Python executable for system mode. Default: python3.10/python3/python
  --requirements PATH   Locked requirements file. Default: requirements-stockverl.txt
  --profile PROFILE     auto|a800_8gpu|debug. Default: auto
  --no-setup            Skip environment setup and model download
  --no-cn-mirror        Do not prefer China mirrors during setup
  --no-download-model   Do not download Qwen3-4B during setup
  -h, --help            Show this help

Examples:
  bash run_stock_agent_rl.sh "your_token"
  bash run_stock_agent_rl.sh "your_token" --profile a800_8gpu
  bash run_stock_agent_rl.sh "your_token" --no-setup
  bash run_stock_agent_rl.sh "your_token" -- --total-epochs 2
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_FILE="$REPO_ROOT/output.log"
if [[ -z "${STOCK_AGENT_LOGGING_STARTED:-}" ]]; then
  : > "$LOG_FILE"
  export STOCK_AGENT_LOGGING_STARTED=1
  bash "$0" "$@" 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
echo "== Logging to $LOG_FILE =="

ENV_NAME="${ENV_NAME:-stockverl}"
ENV_MODE="${ENV_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-stockverl.txt}"
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
  auto|a800_8gpu|debug) ;;
  *)
    echo "--profile must be one of: auto, a800_8gpu, debug" >&2
    exit 2
    ;;
esac

case "$ENV_MODE" in
  auto|conda|system) ;;
  *)
    echo "--env-mode must be one of: auto, conda, system" >&2
    exit 2
    ;;
esac

if [[ "$RUN_SETUP" == "1" ]]; then
  SETUP_ARGS=(--env-mode "$ENV_MODE" --env-name "$ENV_NAME" --torch-cuda cu128 --requirements "$REQUIREMENTS_FILE")
  if [[ -n "$PYTHON_BIN" ]]; then
    SETUP_ARGS+=(--python-bin "$PYTHON_BIN")
  fi
  if [[ "$USE_CN_MIRROR" == "1" ]]; then
    SETUP_ARGS+=(--cn-mirror)
  fi
  if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
    SETUP_ARGS+=(--download-model)
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

readarray -t PROFILE_LINES < <(PROFILE="$PROFILE" "$PYTHON_CMD" - <<'PY'
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
    ppo_micro,
    logprob_micro,
    rollout_n,
    max_prompt,
    max_response,
    gpu_mem_util,
    rollout_max_model_len,
    rollout_max_num_batched_tokens,
    agent_workers,
    reward_workers,
    total_epochs,
):
    print(f"PROFILE_NAME={name}")
    print(f"GPU_COUNT={count}")
    print(f"MIN_GPU_MEM_GB={min_mem:.1f}")
    print(f"N_GPUS={n_gpus}")
    print(f"ROLLOUT_TP={rollout_tp}")
    print(f"TRAIN_BATCH_SIZE={train_batch}")
    print(f"PPO_MINI_BATCH_SIZE={ppo_mini}")
    print(f"PPO_MICRO_BATCH_SIZE_PER_GPU={ppo_micro}")
    print(f"LOG_PROB_MICRO_BATCH_SIZE_PER_GPU={logprob_micro}")
    print(f"ROLLOUT_N={rollout_n}")
    print(f"MAX_PROMPT_LENGTH={max_prompt}")
    print(f"MAX_RESPONSE_LENGTH={max_response}")
    print(f"ROLLOUT_GPU_MEMORY_UTILIZATION={gpu_mem_util}")
    print(f"ROLLOUT_MAX_MODEL_LEN={rollout_max_model_len}")
    print(f"ROLLOUT_MAX_NUM_BATCHED_TOKENS={rollout_max_num_batched_tokens}")
    print(f"ROLLOUT_AGENT_NUM_WORKERS={agent_workers}")
    print(f"REWARD_NUM_WORKERS={reward_workers}")
    print(f"TOTAL_EPOCHS={total_epochs}")

def emit_a800_profile(name: str) -> None:
    emit(
        name=name,
        n_gpus=8,
        rollout_tp=4,
        train_batch=96,
        ppo_mini=48,
        ppo_micro=4,
        logprob_micro=4,
        rollout_n=6,
        max_prompt=4096,
        max_response=1536,
        gpu_mem_util=0.78,
        rollout_max_model_len=5632,
        rollout_max_num_batched_tokens=24576,
        agent_workers=16,
        reward_workers=16,
        total_epochs=3,
    )

if profile in {"auto", "a800_8gpu"}:
    if count < 8 or min_mem < 70:
        raise SystemExit(
            f"Profile {profile!r} expects 8 GPUs with at least 70GB each; "
            f"detected {count} GPU(s), min memory {min_mem:.1f}GB. "
            "Use --profile debug only for a quick non-production smoke test."
        )
    emit_a800_profile("a800_8gpu")
else:
    emit(
        name="debug",
        n_gpus=1,
        rollout_tp=1,
        train_batch=8,
        ppo_mini=8,
        ppo_micro=1,
        logprob_micro=1,
        rollout_n=2,
        max_prompt=2048,
        max_response=512,
        gpu_mem_util=0.50,
        rollout_max_model_len=2560,
        rollout_max_num_batched_tokens=8192,
        agent_workers=1,
        reward_workers=1,
        total_epochs=1,
    )
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
if [[ "$USE_CONDA" == "1" ]]; then
  echo "Env: conda:$ENV_NAME"
else
  echo "Env: system-python"
fi
echo "Python: $PYTHON_EXE"
echo "Selected profile: $PROFILE_NAME"
echo "Detected GPUs: $GPU_COUNT, min memory: ${MIN_GPU_MEM_GB}GB"
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
  --max-tasks 15000 \
  --max-tool-calls 4 \
  --up-quantile 0.70 \
  --down-quantile 0.30 \
  --pnl-scale 0.03 \
  --lora-rank 0 \
  --lora-alpha 0 \
  --actor-lr 1e-6 \
  "${EXTRA_ARGS[@]}"
