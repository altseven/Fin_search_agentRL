#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash setup_stockverl_env.sh [options]

Default behavior:
  - create/use conda env: stockverl
  - Python: 3.10
  - install PyTorch automatically if missing, default CUDA wheel: cu124
  - install this project's Python deps plus verl-main with vLLM support
  - do not install flash-attn by default, because stock_agent_rl_mvp.py defaults to sdpa
  - do not download model weights by default

Options:
  --env-name NAME             Conda env name. Default: stockverl
  --python VERSION            Python version. Default: 3.10
  --install-torch auto|yes|no PyTorch install policy. Default: auto
  --torch-cuda cu121|cu124|cu126|cu128|cpu
                              PyTorch wheel index flavor. Default: cu124
  --cn-mirror                 Use Tsinghua PyPI mirror for normal pip packages
  --install-flash-attn        Try to install flash-attn. Not needed for current sdpa default
  --download-model            Download Qwen/Qwen3-4B to model/Qwen3-4B via ModelScope
  --model-id ID               ModelScope model id. Default: Qwen/Qwen3-4B
  --model-dir DIR             Local model dir. Default: model/Qwen3-4B
  -h, --help                  Show this help

Examples:
  bash setup_stockverl_env.sh
  bash setup_stockverl_env.sh --cn-mirror
  bash setup_stockverl_env.sh --torch-cuda cu128 --download-model
  bash setup_stockverl_env.sh --env-name stockverl_a800 --install-flash-attn
EOF
}

ENV_NAME="${ENV_NAME:-stockverl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
USE_CN_MIRROR="${USE_CN_MIRROR:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-model/Qwen3-4B}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --install-torch)
      INSTALL_TORCH="$2"
      shift 2
      ;;
    --torch-cuda)
      TORCH_CUDA="$2"
      shift 2
      ;;
    --cn-mirror)
      USE_CN_MIRROR=1
      shift
      ;;
    --install-flash-attn)
      INSTALL_FLASH_ATTN=1
      shift
      ;;
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$INSTALL_TORCH" in
  auto|yes|no) ;;
  *)
    echo "--install-torch must be one of: auto, yes, no" >&2
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "== Stock agent RL environment setup =="
echo "Repo: $REPO_ROOT"
echo "Conda env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "Install torch: $INSTALL_TORCH ($TORCH_CUDA)"
echo "Use CN mirror: $USE_CN_MIRROR"
echo "Install flash-attn: $INSTALL_FLASH_ATTN"
echo

echo "== Machine summary =="
uname -a || true
df -h "$REPO_ROOT" || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  nvidia-smi topo -m || true
else
  echo "nvidia-smi not found. This script can still install packages, but GPU checks will fail."
fi
echo

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
  echo "conda not found. Please install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

set +u
eval "$(conda shell.bash hook)"
set -u

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "== Using existing conda env: $ENV_NAME =="
else
  echo "== Creating conda env: $ENV_NAME =="
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

set +u
conda activate "$ENV_NAME"
set -u
echo "Python executable: $(command -v python)"
python --version

PIP_ARGS=()
if [[ "$USE_CN_MIRROR" == "1" ]]; then
  PIP_ARGS=(-i "https://pypi.tuna.tsinghua.edu.cn/simple" --trusted-host "pypi.tuna.tsinghua.edu.cn")
fi

pip_install() {
  python -m pip install "${PIP_ARGS[@]}" "$@"
}

echo "== Upgrading packaging tools =="
pip_install -U pip setuptools wheel packaging ninja

torch_import_ok=0
if python - <<'PY'
try:
    import torch
    print("torch already importable:", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(1)
PY
then
  torch_import_ok=1
fi

if [[ "$INSTALL_TORCH" == "yes" || ( "$INSTALL_TORCH" == "auto" && "$torch_import_ok" == "0" ) ]]; then
  echo "== Installing PyTorch ($TORCH_CUDA) =="
  if [[ "$TORCH_CUDA" == "cpu" ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/$TORCH_CUDA"
  fi
  python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
elif [[ "$INSTALL_TORCH" == "no" ]]; then
  echo "== Skipping PyTorch install by request =="
else
  echo "== PyTorch already exists, keeping it =="
fi

echo "== Installing project runtime packages =="
pip_install tushare pandas "pyarrow>=19.0.0" datasets modelscope

echo "== Installing verl-main with vLLM extra =="
cd "$REPO_ROOT/verl-main"
pip_install -e ".[vllm]"
pip_install "nvidia-ml-py>=12.560.30" "fastapi>=0.115.0" "uvicorn>=0.30.0" "TransferQueue==0.1.7"
cd "$REPO_ROOT"

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "== Installing flash-attn =="
  echo "This can take a long time and may fail if CUDA/Torch/compiler versions do not match."
  MAX_JOBS="${MAX_JOBS:-8}" pip_install flash-attn --no-build-isolation
else
  echo "== Skipping flash-attn =="
  echo "Current MVP defaults to --attn-implementation sdpa, so flash-attn is optional."
fi

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  echo "== Downloading model via ModelScope =="
  mkdir -p "$(dirname "$MODEL_DIR")"
  if [[ -d "$MODEL_DIR" && -n "$(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
    echo "Model directory already exists and is non-empty: $MODEL_DIR"
  else
    modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
  fi
fi

echo "== Import and GPU verification =="
python - <<'PY'
import importlib

mods = [
    "torch",
    "ray",
    "vllm",
    "transformers",
    "pandas",
    "pyarrow",
    "datasets",
    "tushare",
    "modelscope",
]

for name in mods:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: {version}")

import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu[{i}]: {props.name}, {props.total_memory / 1024**3:.1f} GB")

import stock_agent_rl_mvp
print("stock_agent_rl_mvp import: ok")
PY

echo
echo "== Done =="
echo "Activate this environment with:"
echo "  conda activate $ENV_NAME"
echo
echo "Typical next commands:"
echo "  python stock_agent_rl_mvp.py --mode all-train --tushare-token \"YOUR_TOKEN\""
echo "  python stock_agent_rl_mvp.py --mode all-train --no-auto-download-model --tushare-token \"YOUR_TOKEN\""
