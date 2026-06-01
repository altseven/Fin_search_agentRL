#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash setup_stockverl_env.sh [options]

Default behavior:
  - use conda env stockverl when conda exists; otherwise use the current Python
  - Python: 3.10
  - install PyTorch automatically if missing, default CUDA wheel: cu128
  - install this project's Python deps plus verl-main with vLLM support
  - do not install flash-attn by default, because stock_agent_rl_mvp.py defaults to sdpa
  - do not download model weights by default

Options:
  --env-mode MODE            auto|conda|system. Default: auto
  --env-name NAME             Conda env name. Default: stockverl
  --python VERSION            Python version. Default: 3.10
  --python-bin PATH           Python executable for system mode. Default: python3.10/python3/python
  --install-torch auto|yes|no PyTorch install policy. Default: auto
  --torch-cuda cu121|cu124|cu126|cu128|cpu
                              PyTorch wheel index flavor. Default: cu128
  --torch-version VERSION     Torch version. Default: 2.9.0
  --requirements PATH         Locked requirements file. Default: requirements-stockverl.txt
  --cn-mirror                 Use Tsinghua PyPI mirror for normal pip packages
  --install-flash-attn        Try to install flash-attn. Not needed for current sdpa default
  --download-model            Download Qwen/Qwen3-4B to model/Qwen3-4B via ModelScope
  --model-id ID               ModelScope model id. Default: Qwen/Qwen3-4B
  --model-dir DIR             Local model dir. Default: model/Qwen3-4B
  -h, --help                  Show this help

Examples:
  bash setup_stockverl_env.sh
  bash setup_stockverl_env.sh --cn-mirror
  bash setup_stockverl_env.sh --download-model
  bash setup_stockverl_env.sh --env-name stockverl_a800 --install-flash-attn
EOF
}

ENV_NAME="${ENV_NAME:-stockverl}"
ENV_MODE="${ENV_MODE:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.0}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements-stockverl.txt}"
USE_CN_MIRROR="${USE_CN_MIRROR:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-model/Qwen3-4B}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="$2"
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

case "$ENV_MODE" in
  auto|conda|system) ;;
  *)
    echo "--env-mode must be one of: auto, conda, system" >&2
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
echo "Env mode: $ENV_MODE"
echo "Conda env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "Install torch: $INSTALL_TORCH ($TORCH_CUDA, torch==$TORCH_VERSION)"
echo "Requirements lock: $REQUIREMENTS_FILE"
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
  for py in "python$PYTHON_VERSION" python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      command -v "$py"
      return 0
    fi
  done
  echo "No usable Python found. Expected python$PYTHON_VERSION, python3, or python." >&2
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
  PYTHON_CMD="python"
else
  PYTHON_CMD="$(find_python_bin)"
  echo "== Using system Python =="
fi

echo "Python executable: $("$PYTHON_CMD" -c 'import sys; print(sys.executable)')"
"$PYTHON_CMD" --version
"$PYTHON_CMD" -m pip --version >/dev/null 2>&1 || "$PYTHON_CMD" -m ensurepip --upgrade || true

PIP_ARGS=()
if [[ "$USE_CN_MIRROR" == "1" ]]; then
  PIP_ARGS=(-i "https://pypi.tuna.tsinghua.edu.cn/simple" --trusted-host "pypi.tuna.tsinghua.edu.cn")
fi

PIP_BREAK_ARGS=()
if [[ "$USE_CONDA" == "0" ]] && "$PYTHON_CMD" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
  PIP_BREAK_ARGS=(--break-system-packages)
fi

pip_install() {
  "$PYTHON_CMD" -m pip install "${PIP_ARGS[@]}" "${PIP_BREAK_ARGS[@]}" "$@"
}

echo "== Upgrading packaging tools =="
pip_install -U pip setuptools wheel packaging ninja

if [[ "$REQUIREMENTS_FILE" = /* ]]; then
  REQUIREMENTS_PATH="$REQUIREMENTS_FILE"
else
  REQUIREMENTS_PATH="$REPO_ROOT/$REQUIREMENTS_FILE"
fi
if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "Requirements lock file not found: $REQUIREMENTS_PATH" >&2
  exit 1
fi

torch_import_ok=0
if TORCH_VERSION="$TORCH_VERSION" "$PYTHON_CMD" - <<'PY'
import os
import sys

try:
    import torch
    print("torch already importable:", torch.__version__, "cuda:", torch.cuda.is_available())
    expected = os.environ["TORCH_VERSION"]
    installed = torch.__version__.split("+", 1)[0]
    if installed != expected:
        print(f"torch version mismatch: expected {expected}, got {installed}")
        raise SystemExit(1)
except Exception as exc:
    print("torch check failed:", exc)
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
  "$PYTHON_CMD" -m pip install "${PIP_BREAK_ARGS[@]}" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"
elif [[ "$INSTALL_TORCH" == "no" ]]; then
  echo "== Skipping PyTorch install by request =="
else
  echo "== PyTorch already exists, keeping it =="
fi

echo "== Installing project runtime packages =="
pip_install -U --upgrade-strategy eager -r "$REQUIREMENTS_PATH"

echo "== Installing verl-main =="
cd "$REPO_ROOT/verl-main"
pip_install -e . --no-deps
cd "$REPO_ROOT"

echo "== Re-applying locked runtime stack =="
pip_install -U --upgrade-strategy eager -r "$REQUIREMENTS_PATH"

echo "== Locked package verification =="
REQUIREMENTS_PATH="$REQUIREMENTS_PATH" "$PYTHON_CMD" - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

path = os.environ["REQUIREMENTS_PATH"]
mismatches = []
for raw in open(path, encoding="utf-8"):
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    req = Requirement(line)
    expected = None
    for spec in req.specifier:
        if spec.operator == "==":
            expected = spec.version
            break
    if expected is None:
        continue
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        mismatches.append(f"{req.name}: missing, expected {expected}")
        continue
    if installed != expected:
        mismatches.append(f"{req.name}: installed {installed}, expected {expected}")
    else:
        print(f"{req.name}: {installed}")
if mismatches:
    raise SystemExit("Locked package mismatch:\n" + "\n".join(mismatches))
PY

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
    MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" "$PYTHON_CMD" - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(os.environ["MODEL_ID"], local_dir=os.environ["MODEL_DIR"])
PY
  fi
fi

echo "== Import and GPU verification =="
"$PYTHON_CMD" - <<'PY'
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
if [[ "$USE_CONDA" == "1" ]]; then
  echo "Activate this environment with:"
  echo "  conda activate $ENV_NAME"
else
  echo "Using system Python; no conda activation is needed."
fi
echo
echo "Typical next commands:"
echo "  bash run_stock_agent_rl.sh \"YOUR_TOKEN\" --no-setup"
echo "  python stock_agent_rl_mvp.py --mode all-train --no-auto-download-model --tushare-token \"YOUR_TOKEN\""
