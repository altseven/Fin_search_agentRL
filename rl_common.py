from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - import guard for clear runtime errors
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None


DEFAULT_TUSHARE_HTTP_URL = "http://lianghua.nanyangqiankun.top"
DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "model"
DEFAULT_RESULT_DIR = "result"
DEFAULT_INDEX_CODE = "000016.SH"  # SSE 50
DEFAULT_MARKET_INDEX = "000016.SH"
DEFAULT_HF_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_MODEL_PATH = "model/Qwen3-4B"
DATA_SOURCE = "stock_agent_rl_mvp"
CLASSES = ["up", "neutral", "down"]


def require_pandas() -> None:
    if pd is None:
        raise RuntimeError(f"pandas is required but failed to import: {_PANDAS_IMPORT_ERROR}")


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def load_local_config_value(name: str) -> Any:
    path = Path(__file__).resolve().parent / "local_config.py"
    if not path.exists():
        return None
    namespace: dict[str, Any] = {}
    try:
        exec(path.read_text(encoding="utf-8"), namespace)
    except Exception as exc:
        raise RuntimeError(f"Failed to read local_config.py: {exc}") from exc
    return namespace.get(name)


def normalize_date(date: str | int) -> str:
    s = str(date).strip()
    s = re.sub(r"[^0-9]", "", s)
    if len(s) != 8:
        raise ValueError(f"Expected YYYYMMDD-like date, got {date!r}")
    return s


def normalize_ts_code(ts_code: str) -> str:
    code = str(ts_code).strip().upper()
    if re.match(r"^[0-9]{6}\.(SH|SZ|BJ)$", code):
        return code
    digits = re.sub(r"[^0-9]", "", code)
    if len(digits) != 6:
        raise ValueError(f"Cannot normalize stock code: {ts_code!r}")
    if digits.startswith(("6", "9")):
        return f"{digits}.SH"
    if digits.startswith("8"):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def make_timestamp_run_dir(result_base: Path) -> Path:
    stem = time.strftime("%H%M%S_%m%d_result")
    candidate = result_base / stem
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        alt = result_base / f"{stem}_{i}"
        if not alt.exists():
            return alt
    raise RuntimeError(f"Could not allocate result run directory under {result_base}")


def ensure_dirs(cfg: Any) -> dict[str, Path]:
    data_base = Path(cfg.data_dir).expanduser().resolve()
    model_base = Path(cfg.model_dir).expanduser().resolve()
    result_base = Path(cfg.result_dir).expanduser().resolve()
    result_base.mkdir(parents=True, exist_ok=True)
    if cfg.run_dir:
        run_base = Path(cfg.run_dir).expanduser().resolve()
    else:
        run_base = make_timestamp_run_dir(result_base).resolve()
        cfg.run_dir = str(run_base)
    dirs = {
        "data": data_base,
        "raw": data_base / "raw",
        "processed": data_base / "processed",
        "model": model_base,
        "result": result_base,
        "run": run_base,
        "verl": run_base / "verl",
        "rollouts": run_base / "rollouts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def table_path(stem: Path, suffix: str) -> Path:
    return stem.with_suffix(suffix)


def save_table(df: "pd.DataFrame", stem: Path) -> None:
    require_pandas()
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(table_path(stem, ".pkl"))
    df.to_csv(table_path(stem, ".csv"), index=False)
    try:
        df.to_parquet(table_path(stem, ".parquet"), index=False)
    except Exception as exc:
        log(f"Parquet write skipped for {stem.name}: {exc}. Pickle/CSV are available.")


def read_table(stem: Path) -> "pd.DataFrame":
    require_pandas()
    for suffix, reader in (
        (".parquet", pd.read_parquet),
        (".pkl", pd.read_pickle),
        (".csv", pd.read_csv),
    ):
        path = table_path(stem, suffix)
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f"No cached table found for {stem}")


def maybe_read_table(stem: Path) -> "pd.DataFrame | None":
    try:
        return read_table(stem)
    except FileNotFoundError:
        return None
