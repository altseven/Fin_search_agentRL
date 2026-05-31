#!/usr/bin/env python3
"""Single-file MVP for a stock search-agent RLVR project.

This file intentionally keeps the first version small:

1. Pull a small SSE50 universe with Tushare.
2. Build point-in-time price/market factor snapshots.
3. Build T+H relative-return labels.
4. Export a verl-compatible parquet dataset.
5. Provide verl function tools and a custom reward function from this same file.
6. Run a rule-agent baseline locally for smoke testing.

The Tushare token is deliberately not hard-coded. Pass it with:

    export TUSHARE_TOKEN="..."
    python3 stock_agent_rl_mvp.py --mode all

or:

    python3 stock_agent_rl_mvp.py --mode all --tushare-token "..."

You can also call main(tushare_token="...") from Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - import guard for clear runtime errors
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover - this file also runs without verl installed

    def function_tool(arg: Any = None, **_: Any) -> Callable:
        if callable(arg):
            return arg

        def deco(fn: Callable) -> Callable:
            return fn

        return deco


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


@dataclass
class MVPConfig:
    mode: str = "all"
    data_dir: str = DEFAULT_DATA_DIR
    model_dir: str = DEFAULT_MODEL_DIR
    result_dir: str = DEFAULT_RESULT_DIR
    run_dir: str | None = None
    verl_dir: str = "verl-main"
    tushare_token: str | None = None
    tushare_http_url: str = DEFAULT_TUSHARE_HTTP_URL
    index_code: str = DEFAULT_INDEX_CODE
    market_index_code: str = DEFAULT_MARKET_INDEX
    index_date: str = "20260101"
    start_date: str = "20230101"
    end_date: str = "20260531"
    train_end_date: str = "20241231"
    valid_end_date: str = "20250930"
    horizon: int = 5
    sample_stride: int = 5
    max_stocks: int = 50
    max_tasks: int = 3000
    seed: int = 7
    sleep_seconds: float = 0.12
    force_refresh: bool = False
    write_verl_command: bool = True
    model_path: str = DEFAULT_MODEL_PATH
    rollout_n: int = 2
    train_batch_size: int = 8
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 1
    log_prob_micro_batch_size_per_gpu: int = 1
    max_prompt_length: int = 2048
    max_response_length: int = 1024
    total_epochs: int = 1
    n_gpus_per_node: int = 2
    lora_rank: int = 32
    lora_alpha: int = 32
    rollout_tp: int = 2
    rollout_gpu_memory_utilization: float = 0.45
    command_file: str = "run_verl_stock_grpo.sh"


def require_pandas() -> None:
    if pd is None:
        raise RuntimeError(f"pandas is required but failed to import: {_PANDAS_IMPORT_ERROR}")


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


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


def ensure_dirs(cfg: MVPConfig) -> dict[str, Path]:
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


def make_tushare_client(token: str | None, http_url: str = DEFAULT_TUSHARE_HTTP_URL) -> Any:
    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "No Tushare token found. Pass --tushare-token, set TUSHARE_TOKEN, "
            "or call main(tushare_token='...')."
        )
    try:
        import tushare as ts
    except Exception as exc:
        raise RuntimeError("tushare is required on the data-building machine. Install tushare==1.4.24.") from exc

    ts.set_token(token)
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = http_url
    return pro


def call_tushare(
    pro: Any,
    api_name: str,
    sleep_seconds: float = 0.12,
    retries: int = 3,
    **kwargs: Any,
) -> "pd.DataFrame":
    require_pandas()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            fn = getattr(pro, api_name)
            df = fn(**kwargs)
            time.sleep(max(0.0, sleep_seconds))
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "permission" in msg.lower() or "not allowed" in msg.lower() or "No permission" in msg:
                raise
            time.sleep(min(2.0 * attempt, 5.0))
    raise RuntimeError(f"Tushare call failed: {api_name}({kwargs})") from last_exc


def fetch_or_cache(
    stem: Path,
    force_refresh: bool,
    fetch_fn: Callable[[], "pd.DataFrame"],
) -> "pd.DataFrame":
    cached = None if force_refresh else maybe_read_table(stem)
    if cached is not None:
        log(f"Cache hit: {stem.name} rows={len(cached)}")
        return cached
    df = fetch_fn()
    save_table(df, stem)
    log(f"Saved: {stem.name} rows={len(df)}")
    return df


def get_sse50_universe(pro: Any, cfg: MVPConfig, dirs: dict[str, Path]) -> "pd.DataFrame":
    require_pandas()
    index_date = normalize_date(cfg.index_date)
    start = str(int(index_date[:4]) - 1) + "0101"
    end = str(int(index_date[:4]) + 1) + "1231"

    def fetch_index_weight() -> "pd.DataFrame":
        return call_tushare(
            pro,
            "index_weight",
            sleep_seconds=cfg.sleep_seconds,
            index_code=cfg.index_code,
            start_date=start,
            end_date=end,
        )

    weights = fetch_or_cache(dirs["raw"] / f"index_weight_{cfg.index_code}_{start}_{end}", cfg.force_refresh, fetch_index_weight)
    if weights.empty:
        raise RuntimeError(
            "index_weight returned empty data. Try passing --stock-codes manually after extending the script, "
            "or check Tushare permission for index_weight."
        )
    weights["trade_date"] = weights["trade_date"].astype(str)
    eligible = weights[weights["trade_date"] <= index_date].copy()
    if eligible.empty:
        eligible = weights.copy()
    chosen_date = eligible["trade_date"].max()
    universe_codes = sorted(eligible.loc[eligible["trade_date"] == chosen_date, "con_code"].dropna().unique().tolist())
    universe_codes = universe_codes[: cfg.max_stocks]

    def fetch_stock_basic() -> "pd.DataFrame":
        return call_tushare(
            pro,
            "stock_basic",
            sleep_seconds=cfg.sleep_seconds,
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )

    basics = fetch_or_cache(dirs["raw"] / "stock_basic_L", cfg.force_refresh, fetch_stock_basic)
    basics = basics[basics["ts_code"].isin(universe_codes)].copy()
    basics["index_code"] = cfg.index_code
    basics["index_weight_date"] = chosen_date
    basics = basics.sort_values("ts_code").reset_index(drop=True)
    save_table(basics, dirs["processed"] / "universe")
    log(f"Universe built from {cfg.index_code} at nearest date {chosen_date}: rows={len(basics)}")
    return basics


def fetch_market_data(pro: Any, cfg: MVPConfig, dirs: dict[str, Path], universe: "pd.DataFrame") -> None:
    require_pandas()
    start = normalize_date(cfg.start_date)
    end = normalize_date(cfg.end_date)
    codes = sorted(universe["ts_code"].unique().tolist())

    all_daily: list["pd.DataFrame"] = []
    all_adj: list["pd.DataFrame"] = []
    all_basic: list["pd.DataFrame"] = []

    for i, code in enumerate(codes, start=1):
        safe_code = code.replace(".", "_")
        log(f"Fetching {i}/{len(codes)} {code}")

        daily = fetch_or_cache(
            dirs["raw"] / f"daily_{safe_code}_{start}_{end}",
            cfg.force_refresh,
            lambda code=code: call_tushare(
                pro,
                "daily",
                sleep_seconds=cfg.sleep_seconds,
                ts_code=code,
                start_date=start,
                end_date=end,
            ),
        )
        if not daily.empty:
            all_daily.append(daily)

        adj = fetch_or_cache(
            dirs["raw"] / f"adj_factor_{safe_code}_{start}_{end}",
            cfg.force_refresh,
            lambda code=code: call_tushare(
                pro,
                "adj_factor",
                sleep_seconds=cfg.sleep_seconds,
                ts_code=code,
                start_date=start,
                end_date=end,
            ),
        )
        if not adj.empty:
            all_adj.append(adj)

        basic = fetch_or_cache(
            dirs["raw"] / f"daily_basic_{safe_code}_{start}_{end}",
            cfg.force_refresh,
            lambda code=code: call_tushare(
                pro,
                "daily_basic",
                sleep_seconds=cfg.sleep_seconds,
                ts_code=code,
                start_date=start,
                end_date=end,
                fields=(
                    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                    "pe,pe_ttm,pb,total_mv,circ_mv"
                ),
            ),
        )
        if not basic.empty:
            all_basic.append(basic)

    if not all_daily:
        raise RuntimeError("No daily stock data fetched.")

    daily_all = pd.concat(all_daily, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    adj_all = pd.concat(all_adj, ignore_index=True).drop_duplicates(["ts_code", "trade_date"]) if all_adj else pd.DataFrame()
    basic_all = pd.concat(all_basic, ignore_index=True).drop_duplicates(["ts_code", "trade_date"]) if all_basic else pd.DataFrame()

    save_table(daily_all, dirs["processed"] / "stock_daily")
    save_table(adj_all, dirs["processed"] / "adj_factor")
    save_table(basic_all, dirs["processed"] / "daily_basic")

    index_daily = fetch_or_cache(
        dirs["raw"] / f"index_daily_{cfg.market_index_code}_{start}_{end}",
        cfg.force_refresh,
        lambda: call_tushare(
            pro,
            "index_daily",
            sleep_seconds=cfg.sleep_seconds,
            ts_code=cfg.market_index_code,
            start_date=start,
            end_date=end,
        ),
    )
    save_table(index_daily, dirs["processed"] / "index_daily")

    trade_cal = fetch_or_cache(
        dirs["raw"] / f"trade_cal_SSE_{start}_{end}",
        cfg.force_refresh,
        lambda: call_tushare(
            pro,
            "trade_cal",
            sleep_seconds=cfg.sleep_seconds,
            exchange="SSE",
            start_date=start,
            end_date=end,
            fields="exchange,cal_date,is_open,pretrade_date",
        ),
    )
    save_table(trade_cal, dirs["processed"] / "trade_cal")


def build_factor_snapshot(cfg: MVPConfig, dirs: dict[str, Path]) -> "pd.DataFrame":
    require_pandas()
    daily = read_table(dirs["processed"] / "stock_daily")
    adj = read_table(dirs["processed"] / "adj_factor")
    basic = read_table(dirs["processed"] / "daily_basic")
    universe = read_table(dirs["processed"] / "universe")
    index_daily = read_table(dirs["processed"] / "index_daily")

    df = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    if not basic.empty:
        keep = [
            c
            for c in [
                "ts_code",
                "trade_date",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "total_mv",
                "circ_mv",
            ]
            if c in basic.columns
        ]
        df = df.merge(basic[keep], on=["ts_code", "trade_date"], how="left")

    name_cols = [c for c in ["ts_code", "name", "industry", "market", "list_date"] if c in universe.columns]
    df = df.merge(universe[name_cols], on="ts_code", how="left")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "total_mv", "circ_mv"]:
        if col not in df.columns:
            df[col] = float("nan")
    for col in ["open", "high", "low", "close", "vol", "amount", "adj_factor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group = df.groupby("ts_code", group_keys=False)
    df["adj_factor"] = group["adj_factor"].ffill()
    df["adj_close"] = df["close"] * df["adj_factor"]
    df["ret_1d"] = group["adj_close"].pct_change(1)
    df["ret_5d"] = group["adj_close"].pct_change(5)
    df["ret_20d"] = group["adj_close"].pct_change(20)
    df["vol_20d"] = group["ret_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    df["amount_20d_avg"] = group["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    df["turnover_20d_avg"] = group["turnover_rate"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    idx = index_daily.copy()
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx["index_ret_1d"] = idx["close"].pct_change(1)
    idx["index_ret_5d"] = idx["close"].pct_change(5)
    idx["index_ret_20d"] = idx["close"].pct_change(20)
    idx["index_vol_20d"] = idx["index_ret_1d"].rolling(20, min_periods=10).std()
    idx_features = idx[["trade_date", "close", "index_ret_5d", "index_ret_20d", "index_vol_20d"]].rename(
        columns={"close": "index_close"}
    )
    df = df.merge(idx_features, on="trade_date", how="left")
    df["rs_market_5d"] = df["ret_5d"] - df["index_ret_5d"]
    df["rs_market_20d"] = df["ret_20d"] - df["index_ret_20d"]

    rank_cols = {
        "momentum_rank": "ret_20d",
        "rs_rank": "rs_market_20d",
        "liquidity_rank": "amount_20d_avg",
        "vol_rank": "vol_20d",
        "pb_rank": "pb",
    }
    for new_col, old_col in rank_cols.items():
        if old_col in df.columns:
            df[new_col] = df.groupby("trade_date")[old_col].rank(pct=True)

    save_table(df, dirs["processed"] / "factor_snapshot")

    market = idx.copy()
    market["market_ret_5d"] = market["index_ret_5d"]
    market["market_ret_20d"] = market["index_ret_20d"]
    market["market_vol_20d"] = market["index_vol_20d"]
    save_table(market, dirs["processed"] / "market_context")
    log(f"Factor snapshot built: rows={len(df)}")
    return df


def build_tasks_and_labels(cfg: MVPConfig, dirs: dict[str, Path]) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    require_pandas()
    factors = read_table(dirs["processed"] / "factor_snapshot")
    market = read_table(dirs["processed"] / "market_context")
    horizon = int(cfg.horizon)

    factors = factors.sort_values(["ts_code", "trade_date"]).copy()
    group = factors.groupby("ts_code", group_keys=False)
    factors["entry_date"] = group["trade_date"].shift(-1)
    factors["exit_date"] = group["trade_date"].shift(-(1 + horizon))
    factors["entry_adj_close"] = group["adj_close"].shift(-1)
    factors["exit_adj_close"] = group["adj_close"].shift(-(1 + horizon))
    factors["stock_return"] = factors["exit_adj_close"] / factors["entry_adj_close"] - 1.0

    market = market.sort_values("trade_date").copy()
    market["market_entry_date"] = market["trade_date"].shift(-1)
    market["market_exit_date"] = market["trade_date"].shift(-(1 + horizon))
    market["market_entry_close"] = market["close"].shift(-1)
    market["market_exit_close"] = market["close"].shift(-(1 + horizon))
    market["market_future_return"] = market["market_exit_close"] / market["market_entry_close"] - 1.0
    factors = factors.merge(
        market[["trade_date", "market_future_return"]],
        on="trade_date",
        how="left",
    )
    factors["future_relative_return"] = factors["stock_return"] - factors["market_future_return"]
    factors = factors.dropna(subset=["future_relative_return", "ret_20d", "vol_20d"]).copy()
    factors["cross_section_rank"] = factors.groupby("trade_date")["future_relative_return"].rank(pct=True)
    factors["label"] = "neutral"
    factors.loc[factors["cross_section_rank"] >= 0.70, "label"] = "up"
    factors.loc[factors["cross_section_rank"] <= 0.30, "label"] = "down"
    factors["label_id"] = factors["label"].map({c: i for i, c in enumerate(CLASSES)})

    trade_dates = sorted(factors["trade_date"].unique().tolist())
    sampled_dates = set(trade_dates[:: max(1, int(cfg.sample_stride))])
    tasks = factors[factors["trade_date"].isin(sampled_dates)].copy()
    tasks = tasks[tasks["trade_date"] <= normalize_date(cfg.end_date)].copy()

    def split_of(d: str) -> str:
        if d <= normalize_date(cfg.train_end_date):
            return "train"
        if d <= normalize_date(cfg.valid_end_date):
            return "valid"
        return "test"

    tasks["split"] = tasks["trade_date"].map(split_of)
    tasks["sample_id"] = tasks.apply(lambda r: f"{r.ts_code}_{r.trade_date}_T{horizon}", axis=1)
    tasks["as_of"] = tasks["trade_date"].map(lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]} 15:30:00")
    tasks["horizon"] = horizon
    tasks["max_tool_calls"] = 3

    if cfg.max_tasks and len(tasks) > cfg.max_tasks:
        tasks = tasks.sample(n=cfg.max_tasks, random_state=cfg.seed).sort_values(["trade_date", "ts_code"])

    task_cols = [
        "sample_id",
        "ts_code",
        "name",
        "industry",
        "trade_date",
        "as_of",
        "horizon",
        "entry_date",
        "exit_date",
        "max_tool_calls",
        "split",
    ]
    label_cols = [
        "sample_id",
        "ts_code",
        "trade_date",
        "horizon",
        "entry_date",
        "exit_date",
        "stock_return",
        "market_future_return",
        "future_relative_return",
        "cross_section_rank",
        "label",
        "label_id",
    ]
    task_df = tasks[[c for c in task_cols if c in tasks.columns]].reset_index(drop=True)
    label_df = tasks[[c for c in label_cols if c in tasks.columns]].reset_index(drop=True)
    save_table(task_df, dirs["processed"] / "tasks")
    save_table(label_df, dirs["processed"] / "labels")
    log(f"Tasks built: rows={len(task_df)} splits={task_df['split'].value_counts().to_dict()}")
    return task_df, label_df


def build_data(cfg: MVPConfig) -> None:
    dirs = ensure_dirs(cfg)
    pro = make_tushare_client(cfg.tushare_token, cfg.tushare_http_url)
    universe = get_sse50_universe(pro, cfg, dirs)
    fetch_market_data(pro, cfg, dirs, universe)
    build_factor_snapshot(cfg, dirs)
    build_tasks_and_labels(cfg, dirs)


_TOOL_TABLE_CACHE: dict[str, "pd.DataFrame"] = {}


def _tool_processed_dir() -> Path:
    return Path(os.environ.get("STOCK_AGENT_DATA_DIR", DEFAULT_DATA_DIR)).expanduser().resolve() / "processed"


def _load_tool_table(name: str) -> "pd.DataFrame":
    require_pandas()
    key = str(_tool_processed_dir() / name)
    if key not in _TOOL_TABLE_CACHE:
        _TOOL_TABLE_CACHE[key] = read_table(_tool_processed_dir() / name)
    return _TOOL_TABLE_CACHE[key]


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _date_to_yyyymmdd(s: str) -> str:
    return normalize_date(s)


@function_tool
def get_price_factors(ts_code: str, as_of_date: str, lookback_days: int = 20) -> dict:
    """Get point-in-time price and factor summary for one A-share stock.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        lookback_days: Lookback window length; currently used as metadata only.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _date_to_yyyymmdd(as_of_date)
        df = _load_tool_table("factor_snapshot")
        row_df = df[(df["ts_code"] == code) & (df["trade_date"].astype(str) == date)]
        if row_df.empty:
            return {
                "tool_name": "get_price_factors",
                "status": "error",
                "error_type": "not_found",
                "message": f"No factor row for {code} at {date}.",
            }
        row = row_df.iloc[0]
        factors = {
            "ret_1d": _safe_float(row.get("ret_1d")),
            "ret_5d": _safe_float(row.get("ret_5d")),
            "ret_20d": _safe_float(row.get("ret_20d")),
            "vol_20d": _safe_float(row.get("vol_20d")),
            "rs_market_5d": _safe_float(row.get("rs_market_5d")),
            "rs_market_20d": _safe_float(row.get("rs_market_20d")),
            "turnover_20d_avg": _safe_float(row.get("turnover_20d_avg")),
            "amount_20d_avg": _safe_float(row.get("amount_20d_avg")),
            "momentum_rank": _safe_float(row.get("momentum_rank")),
            "rs_rank": _safe_float(row.get("rs_rank")),
            "liquidity_rank": _safe_float(row.get("liquidity_rank")),
            "vol_rank": _safe_float(row.get("vol_rank")),
            "pe_ttm": _safe_float(row.get("pe_ttm")),
            "pb": _safe_float(row.get("pb")),
            "total_mv": _safe_float(row.get("total_mv")),
        }
        summary = (
            f"{code} ret_20d={factors['ret_20d']}, rs_market_20d={factors['rs_market_20d']}, "
            f"momentum_rank={factors['momentum_rank']}, vol_20d={factors['vol_20d']}."
        )
        return {
            "tool_name": "get_price_factors",
            "status": "ok",
            "as_of_date": date,
            "ts_code": code,
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "lookback_days": int(lookback_days),
            "factors": factors,
            "summary": summary,
        }
    except Exception as exc:
        return {
            "tool_name": "get_price_factors",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def get_market_context(as_of_date: str, index_code: str = DEFAULT_MARKET_INDEX) -> dict:
    """Get point-in-time market index context.

    Args:
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        index_code: Market index code. The default is SSE50 "000016.SH".
    """
    try:
        date = _date_to_yyyymmdd(as_of_date)
        df = _load_tool_table("market_context")
        row_df = df[df["trade_date"].astype(str) == date]
        if row_df.empty:
            return {
                "tool_name": "get_market_context",
                "status": "error",
                "error_type": "not_found",
                "message": f"No market row for {date}.",
            }
        row = row_df.iloc[0]
        metrics = {
            "market_ret_5d": _safe_float(row.get("market_ret_5d")),
            "market_ret_20d": _safe_float(row.get("market_ret_20d")),
            "market_vol_20d": _safe_float(row.get("market_vol_20d")),
            "index_close": _safe_float(row.get("close")),
        }
        return {
            "tool_name": "get_market_context",
            "status": "ok",
            "index_code": index_code,
            "as_of_date": date,
            "metrics": metrics,
            "summary": (
                f"Market ret_20d={metrics['market_ret_20d']}, "
                f"vol_20d={metrics['market_vol_20d']}."
            ),
        }
    except Exception as exc:
        return {
            "tool_name": "get_market_context",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def search_announcements(ts_code: str, as_of_date: str, query: str = "", top_k: int = 3) -> dict:
    """Search point-in-time announcements if an announcement table is available.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        query: Keyword query for title or summary.
        top_k: Maximum number of announcement rows to return.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _date_to_yyyymmdd(as_of_date)
        stem = _tool_processed_dir() / "announcements"
        if not table_path(stem, ".pkl").exists() and not table_path(stem, ".parquet").exists() and not table_path(stem, ".csv").exists():
            return {
                "tool_name": "search_announcements",
                "status": "ok",
                "as_of_date": date,
                "ts_code": code,
                "result_count": 0,
                "results": [],
                "warning": "No announcements table is available in this MVP run.",
            }
        df = _load_tool_table("announcements")
        date_col = "publish_date" if "publish_date" in df.columns else "trade_date"
        mask = (df["ts_code"] == code) & (df[date_col].astype(str) <= date)
        if query:
            q_terms = [x.lower() for x in re.split(r"\s+", query.strip()) if x]
            if q_terms:
                title = df["title"].astype(str) if "title" in df.columns else pd.Series([""] * len(df), index=df.index)
                summary = df["summary"].astype(str) if "summary" in df.columns else pd.Series([""] * len(df), index=df.index)
                text = (title + " " + summary).str.lower()
                mask &= text.map(lambda x: any(t in x for t in q_terms))
        out = df[mask].sort_values(date_col, ascending=False).head(max(1, min(int(top_k), 10)))
        results = []
        for _, row in out.iterrows():
            results.append(
                {
                    "id": str(row.get("ann_id", row.get("id", ""))),
                    "publish_date": str(row.get(date_col, "")),
                    "title": str(row.get("title", "")),
                    "summary": str(row.get("summary", ""))[:240],
                }
            )
        return {
            "tool_name": "search_announcements",
            "status": "ok",
            "as_of_date": date,
            "ts_code": code,
            "query": query,
            "result_count": len(results),
            "results": results,
        }
    except Exception as exc:
        return {
            "tool_name": "search_announcements",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def parse_final_answer(text: str) -> dict | None:
    if not text:
        return None
    candidates: list[str] = []
    stack = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and ("p_up" in obj or "prediction" in obj):
            return obj
    return None


def compute_stock_reward(answer: dict | None, extra_info: dict[str, Any]) -> dict[str, Any]:
    if answer is None:
        return {"score": -1.0, "parse_error": 1.0}

    try:
        probs = [float(answer["p_up"]), float(answer["p_neutral"]), float(answer["p_down"])]
    except Exception:
        return {"score": -1.0, "missing_probability": 1.0}

    if any(math.isnan(p) or p < 0.0 or p > 1.0 for p in probs) or abs(sum(probs) - 1.0) > 0.08:
        return {"score": -0.8, "invalid_probability": 1.0}

    s = sum(probs)
    probs = [p / s for p in probs]
    true_label = str(extra_info.get("label", "neutral"))
    if true_label not in CLASSES:
        true_label = "neutral"
    true_idx = CLASSES.index(true_label)
    pred_idx = max(range(3), key=lambda i: probs[i])
    pred_label = CLASSES[pred_idx]

    direction_reward = 1.0 if pred_label == true_label else -0.5
    prob_reward = probs[true_idx]
    brier_error = sum((probs[i] - (1.0 if i == true_idx else 0.0)) ** 2 for i in range(3))
    brier_reward = 1.0 - brier_error / 2.0
    future_rel = float(extra_info.get("future_relative_return", 0.0))
    pnl_scale = max(float(extra_info.get("pnl_scale", 0.03)), 1e-6)
    position = probs[0] - probs[2]
    pnl_reward = max(-1.0, min(1.0, position * future_rel / pnl_scale))

    format_reward = 0.04
    if answer.get("prediction") == pred_label:
        format_reward += 0.03
    if isinstance(answer.get("evidence_summary"), list) and answer["evidence_summary"]:
        format_reward += 0.03

    num_turns = extra_info.get("num_turns")
    try:
        turn_cost = -0.01 * max(0, int(num_turns or 0) - 1)
    except Exception:
        turn_cost = 0.0

    total = (
        0.18 * direction_reward
        + 0.27 * prob_reward
        + 0.27 * brier_reward
        + 0.20 * pnl_reward
        + format_reward
        + turn_cost
    )
    total = max(-1.0, min(1.5, total))
    return {
        "score": float(total),
        "direction_reward": float(direction_reward),
        "prob_reward": float(prob_reward),
        "brier_reward": float(brier_reward),
        "pnl_reward": float(pnl_reward),
        "format_reward": float(format_reward),
        "turn_cost": float(turn_cost),
        "pred_label": pred_label,
        "true_label": true_label,
        "future_relative_return": float(future_rel),
    }


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {"label": ground_truth}
    return {}


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> dict[str, Any]:
    """verl custom reward entrypoint."""
    gt = _parse_ground_truth(ground_truth)
    info = {}
    info.update(gt)
    if extra_info:
        info.update(dict(extra_info))
    answer = parse_final_answer(solution_str)
    return compute_stock_reward(answer, info)


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a stock prediction search agent. You must use only point-in-time "
        "information available at the given as_of date. You can call tools to get "
        "price factors, market context, and optional announcements. Finally output "
        "strict JSON with prediction, p_up, p_neutral, p_down, alpha_score, "
        "confidence, evidence_summary, risk_factors, and search_steps_used."
    )
    user = (
        f"Predict the relative return direction for stock {row.get('ts_code')} "
        f"({row.get('name', '')}) over the next {row.get('horizon')} trading days.\n"
        f"Industry: {row.get('industry', '')}\n"
        f"As-of date: {row.get('trade_date')}\n"
        f"Max tool calls: {row.get('max_tool_calls', 3)}\n"
        "Available tools: get_price_factors, get_market_context, search_announcements.\n"
        "Labels are hidden from you. Do not invent future information."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def export_verl_dataset(cfg: MVPConfig) -> tuple[Path, Path]:
    require_pandas()
    dirs = ensure_dirs(cfg)
    tasks = read_table(dirs["processed"] / "tasks")
    labels = read_table(dirs["processed"] / "labels")
    df = tasks.merge(labels, on=["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"], how="inner")

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        item = row.to_dict()
        gt = {
            "label": item["label"],
            "future_relative_return": float(item["future_relative_return"]),
        }
        rows.append(
            {
                "uid": str(item["sample_id"]),
                "data_source": DATA_SOURCE,
                "agent_name": "tool_agent",
                "prompt": make_prompt(item),
                "ability": "stock_prediction_agent",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
                "extra_info": {
                    "split": item["split"],
                    "sample_id": item["sample_id"],
                    "ts_code": item["ts_code"],
                    "trade_date": item["trade_date"],
                    "as_of": item["as_of"],
                    "horizon": int(item["horizon"]),
                    "entry_date": item["entry_date"],
                    "exit_date": item["exit_date"],
                    "label": item["label"],
                    "label_id": int(item["label_id"]),
                    "future_relative_return": float(item["future_relative_return"]),
                    "pnl_scale": 0.03,
                },
            }
        )

    random.Random(cfg.seed).shuffle(rows)
    train_rows = [r for r in rows if r["extra_info"]["split"] == "train"]
    valid_rows = [r for r in rows if r["extra_info"]["split"] == "valid"]
    if not valid_rows:
        valid_rows = [r for r in rows if r["extra_info"]["split"] == "test"][: max(1, len(rows) // 10)]

    train_path = dirs["verl"] / "train.parquet"
    valid_path = dirs["verl"] / "valid.parquet"
    write_records_parquet(train_rows, train_path)
    write_records_parquet(valid_rows, valid_path)
    log(f"verl dataset exported: train={len(train_rows)} valid={len(valid_rows)}")
    return train_path, valid_path


def write_records_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import datasets

        datasets.Dataset.from_list(rows).to_parquet(str(path))
        return
    except Exception as exc:
        log(f"datasets parquet writer failed: {exc}")

    try:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return
    except Exception as exc:
        jsonl = path.with_suffix(".jsonl")
        with jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"Could not write parquet to {path}. Wrote JSONL fallback at {jsonl}. "
            "Install pyarrow or datasets on the training server."
        ) from exc


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def rule_agent_predict(task: dict[str, Any]) -> dict[str, Any]:
    pf = get_price_factors(str(task["ts_code"]), str(task["trade_date"]))
    mc = get_market_context(str(task["trade_date"]))
    factors = pf.get("factors", {}) if pf.get("status") == "ok" else {}

    momentum = _safe_float(factors.get("momentum_rank"), 0.5) or 0.5
    rs_rank = _safe_float(factors.get("rs_rank"), 0.5) or 0.5
    liq = _safe_float(factors.get("liquidity_rank"), 0.5) or 0.5
    vol = _safe_float(factors.get("vol_rank"), 0.5) or 0.5
    ret_5d = _safe_float(factors.get("ret_5d"), 0.0) or 0.0
    score = 1.2 * (momentum - 0.5) + 1.0 * (rs_rank - 0.5) + 0.3 * (liq - 0.5) - 0.35 * (vol - 0.5)
    score += max(-0.2, min(0.2, ret_5d)) * 1.5
    edge = math.tanh(score)
    p_up = 0.33 + 0.24 * edge
    p_down = 0.33 - 0.24 * edge
    p_neutral = 1.0 - p_up - p_down
    probs = [max(0.02, p_up), max(0.02, p_neutral), max(0.02, p_down)]
    s = sum(probs)
    probs = [p / s for p in probs]
    pred = CLASSES[max(range(3), key=lambda i: probs[i])]
    answer = {
        "prediction": pred,
        "p_up": round(probs[0], 4),
        "p_neutral": round(probs[1], 4),
        "p_down": round(probs[2], 4),
        "alpha_score": round(probs[0] - probs[2], 4),
        "confidence": round(max(probs), 4),
        "evidence_summary": [
            {
                "direction": "positive" if edge >= 0 else "negative",
                "source_type": "price_factor",
                "source_id": f"{task['ts_code']}_{task['trade_date']}",
                "summary": pf.get("summary", ""),
            },
            {
                "direction": "neutral",
                "source_type": "market_context",
                "source_id": str(task["trade_date"]),
                "summary": mc.get("summary", ""),
            },
        ],
        "risk_factors": ["This MVP uses price/market factors only; no verified announcement/news table is loaded."],
        "search_steps_used": 2,
    }
    return answer


def run_rule_rollout(cfg: MVPConfig) -> Path:
    require_pandas()
    dirs = ensure_dirs(cfg)
    os.environ["STOCK_AGENT_DATA_DIR"] = str(Path(cfg.data_dir).expanduser().resolve())
    tasks = read_table(dirs["processed"] / "tasks")
    labels = read_table(dirs["processed"] / "labels")
    df = tasks.merge(labels, on=["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"], how="inner")

    rows = []
    for _, row in df.iterrows():
        task = row.to_dict()
        ans = rule_agent_predict(task)
        reward = compute_stock_reward(ans, task)
        rows.append(
            {
                "sample_id": task["sample_id"],
                "split": task["split"],
                "ts_code": task["ts_code"],
                "trade_date": task["trade_date"],
                "label": task["label"],
                "future_relative_return": task["future_relative_return"],
                "prediction": ans["prediction"],
                "p_up": ans["p_up"],
                "p_neutral": ans["p_neutral"],
                "p_down": ans["p_down"],
                "alpha_score": ans["alpha_score"],
                "reward": reward["score"],
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, dirs["rollouts"] / "rule_baseline_predictions")
    metrics = summarize_predictions(out)
    metrics_path = dirs["rollouts"] / "rule_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Rule rollout metrics: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics_path


def summarize_predictions(df: "pd.DataFrame") -> dict[str, Any]:
    require_pandas()
    out: dict[str, Any] = {}
    for split, sub in df.groupby("split"):
        if sub.empty:
            continue
        acc = float((sub["prediction"] == sub["label"]).mean())
        mean_reward = float(sub["reward"].mean())
        brier = 0.0
        for _, row in sub.iterrows():
            y = [1.0 if row["label"] == c else 0.0 for c in CLASSES]
            p = [float(row["p_up"]), float(row["p_neutral"]), float(row["p_down"])]
            brier += sum((p[i] - y[i]) ** 2 for i in range(3)) / len(sub)
        rank_ics = []
        for _, g in sub.groupby("trade_date"):
            if len(g) >= 5:
                rank_ics.append(float(g["alpha_score"].rank().corr(g["future_relative_return"].rank())))
        out[split] = {
            "n": int(len(sub)),
            "mean_reward": mean_reward,
            "accuracy": acc,
            "brier": float(brier),
            "mean_rank_ic": float(sum(rank_ics) / len(rank_ics)) if rank_ics else None,
        }
    return out


def is_local_model_path(model_path: str) -> bool:
    return model_path.startswith(("/", "./", "../", "model/")) or Path(model_path).exists()


def resolve_model_path_for_command(model_path: str) -> str:
    if is_local_model_path(model_path):
        return str(Path(model_path).expanduser().resolve())
    return model_path


def print_download_hints(model_path: str = DEFAULT_MODEL_PATH, model_dir: str = DEFAULT_MODEL_DIR) -> None:
    local_dir = Path(model_path) if is_local_model_path(model_path) else Path(model_dir) / "Qwen3-4B"
    local_dir = local_dir.expanduser()
    msg = f"""
Model download hints for China mainland networks:

Option A: Hugging Face mirror
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download {DEFAULT_HF_MODEL_ID} --local-dir {local_dir}

Option B: ModelScope
  pip install -U modelscope
  modelscope download --model {DEFAULT_HF_MODEL_ID} --local_dir {local_dir}

Then run this script or the verl command with:
  --model-path {local_dir}
"""
    print(msg.strip())


def make_verl_command(cfg: MVPConfig, train_path: Path, valid_path: Path) -> str:
    script_path = Path(__file__).resolve()
    repo = Path(cfg.verl_dir).expanduser().resolve()
    data_root = Path(cfg.data_dir).expanduser().resolve()
    model_path = resolve_model_path_for_command(cfg.model_path)
    command = f"""#!/usr/bin/env bash
set -xeuo pipefail

export PYTHONPATH="{repo}:${{PYTHONPATH:-}}"
export STOCK_AGENT_DATA_DIR="{data_root}"
export HYDRA_FULL_ERROR=1

cd "{repo}"

python3 -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  algorithm.use_kl_in_reward=False \\
  data.train_files="{train_path}" \\
  data.val_files="{valid_path}" \\
  data.train_batch_size={cfg.train_batch_size} \\
  data.max_prompt_length={cfg.max_prompt_length} \\
  data.max_response_length={cfg.max_response_length} \\
  data.filter_overlong_prompts=True \\
  data.truncation=error \\
  data.return_raw_chat=True \\
  actor_rollout_ref.model.path="{model_path}" \\
  actor_rollout_ref.model.use_remove_padding=True \\
  actor_rollout_ref.model.enable_gradient_checkpointing=True \\
  actor_rollout_ref.model.lora_rank={cfg.lora_rank} \\
  actor_rollout_ref.model.lora_alpha={cfg.lora_alpha} \\
  actor_rollout_ref.model.target_modules=all-linear \\
  actor_rollout_ref.actor.optim.lr=3e-5 \\
  actor_rollout_ref.actor.ppo_mini_batch_size={cfg.ppo_mini_batch_size} \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg.ppo_micro_batch_size_per_gpu} \\
  actor_rollout_ref.actor.use_kl_loss=True \\
  actor_rollout_ref.actor.kl_loss_coef=0.001 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.actor.entropy_coeff=0 \\
  actor_rollout_ref.actor.fsdp_config.param_offload=True \\
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
  actor_rollout_ref.actor.use_dynamic_bsz=True \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.tensor_model_parallel_size={cfg.rollout_tp} \\
  actor_rollout_ref.rollout.gpu_memory_utilization={cfg.rollout_gpu_memory_utilization} \\
  actor_rollout_ref.rollout.n={cfg.rollout_n} \\
  actor_rollout_ref.rollout.load_format=safetensors \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg.log_prob_micro_batch_size_per_gpu} \\
  actor_rollout_ref.rollout.multi_turn.enable=True \\
  actor_rollout_ref.rollout.multi_turn.format=hermes \\
  actor_rollout_ref.rollout.multi_turn.function_tool_path="{script_path}" \\
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \\
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \\
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \\
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg.log_prob_micro_batch_size_per_gpu} \\
  actor_rollout_ref.ref.fsdp_config.param_offload=True \\
  reward.custom_reward_function.path="{script_path}" \\
  reward.custom_reward_function.name=compute_score \\
  reward.reward_manager.name=naive \\
  trainer.critic_warmup=0 \\
  trainer.logger='["console"]' \\
  trainer.project_name=stock_agent_rl_mvp \\
  trainer.experiment_name=qwen3_4b_sse50_grpo_lora \\
  trainer.n_gpus_per_node={cfg.n_gpus_per_node} \\
  trainer.nnodes=1 \\
  trainer.save_freq=5 \\
  trainer.test_freq=1 \\
  trainer.total_epochs={cfg.total_epochs} \\
  "$@"
"""
    return command


def write_verl_command_script(cfg: MVPConfig) -> Path:
    dirs = ensure_dirs(cfg)
    train_path = dirs["verl"] / "train.parquet"
    valid_path = dirs["verl"] / "valid.parquet"
    if not train_path.exists() or not valid_path.exists():
        train_path, valid_path = export_verl_dataset(cfg)
    command = make_verl_command(cfg, train_path.resolve(), valid_path.resolve())
    out_path = Path(cfg.command_file).expanduser()
    out = out_path.resolve() if out_path.is_absolute() else (dirs["run"] / out_path).resolve()
    out.write_text(command, encoding="utf-8")
    out.chmod(0o755)
    log(f"Wrote verl command script: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> MVPConfig:
    p = argparse.ArgumentParser(description="Single-file MVP for stock search-agent RL with Tushare + verl.")
    p.add_argument("--mode", default="all", choices=["all", "build-data", "export-verl", "rule-rollout", "print-verl-command", "download-hints"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--verl-dir", default="verl-main")
    p.add_argument("--tushare-token", default=None)
    p.add_argument("--tushare-http-url", default=DEFAULT_TUSHARE_HTTP_URL)
    p.add_argument("--index-code", default=DEFAULT_INDEX_CODE)
    p.add_argument("--market-index-code", default=DEFAULT_MARKET_INDEX)
    p.add_argument("--index-date", default="20260101")
    p.add_argument("--start-date", default="20230101")
    p.add_argument("--end-date", default="20260531")
    p.add_argument("--train-end-date", default="20241231")
    p.add_argument("--valid-end-date", default="20250930")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--sample-stride", type=int, default=5)
    p.add_argument("--max-stocks", type=int, default=50)
    p.add_argument("--max-tasks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sleep-seconds", type=float, default=0.12)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--rollout-n", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--ppo-mini-batch-size", type=int, default=8)
    p.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    p.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-response-length", type=int, default=1024)
    p.add_argument("--total-epochs", type=int, default=1)
    p.add_argument("--n-gpus-per-node", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--rollout-tp", type=int, default=2)
    p.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--command-file", default="run_verl_stock_grpo.sh")
    p.add_argument("--no-write-verl-command", dest="write_verl_command", action="store_false")
    p.set_defaults(write_verl_command=True)
    args = p.parse_args(argv)
    return MVPConfig(**vars(args))


def main(
    mode: str = "all",
    data_dir: str = DEFAULT_DATA_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    run_dir: str | None = None,
    verl_dir: str = "verl-main",
    tushare_token: str | None = None,
    tushare_http_url: str = DEFAULT_TUSHARE_HTTP_URL,
    index_code: str = DEFAULT_INDEX_CODE,
    market_index_code: str = DEFAULT_MARKET_INDEX,
    index_date: str = "20260101",
    start_date: str = "20230101",
    end_date: str = "20260531",
    train_end_date: str = "20241231",
    valid_end_date: str = "20250930",
    horizon: int = 5,
    sample_stride: int = 5,
    max_stocks: int = 50,
    max_tasks: int = 3000,
    seed: int = 7,
    sleep_seconds: float = 0.12,
    force_refresh: bool = False,
    write_verl_command: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    rollout_n: int = 2,
    train_batch_size: int = 8,
    ppo_mini_batch_size: int = 8,
    ppo_micro_batch_size_per_gpu: int = 1,
    log_prob_micro_batch_size_per_gpu: int = 1,
    max_prompt_length: int = 2048,
    max_response_length: int = 1024,
    total_epochs: int = 1,
    n_gpus_per_node: int = 2,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    rollout_tp: int = 2,
    rollout_gpu_memory_utilization: float = 0.45,
    command_file: str = "run_verl_stock_grpo.sh",
) -> dict[str, Any]:
    cfg = MVPConfig(
        mode=mode,
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        run_dir=run_dir,
        verl_dir=verl_dir,
        tushare_token=tushare_token,
        tushare_http_url=tushare_http_url,
        index_code=index_code,
        market_index_code=market_index_code,
        index_date=index_date,
        start_date=start_date,
        end_date=end_date,
        train_end_date=train_end_date,
        valid_end_date=valid_end_date,
        horizon=horizon,
        sample_stride=sample_stride,
        max_stocks=max_stocks,
        max_tasks=max_tasks,
        seed=seed,
        sleep_seconds=sleep_seconds,
        force_refresh=force_refresh,
        write_verl_command=write_verl_command,
        model_path=model_path,
        rollout_n=rollout_n,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        rollout_tp=rollout_tp,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        command_file=command_file,
    )
    random.seed(cfg.seed)
    dirs = ensure_dirs(cfg)
    config_for_log = {k: ("***" if k == "tushare_token" and v else v) for k, v in asdict(cfg).items()}
    (dirs["run"] / "run_config.json").write_text(
        json.dumps(config_for_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Config: {json.dumps(config_for_log, ensure_ascii=False)}")
    log(f"Run output dir: {dirs['run']}")

    results: dict[str, Any] = {"config": config_for_log}
    if cfg.mode in ("all", "download-hints"):
        print_download_hints(cfg.model_path, cfg.model_dir)
        if cfg.mode == "download-hints":
            return results

    if cfg.mode in ("all", "build-data"):
        build_data(cfg)
        results["data_built"] = True

    if cfg.mode in ("all", "export-verl"):
        train_path, valid_path = export_verl_dataset(cfg)
        results["train_parquet"] = str(train_path)
        results["valid_parquet"] = str(valid_path)

    if cfg.mode in ("all", "rule-rollout"):
        metrics_path = run_rule_rollout(cfg)
        results["rule_metrics"] = str(metrics_path)

    if cfg.mode in ("all", "print-verl-command"):
        if cfg.write_verl_command:
            path = write_verl_command_script(cfg)
            results["verl_command_file"] = str(path)
        else:
            dirs = ensure_dirs(cfg)
            print(make_verl_command(cfg, (dirs["verl"] / "train.parquet").resolve(), (dirs["verl"] / "valid.parquet").resolve()))

    return results


if __name__ == "__main__":
    cli_cfg = parse_args()
    main(**asdict(cli_cfg))
