from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from rl_common import (
    CLASSES,
    DEFAULT_TUSHARE_HTTP_URL,
    ensure_dirs,
    load_local_config_value,
    log,
    maybe_read_table,
    normalize_date,
    pd,
    read_table,
    require_pandas,
    save_table,
)
from rl_config import MVPConfig


def make_tushare_client(token: str | None, http_url: str = DEFAULT_TUSHARE_HTTP_URL) -> Any:
    token = token or load_local_config_value("TUSHARE_TOKEN") or os.environ.get("TUSHARE_TOKEN")
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

    weights = fetch_or_cache(
        dirs["raw"] / f"index_weight_{cfg.index_code}_{start}_{end}",
        cfg.force_refresh,
        lambda: call_tushare(
            pro,
            "index_weight",
            sleep_seconds=cfg.sleep_seconds,
            index_code=cfg.index_code,
            start_date=start,
            end_date=end,
        ),
    )
    if weights.empty:
        raise RuntimeError(
            "index_weight returned empty data. Check Tushare permission for index_weight "
            "or extend the script with a manual stock list."
        )
    weights["trade_date"] = weights["trade_date"].astype(str)
    eligible = weights[weights["trade_date"] <= index_date].copy()
    if eligible.empty:
        eligible = weights.copy()
    chosen_date = eligible["trade_date"].max()
    universe_codes = sorted(eligible.loc[eligible["trade_date"] == chosen_date, "con_code"].dropna().unique().tolist())
    universe_codes = universe_codes[: cfg.max_stocks]

    basics = fetch_or_cache(
        dirs["raw"] / "stock_basic_L",
        cfg.force_refresh,
        lambda: call_tushare(
            pro,
            "stock_basic",
            sleep_seconds=cfg.sleep_seconds,
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        ),
    )
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
    market["market_future_return"] = market["close"].shift(-(1 + horizon)) / market["close"].shift(-1) - 1.0
    factors = factors.merge(market[["trade_date", "market_future_return"]], on="trade_date", how="left")
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
