from __future__ import annotations

import math
import os
import re
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


def optional_tushare_call(
    pro: Any,
    api_name: str,
    sleep_seconds: float = 0.12,
    **kwargs: Any,
) -> "pd.DataFrame":
    """Best-effort Tushare call for non-core data such as announcements/fundamentals."""
    require_pandas()
    try:
        return call_tushare(pro, api_name, sleep_seconds=sleep_seconds, retries=1, **kwargs)
    except Exception as exc:
        log(f"Optional Tushare API skipped: {api_name}({kwargs}) -> {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _date_to_iso(date: Any, close_time: str = "15:30:00") -> str:
    digits = re.sub(r"[^0-9]", "", str(date))
    s = digits[:8] if len(digits) >= 8 else normalize_date(date)
    return f"{s[:4]}-{s[4:6]}-{s[6:]} {close_time}"


def _industry_id(name: Any) -> str:
    text = str(name or "unknown").strip() or "unknown"
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")
    return f"IND_{slug or 'unknown'}"


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _coalesced_column(df: "pd.DataFrame", name: str) -> "pd.Series":
    selected = df.loc[:, name]
    if isinstance(selected, pd.DataFrame):
        return selected.bfill(axis=1).iloc[:, 0].rename(name)
    return selected


def _coalesce_duplicate_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    if not df.columns.has_duplicates:
        return df

    seen: set[str] = set()
    columns = []
    for name in df.columns:
        if name in seen:
            continue
        seen.add(name)
        columns.append(_coalesced_column(df, name).rename(name))
    return pd.concat(columns, axis=1)


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
    basics["company_name"] = basics["name"] if "name" in basics.columns else ""
    basics["industry_name"] = basics["industry"].fillna("unknown") if "industry" in basics.columns else "unknown"
    basics["industry_id"] = basics["industry_name"].map(_industry_id)
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

    name_cols = [
        c
        for c in [
            "ts_code",
            "name",
            "company_name",
            "industry",
            "industry_id",
            "industry_name",
            "market",
            "list_date",
        ]
        if c in universe.columns
    ]
    df = df.merge(universe[name_cols], on="ts_code", how="left")
    if "company_name" not in df.columns:
        df["company_name"] = df.get("name", "")
    if "industry_name" not in df.columns:
        df["industry_name"] = df.get("industry", "unknown")
    if "industry_id" not in df.columns:
        df["industry_id"] = df["industry_name"].map(_industry_id)
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
    df["turnover_5d_avg"] = group["turnover_rate"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    df["turnover_20d_avg"] = group["turnover_rate"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    if "pct_chg" in df.columns:
        df["limit_up"] = (pd.to_numeric(df["pct_chg"], errors="coerce") >= 9.5).astype(int)
        df["limit_down"] = (pd.to_numeric(df["pct_chg"], errors="coerce") <= -9.5).astype(int)
    else:
        df["limit_up"] = 0
        df["limit_down"] = 0
    df["limit_up_recent_5d"] = group["limit_up"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["limit_down_recent_5d"] = group["limit_down"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["suspended"] = df["close"].isna().astype(int)
    df["suspended_recent_20d"] = group["suspended"].rolling(20, min_periods=1).sum().reset_index(level=0, drop=True)

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
            df[f"{new_col}_in_industry"] = df.groupby(["trade_date", "industry_id"])[old_col].rank(pct=True)

    save_table(df, dirs["processed"] / "factor_snapshot")

    market = idx.copy()
    market["market_ret_5d"] = market["index_ret_5d"]
    market["market_ret_20d"] = market["index_ret_20d"]
    market["market_vol_20d"] = market["index_vol_20d"]
    save_table(market, dirs["processed"] / "market_context")
    log(f"Factor snapshot built: rows={len(df)}")
    return df


def build_industry_and_peer_context(cfg: MVPConfig, dirs: dict[str, Path]) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    require_pandas()
    factors = read_table(dirs["processed"] / "factor_snapshot").copy()
    for col in ["industry_id", "industry_name"]:
        if col not in factors.columns:
            factors[col] = "unknown"
    factors["trade_date"] = factors["trade_date"].astype(str)

    agg = (
        factors.groupby(["industry_id", "industry_name", "trade_date"], dropna=False)
        .agg(
            member_count=("ts_code", "nunique"),
            industry_ret_5d=("ret_5d", "mean"),
            industry_ret_20d=("ret_20d", "mean"),
            industry_vol_20d=("vol_20d", "mean"),
            turnover_20d_avg=("turnover_20d_avg", "mean"),
            amount_20d_avg=("amount_20d_avg", "mean"),
            pe_ttm_median=("pe_ttm", "median"),
            pb_median=("pb", "median"),
            momentum_rank_mean=("momentum_rank", "mean"),
            rs_rank_mean=("rs_rank", "mean"),
        )
        .reset_index()
        .sort_values(["industry_id", "trade_date"])
    )
    agg["valuation_pe_percentile_2y"] = agg.groupby("industry_id")["pe_ttm_median"].rank(pct=True)
    agg["turnover_percentile_1y"] = agg.groupby("industry_id")["turnover_20d_avg"].rank(pct=True)
    agg["as_of"] = agg["trade_date"].map(_date_to_iso)
    save_table(agg, dirs["processed"] / "industry_context")

    latest_cols = [
        "ts_code",
        "trade_date",
        "company_name",
        "name",
        "industry_id",
        "industry_name",
        "ret_20d",
        "rs_market_20d",
        "momentum_rank_in_industry",
        "liquidity_rank_in_industry",
        "pe_ttm",
        "pb",
        "total_mv",
    ]
    available = [c for c in latest_cols if c in factors.columns]
    peer = factors[available].copy()
    for col, default in [("momentum_rank_in_industry", 0.5), ("liquidity_rank_in_industry", 0.5), ("rs_market_20d", 0.0)]:
        if col not in peer.columns:
            peer[col] = default
    peer["peer_score"] = (
        peer["momentum_rank_in_industry"].fillna(0.5) * 0.45
        + peer["liquidity_rank_in_industry"].fillna(0.5) * 0.20
        + peer["rs_market_20d"].fillna(0.0).rank(pct=True) * 0.35
    )
    save_table(peer, dirs["processed"] / "peer_context")
    log(f"Industry context built: rows={len(agg)} peer_rows={len(peer)}")
    return agg, peer


def build_fundamental_snapshot(
    cfg: MVPConfig,
    dirs: dict[str, Path],
    pro: Any | None = None,
    universe: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    require_pandas()
    factors = read_table(dirs["processed"] / "factor_snapshot").copy()
    cols = [
        "ts_code",
        "trade_date",
        "company_name",
        "name",
        "industry_id",
        "industry_name",
        "pe",
        "pe_ttm",
        "pb",
        "total_mv",
        "circ_mv",
        "turnover_rate",
        "ret_20d",
        "rs_market_20d",
    ]
    out = factors[[c for c in cols if c in factors.columns]].copy()
    out["as_of_date"] = out["trade_date"].astype(str)
    out["publish_time"] = out["as_of_date"].map(_date_to_iso)
    out["report_period"] = ""
    for col in ["revenue_yoy", "net_profit_yoy", "gross_margin", "roe", "debt_ratio"]:
        out[col] = float("nan")

    if cfg.fetch_fundamentals and pro is not None and universe is not None:
        start = normalize_date(cfg.start_date)
        end = normalize_date(cfg.end_date)
        fina_frames: list["pd.DataFrame"] = []
        for code in sorted(universe["ts_code"].dropna().unique().tolist()):
            safe_code = code.replace(".", "_")
            fina = fetch_or_cache(
                dirs["raw"] / f"fina_indicator_{safe_code}_{start}_{end}",
                cfg.force_refresh,
                lambda code=code: optional_tushare_call(
                    pro,
                    "fina_indicator",
                    sleep_seconds=cfg.sleep_seconds,
                    ts_code=code,
                    start_date=start,
                    end_date=end,
                ),
            )
            if not fina.empty:
                fina_frames.append(fina)

        if fina_frames:
            fina_all = pd.concat(fina_frames, ignore_index=True)
            if "ann_date" in fina_all.columns:
                fina_all["ann_date"] = fina_all["ann_date"].astype(str)
                rename = {
                    "q_gr_yoy": "revenue_yoy",
                    "or_yoy": "revenue_yoy",
                    "netprofit_yoy": "net_profit_yoy",
                    "grossprofit_margin": "gross_margin",
                    "roe": "roe",
                    "debt_to_assets": "debt_ratio",
                }
                fina_all = fina_all.rename(columns={k: v for k, v in rename.items() if k in fina_all.columns})
                fina_all = _coalesce_duplicate_columns(fina_all)
                keep = [
                    c
                    for c in [
                        "ts_code",
                        "ann_date",
                        "end_date",
                        "revenue_yoy",
                        "net_profit_yoy",
                        "gross_margin",
                        "roe",
                        "debt_ratio",
                    ]
                    if c in fina_all.columns
                ]
                fina_all = fina_all[keep].sort_values(["ts_code", "ann_date"]).copy()
                fina_all["_ann_int"] = pd.to_numeric(
                    fina_all["ann_date"].astype(str).str.replace(r"[^0-9]", "", regex=True),
                    errors="coerce",
                )
                invalid_ann_dates = int(fina_all["_ann_int"].isna().sum())
                if invalid_ann_dates:
                    log(f"Skipped fundamental rows with invalid ann_date: rows={invalid_ann_dates}")
                fina_all = fina_all.dropna(subset=["_ann_int"]).copy()

                if not fina_all.empty:
                    out["_as_of_int"] = pd.to_numeric(
                        out["as_of_date"].astype(str).str.replace(r"[^0-9]", "", regex=True),
                        errors="coerce",
                    )
                    invalid_as_of = int(out["_as_of_int"].isna().sum())
                    if invalid_as_of:
                        log(f"Skipped fundamental merge for rows with invalid as_of_date: rows={invalid_as_of}")
                    merge_base = out.dropna(subset=["_as_of_int"]).copy()

                    if not merge_base.empty:
                        merge_base["_as_of_int"] = merge_base["_as_of_int"].astype("int64")
                        fina_all["_ann_int"] = fina_all["_ann_int"].astype("int64")
                        merged = pd.merge_asof(
                            merge_base.sort_values("_as_of_int"),
                            fina_all.sort_values("_ann_int"),
                            left_on="_as_of_int",
                            right_on="_ann_int",
                            by="ts_code",
                            direction="backward",
                        )
                        for col in ["revenue_yoy", "net_profit_yoy", "gross_margin", "roe", "debt_ratio"]:
                            if f"{col}_y" in merged.columns:
                                right = _coalesced_column(merged, f"{col}_y")
                                left = _coalesced_column(merged, f"{col}_x") if f"{col}_x" in merged.columns else None
                                merged[col] = right.combine_first(left) if left is not None else right
                            elif col not in merged.columns:
                                merged[col] = float("nan")
                        if "end_date" in merged.columns:
                            merged["report_period"] = merged["end_date"].fillna("")
                        if "ann_date" in merged.columns:
                            merged["publish_time"] = merged["ann_date"].fillna(merged["as_of_date"]).map(_date_to_iso)
                        out = merged[[c for c in merged.columns if not c.endswith("_x") and not c.endswith("_y")]].copy()
                        out = out.drop(columns=[c for c in ["_as_of_int", "_ann_int"] if c in out.columns])

    save_table(out, dirs["processed"] / "fundamental_snapshot")
    log(f"Fundamental snapshot built: rows={len(out)}")
    return out


def build_document_tables(
    cfg: MVPConfig,
    dirs: dict[str, Path],
    pro: Any | None = None,
    universe: "pd.DataFrame | None" = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    require_pandas()
    factors = read_table(dirs["processed"] / "factor_snapshot").copy()
    factors["trade_date"] = factors["trade_date"].astype(str)

    ann_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []

    if cfg.fetch_optional_docs and pro is not None:
        start = normalize_date(cfg.start_date)
        end = normalize_date(cfg.end_date)
        raw_ann_frames: list["pd.DataFrame"] = []
        raw_codes = sorted(universe["ts_code"].dropna().astype(str).unique().tolist()) if universe is not None else []
        for code in raw_codes:
            safe_code = code.replace(".", "_")
            raw_ann = fetch_or_cache(
                dirs["raw"] / f"anns_d_{safe_code}_{start}_{end}",
                cfg.force_refresh,
                lambda code=code: optional_tushare_call(
                    pro,
                    "anns_d",
                    sleep_seconds=cfg.sleep_seconds,
                    ts_code=code,
                    start_date=start,
                    end_date=end,
                ),
            )
            if not raw_ann.empty:
                raw_ann_frames.append(raw_ann)
        if raw_ann_frames:
            raw_ann_all = pd.concat(raw_ann_frames, ignore_index=True)
            save_table(raw_ann_all, dirs["processed"] / "announcements_raw_tushare")
            universe_set = set(universe["ts_code"].astype(str)) if universe is not None else set()
            for idx, row in raw_ann_all.iterrows():
                code = str(row.get("ts_code", row.get("code", "")))
                if not code or (universe_set and code not in universe_set):
                    continue
                date = str(row.get("ann_date", row.get("trade_date", row.get("publish_time", ""))))
                if not re.sub(r"[^0-9]", "", date):
                    continue
                title = str(row.get("title", row.get("ann_title", "")))
                ann_rows.append(
                    {
                        "doc_id": f"ann_real_{idx}",
                        "ts_code": code,
                        "publish_time": _date_to_iso(date, "18:00:00"),
                        "effective_time": _date_to_iso(date, "18:00:00"),
                        "title": title,
                        "summary": title[:280],
                        "content": str(row.get("content", title))[:1000],
                        "source": "tushare_anns_d",
                        "doc_type": str(row.get("ann_type", "announcement")),
                        "importance_score": 0.80,
                    }
                )

    for _, row in factors.iterrows():
        code = str(row.get("ts_code", ""))
        trade_date = str(row.get("trade_date", ""))
        name = str(row.get("company_name", row.get("name", code)))
        industry = str(row.get("industry_name", row.get("industry", "")))
        ret_5d = _safe_float(row.get("ret_5d"), 0.0) or 0.0
        ret_20d = _safe_float(row.get("ret_20d"), 0.0) or 0.0
        rs_20d = _safe_float(row.get("rs_market_20d"), 0.0) or 0.0
        vol_rank = _safe_float(row.get("vol_rank"), 0.5) or 0.5
        turnover_rank = _safe_float(row.get("liquidity_rank"), 0.5) or 0.5
        pe_ttm = _safe_float(row.get("pe_ttm"))
        pb = _safe_float(row.get("pb"))

        signal_bits = []
        if abs(ret_5d) >= 0.04:
            signal_bits.append(f"近5日收益{ret_5d:.2%}")
        if abs(rs_20d) >= 0.06:
            signal_bits.append(f"近20日相对市场收益{rs_20d:.2%}")
        if vol_rank >= 0.80:
            signal_bits.append("波动率处于横截面较高分位")
        if turnover_rank >= 0.80:
            signal_bits.append("成交活跃度处于横截面较高分位")
        if signal_bits:
            direction = "偏正面" if (ret_20d + rs_20d) >= 0 else "偏负面"
            summary = f"{name}（{code}）{trade_date}量价事件：{'；'.join(signal_bits)}，短期交易信号{direction}。"
            news_rows.append(
                {
                    "news_id": f"news_factor_{code}_{trade_date}",
                    "ts_code": code,
                    "tickers": code,
                    "publish_time": _date_to_iso(trade_date, "15:35:00"),
                    "effective_time": _date_to_iso(trade_date, "15:35:00"),
                    "title": f"{name}量价相对强弱更新",
                    "summary": summary,
                    "content": summary,
                    "source": "derived_price_event",
                    "category": "company",
                    "sentiment_score": max(-1.0, min(1.0, (ret_20d + rs_20d) / 0.20)),
                    "relevance_score": 0.70,
                    "industry_id": row.get("industry_id", ""),
                    "industry_name": industry,
                }
            )

        if pe_ttm is not None or pb is not None:
            val_parts = []
            if pe_ttm is not None:
                val_parts.append(f"PE_TTM={pe_ttm:.2f}")
            if pb is not None:
                val_parts.append(f"PB={pb:.2f}")
            summary = f"{name}（{code}）估值快照：{', '.join(val_parts)}；行业={industry}；动量20日={ret_20d:.2%}。"
            ann_rows.append(
                {
                    "doc_id": f"ann_snapshot_{code}_{trade_date}",
                    "ts_code": code,
                    "company_name": name,
                    "publish_time": _date_to_iso(trade_date, "15:40:00"),
                    "effective_time": _date_to_iso(trade_date, "15:40:00"),
                    "title": f"{name}基本面与估值快照",
                    "summary": summary,
                    "content": summary,
                    "source": "derived_fundamental_snapshot",
                    "doc_type": "fundamental_snapshot",
                    "importance_score": 0.55,
                    "industry_id": row.get("industry_id", ""),
                    "industry_name": industry,
                }
            )

    announcements = pd.DataFrame(ann_rows)
    news = pd.DataFrame(news_rows)
    if announcements.empty:
        announcements = pd.DataFrame(
            columns=[
                "doc_id",
                "ts_code",
                "company_name",
                "publish_time",
                "effective_time",
                "title",
                "summary",
                "content",
                "source",
                "doc_type",
                "importance_score",
                "industry_id",
                "industry_name",
            ]
        )
    if news.empty:
        news = pd.DataFrame(
            columns=[
                "news_id",
                "ts_code",
                "tickers",
                "publish_time",
                "effective_time",
                "title",
                "summary",
                "content",
                "source",
                "category",
                "sentiment_score",
                "relevance_score",
                "industry_id",
                "industry_name",
            ]
        )
    save_table(announcements, dirs["processed"] / "announcements")
    save_table(news, dirs["processed"] / "news")
    log(f"Document tables built: announcements={len(announcements)} news={len(news)}")
    return announcements, news


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
    factors["industry_future_return"] = factors.groupby(["trade_date", "industry_id"])["stock_return"].transform("mean")
    factors["industry_member_count"] = factors.groupby(["trade_date", "industry_id"])["ts_code"].transform("count")
    use_industry = factors["industry_member_count"].fillna(0) >= 3
    factors["benchmark_future_return"] = factors["market_future_return"]
    factors.loc[use_industry, "benchmark_future_return"] = factors.loc[use_industry, "industry_future_return"]
    factors["future_relative_return"] = factors["stock_return"] - factors["benchmark_future_return"]
    factors["relative_benchmark"] = "market"
    factors.loc[use_industry, "relative_benchmark"] = "industry"
    factors["future_volatility"] = factors["vol_20d"].fillna(0.03)
    factors = factors.dropna(subset=["future_relative_return", "ret_20d", "vol_20d"]).copy()
    factors["cross_section_rank"] = factors.groupby("trade_date")["future_relative_return"].rank(pct=True)
    factors["label"] = "neutral"
    up_q = min(max(float(cfg.up_quantile), 0.50), 0.95)
    down_q = min(max(float(cfg.down_quantile), 0.05), 0.50)
    factors.loc[factors["cross_section_rank"] >= up_q, "label"] = "up"
    factors.loc[factors["cross_section_rank"] <= down_q, "label"] = "down"
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
    tasks["max_tool_calls"] = int(cfg.max_tool_calls)
    tasks["tools"] = [
        [
            "get_price_factors",
            "get_market_context",
            "get_industry_context",
            "get_peer_context",
            "get_fundamental_snapshot",
            "search_announcements",
            "search_news",
        ]
        for _ in range(len(tasks))
    ]
    tasks["tradable_at_t_plus_1"] = tasks["entry_adj_close"].notna().astype(int)
    limit_up = tasks["limit_up_recent_5d"] if "limit_up_recent_5d" in tasks.columns else pd.Series(0, index=tasks.index)
    limit_down = (
        tasks["limit_down_recent_5d"] if "limit_down_recent_5d" in tasks.columns else pd.Series(0, index=tasks.index)
    )
    tasks["has_limit_issue"] = (limit_up.fillna(0).astype(float).gt(0) | limit_down.fillna(0).astype(float).gt(0)).astype(int)

    if cfg.max_tasks and len(tasks) > cfg.max_tasks:
        dates = sorted(tasks["trade_date"].unique().tolist())
        approx_per_date = max(1, int(tasks.groupby("trade_date")["ts_code"].nunique().median()))
        max_dates = max(1, int(cfg.max_tasks) // approx_per_date)
        if max_dates < len(dates):
            step = max(1, len(dates) // max_dates)
            keep_dates = set(dates[::step][:max_dates])
            tasks = tasks[tasks["trade_date"].isin(keep_dates)].copy()
        if len(tasks) > cfg.max_tasks:
            tasks = tasks.groupby("trade_date", group_keys=False).head(max(1, int(cfg.max_tasks) // max(1, len(tasks["trade_date"].unique()))))
        tasks = tasks.sort_values(["trade_date", "ts_code"])

    task_cols = [
        "sample_id",
        "ts_code",
        "name",
        "company_name",
        "market",
        "industry_id",
        "industry",
        "industry_name",
        "trade_date",
        "as_of",
        "horizon",
        "entry_date",
        "exit_date",
        "max_tool_calls",
        "tools",
        "tradable_at_t_plus_1",
        "has_limit_issue",
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
        "industry_future_return",
        "market_future_return",
        "benchmark_future_return",
        "relative_benchmark",
        "future_relative_return",
        "future_volatility",
        "cross_section_rank",
        "label",
        "label_id",
        "tradable_at_t_plus_1",
        "has_limit_issue",
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
    build_industry_and_peer_context(cfg, dirs)
    build_fundamental_snapshot(cfg, dirs, pro=pro, universe=universe)
    build_document_tables(cfg, dirs, pro=pro, universe=universe)
    build_tasks_and_labels(cfg, dirs)
