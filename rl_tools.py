from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Callable

from rl_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MARKET_INDEX,
    normalize_date,
    normalize_ts_code,
    pd,
    read_table,
    require_pandas,
    table_path,
)

try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover - this file also runs without verl installed

    def function_tool(arg: Any = None, **_: Any) -> Callable:
        if callable(arg):
            return arg

        def deco(fn: Callable) -> Callable:
            return fn

        return deco


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
