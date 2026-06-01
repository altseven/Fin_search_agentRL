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
    return _as_of_to_date(s)


def _as_of_to_date(as_of: str) -> str:
    digits = re.sub(r"[^0-9]", "", str(as_of))
    if len(digits) >= 8:
        return digits[:8]
    return normalize_date(as_of)


def _clamp_top_k(top_k: int, lo: int = 1, hi: int = 10) -> int:
    try:
        value = int(top_k)
    except Exception:
        value = lo
    return max(lo, min(hi, value))


def _default_start_date(as_of_date: str, lookback_calendar_days: int = 180) -> str:
    try:
        dt = pd.to_datetime(as_of_date)
        start = dt - pd.Timedelta(days=max(1, int(lookback_calendar_days)))
        return start.strftime("%Y%m%d")
    except Exception:
        year = max(1900, int(as_of_date[:4]) - 1)
        return f"{year}{as_of_date[4:]}"


def _keyword_score(text: str, query: str) -> float:
    if not query:
        return 0.0
    terms = [t.lower() for t in re.split(r"\s+", str(query).strip()) if t]
    if not terms:
        return 0.0
    lowered = str(text).lower()
    hits = sum(1 for t in terms if t in lowered)
    return hits / max(1, len(terms))


def _safe_table_exists(name: str) -> bool:
    stem = _tool_processed_dir() / name
    return any(table_path(stem, suffix).exists() for suffix in (".parquet", ".pkl", ".csv"))


def _text_trim(value: Any, limit: int = 240) -> str:
    text = str(value or "")
    return text[:limit]


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
            "turnover_5d_avg": _safe_float(row.get("turnover_5d_avg")),
            "turnover_20d_avg": _safe_float(row.get("turnover_20d_avg")),
            "amount_20d_avg": _safe_float(row.get("amount_20d_avg")),
            "momentum_rank": _safe_float(row.get("momentum_rank")),
            "momentum_rank_in_industry": _safe_float(row.get("momentum_rank_in_industry")),
            "rs_rank": _safe_float(row.get("rs_rank")),
            "rs_rank_in_industry": _safe_float(row.get("rs_rank_in_industry")),
            "liquidity_rank": _safe_float(row.get("liquidity_rank")),
            "liquidity_rank_in_industry": _safe_float(row.get("liquidity_rank_in_industry")),
            "vol_rank": _safe_float(row.get("vol_rank")),
            "limit_up_recent_5d": _safe_float(row.get("limit_up_recent_5d"), 0.0),
            "limit_down_recent_5d": _safe_float(row.get("limit_down_recent_5d"), 0.0),
            "suspended_recent_20d": _safe_float(row.get("suspended_recent_20d"), 0.0),
            "pe_ttm": _safe_float(row.get("pe_ttm")),
            "pb": _safe_float(row.get("pb")),
            "total_mv": _safe_float(row.get("total_mv")),
        }
        summary = (
            f"{code} ret_5d={factors['ret_5d']}, ret_20d={factors['ret_20d']}, "
            f"rs_market_20d={factors['rs_market_20d']}, "
            f"industry_momentum_rank={factors['momentum_rank_in_industry']}, "
            f"vol_20d={factors['vol_20d']}, turnover_20d_avg={factors['turnover_20d_avg']}."
        )
        return {
            "tool_name": "get_price_factors",
            "status": "ok",
            "result_count": 1,
            "as_of_date": date,
            "as_of": as_of_date,
            "ts_code": code,
            "name": row.get("company_name", row.get("name", "")),
            "industry_id": row.get("industry_id", ""),
            "industry": row.get("industry_name", row.get("industry", "")),
            "lookback_days": int(lookback_days),
            "factors": factors,
            "summary": summary,
            "warning": None,
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
            "result_count": 1,
            "index_code": index_code,
            "as_of_date": date,
            "as_of": as_of_date,
            "metrics": metrics,
            "summary": (
                f"Market ret_20d={metrics['market_ret_20d']}, "
                f"vol_20d={metrics['market_vol_20d']}."
            ),
            "warning": None,
        }
    except Exception as exc:
        return {
            "tool_name": "get_market_context",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def get_industry_context(industry_id: str, as_of_date: str, lookback_days: int = 20) -> dict:
    """Get point-in-time industry context for an A-share industry group.

    Args:
        industry_id: Industry identifier from the task prompt, such as "IND_银行".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        lookback_days: Lookback window length; currently used as metadata only.
    """
    try:
        date = _as_of_to_date(as_of_date)
        df = _load_tool_table("industry_context")
        row_df = df[(df["industry_id"].astype(str) == str(industry_id)) & (df["trade_date"].astype(str) == date)]
        if row_df.empty:
            return {
                "tool_name": "get_industry_context",
                "status": "error",
                "error_type": "not_found",
                "as_of_date": date,
                "industry_id": industry_id,
                "message": f"No industry context for {industry_id} at {date}.",
                "results": [],
            }
        row = row_df.iloc[0]
        metrics = {
            "member_count": _safe_float(row.get("member_count"), 0.0),
            "industry_ret_5d": _safe_float(row.get("industry_ret_5d")),
            "industry_ret_20d": _safe_float(row.get("industry_ret_20d")),
            "industry_vol_20d": _safe_float(row.get("industry_vol_20d")),
            "turnover_20d_avg": _safe_float(row.get("turnover_20d_avg")),
            "valuation_pe_percentile_2y": _safe_float(row.get("valuation_pe_percentile_2y")),
            "turnover_percentile_1y": _safe_float(row.get("turnover_percentile_1y")),
            "momentum_rank_mean": _safe_float(row.get("momentum_rank_mean")),
            "rs_rank_mean": _safe_float(row.get("rs_rank_mean")),
        }
        summary = (
            f"{row.get('industry_name', industry_id)} industry ret_20d={metrics['industry_ret_20d']}, "
            f"vol_20d={metrics['industry_vol_20d']}, valuation_percentile={metrics['valuation_pe_percentile_2y']}, "
            f"turnover_percentile={metrics['turnover_percentile_1y']}."
        )
        return {
            "tool_name": "get_industry_context",
            "status": "ok",
            "result_count": 1,
            "as_of_date": date,
            "as_of": as_of_date,
            "industry_id": str(industry_id),
            "industry_name": row.get("industry_name", ""),
            "lookback_days": int(lookback_days),
            "metrics": metrics,
            "summary": summary,
            "warning": None,
        }
    except Exception as exc:
        return {
            "tool_name": "get_industry_context",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def get_peer_context(ts_code: str, as_of_date: str, top_k: int = 5) -> dict:
    """Get same-industry peer comparison visible at the prediction date.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        top_k: Maximum number of peer rows to return.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _as_of_to_date(as_of_date)
        top_k = _clamp_top_k(top_k, hi=8)
        df = _load_tool_table("peer_context")
        row_df = df[(df["ts_code"] == code) & (df["trade_date"].astype(str) == date)]
        if row_df.empty:
            return {
                "tool_name": "get_peer_context",
                "status": "error",
                "error_type": "not_found",
                "as_of_date": date,
                "ts_code": code,
                "message": f"No peer row for {code} at {date}.",
                "results": [],
            }
        row = row_df.iloc[0]
        industry_id = str(row.get("industry_id", ""))
        peers = df[(df["trade_date"].astype(str) == date) & (df["industry_id"].astype(str) == industry_id)].copy()
        peers = peers[peers["ts_code"] != code].sort_values("peer_score", ascending=False).head(top_k)
        results = []
        for _, peer in peers.iterrows():
            results.append(
                {
                    "ts_code": str(peer.get("ts_code", "")),
                    "name": str(peer.get("company_name", peer.get("name", ""))),
                    "ret_20d": _safe_float(peer.get("ret_20d")),
                    "rs_market_20d": _safe_float(peer.get("rs_market_20d")),
                    "momentum_rank_in_industry": _safe_float(peer.get("momentum_rank_in_industry")),
                    "liquidity_rank_in_industry": _safe_float(peer.get("liquidity_rank_in_industry")),
                    "pe_ttm": _safe_float(peer.get("pe_ttm")),
                    "pb": _safe_float(peer.get("pb")),
                }
            )
        return {
            "tool_name": "get_peer_context",
            "status": "ok",
            "as_of_date": date,
            "as_of": as_of_date,
            "ts_code": code,
            "industry_id": industry_id,
            "industry_name": row.get("industry_name", ""),
            "result_count": len(results),
            "results": results,
            "summary": f"Found {len(results)} same-industry peers for {code}; compare momentum, relative strength and valuation.",
            "warning": None,
        }
    except Exception as exc:
        return {
            "tool_name": "get_peer_context",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def get_fundamental_snapshot(ts_code: str, as_of_date: str) -> dict:
    """Get the latest point-in-time fundamental and valuation snapshot for one stock.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _as_of_to_date(as_of_date)
        df = _load_tool_table("fundamental_snapshot")
        sub = df[(df["ts_code"] == code) & (df["as_of_date"].astype(str) <= date)].copy()
        if sub.empty:
            return {
                "tool_name": "get_fundamental_snapshot",
                "status": "error",
                "error_type": "not_found",
                "as_of_date": date,
                "ts_code": code,
                "message": f"No fundamental snapshot for {code} before {date}.",
                "results": [],
            }
        row = sub.sort_values("as_of_date").iloc[-1]
        fundamentals = {
            "report_period": str(row.get("report_period", "")),
            "publish_time": str(row.get("publish_time", "")),
            "revenue_yoy": _safe_float(row.get("revenue_yoy")),
            "net_profit_yoy": _safe_float(row.get("net_profit_yoy")),
            "gross_margin": _safe_float(row.get("gross_margin")),
            "roe": _safe_float(row.get("roe")),
            "debt_ratio": _safe_float(row.get("debt_ratio")),
            "pe_ttm": _safe_float(row.get("pe_ttm")),
            "pb": _safe_float(row.get("pb")),
            "total_mv": _safe_float(row.get("total_mv")),
            "circ_mv": _safe_float(row.get("circ_mv")),
        }
        summary = (
            f"{code} fundamentals as of {date}: PE_TTM={fundamentals['pe_ttm']}, PB={fundamentals['pb']}, "
            f"revenue_yoy={fundamentals['revenue_yoy']}, net_profit_yoy={fundamentals['net_profit_yoy']}."
        )
        return {
            "tool_name": "get_fundamental_snapshot",
            "status": "ok",
            "result_count": 1,
            "as_of_date": date,
            "as_of": as_of_date,
            "ts_code": code,
            "name": row.get("company_name", row.get("name", "")),
            "industry_id": row.get("industry_id", ""),
            "industry_name": row.get("industry_name", ""),
            "fundamentals": fundamentals,
            "summary": summary,
            "warning": None,
        }
    except Exception as exc:
        return {
            "tool_name": "get_fundamental_snapshot",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def search_announcements(
    ts_code: str,
    as_of_date: str,
    query: str = "",
    top_k: int = 5,
    start_time: str = "",
    end_time: str = "",
    doc_types: str = "",
) -> dict:
    """Search point-in-time announcements if an announcement table is available.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        query: Keyword query for title or summary.
        top_k: Maximum number of announcement rows to return.
        start_time: Optional start date in YYYYMMDD or YYYY-MM-DD format.
        end_time: Optional end date in YYYYMMDD or YYYY-MM-DD format; future dates are clipped to as_of_date.
        doc_types: Optional comma-separated document type filter.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _as_of_to_date(as_of_date)
        top_k = _clamp_top_k(top_k)
        if not _safe_table_exists("announcements"):
            return {
                "tool_name": "search_announcements",
                "status": "ok",
                "as_of_date": date,
                "as_of": as_of_date,
                "ts_code": code,
                "result_count": 0,
                "results": [],
                "warning": "No announcements table is available in this MVP run.",
            }
        df = _load_tool_table("announcements")
        date_col = "effective_time" if "effective_time" in df.columns else "publish_time"
        date_text = df[date_col].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:8]
        start = _as_of_to_date(start_time) if start_time else _default_start_date(date, 365)
        requested_end = _as_of_to_date(end_time) if end_time else date
        warning = None
        end = requested_end
        if requested_end > date:
            end = date
            warning = "end_time was later than as_of_date and was clipped to avoid future leakage."
        mask = (df["ts_code"].astype(str) == code) & (date_text <= end) & (date_text >= start)
        if doc_types:
            wanted = {x.strip().lower() for x in doc_types.split(",") if x.strip()}
            if wanted and "doc_type" in df.columns:
                mask &= df["doc_type"].astype(str).str.lower().isin(wanted)
        if query:
            q_terms = [x.lower() for x in re.split(r"\s+", query.strip()) if x]
            if q_terms:
                title = df["title"].astype(str) if "title" in df.columns else pd.Series([""] * len(df), index=df.index)
                summary = df["summary"].astype(str) if "summary" in df.columns else pd.Series([""] * len(df), index=df.index)
                text = (title + " " + summary).str.lower()
                mask &= text.map(lambda x: any(t in x for t in q_terms))
        out = df[mask].copy()
        if not out.empty:
            out["_keyword_score"] = (
                out.get("title", "").astype(str) + " " + out.get("summary", "").astype(str)
            ).map(lambda x: _keyword_score(x, query))
            out["_importance"] = out.get("importance_score", 0.0).map(lambda x: _safe_float(x, 0.0) or 0.0)
            out["_date_text"] = date_text.loc[out.index]
            out = out.sort_values(["_keyword_score", "_importance", "_date_text"], ascending=False)
        out = out.head(top_k)
        results = []
        for _, row in out.iterrows():
            results.append(
                {
                    "doc_id": str(row.get("doc_id", row.get("ann_id", row.get("id", "")))),
                    "publish_time": str(row.get("publish_time", row.get(date_col, ""))),
                    "effective_time": str(row.get("effective_time", row.get(date_col, ""))),
                    "title": str(row.get("title", "")),
                    "doc_type": str(row.get("doc_type", "")),
                    "summary": _text_trim(row.get("summary", ""), 260),
                    "source": str(row.get("source", "")),
                    "importance_score": _safe_float(row.get("importance_score")),
                    "relevance": _safe_float(row.get("_keyword_score"), 0.0),
                }
            )
        return {
            "tool_name": "search_announcements",
            "status": "ok",
            "as_of_date": date,
            "as_of": as_of_date,
            "ts_code": code,
            "query": query,
            "start_time": start,
            "end_time": end,
            "result_count": len(results),
            "results": results,
            "warning": warning,
        }
    except Exception as exc:
        return {
            "tool_name": "search_announcements",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


@function_tool
def search_news(
    ts_code: str,
    as_of_date: str,
    query: str = "",
    top_k: int = 5,
    start_time: str = "",
    end_time: str = "",
) -> dict:
    """Search point-in-time stock, industry, and market news/events.

    Args:
        ts_code: Stock code such as "600519.SH".
        as_of_date: Prediction date in YYYYMMDD or YYYY-MM-DD format.
        query: Keyword query for title, summary, content, industry, or event type.
        top_k: Maximum number of news rows to return.
        start_time: Optional start date in YYYYMMDD or YYYY-MM-DD format.
        end_time: Optional end date in YYYYMMDD or YYYY-MM-DD format; future dates are clipped to as_of_date.
    """
    try:
        code = normalize_ts_code(ts_code)
        date = _as_of_to_date(as_of_date)
        top_k = _clamp_top_k(top_k)
        if not _safe_table_exists("news"):
            return {
                "tool_name": "search_news",
                "status": "ok",
                "as_of_date": date,
                "as_of": as_of_date,
                "ts_code": code,
                "query": query,
                "result_count": 0,
                "results": [],
                "warning": "No news table is available.",
            }
        df = _load_tool_table("news")
        date_col = "effective_time" if "effective_time" in df.columns else "publish_time"
        date_text = df[date_col].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:8]
        start = _as_of_to_date(start_time) if start_time else _default_start_date(date, 180)
        requested_end = _as_of_to_date(end_time) if end_time else date
        warning = None
        end = requested_end
        if requested_end > date:
            end = date
            warning = "end_time was later than as_of_date and was clipped to avoid future leakage."

        ticker_text = df.get("tickers", df.get("ts_code", "")).astype(str)
        mask = ticker_text.str.contains(code, regex=False) | (df.get("ts_code", "").astype(str) == code)
        mask &= (date_text <= end) & (date_text >= start)
        text = (
            df.get("title", "").astype(str)
            + " "
            + df.get("summary", "").astype(str)
            + " "
            + df.get("content", "").astype(str)
            + " "
            + df.get("industry_name", "").astype(str)
            + " "
            + df.get("category", "").astype(str)
        )
        if query:
            scores = text.map(lambda x: _keyword_score(x, query))
            mask &= scores > 0
        else:
            scores = text.map(lambda x: _keyword_score(x, query))

        out = df[mask].copy()
        if not out.empty:
            out["_keyword_score"] = scores.loc[out.index]
            out["_relevance"] = out.get("relevance_score", 0.0).map(lambda x: _safe_float(x, 0.0) or 0.0)
            out["_date_text"] = date_text.loc[out.index]
            out = out.sort_values(["_keyword_score", "_relevance", "_date_text"], ascending=False)
        out = out.head(top_k)
        results = []
        for _, row in out.iterrows():
            results.append(
                {
                    "news_id": str(row.get("news_id", row.get("id", ""))),
                    "publish_time": str(row.get("publish_time", row.get(date_col, ""))),
                    "effective_time": str(row.get("effective_time", row.get(date_col, ""))),
                    "title": str(row.get("title", "")),
                    "summary": _text_trim(row.get("summary", ""), 260),
                    "source": str(row.get("source", "")),
                    "category": str(row.get("category", "")),
                    "relevance": _safe_float(row.get("_keyword_score"), 0.0),
                    "sentiment_score": _safe_float(row.get("sentiment_score")),
                }
            )
        return {
            "tool_name": "search_news",
            "status": "ok",
            "as_of_date": date,
            "as_of": as_of_date,
            "ts_code": code,
            "query": query,
            "start_time": start,
            "end_time": end,
            "result_count": len(results),
            "results": results,
            "warning": warning,
        }
    except Exception as exc:
        return {
            "tool_name": "search_news",
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
