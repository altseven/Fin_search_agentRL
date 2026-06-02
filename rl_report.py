from __future__ import annotations

import ast
import csv
import html
import json
import math
import re
import textwrap
import time
from pathlib import Path
from typing import Any

from rl_common import CLASSES, DEFAULT_DATA_DIR, DEFAULT_RESULT_DIR, log, maybe_read_table, pd, require_pandas


SPLIT_ORDER = ["train", "valid", "test"]
LABEL_COLORS = {"up": "#059669", "neutral": "#d97706", "down": "#dc2626"}
SERIES_COLORS = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#f59e0b"]


def find_latest_run_dir(result_dir: str | Path = DEFAULT_RESULT_DIR) -> Path:
    base = Path(result_dir).expanduser().resolve()
    candidates = [p for p in base.glob("*_result*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def generate_report_for_latest(
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    return generate_report_for_run(find_latest_run_dir(result_dir), data_dir=data_dir)


def generate_report_for_run(
    run_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    extra_log_paths: list[str | Path] | None = None,
) -> Path:
    require_pandas()
    run = Path(run_dir).expanduser().resolve()
    data = Path(data_dir).expanduser().resolve()
    report_dir = run / "report"
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _load_json(run / "rollouts" / "rule_baseline_metrics.json")
    baseline_rows = _baseline_metric_rows(baseline_metrics)
    predictions = _read_cached_table(run / "rollouts" / "rule_baseline_predictions")
    task_labels = _load_task_label_frame(data)
    metric_rows = _load_training_metric_rows(run, extra_log_paths=extra_log_paths)

    dataset_rows = _dataset_summary_rows(task_labels)
    label_rows = _label_distribution_rows(task_labels)
    prediction_rows = _prediction_distribution_rows(predictions)
    reward_progress_rows = _reward_progress_rows(metric_rows)
    baseline_vs_rl_rows = _baseline_vs_rl_rows(baseline_rows, reward_progress_rows)
    top_rows = _top_prediction_rows(predictions)

    artifacts: dict[str, str] = {}
    artifacts["baseline_metrics_csv"] = str(_write_rows(tables_dir / "baseline_metrics.csv", baseline_rows))
    artifacts["baseline_metrics_md"] = str(_write_markdown_table(tables_dir / "baseline_metrics.md", baseline_rows))
    artifacts["dataset_summary_csv"] = str(_write_rows(tables_dir / "dataset_summary.csv", dataset_rows))
    artifacts["dataset_summary_md"] = str(_write_markdown_table(tables_dir / "dataset_summary.md", dataset_rows))
    artifacts["label_distribution_csv"] = str(_write_rows(tables_dir / "label_distribution.csv", label_rows))
    artifacts["prediction_distribution_csv"] = str(_write_rows(tables_dir / "prediction_distribution.csv", prediction_rows))
    artifacts["training_metrics_csv"] = str(_write_rows(tables_dir / "training_metrics.csv", metric_rows))
    artifacts["reward_progress_csv"] = str(_write_rows(tables_dir / "reward_progress.csv", reward_progress_rows))
    artifacts["reward_progress_md"] = str(_write_markdown_table(tables_dir / "reward_progress.md", reward_progress_rows))
    artifacts["baseline_vs_rl_reward_csv"] = str(_write_rows(tables_dir / "baseline_vs_rl_reward.csv", baseline_vs_rl_rows))
    artifacts["baseline_vs_rl_reward_md"] = str(
        _write_markdown_table(tables_dir / "baseline_vs_rl_reward.md", baseline_vs_rl_rows)
    )
    artifacts["top_predictions_csv"] = str(_write_rows(tables_dir / "top_predictions.csv", top_rows))
    artifacts["top_predictions_md"] = str(_write_markdown_table(tables_dir / "top_predictions.md", top_rows))

    chart_paths = _write_charts(
        figures_dir=figures_dir,
        baseline_rows=baseline_rows,
        label_rows=label_rows,
        prediction_rows=prediction_rows,
        predictions=predictions,
        reward_progress_rows=reward_progress_rows,
        metric_rows=metric_rows,
    )
    artifacts.update({name: str(path) for name, path in chart_paths.items()})

    index_path = _write_report_index(
        report_dir=report_dir,
        run_dir=run,
        baseline_rows=baseline_rows,
        dataset_rows=dataset_rows,
        reward_progress_rows=reward_progress_rows,
        baseline_vs_rl_rows=baseline_vs_rl_rows,
        artifacts=artifacts,
    )
    artifacts["report_index_md"] = str(index_path)

    pdf_path = report_dir / "stock_agent_rl_report.pdf"
    _write_pdf_report(
        pdf_path=pdf_path,
        run_dir=run,
        baseline_rows=baseline_rows,
        dataset_rows=dataset_rows,
        reward_progress_rows=reward_progress_rows,
        baseline_vs_rl_rows=baseline_vs_rl_rows,
        label_rows=label_rows,
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        artifacts=artifacts,
    )
    artifacts["summary_pdf"] = str(pdf_path)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run),
        "data_dir": str(data),
        "artifacts": artifacts,
        "notes": [
            "Reward progress is computed from verl file logger JSONL when available, then from output logs as fallback.",
            "If no RL metric rows exist, the report still summarizes data and rule-baseline reward.",
        ],
    }
    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Report written: {pdf_path}")
    return report_dir


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Could not read JSON {path}: {exc}")
        return {}


def _read_cached_table(stem: Path) -> "pd.DataFrame | None":
    try:
        return maybe_read_table(stem)
    except Exception as exc:
        log(f"Could not read table {stem}: {exc}")
        return None


def _load_task_label_frame(data_dir: Path) -> "pd.DataFrame | None":
    tasks = _read_cached_table(data_dir / "processed" / "tasks")
    labels = _read_cached_table(data_dir / "processed" / "labels")
    if tasks is None and labels is None:
        return None
    if tasks is None:
        return labels
    if labels is None:
        return tasks
    keys = [c for c in ["sample_id", "ts_code", "trade_date", "horizon", "entry_date", "exit_date"] if c in tasks and c in labels]
    if not keys:
        return tasks
    try:
        return tasks.merge(labels, on=keys, how="left", suffixes=("", "_label"))
    except Exception as exc:
        log(f"Could not merge tasks/labels for report: {exc}")
        return tasks


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _fmt(value: Any, digits: int = 4) -> str:
    f = _safe_float(value)
    if f is None:
        return ""
    if abs(f) >= 100:
        return f"{f:.2f}"
    return f"{f:.{digits}f}"


def _split_key(split: Any) -> tuple[int, str]:
    s = str(split)
    try:
        return (SPLIT_ORDER.index(s), s)
    except ValueError:
        return (len(SPLIT_ORDER), s)


def _baseline_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in sorted(metrics, key=_split_key):
        item = metrics.get(split) or {}
        rows.append(
            {
                "split": split,
                "n": int(item.get("n", 0) or 0),
                "mean_reward": _safe_float(item.get("mean_reward")),
                "accuracy": _safe_float(item.get("accuracy")),
                "macro_f1": _safe_float(item.get("macro_f1")),
                "brier": _safe_float(item.get("brier")),
                "mean_rank_ic": _safe_float(item.get("mean_rank_ic")),
                "top_bottom_return": _safe_float(item.get("top_bottom_return")),
            }
        )
    return rows


def _dataset_summary_rows(df: "pd.DataFrame | None") -> list[dict[str, Any]]:
    if df is None or df.empty or "split" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for split, sub in df.groupby("split", dropna=False):
        row: dict[str, Any] = {"split": str(split), "n": int(len(sub))}
        if "ts_code" in sub:
            row["stocks"] = int(sub["ts_code"].nunique())
        if "trade_date" in sub:
            row["trade_dates"] = int(sub["trade_date"].nunique())
        if "future_relative_return" in sub:
            row["mean_future_relative_return"] = _safe_float(sub["future_relative_return"].mean())
        for cls in CLASSES:
            if "label" in sub:
                row[f"label_{cls}"] = int((sub["label"] == cls).sum())
        rows.append(row)
    return sorted(rows, key=lambda r: _split_key(r.get("split")))


def _label_distribution_rows(df: "pd.DataFrame | None") -> list[dict[str, Any]]:
    if df is None or df.empty or "split" not in df or "label" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for (split, label), sub in df.groupby(["split", "label"], dropna=False):
        rows.append({"split": str(split), "label": str(label), "n": int(len(sub))})
    return sorted(rows, key=lambda r: (_split_key(r["split"]), CLASSES.index(r["label"]) if r["label"] in CLASSES else 99))


def _prediction_distribution_rows(df: "pd.DataFrame | None") -> list[dict[str, Any]]:
    if df is None or df.empty or "split" not in df or "prediction" not in df:
        return []
    rows: list[dict[str, Any]] = []
    for (split, pred), sub in df.groupby(["split", "prediction"], dropna=False):
        rows.append({"split": str(split), "prediction": str(pred), "n": int(len(sub))})
    return sorted(
        rows, key=lambda r: (_split_key(r["split"]), CLASSES.index(r["prediction"]) if r["prediction"] in CLASSES else 99)
    )


def _top_prediction_rows(df: "pd.DataFrame | None", n_each: int = 15) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = [
        "bucket",
        "split",
        "sample_id",
        "ts_code",
        "trade_date",
        "label",
        "prediction",
        "reward",
        "alpha_score",
        "future_relative_return",
        "p_up",
        "p_neutral",
        "p_down",
    ]
    available = [c for c in cols if c != "bucket" and c in df.columns]
    rows: list[dict[str, Any]] = []
    if "reward" in df:
        ordered = df.sort_values("reward", ascending=False)
        for bucket, sub in (("top_reward", ordered.head(n_each)), ("bottom_reward", ordered.tail(n_each).sort_values("reward"))):
            for _, item in sub.iterrows():
                row = {"bucket": bucket}
                for col in available:
                    value = item.get(col)
                    row[col] = _safe_float(value, value) if col in {"reward", "alpha_score", "future_relative_return", "p_up", "p_neutral", "p_down"} else value
                rows.append(row)
    return rows


def _flatten_dict(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name))
        else:
            out[name] = value
    return out


def _load_training_metric_rows(run_dir: Path, extra_log_paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    candidates: list[Path] = [run_dir / "verl" / "training_metrics.jsonl"]
    candidates.extend(run_dir.rglob("*.jsonl"))
    root = Path(__file__).resolve().parent
    candidates.extend((root / "verl-main" / "stock_agent_rl_mvp").glob("*.jsonl"))
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        rows.extend(_parse_metric_jsonl(resolved))
    log_candidates = [run_dir / "output.log", root / "output.log", root / "output_3090_small.log"]
    if extra_log_paths:
        log_candidates.extend(Path(p) for p in extra_log_paths)
    if not rows:
        for path in log_candidates:
            if path.exists():
                rows.extend(_parse_console_metric_log(path))
    rows.sort(key=lambda r: (_safe_float(r.get("step"), 0.0) or 0.0, str(r.get("source", ""))))
    return rows


def _parse_metric_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        log(f"Could not read metric JSONL {path}: {exc}")
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, dict):
            continue
        flat = _flatten_dict(data)
        numeric = {k: _safe_float(v) for k, v in flat.items()}
        numeric = {k: v for k, v in numeric.items() if v is not None}
        if not numeric:
            continue
        numeric["step"] = int(obj.get("step", len(rows)))
        numeric["source"] = str(path)
        rows.append(numeric)
    return rows


def _parse_console_metric_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step_re = re.compile(r"\bstep:(\d+)\b")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return rows
    for line in lines:
        match = step_re.search(line)
        if not match:
            continue
        row: dict[str, Any] = {"step": int(match.group(1)), "source": str(path)}
        for piece in line.split(" - ")[1:]:
            if ":" not in piece:
                continue
            key, raw_value = piece.split(":", 1)
            value = _safe_float(raw_value.strip())
            if value is not None:
                row[key.strip()] = value
        if len(row) > 2:
            rows.append(row)
    return rows


def _reward_columns(metric_rows: list[dict[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for row in metric_rows:
        for key, value in row.items():
            if key in {"step", "source"}:
                continue
            if _safe_float(value) is None:
                continue
            low = key.lower()
            if "reward" in low or "score" in low:
                columns.add(key)

    def priority(col: str) -> tuple[int, str]:
        low = col.lower()
        patterns = [
            "val",
            "validation",
            "test",
            "critic/rewards/mean",
            "critic/scores/mean",
            "reward",
            "score",
        ]
        for i, pat in enumerate(patterns):
            if pat in low:
                return (i, col)
        return (len(patterns), col)

    return sorted(columns, key=priority)


def _reward_progress_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in _reward_columns(metric_rows):
        points: list[tuple[int, float]] = []
        for row in metric_rows:
            value = _safe_float(row.get(col))
            step = _safe_float(row.get("step"))
            if value is not None and step is not None:
                points.append((int(step), value))
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        best_step, best_value = max(points, key=lambda x: x[1])
        first_step, first_value = points[0]
        last_step, last_value = points[-1]
        rows.append(
            {
                "metric": col,
                "points": len(points),
                "first_step": first_step,
                "first_reward": first_value,
                "last_step": last_step,
                "last_reward": last_value,
                "best_step": best_step,
                "best_reward": best_value,
                "delta_last_minus_first": last_value - first_value,
                "delta_best_minus_first": best_value - first_value,
            }
        )
    return rows


def _baseline_vs_rl_rows(baseline_rows: list[dict[str, Any]], reward_progress_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_valid = _safe_float(_row_for_split(baseline_rows, "valid").get("mean_reward"))
    baseline_train = _safe_float(_row_for_split(baseline_rows, "train").get("mean_reward"))
    baseline_test = _safe_float(_row_for_split(baseline_rows, "test").get("mean_reward"))
    baseline_ref = baseline_valid if baseline_valid is not None else baseline_train
    rows: list[dict[str, Any]] = []
    if not reward_progress_rows:
        rows.append(
            {
                "rl_metric": "",
                "rule_baseline_reference_split": "valid" if baseline_valid is not None else ("train" if baseline_train is not None else ""),
                "rule_baseline_mean_reward": baseline_ref,
                "rl_first_reward": None,
                "rl_last_reward": None,
                "rl_best_reward": None,
                "rl_last_minus_first": None,
                "status": "pending_rl_metrics",
            }
        )
        return rows
    for item in reward_progress_rows:
        first = _safe_float(item.get("first_reward"))
        last = _safe_float(item.get("last_reward"))
        best = _safe_float(item.get("best_reward"))
        rows.append(
            {
                "rl_metric": item["metric"],
                "rule_baseline_reference_split": "valid" if baseline_valid is not None else ("train" if baseline_train is not None else ""),
                "rule_baseline_mean_reward": baseline_ref,
                "rule_baseline_test_reward": baseline_test,
                "rl_first_reward": first,
                "rl_last_reward": last,
                "rl_best_reward": best,
                "rl_last_minus_first": (last - first) if last is not None and first is not None else None,
                "rl_best_minus_first": (best - first) if best is not None and first is not None else None,
                "rl_last_minus_rule_baseline": (last - baseline_ref) if last is not None and baseline_ref is not None else None,
                "status": "ok",
            }
        )
    return rows


def _row_for_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("split")) == split:
            return row
    return {}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell_value(row.get(k)) for k in columns})
    return path


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines: list[str] = []
    if columns:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            values = [_markdown_cell(_cell_value(row.get(col))) for col in columns]
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("_No rows available._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cell_value(value: Any) -> Any:
    if isinstance(value, float):
        return _fmt(value)
    if value is None:
        return ""
    return value


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_charts(
    figures_dir: Path,
    baseline_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    predictions: "pd.DataFrame | None",
    reward_progress_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    splits = [row["split"] for row in baseline_rows]
    charts["baseline_reward_svg"] = _write_bar_svg(
        figures_dir / "baseline_mean_reward_by_split.svg",
        "Rule Baseline Mean Reward",
        splits,
        {"mean_reward": [_safe_float(row.get("mean_reward"), 0.0) or 0.0 for row in baseline_rows]},
        y_label="reward",
    )
    charts["baseline_accuracy_f1_svg"] = _write_bar_svg(
        figures_dir / "baseline_accuracy_macro_f1.svg",
        "Rule Baseline Accuracy and Macro-F1",
        splits,
        {
            "accuracy": [_safe_float(row.get("accuracy"), 0.0) or 0.0 for row in baseline_rows],
            "macro_f1": [_safe_float(row.get("macro_f1"), 0.0) or 0.0 for row in baseline_rows],
        },
        y_label="score",
    )
    charts["baseline_rank_spread_svg"] = _write_bar_svg(
        figures_dir / "baseline_rank_ic_top_bottom.svg",
        "Rule Baseline Rank IC and Top-Bottom Return",
        splits,
        {
            "mean_rank_ic": [_safe_float(row.get("mean_rank_ic"), 0.0) or 0.0 for row in baseline_rows],
            "top_bottom_return": [_safe_float(row.get("top_bottom_return"), 0.0) or 0.0 for row in baseline_rows],
        },
        y_label="value",
    )
    charts["label_distribution_svg"] = _write_count_group_svg(
        figures_dir / "label_distribution.svg", "Task Label Distribution", label_rows, group_col="label"
    )
    charts["prediction_distribution_svg"] = _write_count_group_svg(
        figures_dir / "prediction_distribution.svg", "Rule Prediction Distribution", prediction_rows, group_col="prediction"
    )
    if predictions is not None and not predictions.empty and "reward" in predictions:
        reward_values = [_safe_float(v) for v in predictions["reward"].tolist()]
        charts["reward_histogram_svg"] = _write_histogram_svg(
            figures_dir / "rule_reward_histogram.svg",
            "Rule Baseline Sample Reward Distribution",
            [v for v in reward_values if v is not None],
            bins=24,
        )
    else:
        charts["reward_histogram_svg"] = _write_placeholder_svg(
            figures_dir / "rule_reward_histogram.svg", "Rule Baseline Sample Reward Distribution", "No prediction rows yet."
        )
    if (
        predictions is not None
        and not predictions.empty
        and "alpha_score" in predictions
        and "future_relative_return" in predictions
    ):
        points = [
            (_safe_float(row.get("alpha_score")), _safe_float(row.get("future_relative_return")))
            for _, row in predictions.iterrows()
        ]
        points = [(x, y) for x, y in points if x is not None and y is not None]
        charts["alpha_vs_return_svg"] = _write_scatter_svg(
            figures_dir / "alpha_score_vs_future_relative_return.svg",
            "Rule Alpha Score vs Future Relative Return",
            points,
            x_label="alpha_score",
            y_label="future_relative_return",
        )
    else:
        charts["alpha_vs_return_svg"] = _write_placeholder_svg(
            figures_dir / "alpha_score_vs_future_relative_return.svg",
            "Rule Alpha Score vs Future Relative Return",
            "No alpha/return prediction rows yet.",
        )
    charts["rl_reward_curve_svg"] = _write_rl_reward_curve_svg(
        figures_dir / "rl_reward_curve.svg", metric_rows, reward_progress_rows
    )
    charts["baseline_vs_rl_reward_svg"] = _write_baseline_vs_rl_svg(
        figures_dir / "baseline_vs_rl_reward.svg", baseline_rows, reward_progress_rows
    )
    return charts


def _svg_start(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
    )


def _svg_text(x: float, y: float, text: Any, size: int = 13, color: str = "#111827", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(text))}</text>\n'
    )


def _value_range(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return (0.0, 1.0)
    low = min(0.0, min(vals))
    high = max(0.0, max(vals))
    if abs(high - low) < 1e-9:
        low -= 1.0
        high += 1.0
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _write_bar_svg(path: Path, title: str, labels: list[Any], series: dict[str, list[float]], y_label: str = "") -> Path:
    width, height = 920, 520
    left, right, top, bottom = 86, 32, 64, 88
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_values = [v for values in series.values() for v in values]
    y_min, y_max = _value_range(all_values)

    def y_pos(v: float) -> float:
        return top + chart_h - (v - y_min) / (y_max - y_min) * chart_h

    svg = _svg_start(width, height)
    svg += _svg_text(width / 2, 30, title, 20, anchor="middle")
    if not labels or not series:
        svg += _svg_text(width / 2, height / 2, "No data available.", 16, "#6b7280", "middle")
        svg += "</svg>\n"
        path.write_text(svg, encoding="utf-8")
        return path
    zero_y = y_pos(0.0)
    svg += f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#9ca3af"/>\n'
    svg += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>\n'
    svg += f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>\n'
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = y_pos(v)
        svg += f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>\n'
        svg += _svg_text(left - 10, y + 4, _fmt(v, 3), 11, "#4b5563", "end")
    n_groups = len(labels)
    n_series = max(1, len(series))
    group_w = chart_w / max(1, n_groups)
    bar_w = min(42, group_w * 0.72 / n_series)
    for s_idx, (name, values) in enumerate(series.items()):
        color = SERIES_COLORS[s_idx % len(SERIES_COLORS)]
        for i, label in enumerate(labels):
            value = _safe_float(values[i] if i < len(values) else 0.0, 0.0) or 0.0
            x = left + i * group_w + group_w / 2 - (n_series * bar_w) / 2 + s_idx * bar_w
            y = min(y_pos(value), zero_y)
            h = abs(y_pos(value) - zero_y)
            svg += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.86:.1f}" height="{h:.1f}" fill="{color}"/>\n'
        legend_x = left + s_idx * 180
        svg += f'<rect x="{legend_x}" y="{height - 34}" width="12" height="12" fill="{color}"/>\n'
        svg += _svg_text(legend_x + 18, height - 24, name, 12)
    for i, label in enumerate(labels):
        x = left + i * group_w + group_w / 2
        svg += _svg_text(x, height - bottom + 26, label, 12, "#374151", "middle")
    if y_label:
        svg += _svg_text(20, top + chart_h / 2, y_label, 12, "#4b5563")
    svg += "</svg>\n"
    path.write_text(svg, encoding="utf-8")
    return path


def _write_count_group_svg(path: Path, title: str, rows: list[dict[str, Any]], group_col: str) -> Path:
    if not rows:
        return _write_placeholder_svg(path, title, "No rows available.")
    splits = sorted({str(r.get("split")) for r in rows}, key=_split_key)
    groups = [g for g in CLASSES if any(str(r.get(group_col)) == g for r in rows)]
    groups.extend(sorted({str(r.get(group_col)) for r in rows if str(r.get(group_col)) not in groups}))
    series: dict[str, list[float]] = {}
    for group in groups:
        values = []
        for split in splits:
            total = sum(int(r.get("n", 0) or 0) for r in rows if str(r.get("split")) == split and str(r.get(group_col)) == group)
            values.append(float(total))
        series[group] = values
    return _write_bar_svg(path, title, splits, series, y_label="count")


def _write_histogram_svg(path: Path, title: str, values: list[float], bins: int = 20) -> Path:
    if not values:
        return _write_placeholder_svg(path, title, "No values available.")
    low, high = min(values), max(values)
    if abs(high - low) < 1e-9:
        low -= 0.5
        high += 0.5
    counts = [0] * bins
    for value in values:
        idx = int((value - low) / (high - low) * bins)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    labels = [f"{low + (high - low) * (i + 0.5) / bins:.2f}" for i in range(bins)]
    return _write_bar_svg(path, title, labels, {"count": [float(c) for c in counts]}, y_label="count")


def _write_scatter_svg(path: Path, title: str, points: list[tuple[float, float]], x_label: str, y_label: str) -> Path:
    width, height = 920, 520
    left, right, top, bottom = 86, 32, 64, 82
    chart_w = width - left - right
    chart_h = height - top - bottom
    svg = _svg_start(width, height)
    svg += _svg_text(width / 2, 30, title, 20, anchor="middle")
    if not points:
        svg += _svg_text(width / 2, height / 2, "No data available.", 16, "#6b7280", "middle")
        svg += "</svg>\n"
        path.write_text(svg, encoding="utf-8")
        return path
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = _value_range(xs)
    y_min, y_max = _value_range(ys)

    def x_pos(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * chart_w

    def y_pos(v: float) -> float:
        return top + chart_h - (v - y_min) / (y_max - y_min) * chart_h

    svg += f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>\n'
    svg += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>\n'
    svg += f'<line x1="{left}" y1="{y_pos(0.0):.1f}" x2="{width - right}" y2="{y_pos(0.0):.1f}" stroke="#e5e7eb"/>\n'
    svg += f'<line x1="{x_pos(0.0):.1f}" y1="{top}" x2="{x_pos(0.0):.1f}" y2="{height - bottom}" stroke="#e5e7eb"/>\n'
    step = max(1, len(points) // 1000)
    for x, y in points[::step]:
        svg += f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="2.4" fill="#2563eb" fill-opacity="0.42"/>\n'
    svg += _svg_text(width / 2, height - 26, x_label, 13, "#374151", "middle")
    svg += _svg_text(20, top + chart_h / 2, y_label, 13, "#374151")
    svg += _svg_text(left, height - bottom + 22, _fmt(x_min, 3), 11, "#4b5563", "middle")
    svg += _svg_text(width - right, height - bottom + 22, _fmt(x_max, 3), 11, "#4b5563", "middle")
    svg += _svg_text(left - 10, y_pos(y_min), _fmt(y_min, 3), 11, "#4b5563", "end")
    svg += _svg_text(left - 10, y_pos(y_max) + 4, _fmt(y_max, 3), 11, "#4b5563", "end")
    svg += "</svg>\n"
    path.write_text(svg, encoding="utf-8")
    return path


def _write_rl_reward_curve_svg(path: Path, metric_rows: list[dict[str, Any]], reward_progress_rows: list[dict[str, Any]]) -> Path:
    if not reward_progress_rows:
        return _write_placeholder_svg(path, "RL Reward Curve", "No verl reward metrics found yet.")
    metric = reward_progress_rows[0]["metric"]
    points: list[tuple[int, float]] = []
    for row in metric_rows:
        value = _safe_float(row.get(metric))
        step = _safe_float(row.get("step"))
        if value is not None and step is not None:
            points.append((int(step), value))
    return _write_line_svg(path, f"RL Reward Curve: {metric}", points, y_label="reward")


def _write_baseline_vs_rl_svg(path: Path, baseline_rows: list[dict[str, Any]], reward_progress_rows: list[dict[str, Any]]) -> Path:
    valid = _safe_float(_row_for_split(baseline_rows, "valid").get("mean_reward"))
    train = _safe_float(_row_for_split(baseline_rows, "train").get("mean_reward"))
    baseline = valid if valid is not None else train
    if not reward_progress_rows:
        labels = ["rule_baseline"]
        series = {"reward": [baseline or 0.0]}
        return _write_bar_svg(path, "Rule Baseline vs RL Reward", labels, series, y_label="reward")
    item = reward_progress_rows[0]
    labels = ["rule_baseline", "rl_first", "rl_last", "rl_best"]
    values = [
        baseline or 0.0,
        _safe_float(item.get("first_reward"), 0.0) or 0.0,
        _safe_float(item.get("last_reward"), 0.0) or 0.0,
        _safe_float(item.get("best_reward"), 0.0) or 0.0,
    ]
    return _write_bar_svg(path, "Rule Baseline vs RL Reward", labels, {"reward": values}, y_label="reward")


def _write_line_svg(path: Path, title: str, points: list[tuple[int, float]], y_label: str = "") -> Path:
    width, height = 920, 520
    left, right, top, bottom = 86, 32, 64, 82
    chart_w = width - left - right
    chart_h = height - top - bottom
    svg = _svg_start(width, height)
    svg += _svg_text(width / 2, 30, title, 20, anchor="middle")
    if not points:
        svg += _svg_text(width / 2, height / 2, "No data available.", 16, "#6b7280", "middle")
        svg += "</svg>\n"
        path.write_text(svg, encoding="utf-8")
        return path
    points = sorted(points)
    x_min, x_max = float(points[0][0]), float(points[-1][0])
    if abs(x_max - x_min) < 1e-9:
        x_max += 1.0
    y_min, y_max = _value_range([p[1] for p in points])

    def x_pos(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * chart_w

    def y_pos(v: float) -> float:
        return top + chart_h - (v - y_min) / (y_max - y_min) * chart_h

    svg += f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>\n'
    svg += f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>\n'
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        y = y_pos(yv)
        svg += f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>\n'
        svg += _svg_text(left - 10, y + 4, _fmt(yv, 3), 11, "#4b5563", "end")
    poly = " ".join(f"{x_pos(step):.1f},{y_pos(value):.1f}" for step, value in points)
    svg += f'<polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="2.5"/>\n'
    for step, value in points[-120:]:
        svg += f'<circle cx="{x_pos(step):.1f}" cy="{y_pos(value):.1f}" r="2.6" fill="#2563eb"/>\n'
    svg += _svg_text(left, height - bottom + 24, str(points[0][0]), 11, "#4b5563", "middle")
    svg += _svg_text(width - right, height - bottom + 24, str(points[-1][0]), 11, "#4b5563", "middle")
    if y_label:
        svg += _svg_text(20, top + chart_h / 2, y_label, 13, "#374151")
    svg += "</svg>\n"
    path.write_text(svg, encoding="utf-8")
    return path


def _write_placeholder_svg(path: Path, title: str, message: str) -> Path:
    width, height = 920, 520
    svg = _svg_start(width, height)
    svg += _svg_text(width / 2, 34, title, 20, anchor="middle")
    svg += f'<rect x="110" y="130" width="700" height="220" fill="#f9fafb" stroke="#d1d5db"/>\n'
    svg += _svg_text(width / 2, 240, message, 16, "#6b7280", "middle")
    svg += "</svg>\n"
    path.write_text(svg, encoding="utf-8")
    return path


def _write_report_index(
    report_dir: Path,
    run_dir: Path,
    baseline_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    reward_progress_rows: list[dict[str, Any]],
    baseline_vs_rl_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> Path:
    lines = [
        "# Stock Search-Agent RL Report",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Run dir: `{run_dir}`",
        "",
        "## Reward Evidence",
        "",
    ]
    if reward_progress_rows:
        best = reward_progress_rows[0]
        lines.append(
            f"- Primary RL metric: `{best['metric']}`, first={_fmt(best['first_reward'])}, "
            f"last={_fmt(best['last_reward'])}, best={_fmt(best['best_reward'])}."
        )
        lines.append(f"- Last-first delta: {_fmt(best['delta_last_minus_first'])}.")
    else:
        lines.append("- No RL reward metrics found yet; the report currently shows data and rule-baseline evidence.")
    valid = _row_for_split(baseline_rows, "valid")
    if valid:
        lines.append(f"- Rule-baseline valid mean reward: {_fmt(valid.get('mean_reward'))}.")
    lines.extend(["", "## Tables", ""])
    for name, path in artifacts.items():
        if "/tables/" in path or path.endswith(".csv") or path.endswith(".md"):
            rel = _relative_display_path(Path(path), report_dir)
            lines.append(f"- `{name}`: [{rel}]({rel})")
    lines.extend(["", "## Figures", ""])
    for name, path in artifacts.items():
        if path.endswith(".svg"):
            rel = _relative_display_path(Path(path), report_dir)
            lines.append(f"- `{name}`: [{rel}]({rel})")
    lines.extend(["", "## Baseline Metrics", ""])
    lines.extend(_markdown_lines_for_rows(baseline_rows))
    lines.extend(["", "## Dataset Summary", ""])
    lines.extend(_markdown_lines_for_rows(dataset_rows))
    lines.extend(["", "## Baseline vs RL Reward", ""])
    lines.extend(_markdown_lines_for_rows(baseline_vs_rl_rows))
    path = report_dir / "report_index.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _relative_display_path(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path


def _markdown_lines_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        return ["_No rows available._"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_markdown_cell(_cell_value(row.get(col))) for col in columns) + " |")
    return out


def _write_pdf_report(
    pdf_path: Path,
    run_dir: Path,
    baseline_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    reward_progress_rows: list[dict[str, Any]],
    baseline_vs_rl_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    pdf = SimplePDF()
    pdf.heading("Stock Search-Agent RL Report")
    pdf.text(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.text(f"Run: {run_dir}")
    pdf.spacer(8)
    pdf.heading("Executive Summary", size=16)
    if reward_progress_rows:
        primary = reward_progress_rows[0]
        pdf.text(
            "Primary RL reward metric improved from "
            f"{_fmt(primary.get('first_reward'))} to {_fmt(primary.get('last_reward'))}; "
            f"best observed reward was {_fmt(primary.get('best_reward'))}."
        )
    else:
        pdf.text("No RL reward metric rows were found yet. This PDF currently shows dataset and rule-baseline evidence.")
    valid = _row_for_split(baseline_rows, "valid")
    if valid:
        pdf.text(
            "Rule-baseline valid split: "
            f"mean_reward={_fmt(valid.get('mean_reward'))}, accuracy={_fmt(valid.get('accuracy'))}, "
            f"macro_f1={_fmt(valid.get('macro_f1'))}."
        )
    pdf.spacer(10)
    pdf.table(
        "Baseline Metrics",
        baseline_rows,
        ["split", "n", "mean_reward", "accuracy", "macro_f1", "brier", "mean_rank_ic", "top_bottom_return"],
    )
    pdf.table("Dataset Summary", dataset_rows, ["split", "n", "stocks", "trade_dates", "label_up", "label_neutral", "label_down"])

    pdf.new_page()
    pdf.heading("Reward Evidence")
    pdf.table(
        "RL Reward Progress",
        reward_progress_rows,
        ["metric", "points", "first_reward", "last_reward", "best_reward", "delta_last_minus_first", "delta_best_minus_first"],
        max_rows=8,
    )
    pdf.table(
        "Rule Baseline vs RL Reward",
        baseline_vs_rl_rows,
        ["rl_metric", "rule_baseline_mean_reward", "rl_first_reward", "rl_last_reward", "rl_best_reward", "rl_last_minus_first"],
        max_rows=8,
    )
    pdf.bar_chart(
        "Rule Baseline Mean Reward",
        [str(r.get("split")) for r in baseline_rows],
        [_safe_float(r.get("mean_reward"), 0.0) or 0.0 for r in baseline_rows],
    )
    if reward_progress_rows:
        metric = reward_progress_rows[0]["metric"]
        points = []
        for row in metric_rows:
            step = _safe_float(row.get("step"))
            value = _safe_float(row.get(metric))
            if step is not None and value is not None:
                points.append((step, value))
        pdf.line_chart("RL Reward Curve", points)

    pdf.new_page()
    pdf.heading("Prediction and Data Checks")
    pdf.bar_chart(
        "Baseline Accuracy",
        [str(r.get("split")) for r in baseline_rows],
        [_safe_float(r.get("accuracy"), 0.0) or 0.0 for r in baseline_rows],
    )
    pdf.group_count_table("Label Distribution", label_rows, "label")
    pdf.group_count_table("Prediction Distribution", prediction_rows, "prediction")

    pdf.new_page()
    pdf.heading("Generated Artifacts")
    pdf.text("The run directory contains CSV/Markdown tables, SVG figures, and this PDF.")
    for name, path in sorted(artifacts.items()):
        if name.endswith("_svg") or name.endswith("_csv") or name.endswith("_md"):
            pdf.text(f"- {name}: {Path(path).name}", size=9)
    pdf.save(pdf_path)


class SimplePDF:
    def __init__(self, width: int = 612, height: int = 792):
        self.width = width
        self.height = height
        self.margin = 50
        self.pages: list[list[str]] = []
        self.current: list[str] = []
        self.y = self.height - self.margin
        self.new_page()

    def new_page(self) -> None:
        if self.current:
            self.pages.append(self.current)
        self.current = []
        self.y = self.height - self.margin

    def ensure(self, needed: float) -> None:
        if self.y - needed < self.margin:
            self.new_page()

    def heading(self, text: str, size: int = 20) -> None:
        self.ensure(size + 18)
        self.text(text, size=size)
        self.y -= 4
        self.line(self.margin, self.y, self.width - self.margin, self.y, "#d1d5db")
        self.y -= 14

    def text(self, text: str, size: int = 11, color: str = "#111827") -> None:
        max_chars = max(30, int((self.width - 2 * self.margin) / (size * 0.52)))
        for line in textwrap.wrap(str(text), width=max_chars) or [""]:
            self.ensure(size + 6)
            self.current.append(f"{_pdf_color(color)} BT /F1 {size} Tf {self.margin} {self.y:.1f} Td ({_pdf_escape(line)}) Tj ET\n")
            self.y -= size + 5

    def spacer(self, amount: float) -> None:
        self.y -= amount

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#111827") -> None:
        self.current.append(f"{_pdf_color(color)} {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S\n")

    def rect(self, x: float, y: float, w: float, h: float, color: str = "#2563eb") -> None:
        self.current.append(f"{_pdf_color(color)} {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f\n")

    def table(self, title: str, rows: list[dict[str, Any]], columns: list[str], max_rows: int = 12) -> None:
        self.ensure(48)
        self.text(title, size=14)
        if not rows:
            self.text("No rows available.", size=10, color="#6b7280")
            return
        usable_cols = [c for c in columns if any(c in row for row in rows)]
        if not usable_cols:
            self.text("No columns available.", size=10, color="#6b7280")
            return
        self.ensure(18 + min(max_rows, len(rows)) * 14)
        col_w = (self.width - 2 * self.margin) / len(usable_cols)
        y = self.y
        for idx, col in enumerate(usable_cols):
            self._draw_text_at(self.margin + idx * col_w, y, col[:18], 8, "#374151")
        y -= 13
        self.line(self.margin, y + 4, self.width - self.margin, y + 4, "#d1d5db")
        for row in rows[:max_rows]:
            for idx, col in enumerate(usable_cols):
                self._draw_text_at(self.margin + idx * col_w, y, str(_cell_value(row.get(col)))[:18], 8, "#111827")
            y -= 13
        self.y = y - 8

    def bar_chart(self, title: str, labels: list[str], values: list[float]) -> None:
        self.ensure(190)
        self.text(title, size=14)
        if not labels or not values:
            self.text("No chart data.", size=10, color="#6b7280")
            return
        x0, y0, w, h = self.margin, self.y - 145, self.width - 2 * self.margin, 120
        vals = [_safe_float(v, 0.0) or 0.0 for v in values]
        y_min, y_max = _value_range(vals)

        def yp(v: float) -> float:
            return y0 + (v - y_min) / (y_max - y_min) * h

        zero_y = yp(0.0)
        self.line(x0, zero_y, x0 + w, zero_y, "#9ca3af")
        gap = w / max(1, len(vals))
        bar_w = min(42, gap * 0.55)
        for i, value in enumerate(vals):
            x = x0 + i * gap + gap / 2 - bar_w / 2
            y = min(yp(value), zero_y)
            self.rect(x, y, bar_w, abs(yp(value) - zero_y), SERIES_COLORS[i % len(SERIES_COLORS)])
            self._draw_text_at(x, y0 - 14, labels[i][:10], 8, "#374151")
        self.y = y0 - 28

    def line_chart(self, title: str, points: list[tuple[float, float]]) -> None:
        self.ensure(190)
        self.text(title, size=14)
        if not points:
            self.text("No chart data.", size=10, color="#6b7280")
            return
        points = sorted(points)
        x0, y0, w, h = self.margin, self.y - 145, self.width - 2 * self.margin, 120
        x_min, x_max = points[0][0], points[-1][0]
        if abs(x_max - x_min) < 1e-9:
            x_max += 1
        y_min, y_max = _value_range([p[1] for p in points])

        def xp(v: float) -> float:
            return x0 + (v - x_min) / (x_max - x_min) * w

        def yp(v: float) -> float:
            return y0 + (v - y_min) / (y_max - y_min) * h

        self.line(x0, y0, x0 + w, y0, "#374151")
        self.line(x0, y0, x0, y0 + h, "#374151")
        if len(points) >= 2:
            cmd = _pdf_color("#2563eb") + f" {xp(points[0][0]):.1f} {yp(points[0][1]):.1f} m "
            for step, value in points[1:]:
                cmd += f"{xp(step):.1f} {yp(value):.1f} l "
            cmd += "S\n"
            self.current.append(cmd)
        self.y = y0 - 28

    def group_count_table(self, title: str, rows: list[dict[str, Any]], group_col: str) -> None:
        splits = sorted({str(r.get("split")) for r in rows}, key=_split_key)
        groups = [g for g in CLASSES if any(str(r.get(group_col)) == g for r in rows)]
        table_rows = []
        for split in splits:
            item = {"split": split}
            for group in groups:
                item[group] = sum(int(r.get("n", 0) or 0) for r in rows if str(r.get("split")) == split and str(r.get(group_col)) == group)
            table_rows.append(item)
        self.table(title, table_rows, ["split", *groups], max_rows=8)

    def _draw_text_at(self, x: float, y: float, text: str, size: int, color: str) -> None:
        self.current.append(f"{_pdf_color(color)} BT /F1 {size} Tf {x:.1f} {y:.1f} Td ({_pdf_escape(text)}) Tj ET\n")

    def save(self, path: Path) -> None:
        if self.current:
            self.pages.append(self.current)
            self.current = []
        path.parent.mkdir(parents=True, exist_ok=True)
        objects: list[bytes] = []
        n_pages = len(self.pages)
        font_id = 3
        content_ids = [4 + i * 2 for i in range(n_pages)]
        page_ids = [5 + i * 2 for i in range(n_pages)]
        catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
        kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode("ascii")
        pages_obj = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n_pages).encode("ascii") + b" >>"
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        objects.extend([catalog, pages_obj, font])
        for content_id, page_id, commands in zip(content_ids, page_ids, self.pages):
            stream = "".join(commands).encode("latin-1", errors="replace")
            content = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
            page = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
            objects.extend([content, page])
        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_pos}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(bytes(out))


def _pdf_escape(text: Any) -> str:
    s = str(text).encode("latin-1", errors="replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_color(color: str) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "0 0 0 rg 0 0 0 RG"
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG"


def parse_metric_dict_from_line(line: str) -> dict[str, Any]:
    """Small helper for ad-hoc debugging of console metric lines."""
    try:
        return ast.literal_eval(line)
    except Exception:
        return {}
