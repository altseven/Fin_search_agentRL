from __future__ import annotations

import ast
import csv
import json
import math
import os
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

    def run_quality(path: Path) -> int:
        score = 0
        if (path / "verl" / "training_metrics.jsonl").exists():
            score += 4
        if (path / "rollouts" / "rule_baseline_metrics.json").exists():
            score += 3
        if (path / "verl" / "train.parquet").exists() or (path / "verl" / "valid.parquet").exists():
            score += 2
        if (path / "report" / "stock_agent_rl_report.pdf").exists():
            score += 1
        return score

    candidates.sort(key=lambda p: (run_quality(p), p.stat().st_mtime), reverse=True)
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
            "Figures are exported as PDF files under report/figures for presentation-quality rendering.",
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
    charts["workflow_pdf"] = _save_chart_pdf(figures_dir / "workflow.pdf", lambda: _fig_workflow())
    charts["baseline_reward_pdf"] = _save_chart_pdf(
        figures_dir / "baseline_mean_reward_by_split.pdf",
        lambda: _fig_baseline_reward(baseline_rows),
    )
    charts["baseline_accuracy_f1_pdf"] = _save_chart_pdf(
        figures_dir / "baseline_accuracy_macro_f1.pdf",
        lambda: _fig_baseline_accuracy_f1(baseline_rows),
    )
    charts["baseline_rank_spread_pdf"] = _save_chart_pdf(
        figures_dir / "baseline_rank_ic_top_bottom.pdf",
        lambda: _fig_baseline_rank_spread(baseline_rows),
    )
    charts["label_distribution_pdf"] = _save_chart_pdf(
        figures_dir / "label_distribution.pdf",
        lambda: _fig_count_group("Task Label Distribution", label_rows, "label"),
    )
    charts["prediction_distribution_pdf"] = _save_chart_pdf(
        figures_dir / "prediction_distribution.pdf",
        lambda: _fig_count_group("Rule Prediction Distribution", prediction_rows, "prediction"),
    )
    charts["reward_histogram_pdf"] = _save_chart_pdf(
        figures_dir / "rule_reward_histogram.pdf",
        lambda: _fig_rule_reward_histogram(predictions),
    )
    charts["alpha_vs_return_pdf"] = _save_chart_pdf(
        figures_dir / "alpha_score_vs_future_relative_return.pdf",
        lambda: _fig_alpha_vs_return(predictions),
    )
    charts["rl_reward_curve_pdf"] = _save_chart_pdf(
        figures_dir / "rl_reward_curve.pdf",
        lambda: _fig_rl_reward_curve(metric_rows, reward_progress_rows),
    )
    charts["baseline_vs_rl_reward_pdf"] = _save_chart_pdf(
        figures_dir / "baseline_vs_rl_reward.pdf",
        lambda: _fig_baseline_vs_rl_reward(baseline_rows, reward_progress_rows),
    )
    charts["tool_usage_pdf"] = _save_chart_pdf(
        figures_dir / "tool_usage_over_training.pdf",
        lambda: _fig_tool_usage(metric_rows),
    )
    charts["reward_components_pdf"] = _save_chart_pdf(
        figures_dir / "reward_components_over_training.pdf",
        lambda: _fig_reward_components(metric_rows),
    )
    charts["optimization_health_pdf"] = _save_chart_pdf(
        figures_dir / "optimization_health_metrics.pdf",
        lambda: _fig_optimization_health(metric_rows),
    )
    return charts


def _get_matplotlib():
    cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "figure.max_open_warning": 0,
        }
    )
    return plt, PdfPages


def _save_chart_pdf(path: Path, fig_factory: Any) -> Path:
    plt, _ = _get_matplotlib()
    fig = fig_factory()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _blank_figure(title: str, message: str):
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    fig.suptitle(title, y=0.94, fontweight="bold")
    ax.text(0.5, 0.52, message, ha="center", va="center", fontsize=13, color="#4b5563", wrap=True)
    return fig


def _style_axis(ax: Any, title: str, x_label: str = "", y_label: str = "", note: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if note:
        ax.text(
            0,
            -0.20,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#4b5563",
            wrap=True,
        )


def _fig_workflow():
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.axis("off")
    fig.suptitle("Stock Search-Agent RL Workflow", y=0.96, fontweight="bold")
    nodes = [
        ("Tushare cache", "SSE50 universe, daily, adj factors,\ndaily_basic, index data, trade calendar"),
        ("Point-in-time tables", "factor_snapshot, market_context,\nindustry_context, peer_context,\nfundamental_snapshot, announcements, news"),
        ("RL task construction", "as_of date + hidden future label\nup / neutral / down from future relative return"),
        ("Tool agent rollout", "Qwen policy can call local tools\nthen outputs strict prediction JSON"),
        ("Verifiable reward", "direction, probability, Brier, PnL,\nevidence/tool use, penalties"),
        ("GRPO update", "verl updates policy; compare reward\nbefore/after and monitor tool usage"),
        ("Mentor report", "PDF charts/tables show data quality,\nreward improvement, tool behavior, stability"),
    ]
    xs = [0.08, 0.36, 0.64, 0.08, 0.36, 0.64, 0.36]
    ys = [0.76, 0.76, 0.76, 0.42, 0.42, 0.42, 0.16]
    colors = ["#dbeafe", "#e0f2fe", "#ecfdf5", "#fef3c7", "#fee2e2", "#ede9fe", "#f3f4f6"]
    for idx, ((title, body), x, y, color) in enumerate(zip(nodes, xs, ys, colors)):
        ax.text(
            x,
            y,
            f"{idx + 1}. {title}\n{body}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#6b7280", linewidth=1.0),
        )
    arrows = [((0.27, 0.76), (0.34, 0.76)), ((0.55, 0.76), (0.62, 0.76)), ((0.68, 0.68), (0.20, 0.50)), ((0.27, 0.42), (0.34, 0.42)), ((0.55, 0.42), (0.62, 0.42)), ((0.70, 0.35), (0.49, 0.23))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    ax.text(
        0.04,
        0.04,
        "How to read: a good run should show reward increasing, tool-call metrics above zero when tool use is required, "
        "valid/test metrics not collapsing, and stable optimization diagnostics.",
        transform=ax.transAxes,
        fontsize=10,
        color="#374151",
        wrap=True,
    )
    return fig


def _fig_baseline_reward(baseline_rows: list[dict[str, Any]]):
    splits = [str(row.get("split")) for row in baseline_rows]
    values = [_safe_float(row.get("mean_reward"), 0.0) or 0.0 for row in baseline_rows]
    return _fig_bar_single(
        "Rule Baseline Mean Reward by Split",
        splits,
        values,
        "Data split",
        "Mean reward",
        "This is the non-RL reference. RL should improve from its own first reward and ideally approach or exceed this rule baseline.",
    )


def _fig_baseline_accuracy_f1(baseline_rows: list[dict[str, Any]]):
    splits = [str(row.get("split")) for row in baseline_rows]
    series = {
        "accuracy": [_safe_float(row.get("accuracy"), 0.0) or 0.0 for row in baseline_rows],
        "macro_f1": [_safe_float(row.get("macro_f1"), 0.0) or 0.0 for row in baseline_rows],
    }
    return _fig_group_bar(
        "Rule Baseline Classification Metrics",
        splits,
        series,
        "Data split",
        "Score",
        "Accuracy shows direct label hit-rate. Macro-F1 checks whether one class dominates the predictions.",
    )


def _fig_baseline_rank_spread(baseline_rows: list[dict[str, Any]]):
    splits = [str(row.get("split")) for row in baseline_rows]
    series = {
        "mean_rank_ic": [_safe_float(row.get("mean_rank_ic"), 0.0) or 0.0 for row in baseline_rows],
        "top_bottom_return": [_safe_float(row.get("top_bottom_return"), 0.0) or 0.0 for row in baseline_rows],
    }
    return _fig_group_bar(
        "Rule Baseline Ranking Metrics",
        splits,
        series,
        "Data split",
        "Value",
        "Positive Rank IC means scores rank better future relative returns. Positive top-bottom return means high-score stocks outperform low-score stocks.",
    )


def _fig_bar_single(title: str, labels: list[str], values: list[float], x_label: str, y_label: str, note: str):
    plt, _ = _get_matplotlib()
    if not labels:
        return _blank_figure(title, "No rows available.")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(values))]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.axhline(0, color="#6b7280", lw=1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, _fmt(value, 3), ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    _style_axis(ax, title, x_label, y_label, note)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _fig_group_bar(title: str, labels: list[str], series: dict[str, list[float]], x_label: str, y_label: str, note: str):
    plt, _ = _get_matplotlib()
    if not labels or not series:
        return _blank_figure(title, "No rows available.")
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    x = list(range(len(labels)))
    n_series = max(1, len(series))
    width = min(0.24, 0.72 / n_series)
    offset0 = -width * (n_series - 1) / 2
    for i, (name, values) in enumerate(series.items()):
        vals = [_safe_float(values[j] if j < len(values) else 0.0, 0.0) or 0.0 for j in range(len(labels))]
        pos = [v + offset0 + i * width for v in x]
        ax.bar(pos, vals, width=width, label=name, color=SERIES_COLORS[i % len(SERIES_COLORS)])
    ax.axhline(0, color="#6b7280", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="best", frameon=False)
    _style_axis(ax, title, x_label, y_label, note)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _fig_count_group(title: str, rows: list[dict[str, Any]], group_col: str):
    if not rows:
        return _blank_figure(title, "No rows available.")
    splits = sorted({str(r.get("split")) for r in rows}, key=_split_key)
    groups = [g for g in CLASSES if any(str(r.get(group_col)) == g for r in rows)]
    groups.extend(sorted({str(r.get(group_col)) for r in rows if str(r.get(group_col)) not in groups}))
    series = {}
    for group in groups:
        values = []
        for split in splits:
            total = sum(int(r.get("n", 0) or 0) for r in rows if str(r.get("split")) == split and str(r.get(group_col)) == group)
            values.append(float(total))
        series[group] = values
    return _fig_group_bar(
        title,
        splits,
        series,
        "Data split",
        "Number of samples",
        "Use this as a sanity check. Severe class imbalance makes reward improvement less trustworthy.",
    )


def _fig_rule_reward_histogram(predictions: "pd.DataFrame | None"):
    if predictions is None or predictions.empty or "reward" not in predictions:
        return _blank_figure("Rule Baseline Sample Reward Distribution", "No rule-baseline prediction rows are available.")
    values = [_safe_float(v) for v in predictions["reward"].tolist()]
    values = [v for v in values if v is not None]
    if not values:
        return _blank_figure("Rule Baseline Sample Reward Distribution", "No numeric reward values are available.")
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.hist(values, bins=24, color="#2563eb", edgecolor="white", alpha=0.86)
    ax.axvline(sum(values) / len(values), color="#dc2626", linestyle="--", linewidth=1.8, label=f"mean={_fmt(sum(values) / len(values), 3)}")
    ax.legend(frameon=False)
    _style_axis(
        ax,
        "Rule Baseline Sample Reward Distribution",
        "Sample reward",
        "Count",
        "A healthier baseline has rewards spread around useful positive values rather than collapsing near a single constant.",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _fig_alpha_vs_return(predictions: "pd.DataFrame | None"):
    if predictions is None or predictions.empty or "alpha_score" not in predictions or "future_relative_return" not in predictions:
        return _blank_figure("Rule Alpha Score vs Future Relative Return", "No alpha_score/future_relative_return rows are available.")
    points = [
        (_safe_float(row.get("alpha_score")), _safe_float(row.get("future_relative_return")))
        for _, row in predictions.iterrows()
    ]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        return _blank_figure("Rule Alpha Score vs Future Relative Return", "No numeric points are available.")
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=12, color="#2563eb", alpha=0.45, edgecolors="none")
    ax.axhline(0, color="#9ca3af", lw=1)
    ax.axvline(0, color="#9ca3af", lw=1)
    corr = None
    if len(points) >= 3:
        try:
            corr = pd.Series(xs).corr(pd.Series(ys), method="spearman")
        except Exception:
            corr = None
    note = "Positive slope / positive Spearman correlation means the score ranks better future relative returns."
    if corr is not None and math.isfinite(float(corr)):
        note += f" Spearman={_fmt(corr, 3)}."
    _style_axis(ax, "Rule Alpha Score vs Future Relative Return", "Alpha score", "Future relative return", note)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _metric_points(metric_rows: list[dict[str, Any]], metric: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in metric_rows:
        value = _safe_float(row.get(metric))
        step = _safe_float(row.get("step"))
        if value is not None and step is not None:
            points.append((int(step), value))
    points.sort(key=lambda x: x[0])
    return points


def _metric_columns(metric_rows: list[dict[str, Any]], includes: list[str], excludes: list[str] | None = None) -> list[str]:
    excludes = excludes or []
    columns: set[str] = set()
    for row in metric_rows:
        for key, value in row.items():
            low = key.lower()
            if key in {"step", "source"} or _safe_float(value) is None:
                continue
            if all(term in low for term in includes) and not any(term in low for term in excludes):
                columns.add(key)
    return sorted(columns)


def _fig_metric_lines(title: str, metric_rows: list[dict[str, Any]], metrics: list[str], y_label: str, note: str):
    plt, _ = _get_matplotlib()
    metrics = [m for m in metrics if _metric_points(metric_rows, m)]
    if not metrics:
        return _blank_figure(title, "No matching training metric rows were found.")
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    for i, metric in enumerate(metrics[:6]):
        points = _metric_points(metric_rows, metric)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        label = _short_metric_name(metric)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=label)
        if len(points) >= 2:
            ax.annotate(
                f"{_fmt(ys[0], 3)} -> {_fmt(ys[-1], 3)}",
                xy=(xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=SERIES_COLORS[i % len(SERIES_COLORS)],
            )
    ax.legend(loc="best", frameon=False)
    _style_axis(ax, title, "Training step", y_label, note)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def _short_metric_name(metric: str, limit: int = 46) -> str:
    text = str(metric)
    replacements = [
        ("rollout_reward_scores/", ""),
        ("critic/rewards/", "reward/"),
        ("critic/scores/", "score/"),
        ("global_seqlen/", "seqlen/"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _fig_rl_reward_curve(metric_rows: list[dict[str, Any]], reward_progress_rows: list[dict[str, Any]]):
    metrics = [str(r.get("metric")) for r in reward_progress_rows[:4] if r.get("metric")]
    if not metrics:
        metrics = _reward_columns(metric_rows)[:4]
    return _fig_metric_lines(
        "RL Reward Curve",
        metric_rows,
        metrics,
        "Reward / score",
        "Good sign: validation or rollout reward increases from the first logged point and does not collapse later. Best-first is useful for short smoke tests; last-first is more important for stable long runs.",
    )


def _fig_baseline_vs_rl_reward(baseline_rows: list[dict[str, Any]], reward_progress_rows: list[dict[str, Any]]):
    valid = _safe_float(_row_for_split(baseline_rows, "valid").get("mean_reward"))
    train = _safe_float(_row_for_split(baseline_rows, "train").get("mean_reward"))
    baseline = valid if valid is not None else train
    labels = ["rule baseline"]
    values = [baseline or 0.0]
    if reward_progress_rows:
        item = reward_progress_rows[0]
        labels.extend(["RL first", "RL last", "RL best"])
        values.extend([
            _safe_float(item.get("first_reward"), 0.0) or 0.0,
            _safe_float(item.get("last_reward"), 0.0) or 0.0,
            _safe_float(item.get("best_reward"), 0.0) or 0.0,
        ])
    return _fig_bar_single(
        "Rule Baseline vs RL Reward",
        labels,
        values,
        "Reference point",
        "Reward",
        "For the mentor story, the minimum evidence is RL last > RL first. Stronger evidence is RL best/last approaching or exceeding the rule baseline on validation.",
    )


def _fig_tool_usage(metric_rows: list[dict[str, Any]]):
    metrics = []
    preferred_terms = [
        ["num_tool_calls"],
        ["tool_use_reward"],
        ["missing_tool_penalty"],
        ["tool_calls"],
        ["num_turns"],
    ]
    for terms in preferred_terms:
        for col in _metric_columns(metric_rows, terms):
            if col not in metrics:
                metrics.append(col)
    return _fig_metric_lines(
        "Tool Usage During RL",
        metric_rows,
        metrics[:6],
        "Metric value",
        "Good sign for a search agent: num_tool_calls rises above zero when min_tool_calls is enabled. If missing_tool_penalty stays negative, the model is still avoiding or failing tool calls.",
    )


def _fig_reward_components(metric_rows: list[dict[str, Any]]):
    names = [
        "direction_reward",
        "prob_reward",
        "brier_reward",
        "pnl_reward",
        "evidence_reward",
        "format_reward",
        "tool_use_reward",
        "missing_tool_penalty",
        "search_cost",
    ]
    metrics = []
    for name in names:
        for col in _metric_columns(metric_rows, [name]):
            if col not in metrics:
                metrics.append(col)
    return _fig_metric_lines(
        "Reward Components",
        metric_rows,
        metrics[:6],
        "Component value",
        "Good sign: useful components such as probability/Brier/PnL/tool-use improve while penalty components move toward zero.",
    )


def _fig_optimization_health(metric_rows: list[dict[str, Any]]):
    terms = [
        ["kl"],
        ["entropy"],
        ["clip"],
        ["loss"],
        ["response"],
        ["prompt"],
    ]
    metrics = []
    for term in terms:
        for col in _metric_columns(metric_rows, term):
            low = col.lower()
            if "reward" in low or "score" in low:
                continue
            if col not in metrics:
                metrics.append(col)
    return _fig_metric_lines(
        "Optimization Health Metrics",
        metric_rows,
        metrics[:6],
        "Metric value",
        "Use this page to diagnose training stability. Watch for extreme KL/loss spikes, response length collapse, or clipping metrics saturating.",
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
        if path.endswith(".pdf") and Path(path).name != "stock_agent_rl_report.pdf":
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
    plt, PdfPages = _get_matplotlib()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for fig in _report_figures(
            run_dir=run_dir,
            baseline_rows=baseline_rows,
            dataset_rows=dataset_rows,
            reward_progress_rows=reward_progress_rows,
            baseline_vs_rl_rows=baseline_vs_rl_rows,
            label_rows=label_rows,
            prediction_rows=prediction_rows,
            metric_rows=metric_rows,
            artifacts=artifacts,
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _report_figures(
    run_dir: Path,
    baseline_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    reward_progress_rows: list[dict[str, Any]],
    baseline_vs_rl_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> list[Any]:
    figures: list[Any] = []
    figures.append(_fig_title_summary(run_dir, baseline_rows, dataset_rows, reward_progress_rows))
    figures.append(_fig_workflow())
    figures.append(
        _fig_text_page(
            "How to Read This Report",
            [
                (
                    "Primary success criterion",
                    "For the mentor task, the most direct evidence is RL reward improvement: RL last reward should be higher than RL first reward; RL best reward shows the best checkpoint seen during training.",
                ),
                (
                    "Search-agent evidence",
                    "If tool-use training is enabled, num_tool_calls should be above zero and missing_tool_penalty should move toward zero. If reward rises but tool calls stay at zero, the run is an RL smoke test, not yet strong evidence of search-agent behavior.",
                ),
                (
                    "Prediction evidence",
                    "Accuracy and macro-F1 measure direction classification. Rank IC and top-bottom return measure stock-selection/ranking quality. These are complementary; reward is the training objective.",
                ),
                (
                    "Stability evidence",
                    "KL, loss, clipping, and response-length metrics should not spike or collapse. A short 3090 run can prove the pipeline; a longer A800 run is needed for convincing predictive improvement.",
                ),
            ],
        )
    )
    figures.append(
        _fig_table_page(
            "Dataset Summary",
            dataset_rows,
            ["split", "n", "stocks", "trade_dates", "label_up", "label_neutral", "label_down", "mean_future_relative_return"],
            "Checks whether train/valid/test have enough samples and reasonably balanced labels.",
        )
    )
    figures.append(_fig_count_group("Task Label Distribution", label_rows, "label"))
    figures.append(
        _fig_table_page(
            "Function Tools Used by the Agent",
            _tool_description_rows(),
            ["tool", "meaning", "local_table", "online_during_rl", "interpretation"],
            "All tools are local table lookups during RL rollout. Tushare is used only during data building/cache refresh.",
            max_rows=10,
        )
    )
    figures.append(
        _fig_text_page(
            "Reward Design",
            [
                (
                    "Verifiable target",
                    "Each task hides the future relative return label from the model. The reward function can verify the final prediction against the label after rollout.",
                ),
                (
                    "Main components",
                    "Total reward combines direction correctness, calibrated class probabilities, Brier score, simple PnL-style alpha payoff, evidence/format rewards, tool-use rewards, and penalties.",
                ),
                (
                    "Why this is RLVR-like",
                    "The reward is computed from observable future returns and strict JSON outputs, so it does not require a human preference model.",
                ),
                (
                    "What improves in a good run",
                    "Reward should rise because the policy learns output format, probability calibration, class direction, and when enabled, tool-using behavior.",
                ),
            ],
        )
    )
    figures.append(
        _fig_table_page(
            "Rule Baseline Metrics",
            baseline_rows,
            ["split", "n", "mean_reward", "accuracy", "macro_f1", "brier", "mean_rank_ic", "top_bottom_return"],
            "Rule baseline is a non-RL reference built from deterministic factors. It is not the main claim, but it tells us whether the constructed data has usable signal.",
        )
    )
    figures.append(_fig_baseline_reward(baseline_rows))
    figures.append(_fig_baseline_accuracy_f1(baseline_rows))
    figures.append(_fig_baseline_rank_spread(baseline_rows))
    figures.append(
        _fig_table_page(
            "RL Reward Progress",
            reward_progress_rows,
            ["metric", "points", "first_step", "first_reward", "last_step", "last_reward", "best_step", "best_reward", "delta_last_minus_first", "delta_best_minus_first"],
            "This is the key table for mentor review. Positive last-first means the trained policy improved over the logged run.",
            max_rows=12,
        )
    )
    figures.append(
        _fig_table_page(
            "Rule Baseline vs RL Reward",
            baseline_vs_rl_rows,
            ["rl_metric", "rule_baseline_reference_split", "rule_baseline_mean_reward", "rl_first_reward", "rl_last_reward", "rl_best_reward", "rl_last_minus_first", "rl_last_minus_rule_baseline"],
            "For small 3090 smoke tests, focus on RL last-first. For the final A800 run, also compare against the rule baseline.",
            max_rows=8,
        )
    )
    figures.append(_fig_rl_reward_curve(metric_rows, reward_progress_rows))
    figures.append(_fig_baseline_vs_rl_reward(baseline_rows, reward_progress_rows))
    figures.append(_fig_tool_usage(metric_rows))
    figures.append(_fig_reward_components(metric_rows))
    figures.append(_fig_optimization_health(metric_rows))
    figures.append(_fig_count_group("Rule Prediction Distribution", prediction_rows, "prediction"))
    figures.append(_fig_rule_reward_histogram(_read_cached_table(run_dir / "rollouts" / "rule_baseline_predictions")))
    figures.append(_fig_alpha_vs_return(_read_cached_table(run_dir / "rollouts" / "rule_baseline_predictions")))
    figures.append(
        _fig_text_page(
            "Artifacts and Reproducibility",
            _artifact_sections(artifacts),
            footer="The report directory contains CSV tables for exact values and PDF figures for presentation. Re-run with --mode report-latest to regenerate this PDF after training.",
        )
    )
    return figures


def _tool_description_rows() -> list[dict[str, str]]:
    return [
        {
            "tool": "get_price_factors",
            "meaning": "Stock price/factor snapshot",
            "local_table": "factor_snapshot",
            "online_during_rl": "No",
            "interpretation": "Momentum, relative strength, volatility, turnover, PE/PB, market cap.",
        },
        {
            "tool": "get_market_context",
            "meaning": "Market index context",
            "local_table": "market_context",
            "online_during_rl": "No",
            "interpretation": "Whether the market benchmark is strong, weak, or volatile.",
        },
        {
            "tool": "get_industry_context",
            "meaning": "Industry context",
            "local_table": "industry_context",
            "online_during_rl": "No",
            "interpretation": "Industry momentum, valuation percentile, turnover, volatility.",
        },
        {
            "tool": "get_peer_context",
            "meaning": "Peer comparison",
            "local_table": "peer_context",
            "online_during_rl": "No",
            "interpretation": "Compare the target stock with same-industry peers.",
        },
        {
            "tool": "get_fundamental_snapshot",
            "meaning": "Fundamental snapshot",
            "local_table": "fundamental_snapshot",
            "online_during_rl": "No",
            "interpretation": "Revenue/profit/ROE if available; PE/PB and size fallback otherwise.",
        },
        {
            "tool": "search_announcements",
            "meaning": "Announcement search",
            "local_table": "announcements",
            "online_during_rl": "No",
            "interpretation": "Point-in-time local announcement/event lookup.",
        },
        {
            "tool": "search_news",
            "meaning": "News/event search",
            "local_table": "news",
            "online_during_rl": "No",
            "interpretation": "Point-in-time local news or derived event lookup.",
        },
    ]


def _artifact_sections(artifacts: dict[str, str]) -> list[tuple[str, str]]:
    table_items = []
    figure_items = []
    other_items = []
    for name, path in sorted(artifacts.items()):
        file_name = Path(path).name
        if path.endswith(".csv") or path.endswith(".md"):
            table_items.append(f"{name}: {file_name}")
        elif path.endswith(".pdf") and file_name != "stock_agent_rl_report.pdf":
            figure_items.append(f"{name}: {file_name}")
        else:
            other_items.append(f"{name}: {file_name}")
    sections = []
    sections.append(("Tables", "\n".join(table_items[:16]) or "No table artifacts."))
    sections.append(("Figure PDFs", "\n".join(figure_items[:18]) or "No figure artifacts."))
    sections.append(("Other", "\n".join(other_items[:8]) or "No other artifacts."))
    return sections


def _fig_title_summary(
    run_dir: Path,
    baseline_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    reward_progress_rows: list[dict[str, Any]],
):
    summary_lines = [
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run directory: {run_dir}",
        "",
    ]
    total_n = sum(int(row.get("n", 0) or 0) for row in dataset_rows)
    stocks = sorted({_cell_value(row.get("stocks")) for row in dataset_rows if row.get("stocks") not in (None, "")})
    summary_lines.append(f"Dataset: {total_n} tasks across splits; stocks per split: {', '.join(map(str, stocks)) if stocks else 'unknown'}.")
    valid = _row_for_split(baseline_rows, "valid")
    if valid:
        summary_lines.append(
            "Rule baseline valid: "
            f"mean_reward={_fmt(valid.get('mean_reward'))}, accuracy={_fmt(valid.get('accuracy'))}, macro_f1={_fmt(valid.get('macro_f1'))}."
        )
    if reward_progress_rows:
        primary = reward_progress_rows[0]
        summary_lines.append(
            "Primary RL metric: "
            f"{_short_metric_name(primary.get('metric'))}; first={_fmt(primary.get('first_reward'))}, "
            f"last={_fmt(primary.get('last_reward'))}, best={_fmt(primary.get('best_reward'))}, "
            f"last-first={_fmt(primary.get('delta_last_minus_first'))}."
        )
        delta = _safe_float(primary.get("delta_last_minus_first"))
        verdict = "PASS: reward improved over the logged run." if delta is not None and delta > 0 else "CHECK: reward did not improve from first to last."
        summary_lines.append(verdict)
    else:
        summary_lines.append("No RL metric rows were found. The report can still document data and baseline quality, but not RL improvement.")

    return _fig_text_page(
        "Stock Search-Agent RL Report",
        [
            ("Executive summary", "\n".join(summary_lines)),
            (
                "Project goal",
                "Build a function-call search agent for A-share stock direction prediction, then use agentic RL with a verifiable reward to improve prediction behavior.",
            ),
            (
                "Core claim to show",
                "The strongest evidence is a positive reward trajectory together with non-zero tool usage and stable optimization diagnostics.",
            ),
        ],
        footer="This PDF is generated automatically from the run directory; exact numbers are available in report/tables/*.csv.",
    )


def _fig_text_page(title: str, sections: list[tuple[str, str]], footer: str = ""):
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.04, 0.94, title, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")
    y = 0.86
    for heading, body in sections:
        ax.text(0.05, y, heading, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#111827", va="top")
        y -= 0.045
        wrapped = "\n".join(textwrap.wrap(str(body), width=115, replace_whitespace=False))
        ax.text(0.07, y, wrapped, transform=ax.transAxes, fontsize=10.5, color="#374151", va="top", linespacing=1.35)
        y -= max(0.10, 0.030 * (wrapped.count("\n") + 1) + 0.055)
        if y < 0.10:
            break
    if footer:
        ax.text(0.04, 0.035, footer, transform=ax.transAxes, fontsize=9, color="#6b7280", va="bottom")
    return fig


def _fig_table_page(title: str, rows: list[dict[str, Any]], columns: list[str], note: str, max_rows: int = 14):
    plt, _ = _get_matplotlib()
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.text(0.03, 0.95, title, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(0.03, 0.89, "\n".join(textwrap.wrap(note, width=130)), transform=ax.transAxes, fontsize=10, color="#4b5563", va="top")
    if not rows:
        ax.text(0.5, 0.50, "No rows available.", transform=ax.transAxes, ha="center", va="center", fontsize=13, color="#6b7280")
        return fig
    usable_cols = [c for c in columns if any(c in row for row in rows)]
    data_rows = []
    for row in rows[:max_rows]:
        data_rows.append([_table_cell(row.get(col), col) for col in usable_cols])
    table = ax.table(
        cellText=data_rows,
        colLabels=[_wrap_header(c) for c in usable_cols],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0.03, 0.12, 0.94, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.18)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f9fafb")
    if len(rows) > max_rows:
        ax.text(0.03, 0.06, f"Showing first {max_rows} of {len(rows)} rows. See CSV for the full table.", transform=ax.transAxes, fontsize=9, color="#6b7280")
    return fig


def _wrap_header(text: str) -> str:
    return "\n".join(textwrap.wrap(str(text).replace("_", " "), width=14)) or str(text)


def _table_cell(value: Any, col: str) -> str:
    value = _cell_value(value)
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    if col == "metric" or col == "rl_metric":
        text = _short_metric_name(text, 36)
    text = text.replace("\n", " ")
    if len(text) > 54:
        return text[:51] + "..."
    return text


def parse_metric_dict_from_line(line: str) -> dict[str, Any]:
    """Small helper for ad-hoc debugging of console metric lines."""
    try:
        return ast.literal_eval(line)
    except Exception:
        return {}
