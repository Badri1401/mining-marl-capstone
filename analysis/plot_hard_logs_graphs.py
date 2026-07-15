#!/usr/bin/env python3
"""Create polished, separate plots for every metric in training_log_hard.csv.

- One figure per metric (no combined dashboard).
- Uses training progress (%) on x-axis (not episode id).
- Saves all figures into ./hard_logs_graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Rolling windows are metric-specific so each chart is readable at 218k rows.
ROLLING_WINDOWS: Dict[str, int] = {
    "reward": 1000,
    "actor_loss": 1000,
    "critic_loss": 1000,
    "steps": 1000,
    "collisions": 1000,
    "avg_final_distance": 1000,
    "success_fraction": 500,
    "team_success": 500,
    "partial_success": 500,
    "goals_reached": 500,
    "replay_skipped_collision": 500,
    "robot_0_reward": 1000,
    "robot_1_reward": 1000,
}

# Columns best shown directly because smoothing would hide intentional schedules
# or phase-internal counters.
DIRECT_PLOT_COLUMNS = {
    "noise",
    "phase_collision_skips",
    "phase_episode_count",
    "phase_replay_stored",
}

# Explicit order to guarantee the requested key metrics appear first.
PREFERRED_ORDER = [
    "reward",
    "actor_loss",
    "critic_loss",
    "steps",
    "collisions",
    "avg_final_distance",
]


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#f8f9fb",
            "axes.facecolor": "#fdfdfd",
            "axes.edgecolor": "#d9dee7",
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.alpha": 0.22,
            "grid.color": "#9aa5b1",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "/"}:
            out.append("_")
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def get_progress_axis(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 100.0, n)


def downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], y[::step]


def metric_window(metric: str, default_window: int) -> int:
    return ROLLING_WINDOWS.get(metric, default_window)


def plot_numeric_metric(
    x: np.ndarray,
    series: pd.Series,
    metric: str,
    output_dir: Path,
    default_window: int,
    max_points: int,
) -> str:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return "skipped (no numeric data)"

    fig, ax = plt.subplots(figsize=(13.5, 6.5))

    raw_color = "#8da0cb"
    smooth_color = "#1f4e79"
    accent_color = "#f97316"

    x_plot, y_plot = downsample(x, y, max_points)

    if metric in DIRECT_PLOT_COLUMNS:
        ax.plot(x_plot, y_plot, color=accent_color, linewidth=2.2, label="Direct")
        strategy = "direct"
    else:
        window = metric_window(metric, default_window)
        y_roll = pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()
        x_roll, y_roll_plot = downsample(x, y_roll, max_points)

        ax.plot(x_plot, y_plot, color=raw_color, linewidth=0.8, alpha=0.25, label="Raw")
        ax.plot(x_roll, y_roll_plot, color=smooth_color, linewidth=2.4, label=f"Rolling Mean ({window})")

        # Add an uncertainty ribbon for visual depth and quick volatility reading.
        y_std = pd.Series(y).rolling(window=window, min_periods=1).std().fillna(0.0).to_numpy()
        y_hi = y_roll + y_std
        y_lo = y_roll - y_std
        x_band, y_hi_band = downsample(x, y_hi, max_points)
        _, y_lo_band = downsample(x, y_lo, max_points)
        ax.fill_between(x_band, y_lo_band, y_hi_band, color=smooth_color, alpha=0.10, label="Rolling +/-1 std")
        strategy = f"rolling({window}) + raw"

    ax.set_title(f"{metric.replace('_', ' ').title()} Over Training Progress")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlim(0, 100)
    ax.legend(loc="best")

    out = output_dir / f"{slugify(metric)}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return strategy


def plot_phase_metric(x: np.ndarray, series: pd.Series, output_dir: Path) -> str:
    s = series.fillna("<NA>").astype(str)
    categories = list(dict.fromkeys(s.tolist()))
    mapping = {cat: i for i, cat in enumerate(categories)}
    y = s.map(mapping).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    cmap = plt.get_cmap("tab20", max(len(categories), 2))

    ax.plot(x, y, color="#111827", linewidth=1.8)
    for i, cat in enumerate(categories):
        ax.axhspan(i - 0.45, i + 0.45, color=cmap(i), alpha=0.20)

    ax.set_title("Curriculum Phase Over Training Progress")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Phase")
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    out = output_dir / "phase.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return "categorical timeline"


def plot_end_reason_metric(x: np.ndarray, series: pd.Series, output_dir: Path, window: int) -> str:
    s = series.fillna("<NA>").astype(str)
    one_hot = pd.get_dummies(s)

    # Rolling proportion by reason gives a stable, interpretable trend.
    rolling_prop = one_hot.rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    colors = plt.get_cmap("Set2", rolling_prop.shape[1])

    stacked = np.zeros(len(rolling_prop))
    for i, col in enumerate(rolling_prop.columns):
        vals = rolling_prop[col].to_numpy(dtype=float)
        ax.fill_between(x, stacked, stacked + vals, color=colors(i), alpha=0.7, label=col)
        stacked += vals

    ax.set_title(f"End Reason Mix Over Training Progress (Rolling {window})")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Proportion")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")

    out = output_dir / "end_reason.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"rolling({window}) stacked proportions"


def ordered_numeric_columns(cols: List[str]) -> List[str]:
    first = [c for c in PREFERRED_ORDER if c in cols]
    rest = [c for c in cols if c not in set(first)]
    return first + rest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create separate polished graphs for hard training logs")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "training_log.csv",
        help="Path to training CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for graphs")
    parser.add_argument("--default-window", type=int, default=1000, help="Default rolling window")
    parser.add_argument("--max-points", type=int, default=14000, help="Max points per plotted line")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"CSV is empty: {args.csv}")

    out_dir = args.output_dir or (args.csv.parent / "hard_logs_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    x = get_progress_axis(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "episode" in numeric_cols:
        numeric_cols.remove("episode")
    numeric_cols = ordered_numeric_columns(numeric_cols)

    strategies: Dict[str, str] = {}

    for metric in numeric_cols:
        strategies[metric] = plot_numeric_metric(
            x=x,
            series=df[metric],
            metric=metric,
            output_dir=out_dir,
            default_window=max(1, args.default_window),
            max_points=max(2000, args.max_points),
        )

    if "phase" in df.columns:
        strategies["phase"] = plot_phase_metric(x=x, series=df["phase"], output_dir=out_dir)

    if "end_reason" in df.columns:
        strategies["end_reason"] = plot_end_reason_metric(
            x=x,
            series=df["end_reason"],
            output_dir=out_dir,
            window=500,
        )

    report_path = out_dir / "plot_strategy_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Total CSV columns: {len(df.columns)}\n")
        f.write(f"Plotted metrics (excluding episode): {len(strategies)}\n")
        f.write("\nMetric -> Strategy\n")
        f.write("-" * 64 + "\n")
        for k, v in strategies.items():
            f.write(f"{k}: {v}\n")

    print(f"Rows loaded: {len(df):,}")
    print(f"CSV columns: {len(df.columns)}")
    print(f"Graphs generated: {len(strategies)}")
    print(f"Output directory: {out_dir}")
    print(f"Strategy report: {report_path}")


if __name__ == "__main__":
    main()
