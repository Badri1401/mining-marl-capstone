#!/usr/bin/env python3
"""Plot all columns from a training CSV in one run.

Usage:
    python3 plot_training_log_hard_all.py
    python3 plot_training_log_hard_all.py --csv training_log_hard.csv --window 1000 --max-points 12000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, math.ceil(len(x) / max_points))
    return x[::step], y[::step]


def plot_numeric_columns(
    df: pd.DataFrame,
    episode_col: str,
    numeric_cols: list[str],
    rolling_window: int,
    max_points: int,
    out_path: Path,
) -> None:
    n = len(numeric_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.2 * nrows), squeeze=False)
    x_full = df[episode_col].to_numpy()

    for i, col in enumerate(numeric_cols):
        ax = axes[i // ncols][i % ncols]
        y_full = df[col].to_numpy(dtype=float)

        valid_mask = np.isfinite(x_full) & np.isfinite(y_full)
        x = x_full[valid_mask]
        y = y_full[valid_mask]

        if len(x) == 0:
            ax.set_title(f"{col} (no numeric data)")
            ax.grid(alpha=0.25)
            continue

        x_plot, y_plot = downsample_xy(x, y, max_points)
        ax.plot(x_plot, y_plot, linewidth=0.8, alpha=0.35, label="raw")

        if rolling_window > 1 and len(y) >= rolling_window:
            y_roll = pd.Series(y).rolling(rolling_window, min_periods=1).mean().to_numpy()
            x_roll, y_roll_plot = downsample_xy(x, y_roll, max_points)
            ax.plot(x_roll, y_roll_plot, linewidth=1.4, alpha=0.95, label=f"rolling({rolling_window})")

        ax.set_title(col)
        ax.set_xlabel(episode_col)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.suptitle("All Numeric Metrics", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def plot_categorical_columns(
    df: pd.DataFrame,
    episode_col: str,
    categorical_cols: list[str],
    max_points: int,
    out_path: Path,
) -> None:
    if not categorical_cols:
        return

    n = len(categorical_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.8 * nrows), squeeze=False)
    x_full = df[episode_col].to_numpy()

    for i, col in enumerate(categorical_cols):
        ax = axes[i // ncols][i % ncols]

        s = df[col].fillna("<NA>").astype(str)
        # Preserve first-seen category order for readable y-axis labels.
        categories = list(dict.fromkeys(s.tolist()))
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        y_full = s.map(mapping).to_numpy(dtype=float)

        valid_mask = np.isfinite(x_full) & np.isfinite(y_full)
        x = x_full[valid_mask]
        y = y_full[valid_mask]
        x_plot, y_plot = downsample_xy(x, y, max_points)

        ax.plot(x_plot, y_plot, linewidth=0.9)
        ax.set_title(col)
        ax.set_xlabel(episode_col)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.grid(alpha=0.25)

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.suptitle("All Categorical Metrics (Encoded Over Episodes)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all columns from training log CSV")
    default_csv = Path(__file__).resolve().parents[1] / "results" / "training_log.csv"
    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to CSV file")
    parser.add_argument("--window", type=int, default=1000, help="Rolling mean window size")
    parser.add_argument("--max-points", type=int, default=12000, help="Max points per plotted line")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive figures")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"CSV is empty: {args.csv}")

    episode_col = "episode" if "episode" in df.columns else df.columns[0]

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c != episode_col
    ]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [episode_col]]

    out_dir = args.csv.parent / "all_plots"
    out_dir.mkdir(exist_ok=True)

    numeric_out = out_dir / "all_numeric_metrics.png"
    plot_numeric_columns(
        df=df,
        episode_col=episode_col,
        numeric_cols=numeric_cols,
        rolling_window=max(1, args.window),
        max_points=max(1000, args.max_points),
        out_path=numeric_out,
    )

    categorical_out = out_dir / "all_categorical_metrics.png"
    plot_categorical_columns(
        df=df,
        episode_col=episode_col,
        categorical_cols=categorical_cols,
        max_points=max(1000, args.max_points),
        out_path=categorical_out,
    )

    print(f"Loaded rows: {len(df):,}")
    print(f"Episode column: {episode_col}")
    print(f"Numeric columns plotted: {len(numeric_cols)}")
    print(f"Categorical columns plotted: {len(categorical_cols)}")
    print(f"Saved: {numeric_out}")
    if categorical_cols:
        print(f"Saved: {categorical_out}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
