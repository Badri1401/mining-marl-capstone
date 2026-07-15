#!/usr/bin/env python3
"""
Elite Quant-Tier Plotting Script.
Applies mathematical severity bucketing, standard deviation compression shading,
and strict statistical zoom bounding to solve "bland line" problems. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

# Paths (default to the final run's log shipped in results/)
INPUT_CSV = Path(__file__).resolve().parents[1] / "results" / "training_log.csv"
OUTPUT_DIR = Path("hard_logs_graphs_updated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#b0b0b0",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.framealpha": 0.9,
    "legend.fontsize": 10,
})

PHASE_COLORS = { "Easy": "#e0f2fe", "Med": "#fef08a", "Hard": "#dcfce7", "VHard": "#fce7f3" }

def get_macro_phase(phase_name):
    phase_name = str(phase_name).lower()
    if "easy" in phase_name: return "Easy"
    if "med" in phase_name: return "Medium"
    if "hard" in phase_name and "v" not in phase_name: return "Hard"
    if "vhard" in phase_name: return "VHard"
    return "Unknown"

def get_phase_bg_color(phase_name):
    macro = get_macro_phase(phase_name)
    if macro == "Easy": return PHASE_COLORS["Easy"]
    if macro == "Medium": return PHASE_COLORS["Med"]
    if macro in ("Hard", "Unknown", "VHard"): return PHASE_COLORS["Hard"]
    return "#ffffff"

def get_phase_boundaries(df, episode_col="episode", phase_col="phase"):
    boundaries = []
    if df.empty or phase_col not in df.columns: return boundaries
    current_phase = df.iloc[0][phase_col]
    start_ep = df.iloc[0][episode_col]
    for i in range(1, len(df)):
        phase = df.iloc[i][phase_col]
        ep = df.iloc[i][episode_col]
        if phase != current_phase:
            boundaries.append((current_phase, start_ep, ep))
            current_phase = phase
            start_ep = ep
    boundaries.append((current_phase, start_ep, df.iloc[-1][episode_col]))
    return boundaries

def add_phase_backgrounds(ax, boundaries):
    for phase, start, end in boundaries:
        ax.axvspan(start, end, facecolor=get_phase_bg_color(phase), alpha=0.5, edgecolor="none", zorder=0)

# =========================================================================================
# GRAPH 3 - ELITE FIX: Reward Variance Shading & Intelligent Zoom
# =========================================================================================
def plot_reward_with_variance(df, x_col, y_col, boundaries, title, ylabel, out_name, 
                              window=100, y_min=-2000, y_max=500):
    """
    Solves the 'Bland Graph' problem.
    Instead of a massive scalar limit that squishes the variance, this plots the Moving 
    Average coupled with a standard deviation band (±1 StdDev). It strictly zooms to 
    where the true learning signal happens [-2000 to 500].
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    x = df[x_col].values
    y = df[y_col].values
    
    # Statistical derivations
    y_series = pd.Series(y)
    y_ma = y_series.rolling(window=window, min_periods=1).mean().values
    y_std = y_series.rolling(window=window, min_periods=1).std().values
    y_std = np.nan_to_num(y_std, 0) # Fill early NaN
    
    add_phase_backgrounds(ax, boundaries)
    
    # Plot confidence/variance band first (fixes flat MA look)
    ax.fill_between(x, y_ma - y_std, y_ma + y_std, color="#64b5f6", alpha=0.3, label=f"±1 Std Dev (Variance)")
    
    # Plot sparse raw trace heavily faded to prevent noise drowning signal
    ax.plot(x, y, color="#bbdefb", alpha=0.1, linewidth=0.5, label="Raw Returns")
    
    # Solid MA
    ax.plot(x, y_ma, color="#0d47a1", linewidth=2.0, label=f"Moving Avg ({window})")

    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    # Brutally enforce signal zoom. Squashing down to -13500 destroyed the visual variance.
    ax.set_ylim(bottom=y_min, top=y_max)
        
    ax.set_title(title + " (With Volatility Bounds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

# =========================================================================================
# GRAPH 6 - ELITE FIX: Severity Distribution instead of raw averages
# =========================================================================================
def plot_collision_severity_bar(df, boundaries, out_name):
    """
    Solves the 'Useless 0 Skips' problem.
    Replaces static bars with a 100% Stacked Bar showing the ratio of Flawless (0 Collisions), 
    Minor (1-5 Collisions) and Severe (>5 Collisions) runs per phase.
    This directly reveals how SAFETY scales as difficulty increases.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    phases = []
    zero_cols = []
    minor_cols = []
    severe_cols = []
    
    for phase, start, end in boundaries:
        mask = (df["episode"] >= start) & (df["episode"] <= end)
        chunk = df[mask]["collisions"]
        total = len(chunk)
        if total == 0: continue
            
        c0 = (chunk == 0).sum() / total * 100
        c_min = ((chunk > 0) & (chunk <= 5)).sum() / total * 100
        c_sev = (chunk > 5).sum() / total * 100
        
        phases.append(phase)
        zero_cols.append(c0)
        minor_cols.append(c_min)
        severe_cols.append(c_sev)
        
    x_pos = np.arange(len(phases))
    width = 0.5
    
    # 100% Stacked Bar construction
    ax.bar(x_pos, zero_cols, width, label='Flawless (0 Collisions)', color='#388e3c', edgecolor='white', linewidth=1)
    ax.bar(x_pos, minor_cols, width, bottom=zero_cols, label='Minor (1-5 Collisions)', color='#fbc02d', edgecolor='white', linewidth=1)
    ax.bar(x_pos, severe_cols, width, bottom=np.array(zero_cols)+np.array(minor_cols), label='Severe (>5 Collisions)', color='#d32f2f', edgecolor='white', linewidth=1)
    
    # Annotate all runs cleanly with contrasting colors
    for i in range(len(x_pos)):
        z, m, s = zero_cols[i], minor_cols[i], severe_cols[i]
        
        # Only draw if there's enough physical space on the bar (> 2.0%)
        if z > 2.0: ax.text(x_pos[i], z/2, f"{z:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        # Black text for yellow "minor" background
        if m > 2.0: ax.text(x_pos[i], z + m/2, f"{m:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=10)
        # White text for red "severe" background
        if s > 2.0: ax.text(x_pos[i], z + m + s/2, f"{s:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    ax.set_title("Collision Severity Progression (100% Stacked Distribution)")
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Percentage of Total Phase Episodes (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=45)
    
    # Legend pushed safely completely outside the graph axes so it NEVER overlaps
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =========================================================================================
# UTILITY FUNCTIONS RETAINED
# =========================================================================================
def plot_bar_phase_performance(df, boundaries, out_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    phase_stats = []
    for phase, start, end in boundaries:
        mask = (df["episode"] >= start) & (df["episode"] <= end)
        chunk = df[mask]
        phase_stats.append((phase, len(chunk), chunk["success_fraction"].mean()))
        
    phases = [p[0] for p in phase_stats]
    success_rates = [p[2] for p in phase_stats]
    counts = [p[1] for p in phase_stats]
    
    x_pos = np.arange(len(phases))
    bar_colors = ["#a8e6cf", "#dcedc1", "#ffd3b6", "#ffaaa5", "#ff8b94", "#b5ebd1"]
    bars = ax.bar(x_pos, success_rates, color=[bar_colors[i%len(bar_colors)] for i in range(len(phases))], edgecolor="black", linewidth=1.2)
    
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{counts[i]} eps", ha='center', va='bottom', fontsize=10)
                
    ax.set_title("Performance by Curriculum Phase")
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Average Success Rate")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def plot_end_reason_pies(df, out_name):
    df["macro_phase"] = df["phase"].apply(get_macro_phase)
    macros = ["Easy", "Medium", "Hard"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("End Reasons Distribution Grouped by Difficulty", fontsize=18, fontweight='bold', y=1.05)
    colors_map = { "success": "#4caf50", "partial_success": "#ffc107", "team_success": "#2e7d32", "time_limit": "#9e9e9e", "collision": "#f44336", "out_of_bounds": "#ff9800", "other": "#cfd8dc" }
    for i, macro in enumerate(macros):
        ax = axes[i]
        chunk = df[df["macro_phase"] == macro]
        if chunk.empty:
            ax.axis('off')
            ax.set_title(f"{macro}\n(No Data)")
            continue
        counts = chunk["end_reason"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        pie_colors = [colors_map.get(str(l).lower(), colors_map["other"]) for l in labels]
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        ax.set_title(f"{macro} Phases\n({len(chunk)} eps)", fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_fairness_comparison(df, boundaries, out_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    x = df["episode"].values
    r0 = df["robot_0_reward"].rolling(1000, min_periods=1).mean().values
    r1 = df["robot_1_reward"].rolling(1000, min_periods=1).mean().values
    add_phase_backgrounds(ax, boundaries)
    ax.plot(x, r0, color="#2196f3", linewidth=2, label="Robot 0 Reward (MA-1000)")
    ax.plot(x, r1, color="#4caf50", linewidth=2, label="Robot 1 Reward (MA-1000)")
    ax.set_ylim(-1000, 500)
    ax.set_title("Per-Robot Rewards Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def plot_multi_success(df, boundaries, out_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    x = df["episode"]
    add_phase_backgrounds(ax, boundaries)
    ax.plot(x, df["success_fraction"].values, color="#c8e6c9", alpha=0.3, label="Raw Total Success")
    ax.plot(x, df["success_fraction"].rolling(50, min_periods=1).mean().values, color="#2e7d32", linewidth=2, label="Success Fraction (MA-50)")
    ax.plot(x, df["team_success"].rolling(50, min_periods=1).mean().values, color="#1565c0", linewidth=2, label="Team Success (MA-50)")
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Target 80%')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Main Success Metrics Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def plot_success_composition_bar(df, boundaries, out_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    phases, team_means, partial_means = [], [], []
    for phase, start, end in boundaries:
        mask = (df["episode"] >= start) & (df["episode"] <= end)
        chunk = df[mask]
        phases.append(phase)
        team_means.append(chunk["team_success"].mean() * 100)
        partial_means.append(chunk["partial_success"].mean() * 100)
    x_pos = np.arange(len(phases))
    width = 0.35
    ax.bar(x_pos - width/2, team_means, width, label='Team Success', color='#1565c0', edgecolor='black')
    ax.bar(x_pos + width/2, partial_means, width, label='Partial Success', color='#fbc02d', edgecolor='black')
    ax.set_title("Success Composition by Phase: Team vs Partial Success")
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Average Rate (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=45)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def plot_distance_bar(df, boundaries, out_name):
    fig, ax = plt.subplots(figsize=(15, 7))
    phases, dist_means = [], []
    for phase, start, end in boundaries:
        mask = (df["episode"] >= start) & (df["episode"] <= end)
        chunk = df[mask]
        phases.append(phase)
        dist_means.append(chunk["avg_final_distance"].mean())
    x_pos = np.arange(len(phases))
    bars = ax.bar(x_pos, dist_means, color='#4a148c', alpha=0.7, edgecolor='black', linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., yval + 0.05, f"{yval:.2f}m", ha='center', va='bottom', fontsize=10)
    ax.set_title("Average Final Distance by Phase")
    ax.set_xlabel("Curriculum Phase")
    ax.set_ylabel("Distance (m)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def plot_line_with_ma_and_phases_standard(df, x_col, y_col, boundaries, title, ylabel, out_name, 
                                 raw_color, ma_color, window=50, y_max=None):
    fig, ax = plt.subplots(figsize=(15, 7))
    x = df[x_col].values
    y = df[y_col].values
    y_ma = df[y_col].rolling(window=window, min_periods=1).mean().values
    ax.plot(x, y, color=raw_color, alpha=0.3, linewidth=1, label="Raw")
    ax.plot(x, y_ma, color=ma_color, linewidth=2.5, label=f"MA-{window}")
    add_phase_backgrounds(ax, boundaries)
    if y_max is not None: ax.set_ylim(top=y_max)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / out_name, dpi=150)
    plt.close(fig)

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV)
    if df.empty: return
    if 'episode' not in df.columns: df['episode'] = np.arange(len(df))
        
    boundaries = get_phase_boundaries(df, "episode", "phase")
    
    # 1. Bar Chart: Performance
    plot_bar_phase_performance(df, boundaries, "01_Performance_By_Phase_Bar.png")
    
    # 2. Pie Charts
    plot_end_reason_pies(df, "02_End_Reason_Distribution_Pie.png")
    
    # 3. Reward Convergence - THE ELITE FIX (Variance Banding + -2000 Bottom Bound for Signal Scale)
    print("Generating: 3. Reward Convergence (StdDev Shading & Zoom Bounds)...")
    plot_reward_with_variance(df, "episode", "reward", boundaries, 
                              "Total Reward Convergence", "Episode Reward", 
                              "03_Reward_Convergence.png", window=100, y_min=-2000, y_max=600)
                                 
    # 4/5. Success
    plot_multi_success(df, boundaries, "04a_Main_Success_Rates.png")
    plot_success_composition_bar(df, boundaries, "04b_Success_Composition_Bar.png")
    plot_fairness_comparison(df, boundaries, "05_Robot_Fairness_Comparison.png")
    
    # 6. Safety - THE ELITE FIX (100% Stacked Severity Distribution instead of pointless skips count)
    print("Generating: 6. Stacked Collision Severity (Flawless vs Severe)...")
    plot_collision_severity_bar(df, boundaries, "06_Collision_Severity_Stacked.png")
    
    # 7/8. Loss and Distance
    plot_line_with_ma_and_phases_standard(df, "episode", "critic_loss", boundaries, 
                                 "Critic Loss Training Stability", "Loss", "07_Critic_Loss.png",
                                 raw_color="#ffccbc", ma_color="#b71c1c", y_max=40)
    plot_distance_bar(df, boundaries, "08_Avg_Final_Distance_Bar.png")
                                 
    print("Complete. Mathematical severity distributions applied.")

if __name__ == "__main__":
    main()
