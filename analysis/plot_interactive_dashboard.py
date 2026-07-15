#!/usr/bin/env python3
"""Builds a modern, interactive HTML dashboard using Plotly for MARL Training Metrics."""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set dark theme globally
import plotly.io as pio
pio.templates.default = "plotly_dark"


def downsample(df: pd.DataFrame, max_rows: int = 15000) -> pd.DataFrame:
    """Downsamples dataframe to avoid massive HTML files and browser lag."""
    if len(df) <= max_rows:
        return df
    step = math.ceil(len(df) / max_rows)
    return df.iloc[::step].copy()


def create_reward_noise_chart(df: pd.DataFrame, episode_col: str, window: int = 500) -> str:
    """Creates a line chart with rolled avg + std dev shaded bands for Rewards, overlaid with Noise."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate rolling statistics
    rolled_mean = df[['reward', 'robot_0_reward', 'robot_1_reward', 'noise']].rolling(window).mean()
    rolled_std = df[['reward', 'robot_0_reward', 'robot_1_reward']].rolling(window).std()
    
    # Extract
    x = df[episode_col].to_numpy()
    r_mean = rolled_mean['reward'].to_numpy()
    r_std = rolled_std['reward'].to_numpy()
    noise = rolled_mean['noise'].to_numpy()
    
    # Downsample
    mask = ~pd.isna(r_mean)
    x, r_mean, r_std, noise = x[mask], r_mean[mask], r_std[mask], noise[mask]
    
    step = max(1, len(x) // 5000)
    x, r_mean, r_std, noise = x[::step], r_mean[::step], r_std[::step], noise[::step]
    
    # Overall Reward Ribbon
    fig.add_trace(go.Scatter(x=x, y=r_mean + r_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=r_mean - r_std, mode='lines', fill='tonexty', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(width=0), name='Reward Variance', hoverinfo='skip'), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=r_mean, mode='lines', name='Total Reward', line=dict(color='#38bdf8', width=2)), secondary_y=False)
    
    # Noise
    fig.add_trace(go.Scatter(x=x, y=noise, mode='lines', name='Noise Level', line=dict(color='#f43f5e', dash='dot', width=1.5)), secondary_y=True)

    fig.update_layout(
        title="DDPG Reward Convergence & Exploration Decay",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Exploration Noise", secondary_y=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_success_area_chart(df: pd.DataFrame, episode_col: str, window: int = 500) -> str:
    """100% Stacked Area tracking outcomes."""
    df_rolled = df[[episode_col, 'team_success', 'partial_success']].rolling(window, on=episode_col).mean().dropna()
    
    x = df_rolled[episode_col]
    team = np.clip(df_rolled['team_success'], 0, 1)
    partial = np.clip(df_rolled['partial_success'], 0, 1)
    failures = np.clip(1.0 - (team + partial), 0, 1)
    
    step = max(1, len(x) // 5000)
    x, team, partial, failures = x[::step], team[::step], partial[::step], failures[::step]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=team, mode='lines', stackgroup='one', name='Team Success (2 Goals)', line=dict(color='#10b981', width=0), fillcolor='rgba(16, 185, 129, 0.8)'))
    fig.add_trace(go.Scatter(x=x, y=partial, mode='lines', stackgroup='one', name='Partial Success (1 Goal)', line=dict(color='#f59e0b', width=0), fillcolor='rgba(245, 158, 11, 0.8)'))
    fig.add_trace(go.Scatter(x=x, y=failures, mode='lines', stackgroup='one', name='Failures', line=dict(color='#ef4444', width=0), fillcolor='rgba(239, 68, 68, 0.8)'))
    
    fig.update_layout(
        title="Training Success Outcomes Over Time",
        yaxis=dict(title="Proportion of Episodes", range=[0, 1]),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_sunburst_end_reasons(df: pd.DataFrame) -> str:
    """Flow of end reasons distributed by phase."""
    if 'end_reason' not in df.columns or 'phase' not in df.columns:
        return "<i>End reason or phase column missing!</i>"
        
    counts = df.groupby(['phase', 'end_reason']).size().reset_index(name='count')
    fig = px.sunburst(counts, path=['phase', 'end_reason'], values='count', 
                      title="Episode Terminations per Curriculum Phase",
                      color='end_reason',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_violins(df: pd.DataFrame) -> str:
    """Violin plots comparing distributions of loss and distance per phase."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Avg Final Distance", "Actor Loss", "Critic Loss"))
    
    # We downsample the actual raw points going to violins drastically so hover doesn't crash HTML
    df_violin = df.sample(n=min(len(df), 15000), random_state=42)
    phases = df_violin['phase'].unique()
    
    for c, phase in enumerate(phases):
        phase_df = df_violin[df_violin['phase'] == phase]
        
        # Avg distance
        fig.add_trace(go.Violin(x=["Distance"]*len(phase_df), y=phase_df['avg_final_distance'], legendgroup=phase, scalegroup=phase, name=phase, side='positive', line_color=px.colors.qualitative.Vivid[c % 10], showlegend=(c==0)), row=1, col=1)
        # Actor Loss
        fig.add_trace(go.Violin(x=["Actor Loss"]*len(phase_df), y=phase_df['actor_loss'], legendgroup=phase, scalegroup=phase, name=phase, side='positive', line_color=px.colors.qualitative.Vivid[c % 10], showlegend=False), row=1, col=2)
        # Critic Loss
        fig.add_trace(go.Violin(x=["Critic Loss"]*len(phase_df), y=phase_df['critic_loss'], legendgroup=phase, scalegroup=phase, name=phase, side='positive', line_color=px.colors.qualitative.Vivid[c % 10], showlegend=False), row=1, col=3)

    fig.update_layout(
        violingap=0, violinmode='overlay',
        title="Distributions & Variance Tightening Across Phases",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.update_traces(meanline_visible=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_heatmap(df: pd.DataFrame, episode_col: str, y_col: str, title: str, colorscale: str) -> str:
    """2D Density Histograms for heavily clustered data like collisions."""
    # Compute histogram server side rather than dumping 200k points to client
    x = df[episode_col].to_numpy()
    y = df[y_col].fillna(0).to_numpy()
    
    x_bins = min(200, max(10, len(x) // 500))
    # We want y bins covering typical integer ranges without too many bins
    y_max = np.percentile(y, 99.5) if np.max(y) > 0 else 10 # ignore upper extreme outliers for binning
    y_bins = 40
    
    H, xedges, yedges = np.histogram2d(x, np.clip(y, 0, y_max), bins=[x_bins, y_bins])
    
    # Plotly heatmap
    fig = go.Figure(go.Heatmap(
        z=H.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale=colorscale,
        reversescale=(colorscale == 'YlOrRd'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title=y_col.replace('_', ' ').title(),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_radar(df: pd.DataFrame) -> str:
    """Radar chart comparing resource expenditure per phase."""
    cols = ['phase_collision_skips', 'phase_episode_count', 'phase_replay_stored']
    if not all(c in df.columns for c in cols):
        return "<i>Phase resource columns missing.</i>"
        
    # Get max of each phase resource per phase (since they are likely cumulative per phase)
    res = df.groupby('phase')[cols].max()
    
    # Normalize across phases for the radar polygon shape
    normalized = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Pastel
    for count, phase in enumerate(res.index):
        fig.add_trace(go.Scatterpolar(
            r=normalized.loc[phase].values.tolist() + [normalized.loc[phase].values[0]],
            theta=[c.replace("phase_", "").title() for c in cols] + [cols[0].replace("phase_", "").title()],
            fill='toself',
            name=phase,
            hovertemplate="Normalized: %{r:.2f}<br>Raw Value: Not Shown Here",
            line_color=colors[count % len(colors)]
        ))
        
    fig.update_layout(
        title="Relative Computational Effort / Resource Usage Per Phase",
        polar=dict(radialaxis=dict(visible=False)),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def main():
    parser = argparse.ArgumentParser()
    # Default exactly on the expected log in that directory
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "training_log.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "interactive_dashboard.html")
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"Error: {args.csv} does not exist.")
        return
        
    print(f"Loading {args.csv.name}...")
    df = pd.read_csv(args.csv)
    
    if df.empty:
        print("CSV is empty.")
        return
        
    episode_col = 'episode' if 'episode' in df.columns else df.columns[0]
    
    print("Generating Interactive Figures (this might take a few seconds)...")
    
    div_reward = create_reward_noise_chart(df, episode_col)
    div_success = create_success_area_chart(df, episode_col)
    div_sunburst = create_sunburst_end_reasons(df)
    div_violins = create_violins(df)
    div_collide_heat = create_heatmap(df, episode_col, 'collisions', "Collision Map (Density Cooling Over Time)", 'Hot')
    div_steps_heat = create_heatmap(df, episode_col, 'steps', "Steps Density (Faster Goal Reaching)", 'Tealgrn')
    div_radar = create_radar(df)
    
    print(f"Writing dashboard to {args.output.name}...")
    
    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MARL Training Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            :root {{
                --bg: #0f172a; --panel: #1e293b; --text: #f8fafc; --accent: #38bdf8;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0; padding: 20px;
            }}
            .header {{
                text-align: center; padding: 20px 0; margin-bottom: 30px;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px;
                border: 1px solid #334155;
            }}
            .header h1 {{ margin: 0 0 10px 0; color: var(--accent); }}
            .header p {{ margin: 0; color: #94a3b8; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 24px;
            }}
            .card {{
                background-color: var(--panel);
                border-radius: 12px; padding: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
                border: 1px solid #334155;
                display: flex; flex-direction: column;
            }}
            .full-width {{ grid-column: 1 / -1; }}
            .plotly-graph-div {{ height: 100%; width: 100%; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Advanced Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Dashboard</h1>
            <p>Interactive exploratory metrics — Zoom, Hover, and Click Legends to isolate data</p>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                {div_reward}
            </div>
            
            <div class="card">
                {div_success}
            </div>
            <div class="card">
                {div_sunburst}
            </div>
            
            <div class="card full-width">
                {div_violins}
            </div>
            
            <div class="card">
                {div_collide_heat}
            </div>
            <div class="card">
                {div_steps_heat}
            </div>
            
            <div class="card full-width" style="max-height: 500px; justify-content: center; align-items: center;">
                {div_radar}
            </div>
        </div>
        
        <div class="header" style="margin-top: 30px; padding: 10px; font-size: 0.8em;">
            Generated dynamically by Modern Python Dashboard Tools
        </div>
    </body>
    </html>
    """
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(template)
        
    print(f"✅ Success! Open '{args.output.absolute()}' in your browser!")

if __name__ == "__main__":
    main()
