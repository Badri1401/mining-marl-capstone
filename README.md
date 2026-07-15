# Multi-Robot Mining Navigation with MADDPG

[![CI](https://github.com/Badri1401/mining-marl-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/Badri1401/mining-marl-capstone/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-22314e.svg)](https://docs.ros.org/en/humble/)

A **multi-agent reinforcement learning** system that trains a team of 3 autonomous
robots to navigate a GPS-denied underground mining arena and reach their goals
without colliding. Training uses **MADDPG** (Multi-Agent Deep Deterministic Policy
Gradient) with centralized critics, decentralized execution, and an 8-phase
curriculum. A matching **ROS 2 / Gazebo** simulation is included for sim-to-real work.

> Capstone project — *"Digital Adaptive Twin for Autonomous Mining Robots."*

---

## Table of Contents

1. [Highlights](#highlights)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)
4. [The Environment](#the-environment)
5. [The Algorithm](#the-algorithm)
6. [Results](#results)
7. [Gazebo / ROS 2 Simulation](#gazebo--ros-2-simulation)
8. [Documentation](#documentation)
9. [License](#license)

---

## Highlights

- **3 differential-drive robots** trained jointly with MADDPG (centralized critic, decentralized actors).
- **PyBullet physics** environment with 16-ray LiDAR (0.12–10 m), 44 obstacles, and a 50 m × 50 m arena.
- **8-phase curriculum** (`Easy1 → … → VHard`) that automatically ramps difficulty as the team improves.
- **Parallel rollout collection** for fast training (see [`docs/PARALLEL_ROLLOUT.md`](docs/PARALLEL_ROLLOUT.md)).
- **Reproducible final run** shipped in [`results/`](results/): training log, plots, interactive dashboard, and the final policy checkpoint.
- **ROS 2 (Humble) + Gazebo** robot description and arena world for high-fidelity simulation.

---

## Repository Layout

```
mining-marl-capstone/
├── src/                          # The training code (final MADDPG v3)
│   ├── environment_maddpg_v3.py     # PyBullet 3-robot mining environment
│   ├── environment_maddpg_terrain.py# Heightmap-terrain environment variant
│   ├── parallel_env_manager.py      # Parallel episode-collection workers
│   ├── train_maddpg.py              # MADDPG trainer (entry point)
│   └── analyze_results.py           # Result plotting / statistics helper
│
├── analysis/                     # Standalone plotting scripts for a run's log
│   ├── plot_hard_logs_updated.py    # Polished per-metric figures
│   ├── plot_hard_logs_graphs.py     # Per-metric figures (base)
│   ├── plot_interactive_dashboard.py# Plotly interactive HTML dashboard
│   └── plot_training_log_hard_all.py# Every column, quick overview
│
├── simulation/                   # ROS 2 / Gazebo packages
│   ├── mining_robot_description/    # URDF, meshes, sensors, launch files
│   └── mining_arena_world/          # 50 m arena world + 5 terrain heightmaps
│
├── results/                      # The final training run (reproducible)
│   ├── checkpoints/                 # final_episode_210000.pt (trained policy)
│   ├── plots/                       # Final figures used in the report
│   ├── training_log.csv             # Full per-episode training log
│   └── interactive_dashboard.html   # Open in a browser to explore the run
│
├── docs/                         # Design & technical documentation
├── scripts/                      # Convenience run scripts
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train

```bash
./scripts/train.sh            # final config (v2_baseline + v3_smooth)
# or run the module directly:
cd src && python train_maddpg.py
```

Training writes a timestamped run to `src/results/maddpg_v3_<timestamp>_final/`
(checkpoints, per-episode `training_log.csv`, and plots). `train_maddpg.py`
already defaults to the final reward/curriculum configuration; see its `--help`
for all flags.

### 3. Reproduce the result plots

The scripts default to the shipped final run in `results/training_log.csv`:

```bash
python analysis/plot_hard_logs_updated.py       # polished per-metric figures
python analysis/plot_interactive_dashboard.py   # interactive HTML dashboard
python analysis/plot_training_log_hard_all.py    # all metrics at a glance
```

Point any of them at a different run with `--csv path/to/training_log.csv`.

---

## The Environment

`MiningEnvironmentMADDPG_V3` (PyBullet, Gymnasium API) — an underground mine arena
where each robot must drive from its start to its goal.

| Property | Value |
|---|---|
| Robots | 3 (differential drive) |
| Arena | 50 m × 50 m |
| Obstacles | 28 pillars + 10 ore piles + 6 equipment blocks |
| LiDAR | 16 rays, 0.12–10 m range, Gaussian noise (σ = 0.01) |
| Observation | `5 + 16 + 3 + 4·(N−1)` = **32-D** per robot (proprioception + LiDAR + goal + neighbours) |
| Action | **2-D continuous** `[linear, angular]` velocity, each in `[−1, 1]` |
| Execution | Decentralized — each robot acts on its own local observation |

**Reward** (`v2_baseline`, goal-first hierarchy, clipped to `[−20, +40]`):

| Event | Reward |
|---|---|
| Individual goal reached | **+20** |
| Full-team completion | **+12** |
| Partial-team progress | **+2** |
| Collision | **−15** |

Dense distance shaping is capped so it can never dominate goal completion.

---

## The Algorithm

**MADDPG** — each robot has its own actor; a **centralized critic** sees the joint
state and actions during training, while execution stays fully decentralized.

| Hyperparameter | Value |
|---|---|
| Actor learning rate | 1e-4 |
| Critic learning rate | 3e-4 |
| Discount γ | 0.99 |
| Soft-update τ | 0.005 |
| Hidden dimension | 384 |
| Batch size | 128 |
| Replay | Phase-aware buffer with success/catastrophe retention |
| Curriculum | `v3_smooth` — 8 phases, unified 0.75 m goal threshold, smoothed ramp |

The **curriculum** advances a phase only once the team sustains a success threshold,
progressing `Easy1 → Easy2 → Easy3 → Med1 → Med2 → Hard1 → Hard2 → VHard`.

---

## Results

The shipped final run trained through the full curriculum to the **`Hard`/`VHard`
phases (~218k episodes)**. All figures below live in [`results/plots/`](results/plots/):

| Figure | What it shows |
|---|---|
| `03_Reward_Convergence.png` | Reward convergence over training |
| `04a_Main_Success_Rates.png` | Success rate (individual / team) |
| `04b_Success_Composition_Bar.png` | Full vs. partial vs. failed episodes |
| `01_Performance_By_Phase_Bar.png` | Performance across curriculum phases |
| `05_Robot_Fairness_Comparison.png` | Per-robot reward fairness |
| `06_Collision_Severity_Stacked.png` | Collision counts by severity |
| `07_Critic_Loss.png` | Critic-loss curve |

For an interactive view, open [`results/interactive_dashboard.html`](results/interactive_dashboard.html)
in a browser. The full per-episode data is in `results/training_log.csv`, and the
final trained policy is `results/checkpoints/final_episode_210000.pt`.

---

## Gazebo / ROS 2 Simulation

A matching **ROS 2 Humble + Gazebo** stack lives in [`simulation/`](simulation/):

```bash
# In a colcon workspace, symlink the two packages into src/ and build:
ln -s "$(pwd)/simulation/mining_robot_description" ~/ros2_ws/src/
ln -s "$(pwd)/simulation/mining_arena_world"       ~/ros2_ws/src/
cd ~/ros2_ws && colcon build

# Terminal 1 — launch the arena world:
ros2 launch mining_arena_world gazebo_world.launch.py
# Terminal 2 — spawn the 3-robot team:
ros2 launch mining_robot_description spawn_multi_robot.launch.py
```

The robot (≈200 mm chassis, differential drive, RPLidar A2 + RealSense D435i) and
5 terrain heightmaps (`gentle_rolling → complex_mixed`) support domain
randomization for sim-to-real transfer. Meshes and heightmaps are generated by the
pure-Python scripts under each package's `scripts/` folder.

---

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — robot, arena, environment, and training design reference.
- [`docs/PARALLEL_ROLLOUT.md`](docs/PARALLEL_ROLLOUT.md) — how parallel episode collection works.

---

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Bug reports and feature requests can be
filed via the issue templates.

## Citation

If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff)):

> Praharaj, B. and Sreeram, M. V. *Multi-Robot Mining Navigation with MADDPG.* 2026.
> https://github.com/Badri1401/mining-marl-capstone

## License

Released under the [MIT License](LICENSE).
