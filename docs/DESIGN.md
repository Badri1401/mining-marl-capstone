# Design Reference

Technical reference for the robot, arena, environment, and training setup.
All values below reflect the code in [`../src`](../src) and the ROS 2 packages in
[`../simulation`](../simulation).

---

## 1. Robot

| Property | Value |
|---|---|
| Chassis | ~200 mm diameter × 52 mm (dual-layer) |
| Drive | Differential (2 independent wheels) |
| Wheels | 65 mm diameter × 26 mm, 160 mm separation |
| Caster | 15 mm ball caster |
| Mass | ~650 g |
| LiDAR | RPLidar A2 — 360°, 0.12–10 m |
| Camera | Intel RealSense D435i (depth) |
| Max velocity | 1.5 m/s · Max wheel torque 5 N·m · Wheel friction μ = 0.8 |

URDF: `simulation/mining_robot_description/urdf/mining_robot.urdf.xacro`
Meshes and physics config are generated / stored under the same package.

---

## 2. Arena

| Property | Value |
|---|---|
| Size | 50 m × 50 m |
| Obstacles (RL env) | 28 pillars (h = 3 m), 10 ore piles, 6 equipment blocks |
| Terrain (Gazebo) | 5 Perlin heightmaps, elevation ±0.15 m, 512×512 px |

Terrain variants (easy → hard): `gentle_rolling`, `moderate_hills`,
`deep_tunnels`, `rocky_ridges`, `complex_mixed`. Used for domain randomization.

World: `simulation/mining_arena_world/worlds/mining_arena.sdf`

---

## 3. RL Environment (`MiningEnvironmentMADDPG_V3`)

- **Backend:** PyBullet, Gymnasium API. Defined in `src/environment_maddpg_v3.py`.
  A heightmap-terrain variant is in `src/environment_maddpg_terrain.py`.
- **Robots:** 3 (configurable via `num_robots`).
- **Observation (per robot, 32-D for 3 robots):**
  `5` proprioceptive + `16` LiDAR rays + `3` goal features + `4·(N−1)` neighbour features.
  Normalised to `[−1, 1]`.
- **Action (per robot):** 2-D continuous `[linear, angular]` velocity in `[−1, 1]`.
- **LiDAR:** 16 rays, range 0.12–10 m, Gaussian noise σ = 0.01.

### Reward (`v2_baseline`)

Goal-first hierarchy, clipped to `[−20, +40]`:

| Event | Reward |
|---|---|
| Individual goal reached | +20 |
| Full-team completion | +12 |
| Partial-team progress | +2 |
| Collision | −15 |

Dense distance shaping is capped/sparse so it cannot dominate goal completion.

---

## 4. Training (MADDPG)

Defined in `src/train_maddpg.py`.

| Component | Setting |
|---|---|
| Actors | One per robot (decentralized execution) |
| Critic | Centralized (sees joint state + actions) |
| Actor LR / Critic LR | 1e-4 / 3e-4 |
| Discount γ | 0.99 |
| Soft-update τ | 0.005 |
| Hidden dim | 384 |
| Batch size | 128 |
| Replay buffer | Phase-aware, with success + catastrophe retention |

### Curriculum (`v3_smooth`)

8 phases with a unified 0.75 m goal threshold and a smoothed difficulty ramp.
A phase advances only after the team sustains its success threshold:

```
Easy1 → Easy2 → Easy3 → Med1 → Med2 → Hard1 → Hard2 → VHard
```

Each phase increases start–goal distance and step budget. See the curriculum
tables at the top of `src/train_maddpg.py` and `src/environment_maddpg_v3.py`.

---

## 5. Outputs

Each run writes to `src/results/maddpg_v3_<timestamp>/`:

- `checkpoints/episode_*.pt` — periodic policy snapshots
- `training_log.csv` — one row per episode (reward, success, losses, collisions, phase, …)
- `graphs/` — auto-generated training figures

The finalized run used for the report is preserved in [`../results`](../results).
