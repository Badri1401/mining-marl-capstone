"""
MINING ENVIRONMENT V3 - REWARD+SMOOTH
File: environment_maddpg_v3.py

Inherits V2 arena, physics, and observations exactly.

REWARD MODE: v2_baseline
    Goal-first reward hierarchy.
    Individual goal completion is the largest per-robot event.
    Full-team completion is the largest shared event.
    Dense shaping is capped/sparse so it cannot dominate completion.
    Goal +20, collision -15, partial +2, full +12.

CURRICULUM MODE: v3_smooth
  8 phases, unified 0.75m threshold, smoothed difficulty ramp.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List


# ==================== EXACT SDF OBSTACLE DATA ====================

# 28 Pillars: (x, y, radius) - all height 3.0m (reduced 30% from 40)
PILLAR_DATA = [
    (-10, -18, 0.6), (-2, -18, 0.55), (6, -18, 0.7),
    (-10, -10, 0.55), (-2, -10, 0.5), (6, -10, 0.6),
    (-18, -2, 0.5), (-10, -2, 0.7), (6, -2, 0.6), (14, -2, 0.55),
    (-6, -6, 0.5), (2, -6, 0.6), (10, -6, 0.7),
    (-10, 6, 0.5), (-2, 6, 0.6), (6, 6, 0.55),
    (-10, 14, 0.6), (-2, 14, 0.55), (6, 14, 0.5),
    (-6, 2, 0.55), (2, 2, 0.5), (10, 2, 0.6),
    (-6, 10, 0.6), (2, 10, 0.55), (10, 10, 0.5),
    (-6, 18, 0.5), (2, 18, 0.6), (10, 18, 0.55),
]

# 10 Ore Piles: (x, y, size_x, size_y, size_z, yaw)
ORE_PILE_DATA = [
    (-8, -16, 1.0, 1.0, 0.8, -0.2),
    (-16, -8, 0.9, 0.7, 0.7, 0.5),
    (-16, 0, 1.0, 0.8, 0.8, 0.2),
    (-4, -4, 1.3, 0.9, 1.0, -0.1),
    (16, 0, 1.0, 0.8, 0.8, 0.6),
    (-16, 8, 0.9, 0.7, 0.7, -0.3),
    (8, 4, 1.2, 1.0, 0.9, 0.4),
    (-8, 16, 1.3, 0.8, 1.0, -0.2),
    (0, 16, 1.1, 0.9, 0.8, 0.5),
    (8, 16, 0.9, 0.8, 0.7, -0.1),
]

# 6 Equipment pieces: (x, y, size_x, size_y, size_z, yaw)
EQUIPMENT_DATA = [
    (-20, -20, 3.5, 1.8, 2.0, 0.8),
    (20, -20, 3.0, 2.0, 2.0, -0.6),
    (-20, 0, 2.5, 1.5, 1.6, 0.3),
    (20, 0, 3.2, 1.9, 2.0, -0.2),
    (-20, 20, 2.8, 1.7, 1.8, 0.5),
    (20, 20, 3.0, 2.0, 2.0, -0.7),
]


# ==================== CURRICULUM DEFINITIONS ====================

CURRICULUM_V2_ORIGINAL = {
    "Easy1": {"range": [1.5, 2.0], "threshold": 1.25, "steps": 300},
    "Easy2": {"range": [2.0, 2.5], "threshold": 1.0,  "steps": 350},
    "Easy3": {"range": [2.5, 3.0], "threshold": 1.0,  "steps": 400},
    "Med1":  {"range": [3.0, 4.0], "threshold": 0.75, "steps": 350},
    "Med2a": {"range": [4.0, 5.0], "threshold": 0.75, "steps": 400},
    "Med2b": {"range": [5.0, 6.0], "threshold": 0.75, "steps": 450},
    "Hard1": {"range": [4.5, 5.5], "threshold": 1.0,  "steps": 800},
    "Hard2": {"range": [5.5, 6.5], "threshold": 0.75, "steps": 1000},
    "VHard": {"range": [6.0, 8.0], "threshold": 0.75, "steps": 1200},
}

CURRICULUM_V3_SMOOTH = {
    # Unified 0.75m threshold everywhere — monotonically harder
    # Each phase increases travel by ~0.75-1.0m
    "Easy1": {"range": [1.0, 1.5],  "threshold": 0.75, "steps": 300},   # travel 0.25-0.75m
    "Easy2": {"range": [1.5, 2.0],  "threshold": 0.75, "steps": 350},   # travel 0.75-1.25m
    "Easy3": {"range": [2.0, 2.75], "threshold": 0.75, "steps": 400},   # travel 1.25-2.0m
    "Med1":  {"range": [2.75, 3.5], "threshold": 0.75, "steps": 500},   # travel 2.0-2.75m
    "Med2":  {"range": [3.5, 4.5],  "threshold": 0.75, "steps": 600},   # travel 2.75-3.75m
    "Hard1": {"range": [4.5, 5.5],  "threshold": 0.75, "steps": 800},   # travel 3.75-4.75m
    "Hard2": {"range": [5.5, 7.0],  "threshold": 0.75, "steps": 1000},  # travel 4.75-6.25m
    "VHard": {"range": [7.0, 9.0],  "threshold": 0.75, "steps": 1200},  # travel 6.25-8.25m
}


REWARD_CONFIG = {
    'progress_scale': 7.5,
    'team_progress_scale': 0.10,
    'time_penalty': 0.001,
    'near_goal_band': 0.50,
    'near_goal_entry_bonus': 2.0,
    'milestone_50_bonus': 1.0,
    'milestone_75_bonus': 1.0,
    'milestone_90_bonus': 2.0,
    'goal_bonus': 20.0,
    'partial_team_bonus': 2.0,
    'full_team_bonus': 12.0,
    'collision_penalty': 15.0,
    'teammate_spacing_penalty': 0.5,
    'reward_clip_min': -20.0,
    'reward_clip_max': 40.0,
}


class MiningEnvironmentMADDPG_V3(gym.Env):
    """
    Multi-Robot Navigation Environment V3 — Reward+Smooth

    Identical physics/arena/observations to V2.
    Uses v2_baseline reward + v3_smooth curriculum.

    Args:
        reward_mode:     'v2_baseline' (only supported mode)
        curriculum_mode: 'v2_original' or 'v3_smooth'
    """

    def __init__(self, num_robots: int = 3, arena_size: float = 50.0,
                 max_episode_steps: int = 500, visualize: bool = False,
                 episode_count: int = 0,
                 enable_waiting_reward: bool = False,
                 enable_frozen_bonus: bool = False,
                 reward_mode: str = 'v2_baseline',
                 curriculum_mode: str = 'v3_smooth'):

        self.num_robots = num_robots
        self.arena_size = arena_size
        self.max_episode_steps = max_episode_steps
        self.visualize = visualize
        self.timestep = 0
        self.episode_count = episode_count

        # ── MODE SWITCHES ──────────────────────────────────────────────────
        assert reward_mode == 'v2_baseline', \
            f"reward_mode must be 'v2_baseline', got '{reward_mode}'"
        assert curriculum_mode in ('v2_original', 'v3_smooth'), \
            f"curriculum_mode must be 'v2_original' or 'v3_smooth', got '{curriculum_mode}'"
        self.reward_mode = reward_mode
        self.curriculum_mode = curriculum_mode

        # Select curriculum table
        if curriculum_mode == 'v2_original':
            self._curriculum = CURRICULUM_V2_ORIGINAL
        else:
            self._curriculum = CURRICULUM_V3_SMOOTH

        # Ablation toggles (carried from V2)
        self.enable_waiting_reward = bool(enable_waiting_reward)
        self.enable_frozen_bonus = bool(enable_frozen_bonus)
        self.last_reward_terms = {}

        # ==================== ROBOT SPECS (from URDF/YAML) ====================
        self.chassis_radius = 0.100
        self.chassis_height = 0.052
        self.chassis_mass = 0.50
        self.wheel_radius = 0.0325
        self.wheel_width = 0.026
        self.wheel_mass = 0.05
        self.wheel_separation = 0.160
        self.robot_mass = 0.65
        self.robot_radius = self.chassis_radius

        # Motion limits
        self.v_max = 1.5
        self.omega_max = np.pi
        self.max_linear_accel = 2.0
        self.max_angular_accel = 6.0
        self.linear_damping = 0.3
        self.angular_damping = 0.3
        self.wheel_friction = 0.8

        # ==================== LIDAR SPECS ====================
        self.num_lidar_rays = 16
        self.lidar_range_min = 0.12
        self.lidar_range_max = 10.0
        self.lidar_noise_stddev = 0.01
        self.lidar_obs_rays = 16

        # ==================== GOAL PARAMETERS ====================
        self.goal_distance_range = [3.0, 8.0]  # overwritten by set_phase
        self.goal_threshold = 0.75

        # ==================== PyBullet ====================
        if self.visualize:
            self.client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=35, cameraYaw=45, cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.client_id)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.01, numSubSteps=4, physicsClientId=self.client_id)

        # State storage
        self.robot_states = {}
        self.robot_bodies = {}
        self.prev_distances = {}
        self.obstacles = []
        self.obstacle_data = []

        # Setup
        self._setup_arena()
        self._setup_robots()
        self._setup_spaces()

        # Initialize curriculum phase
        first_phase = list(self._curriculum.keys())[0]
        self.current_phase = first_phase
        self.set_phase(first_phase)

        print(f"[ENV-V3] Mining Environment V3 initialized:")
        print(f"         reward_mode={self.reward_mode}, curriculum_mode={self.curriculum_mode}")
        print(f"         Arena: {self.arena_size}m, Obstacles: {len(self.obstacles)}")
        print(f"         Robots: {self.num_robots}, Obs dim: {self.obs_dim}D")
        print(f"         Phases: {list(self._curriculum.keys())}")

    # ================================================================
    #  SPACES
    # ================================================================
    def _setup_spaces(self):
        self.obs_dim = 5 + 16 + 3 + 4 * (self.num_robots - 1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0], dtype=np.float32), dtype=np.float32)

    # ================================================================
    #  ARENA (identical to V2)
    # ================================================================
    def _setup_arena(self):
        p.loadURDF("plane.urdf", [0, 0, -0.1], physicsClientId=self.client_id)

        wall_height = 2.0
        wall_thickness = 0.5
        half_arena = self.arena_size / 2

        ns_wall_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_arena, wall_thickness/2, wall_height/2],
            physicsClientId=self.client_id)
        ew_wall_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[wall_thickness/2, half_arena, wall_height/2],
            physicsClientId=self.client_id)

        for shape, pos in [
            (ns_wall_shape, [0, half_arena, wall_height/2]),
            (ns_wall_shape, [0, -half_arena, wall_height/2]),
            (ew_wall_shape, [half_arena, 0, wall_height/2]),
            (ew_wall_shape, [-half_arena, 0, wall_height/2]),
        ]:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                             basePosition=pos, physicsClientId=self.client_id)

        pillar_height = 3.0
        for x, y, radius in PILLAR_DATA:
            cs = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=pillar_height,
                                        physicsClientId=self.client_id)
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=pillar_height,
                                     rgbaColor=[0.5, 0.45, 0.4, 1.0], physicsClientId=self.client_id)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                                     basePosition=[x, y, pillar_height/2], physicsClientId=self.client_id)
            self.obstacles.append(body)
            self.obstacle_data.append((x, y, radius, 'cylinder'))

        for x, y, sx, sy, sz, yaw in ORE_PILE_DATA:
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2],
                                        physicsClientId=self.client_id)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2],
                                     rgbaColor=[0.4, 0.3, 0.2, 1.0], physicsClientId=self.client_id)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                                     basePosition=[x, y, sz/2],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                                     physicsClientId=self.client_id)
            self.obstacles.append(body)
            self.obstacle_data.append((x, y, max(sx, sy)/2, 'box'))

        for x, y, sx, sy, sz, yaw in EQUIPMENT_DATA:
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2],
                                        physicsClientId=self.client_id)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2],
                                     rgbaColor=[0.7, 0.6, 0.2, 1.0], physicsClientId=self.client_id)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                                     basePosition=[x, y, sz/2],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                                     physicsClientId=self.client_id)
            self.obstacles.append(body)
            self.obstacle_data.append((x, y, max(sx, sy)/2, 'box'))

    # ================================================================
    #  ROBOTS (identical to V2)
    # ================================================================
    def _setup_robots(self):
        colors = [
            [0.2, 0.8, 0.2, 1.0],
            [0.2, 0.2, 0.8, 1.0],
            [0.8, 0.2, 0.2, 1.0],
        ]
        for robot_id in range(self.num_robots):
            agent_name = f"robot_{robot_id}"
            angle = (2 * np.pi * robot_id) / self.num_robots
            spawn_x = 8.0 * np.cos(angle)
            spawn_y = 8.0 * np.sin(angle)
            spawn_theta = angle

            cs = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.chassis_radius,
                                        height=self.chassis_height, physicsClientId=self.client_id)
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=self.chassis_radius,
                                     length=self.chassis_height,
                                     rgbaColor=colors[robot_id % len(colors)],
                                     physicsClientId=self.client_id)
            robot_z = self.wheel_radius + self.chassis_height / 2
            robot_body = p.createMultiBody(
                baseMass=self.robot_mass, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                basePosition=[spawn_x, spawn_y, robot_z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, spawn_theta]),
                physicsClientId=self.client_id)

            p.changeDynamics(robot_body, -1,
                             linearDamping=self.linear_damping, angularDamping=self.angular_damping,
                             lateralFriction=self.wheel_friction, physicsClientId=self.client_id)

            goal_x, goal_y = self._generate_valid_goal(spawn_x, spawn_y)
            self.robot_bodies[agent_name] = robot_body
            self.robot_states[agent_name] = {
                'x': spawn_x, 'y': spawn_y, 'theta': spawn_theta,
                'vx': 0, 'vy': 0, 'v': 0, 'omega': 0,
                'goal_x': goal_x, 'goal_y': goal_y,
                'reached': False, 'collided': False,
                'frozen': False, 'just_froze': False,
                'collision_count': 0, 'distance_to_goal': 0.0,
                'got_near_goal_bonus': False,
            }
            self.prev_distances[agent_name] = self._compute_distance_to_goal(agent_name)
            self.robot_states[agent_name]['distance_to_goal'] = self.prev_distances[agent_name]

    # ================================================================
    #  GOAL GENERATION
    # ================================================================
    def _generate_valid_goal(self, spawn_x: float, spawn_y: float) -> Tuple[float, float]:
        min_dist, max_dist = self.goal_distance_range
        margin = self.arena_size / 2 - 2.0
        for _ in range(100):
            dist = np.random.uniform(min_dist, max_dist)
            angle = np.random.uniform(-np.pi, np.pi)
            goal_x = spawn_x + dist * np.cos(angle)
            goal_y = spawn_y + dist * np.sin(angle)
            goal_x = np.clip(goal_x, -margin, margin)
            goal_y = np.clip(goal_y, -margin, margin)
            valid = True
            for ox, oy, r, _ in self.obstacle_data:
                if np.sqrt((goal_x - ox)**2 + (goal_y - oy)**2) < r + 1.0:
                    valid = False
                    break
            if valid:
                return goal_x, goal_y
        return goal_x, goal_y

    # ================================================================
    #  CURRICULUM PHASE SETTING
    # ================================================================
    def set_phase(self, phase_name: str):
        """Set curriculum phase from the active curriculum table."""
        if phase_name not in self._curriculum:
            print(f"[ENV-V3] Warning: phase '{phase_name}' not in {self.curriculum_mode} curriculum")
            return
        cfg = self._curriculum[phase_name]
        self.current_phase = phase_name
        self.goal_distance_range = cfg["range"]
        self.goal_threshold = cfg["threshold"]
        self.max_episode_steps = cfg["steps"]
        print(f"[ENV-V3] Phase: {phase_name}, range: {self.goal_distance_range}, "
              f"threshold: {self.goal_threshold:.2f}m, steps: {self.max_episode_steps}")

    # ================================================================
    #  RESET
    # ================================================================
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.timestep = 0
        self.episode_count += 1
        robot_z = self.wheel_radius + self.chassis_height / 2
        base_angle = np.random.uniform(0, 2 * np.pi)

        for agent_name in self.robot_states.keys():
            robot_body = self.robot_bodies[agent_name]
            robot_id = int(agent_name.split('_')[1])
            angle = base_angle + (2 * np.pi * robot_id) / self.num_robots + np.random.uniform(-0.3, 0.3)
            spawn_x = 8.0 * np.cos(angle)
            spawn_y = 8.0 * np.sin(angle)
            spawn_theta = angle

            p.resetBasePositionAndOrientation(
                robot_body, [spawn_x, spawn_y, robot_z],
                p.getQuaternionFromEuler([0, 0, spawn_theta]),
                physicsClientId=self.client_id)
            p.resetBaseVelocity(robot_body, [0, 0, 0], [0, 0, 0],
                               physicsClientId=self.client_id)

            goal_x, goal_y = self._generate_valid_goal(spawn_x, spawn_y)
            self.robot_states[agent_name] = {
                'x': spawn_x, 'y': spawn_y, 'theta': spawn_theta,
                'vx': 0, 'vy': 0, 'v': 0, 'omega': 0,
                'goal_x': goal_x, 'goal_y': goal_y,
                'reached': False, 'collided': False,
                'frozen': False, 'just_froze': False,
                'collision_count': 0, 'distance_to_goal': 0.0,
                'got_near_goal_bonus': False,
                'got_goal_bonus': False,
                'got_team_bonus': False,
                'got_partial_team_bonus': False,
            }
            self._prev_frozen_count = 0
            self.prev_distances[agent_name] = self._compute_distance_to_goal(agent_name)
            self.robot_states[agent_name]['distance_to_goal'] = self.prev_distances[agent_name]
            self.robot_states[agent_name]['initial_distance'] = self.prev_distances[agent_name]
            self.robot_states[agent_name]['milestone_50'] = False
            self.robot_states[agent_name]['milestone_25'] = False
            self.robot_states[agent_name]['milestone_10'] = False

        return self._get_observations(), {}

    # ================================================================
    #  STEP
    # ================================================================
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        for state in self.robot_states.values():
            state['just_froze'] = False

        for agent_name, action in actions.items():
            state = self.robot_states[agent_name]
            robot_body = self.robot_bodies[agent_name]
            if state['frozen']:
                p.resetBaseVelocity(robot_body, [0, 0, 0], [0, 0, 0],
                                   physicsClientId=self.client_id)
                continue
            v, omega = action
            linear_vel = v * self.v_max
            angular_vel = omega * self.omega_max
            pos, orn = p.getBasePositionAndOrientation(robot_body, physicsClientId=self.client_id)
            theta = p.getEulerFromQuaternion(orn)[2]
            vx = linear_vel * np.cos(theta)
            vy = linear_vel * np.sin(theta)
            p.resetBaseVelocity(robot_body, linearVelocity=[vx, vy, 0],
                               angularVelocity=[0, 0, angular_vel],
                               physicsClientId=self.client_id)

        p.stepSimulation(physicsClientId=self.client_id)
        self.timestep += 1

        self._update_robot_states()
        collisions = self._check_collisions()

        goals_reached = {}
        for agent_name in self.robot_states.keys():
            state = self.robot_states[agent_name]
            if state['frozen']:
                goals_reached[agent_name] = True
                continue
            dist = self._compute_distance_to_goal(agent_name)
            state['distance_to_goal'] = dist
            reached = dist < self.goal_threshold
            goals_reached[agent_name] = reached
            state['reached'] = reached
            state['collided'] = collisions[agent_name]
            if collisions[agent_name]:
                state['collision_count'] += 1
            if reached and not state['frozen']:
                state['frozen'] = True
                state['just_froze'] = True
                robot_body = self.robot_bodies[agent_name]
                p.resetBaseVelocity(robot_body, [0, 0, 0], [0, 0, 0],
                                   physicsClientId=self.client_id)

        rewards, reward_terms = self._compute_rewards(goals_reached, collisions)

        all_frozen = all(s['frozen'] for s in self.robot_states.values())
        terminations = {n: self.robot_states[n]['frozen'] for n in self.robot_states}
        truncations = {n: (self.timestep >= self.max_episode_steps) and not all_frozen
                       for n in self.robot_states}
        observations = self._get_observations()
        infos = {n: {'reached': goals_reached[n], 'all_frozen': all_frozen,
                 'reward_terms': reward_terms[n]}
                 for n in self.robot_states}
        return observations, rewards, terminations, truncations, infos

    # ================================================================
    #  PHYSICS HELPERS (identical to V2)
    # ================================================================
    def _update_robot_states(self):
        for agent_name, robot_body in self.robot_bodies.items():
            if self.robot_states[agent_name]['frozen']:
                continue
            pos, orn = p.getBasePositionAndOrientation(robot_body, physicsClientId=self.client_id)
            vel, ang_vel = p.getBaseVelocity(robot_body, physicsClientId=self.client_id)
            theta = p.getEulerFromQuaternion(orn)[2]
            self.robot_states[agent_name].update({
                'x': pos[0], 'y': pos[1], 'theta': theta,
                'vx': vel[0], 'vy': vel[1],
                'v': np.sqrt(vel[0]**2 + vel[1]**2),
                'omega': ang_vel[2]
            })

    def _compute_distance_to_goal(self, agent_name: str) -> float:
        state = self.robot_states[agent_name]
        return np.sqrt((state['x'] - state['goal_x'])**2 +
                       (state['y'] - state['goal_y'])**2)

    def _check_collisions(self) -> Dict[str, bool]:
        collisions = {n: False for n in self.robot_states}
        robot_positions = {}
        for n in self.robot_states:
            s = self.robot_states[n]
            robot_positions[n] = (s['x'], s['y'])

        for n in self.robot_states:
            if self.robot_states[n]['frozen']:
                continue
            rx, ry = robot_positions[n]
            for ox, oy, r, _ in self.obstacle_data:
                if (rx - ox)**2 + (ry - oy)**2 < (self.robot_radius + r + 0.05)**2:
                    collisions[n] = True
                    break

        agent_list = list(robot_positions.keys())
        for i in range(len(agent_list)):
            n1 = agent_list[i]
            if self.robot_states[n1]['frozen'] or collisions[n1]:
                continue
            x1, y1 = robot_positions[n1]
            for j in range(i + 1, len(agent_list)):
                n2 = agent_list[j]
                if self.robot_states[n2]['frozen']:
                    continue
                x2, y2 = robot_positions[n2]
                if (x1 - x2)**2 + (y1 - y2)**2 < (2 * self.robot_radius + 0.05)**2:
                    collisions[n1] = True
                    collisions[n2] = True
        return collisions

    # ================================================================
    #  LIDAR (identical to V2)
    # ================================================================
    def _get_lidar_readings(self, agent_name: str) -> np.ndarray:
        state = self.robot_states[agent_name]
        x, y, theta = state['x'], state['y'], state['theta']
        ray_angles = theta + np.linspace(0, 2*np.pi, 16, endpoint=False)
        cos_a = np.cos(ray_angles)
        sin_a = np.sin(ray_angles)
        min_distances = np.full(16, self.lidar_range_max)

        if not hasattr(self, '_obs_xyz'):
            self._obs_xyz = np.array([[o[0], o[1], o[2]] for o in self.obstacle_data])

        other_robots = []
        for other_name, other_state in self.robot_states.items():
            if other_name != agent_name and not other_state.get('frozen', False):
                other_robots.append([other_state['x'], other_state['y'], self.chassis_radius])

        if other_robots:
            all_obs = np.vstack([self._obs_xyz, np.array(other_robots)])
        else:
            all_obs = self._obs_xyz

        dist_sq = (x - all_obs[:, 0])**2 + (y - all_obs[:, 1])**2
        max_check_dist = (self.lidar_range_max + np.max(all_obs[:, 2]) + 1.0)**2
        close_mask = dist_sq < max_check_dist
        close_obs = all_obs[close_mask]

        if len(close_obs) == 0:
            min_distances += np.random.normal(0, self.lidar_noise_stddev, 16)
            return np.clip(min_distances, self.lidar_range_min, self.lidar_range_max) / self.lidar_range_max

        N = len(close_obs)
        ox = close_obs[:, 0]
        oy = close_obs[:, 1]
        r = close_obs[:, 2]
        dx = self.lidar_range_max * cos_a
        dy = self.lidar_range_max * sin_a
        fx = x - ox[:, None]
        fy = y - oy[:, None]
        dx_2d = dx[None, :]
        dy_2d = dy[None, :]
        a = dx_2d * dx_2d + dy_2d * dy_2d
        b = 2 * (fx * dx_2d + fy * dy_2d)
        c = fx * fx + fy * fy - (r[:, None])**2
        discriminant = b * b - 4 * a * c
        valid = discriminant >= 0
        t = np.full((N, 16), np.inf)
        t[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2 * a.flatten()[0])
        hit_dist = np.clip(t, 0, 1) * self.lidar_range_max
        valid_hit = (hit_dist >= self.lidar_range_min) & (t > 0) & (t < 1)
        hit_dist[~valid_hit] = self.lidar_range_max
        min_distances = np.minimum(min_distances, np.min(hit_dist, axis=0))
        min_distances += np.random.normal(0, self.lidar_noise_stddev, 16)
        min_distances = np.clip(min_distances, self.lidar_range_min, self.lidar_range_max)
        return min_distances / self.lidar_range_max

    # ================================================================
    #  OBSERVATIONS (identical to V2)
    # ================================================================
    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        agent_names = list(self.robot_states.keys())
        for agent_name in agent_names:
            state = self.robot_states[agent_name]
            goal_dx = state['goal_x'] - state['x']
            goal_dy = state['goal_y'] - state['y']
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
            goal_angle = np.arctan2(goal_dy, goal_dx) - state['theta']
            goal_obs = np.array([
                np.clip(goal_dx / self.arena_size, -1, 1),
                np.clip(goal_dy / self.arena_size, -1, 1),
                np.clip(goal_dist / self.arena_size, 0, 1),
                np.cos(goal_angle), np.sin(goal_angle),
            ])
            lidar_obs = self._get_lidar_readings(agent_name)
            vel_obs = np.array([
                np.clip(state['vx'] / self.v_max, -1, 1),
                np.clip(state['vy'] / self.v_max, -1, 1),
                np.clip(state['omega'] / self.omega_max, -1, 1),
            ])
            teammate_obs = []
            for other_name in agent_names:
                if other_name == agent_name:
                    continue
                other = self.robot_states[other_name]
                dx = other['x'] - state['x']
                dy = other['y'] - state['y']
                dist = np.sqrt(dx**2 + dy**2)
                teammate_obs.extend([
                    np.clip(dx / self.arena_size, -1, 1),
                    np.clip(dy / self.arena_size, -1, 1),
                    np.clip(dist / self.arena_size, 0, 1),
                    1.0 if other['reached'] else 0.0,
                ])
            teammate_obs = np.array(teammate_obs)
            obs[agent_name] = np.concatenate([
                goal_obs, lidar_obs, vel_obs, teammate_obs,
            ]).astype(np.float32)
        return obs

    # ================================================================
    #  REWARD FUNCTION — V2_BASELINE
    # ================================================================
    def _compute_rewards(
        self,
        goals_reached: Dict[str, bool],
        collisions: Dict[str, bool],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Compute per-agent rewards using a goal-first hierarchy."""
        rewards: Dict[str, float] = {}
        reward_terms: Dict[str, Dict[str, float]] = {}
        cfg = REWARD_CONFIG
        max_dist = self.goal_distance_range[1]

        teammates_at_goal = sum(1 for _, reached in goals_reached.items() if reached)
        frozen_teammates_count = sum(
            1 for _, state in self.robot_states.items() if state.get('frozen', False)
        )

        team_distances = []
        team_prev_distances = []
        for agent_name in self.robot_states:
            if not self.robot_states[agent_name].get('frozen', False):
                team_distances.append(self._compute_distance_to_goal(agent_name))
                team_prev_distances.append(self.prev_distances.get(agent_name, 0.0))

        team_progress = 0.0
        if team_distances and team_prev_distances:
            team_progress = (np.mean(team_prev_distances) - np.mean(team_distances)) / max_dist

        robot_positions = {
            agent_name: (state['x'], state['y'])
            for agent_name, state in self.robot_states.items()
        }

        for agent_name, state in self.robot_states.items():
            terms = {
                'progress_reward': 0.0,
                'team_progress_reward': 0.0,
                'time_penalty': 0.0,
                'near_goal_entry_bonus': 0.0,
                'milestone_bonus': 0.0,
                'goal_bonus': 0.0,
                'partial_team_bonus': 0.0,
                'full_team_bonus': 0.0,
                'waiting_reward': 0.0,
                'teammate_spacing_penalty': 0.0,
                'collision_penalty': 0.0,
            }

            if state['frozen'] and state.get('got_goal_bonus', False):
                other_frozen = frozen_teammates_count - 1
                waiting_reward = (0.05 + 0.02 * other_frozen) if self.enable_waiting_reward else 0.0
                terms['waiting_reward'] = waiting_reward
                rewards[agent_name] = waiting_reward
                reward_terms[agent_name] = terms
                continue

            reward_value = 0.0
            dist = self._compute_distance_to_goal(agent_name)
            prev = self.prev_distances[agent_name]

            delta = (prev - dist) / max_dist
            progress_reward = float(cfg['progress_scale'] * delta)
            reward_value += progress_reward
            terms['progress_reward'] += progress_reward

            if team_progress > 0 and not state['frozen']:
                team_progress_reward = float(cfg['team_progress_scale'] * team_progress)
                reward_value += team_progress_reward
                terms['team_progress_reward'] += team_progress_reward

            reward_value -= cfg['time_penalty']
            terms['time_penalty'] -= cfg['time_penalty']

            near_goal_threshold = self.goal_threshold + cfg['near_goal_band']
            if dist < near_goal_threshold and not state.get('got_near_goal_bonus', False):
                reward_value += cfg['near_goal_entry_bonus']
                terms['near_goal_entry_bonus'] += cfg['near_goal_entry_bonus']
                state['got_near_goal_bonus'] = True

            init_dist = state.get('initial_distance', dist)
            if init_dist > 0:
                progress_pct = 1.0 - (dist / init_dist)
                if progress_pct >= 0.50 and not state.get('milestone_50', False):
                    reward_value += cfg['milestone_50_bonus']
                    terms['milestone_bonus'] += cfg['milestone_50_bonus']
                    state['milestone_50'] = True
                if progress_pct >= 0.75 and not state.get('milestone_25', False):
                    reward_value += cfg['milestone_75_bonus']
                    terms['milestone_bonus'] += cfg['milestone_75_bonus']
                    state['milestone_25'] = True
                if progress_pct >= 0.90 and not state.get('milestone_10', False):
                    reward_value += cfg['milestone_90_bonus']
                    terms['milestone_bonus'] += cfg['milestone_90_bonus']
                    state['milestone_10'] = True

            if goals_reached[agent_name] and not state.get('got_goal_bonus', False):
                reward_value += cfg['goal_bonus']
                terms['goal_bonus'] += cfg['goal_bonus']
                state['got_goal_bonus'] = True

            rx, ry = robot_positions[agent_name]
            for other_name, (ox, oy) in robot_positions.items():
                if other_name == agent_name:
                    continue
                if self.robot_states[other_name].get('frozen', False):
                    continue
                teammate_dist = np.sqrt((rx - ox) ** 2 + (ry - oy) ** 2)
                if teammate_dist < 1.5:
                    spacing_penalty = cfg['teammate_spacing_penalty'] * (1.5 - teammate_dist) / 1.5
                    reward_value -= spacing_penalty
                    terms['teammate_spacing_penalty'] -= spacing_penalty

            if collisions[agent_name]:
                reward_value -= cfg['collision_penalty']
                terms['collision_penalty'] -= cfg['collision_penalty']

            rewards[agent_name] = float(
                np.clip(reward_value, cfg['reward_clip_min'], cfg['reward_clip_max'])
            )
            reward_terms[agent_name] = terms
            self.prev_distances[agent_name] = dist

        partial_bonus = cfg['partial_team_bonus']
        full_bonus = cfg['full_team_bonus']

        partial_target = max(1, self.num_robots - 1)
        if partial_target <= teammates_at_goal < self.num_robots:
            for agent_name in rewards:
                if not self.robot_states[agent_name].get('got_partial_team_bonus', False):
                    rewards[agent_name] += partial_bonus
                    reward_terms[agent_name]['partial_team_bonus'] += partial_bonus
                    self.robot_states[agent_name]['got_partial_team_bonus'] = True

        if all(goals_reached.values()):
            for agent_name in rewards:
                if not self.robot_states[agent_name].get('got_team_bonus', False):
                    rewards[agent_name] += full_bonus
                    reward_terms[agent_name]['full_team_bonus'] += full_bonus
                    self.robot_states[agent_name]['got_team_bonus'] = True

        self.last_reward_terms = reward_terms
        return rewards, reward_terms

    # ================================================================
    #  CLOSE
    # ================================================================
    def close(self):
        p.disconnect(self.client_id)


# ================================================================
#  QUICK SMOKE TEST
# ================================================================
def test_environment():
    print("\n" + "="*70)
    print("TESTING MADDPG MINING ENVIRONMENT V3 (REWARD+SMOOTH)")
    print("="*70)

    for cmode in ('v2_original', 'v3_smooth'):
        print(f"\n--- reward_mode=v2_baseline, curriculum_mode={cmode} ---")
        env = MiningEnvironmentMADDPG_V3(
            num_robots=2, visualize=False,
            reward_mode='v2_baseline', curriculum_mode=cmode)
        obs, _ = env.reset()
        print(f"  Obs shape: {obs['robot_0'].shape}")
        print(f"  Phases: {list(env._curriculum.keys())}")
        for step in range(10):
            actions = {f"robot_{i}": env.action_space.sample() for i in range(2)}
            obs, rewards, term, trunc, info = env.step(actions)
        print(f"  10-step sample rewards: { {k: round(v, 3) for k, v in rewards.items()} }")
        env.close()
    print("\n✓ V3 test complete!\n")


if __name__ == "__main__":
    test_environment()
