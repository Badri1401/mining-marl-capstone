"""
ENHANCED MINING ENVIRONMENT WITH HEIGHTMAP TERRAIN
File: environment_maddpg_terrain.py

This is an enhanced version of environment_maddpg.py that adds:
1. Heightmap terrain loading from PNG files
2. Random terrain variant selection per episode
3. Friction zones (puddles with low friction)
4. More realistic physics matching Gazebo configuration

For use with MADDPG training with domain randomization.

Terrain Specifications:
- 50×50m arena (expanded from 30m for heightmap compatibility)
- 5 terrain variants (randomly selected per reset)
- Elevation range: ±0.15m
- Friction zones: normal (μ=0.8), puddles (μ=0.1)
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional
import os
from pathlib import Path


class MiningEnvironmentTerrain(gym.Env):
    """
    Multi-Robot Navigation Environment with Heightmap Terrain
    
    Features:
    - 3 differential-drive robots
    - Heightmap terrain with random variant selection
    - Friction zones for domain randomization
    - 8D observation per robot
    - Continuous action space [v, omega]
    - Built-in curriculum learning
    """
    
    def __init__(
        self, 
        num_robots: int = 3, 
        arena_size: float = 50.0,  # Expanded for heightmap
        max_episode_steps: int = 400, 
        visualize: bool = False,
        episode_count: int = 0,
        heightmap_dir: Optional[str] = None,
        num_terrain_variants: int = 5,
        use_terrain: bool = True
    ):
        
        self.num_robots = num_robots
        self.arena_size = arena_size
        self.max_episode_steps = max_episode_steps
        self.visualize = visualize
        self.timestep = 0
        self.episode_count = episode_count
        self.use_terrain = use_terrain
        self.num_terrain_variants = num_terrain_variants
        
        # Terrain paths
        if heightmap_dir is None:
            # Default path relative to this file
            self.heightmap_dir = Path(__file__).parent.parent / "mining_arena_world" / "heightmaps"
        else:
            self.heightmap_dir = Path(heightmap_dir)
        
        # Robot dynamics (matching physics_params.yaml)
        self.robot_radius = 0.10  # 200mm diameter / 2
        self.robot_mass = 0.65  # kg
        self.wheel_radius = 0.0325  # 65mm / 2
        self.wheel_separation = 0.160  # meters
        self.v_max = 1.5  # m/s
        self.omega_max = np.pi  # rad/s
        self.linear_damping = 0.3
        self.angular_damping = 0.3
        self.wheel_friction = 0.8
        
        # Goal parameters (will be set by curriculum)
        self.goal_distance_range = [2.0, 5.0]
        self.goal_threshold = 0.75
        
        # Terrain state
        self.current_terrain_variant = 1
        self.terrain_body = None
        self.friction_map = None
        
        # Initialize PyBullet
        if self.visualize:
            self.client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=25, cameraYaw=45, cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0], physicsClientId=self.client_id
            )
        else:
            self.client_id = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.01, 
            numSubSteps=10,
            physicsClientId=self.client_id
        )
        
        # Initialize state storage
        self.robot_states = {}
        self.robot_bodies = {}
        self.prev_distances = {}
        self.obstacles = []
        self.pillars = []
        
        # Setup environment
        self._setup_arena()
        self._setup_robots()
        self._setup_spaces()
        
        # Initialize curriculum phase
        self.current_phase = "Easy1"
        self.set_phase("Easy1")
        
        print(f"[ENV] Terrain Environment initialized:")
        print(f"      - Robots: {self.num_robots}")
        print(f"      - Arena: {self.arena_size}m × {self.arena_size}m")
        print(f"      - Terrain: {'Enabled' if self.use_terrain else 'Flat'}")
        print(f"      - Terrain variants: {self.num_terrain_variants}")
        print(f"      - Obs dim: 8D")
        print(f"      - Action dim: 2D (continuous)")
    
    def _setup_spaces(self):
        """Define observation and action spaces"""
        
        # 8D observation: [goal_dx, goal_dy, dist_to_goal, cos(angle), sin(angle), vx, vy, omega]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # Continuous action: [v, omega]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def _load_heightmap(self, variant: int) -> Optional[np.ndarray]:
        """
        Load heightmap from file for terrain generation.
        
        Args:
            variant: Terrain variant number (1-5)
        
        Returns:
            np.ndarray: Heightmap data or None if not found
        """
        heightmap_path = self.heightmap_dir / f"terrain_raw_v{variant}.npy"
        
        if not heightmap_path.exists():
            # Try to generate if not exists
            print(f"[ENV] Heightmap not found: {heightmap_path}")
            print(f"[ENV] Run generate_heightmap.py first, or using flat terrain")
            return None
        
        try:
            heightmap = np.load(str(heightmap_path))
            return heightmap.astype(np.float32)
        except Exception as e:
            print(f"[ENV] Error loading heightmap: {e}")
            return None
    
    def _load_friction_map(self, variant: int) -> Optional[np.ndarray]:
        """Load friction coefficient map."""
        friction_path = self.heightmap_dir / f"friction_v{variant}.npy"
        
        if friction_path.exists():
            try:
                return np.load(str(friction_path))
            except Exception as e:
                print(f"[ENV] Error loading friction map: {e}")
        
        return None
    
    def _create_heightfield_terrain(self, variant: int):
        """
        Create terrain from heightmap using PyBullet heightfield.
        
        Args:
            variant: Terrain variant number
        """
        heightmap = self._load_heightmap(variant)
        
        if heightmap is None:
            # Fallback to flat plane
            self._create_flat_ground()
            return
        
        # Remove existing terrain if any
        if self.terrain_body is not None:
            p.removeBody(self.terrain_body, physicsClientId=self.client_id)
            self.terrain_body = None
        
        # PyBullet heightfield expects 1D array
        height_data = heightmap.flatten()
        num_rows = heightmap.shape[0]
        num_cols = heightmap.shape[1]
        
        # Scale factors
        mesh_scale = [
            self.arena_size / num_cols,  # X scale
            self.arena_size / num_rows,  # Y scale
            1.0  # Z scale (heightmap already in meters)
        ]
        
        # Create heightfield collision shape
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldData=height_data,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_cols,
            replaceHeightfieldIndex=-1,
            physicsClientId=self.client_id
        )
        
        # Create terrain body
        self.terrain_body = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client_id
        )
        
        # Set terrain friction
        p.changeDynamics(
            self.terrain_body, -1,
            lateralFriction=self.wheel_friction,
            physicsClientId=self.client_id
        )
        
        # Load friction map for puddle zones
        self.friction_map = self._load_friction_map(variant)
        
        self.current_terrain_variant = variant
        print(f"[ENV] Loaded terrain variant {variant}")
    
    def _create_flat_ground(self):
        """Create simple flat ground plane."""
        if self.terrain_body is not None:
            p.removeBody(self.terrain_body, physicsClientId=self.client_id)
        
        self.terrain_body = p.loadURDF(
            "plane.urdf",
            [0, 0, -0.1],
            physicsClientId=self.client_id
        )
        
        p.changeDynamics(
            self.terrain_body, -1,
            lateralFriction=self.wheel_friction,
            physicsClientId=self.client_id
        )
    
    def _setup_arena(self):
        """Create mining arena with obstacles and terrain."""
        
        # Initial terrain (will be randomized on reset)
        if self.use_terrain:
            initial_variant = np.random.randint(1, self.num_terrain_variants + 1)
            self._create_heightfield_terrain(initial_variant)
        else:
            self._create_flat_ground()
        
        # Perimeter walls
        wall_height = 2.0
        wall_thickness = 0.5
        
        wall_configs = [
            # (position, half_extents)
            ((-self.arena_size/2, 0, wall_height/2), [wall_thickness/2, self.arena_size/2, wall_height/2]),
            ((self.arena_size/2, 0, wall_height/2), [wall_thickness/2, self.arena_size/2, wall_height/2]),
            ((0, -self.arena_size/2, wall_height/2), [self.arena_size/2, wall_thickness/2, wall_height/2]),
            ((0, self.arena_size/2, wall_height/2), [self.arena_size/2, wall_thickness/2, wall_height/2]),
        ]
        
        for pos, extents in wall_configs:
            wall_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=extents,
                physicsClientId=self.client_id
            )
            
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=extents,
                rgbaColor=[0.4, 0.35, 0.3, 1.0],
                physicsClientId=self.client_id
            )
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos,
                physicsClientId=self.client_id
            )
        
        # Create support pillars (room-and-pillar mining pattern)
        self._create_pillars()
        
        # Create ore piles and equipment
        self._create_obstacles()
        
        print(f"[ENV] Arena: {self.arena_size}m × {self.arena_size}m")
    
    def _create_pillars(self):
        """Create support pillars in grid pattern."""
        np.random.seed(42)  # Reproducible layout
        
        pillar_positions = []
        grid_spacing = 8.0  # 8m between pillars
        
        # Create grid of pillars
        for x in np.arange(-self.arena_size/2 + 6, self.arena_size/2 - 6, grid_spacing):
            for y in np.arange(-self.arena_size/2 + 6, self.arena_size/2 - 6, grid_spacing):
                # Add some randomness to position
                px = x + np.random.uniform(-1.0, 1.0)
                py = y + np.random.uniform(-1.0, 1.0)
                pillar_positions.append((px, py))
        
        for px, py in pillar_positions:
            radius = np.random.uniform(0.5, 0.7)
            height = 3.0
            
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self.client_id
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[0.5, 0.45, 0.4, 1.0],
                physicsClientId=self.client_id
            )
            
            pillar = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[px, py, height/2],
                physicsClientId=self.client_id
            )
            
            self.pillars.append(pillar)
        
        print(f"[ENV] Created {len(self.pillars)} support pillars")
    
    def _create_obstacles(self):
        """Create ore piles and equipment obstacles."""
        np.random.seed(43)  # Different seed for obstacles
        
        # Ore piles (15 boxes)
        num_ore_piles = 15
        for i in range(num_ore_piles):
            x = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            y = np.random.uniform(-self.arena_size/2 + 3, self.arena_size/2 - 3)
            
            # Random box dimensions
            w = np.random.uniform(0.9, 1.4)
            d = np.random.uniform(0.7, 1.0)
            h = np.random.uniform(0.7, 1.0)
            
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[w/2, d/2, h/2],
                physicsClientId=self.client_id
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[w/2, d/2, h/2],
                rgbaColor=[0.4, 0.3, 0.2, 1.0],
                physicsClientId=self.client_id
            )
            
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, h/2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(0, np.pi)]),
                physicsClientId=self.client_id
            )
            
            self.obstacles.append(obstacle)
        
        # Large equipment (8 boxes)
        num_equipment = 8
        equipment_positions = [
            (-20, -20), (20, -20), (-20, 0), (20, 0),
            (-20, 20), (20, 20), (0, -22), (0, 22)
        ]
        
        for px, py in equipment_positions[:num_equipment]:
            # Large box representing parked equipment
            w = np.random.uniform(2.5, 3.5)
            d = np.random.uniform(1.5, 2.0)
            h = np.random.uniform(1.6, 2.0)
            
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[w/2, d/2, h/2],
                physicsClientId=self.client_id
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[w/2, d/2, h/2],
                rgbaColor=[0.7, 0.6, 0.2, 1.0],
                physicsClientId=self.client_id
            )
            
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[px, py, h/2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.random.uniform(-0.8, 0.8)]),
                physicsClientId=self.client_id
            )
            
            self.obstacles.append(obstacle)
        
        print(f"[ENV] Created {len(self.obstacles)} obstacles (ore piles + equipment)")
    
    def _setup_robots(self):
        """Create 3 differential-drive robots with realistic parameters."""
        
        for robot_id in range(self.num_robots):
            agent_name = f"robot_{robot_id}"
            
            # Spawn position: circular arrangement
            angle = (2 * np.pi * robot_id) / self.num_robots
            spawn_radius = self.arena_size / 5
            spawn_x = spawn_radius * np.cos(angle)
            spawn_y = spawn_radius * np.sin(angle)
            spawn_theta = angle
            
            # Get terrain height at spawn position
            spawn_z = self._get_terrain_height(spawn_x, spawn_y) + self.wheel_radius + 0.01
            
            # Create robot body (simplified as sphere for now)
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.robot_radius,
                height=0.052,  # Chassis height
                physicsClientId=self.client_id
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self.robot_radius,
                length=0.052,
                rgbaColor=[0.2, 0.2, 0.2, 1.0],
                physicsClientId=self.client_id
            )
            
            robot_body = p.createMultiBody(
                baseMass=self.robot_mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[spawn_x, spawn_y, spawn_z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, spawn_theta]),
                physicsClientId=self.client_id
            )
            
            # Apply realistic damping
            p.changeDynamics(
                robot_body, -1,
                linearDamping=self.linear_damping,
                angularDamping=self.angular_damping,
                lateralFriction=self.wheel_friction,
                physicsClientId=self.client_id
            )
            
            # Set goal
            goal_dist = np.random.uniform(*self.goal_distance_range)
            goal_angle = np.random.uniform(-np.pi, np.pi)
            goal_x = spawn_x + goal_dist * np.cos(goal_angle)
            goal_y = spawn_y + goal_dist * np.sin(goal_angle)
            
            # Clip to arena bounds
            margin = 2.0
            goal_x = np.clip(goal_x, -self.arena_size/2 + margin, self.arena_size/2 - margin)
            goal_y = np.clip(goal_y, -self.arena_size/2 + margin, self.arena_size/2 - margin)
            
            self.robot_bodies[agent_name] = robot_body
            self.robot_states[agent_name] = {
                'x': spawn_x, 'y': spawn_y, 'theta': spawn_theta,
                'vx': 0, 'vy': 0, 'v': 0, 'omega': 0,
                'goal_x': goal_x, 'goal_y': goal_y,
                'reached': False, 'collided': False
            }
            self.prev_distances[agent_name] = self._compute_distance_to_goal(agent_name)
        
        print(f"[ENV] Created {self.num_robots} robots")
    
    def _get_terrain_height(self, x: float, y: float) -> float:
        """
        Get terrain height at given world coordinates.
        
        Args:
            x, y: World coordinates
        
        Returns:
            float: Terrain height at that point (0 if no heightmap loaded)
        """
        if self.friction_map is None or not self.use_terrain:
            return 0.0
        
        # This is a simplified approach - for accurate height, we'd need the raw heightmap
        # PyBullet heightfield handles the actual collision
        return 0.0
    
    def _get_friction_at_position(self, x: float, y: float) -> float:
        """
        Get friction coefficient at given world coordinates.
        
        Args:
            x, y: World coordinates
        
        Returns:
            float: Friction coefficient (0.8 normal, 0.1 puddle)
        """
        if self.friction_map is None:
            return self.wheel_friction
        
        # Convert world coords to friction map indices
        map_size = self.friction_map.shape[0]
        
        # World to map conversion
        map_x = int((x + self.arena_size/2) / self.arena_size * map_size)
        map_y = int((y + self.arena_size/2) / self.arena_size * map_size)
        
        # Clamp to valid range
        map_x = np.clip(map_x, 0, map_size - 1)
        map_y = np.clip(map_y, 0, map_size - 1)
        
        # Get friction multiplier and convert to actual friction
        friction_ratio = self.friction_map[map_y, map_x]
        return self.wheel_friction * friction_ratio
    
    def set_phase(self, phase_name: str):
        """Manually set the curriculum phase."""
        phases = {
            "Easy1": {"range": [2.0, 3.0], "threshold": 1.5},
            "Easy2": {"range": [3.0, 4.0], "threshold": 1.7},
            "Med1": {"range": [4.0, 6.0], "threshold": 2.0},
            "Hard1": {"range": [5.0, 8.0], "threshold": 2.5},
            "Hard2": {"range": [6.0, 10.0], "threshold": 3.0},
            "VHard": {"range": [8.0, 15.0], "threshold": 3.5},
        }
        
        if phase_name not in phases:
            print(f"[ENV] Warning: phase '{phase_name}' not found")
            return
        
        self.current_phase = phase_name
        self.goal_distance_range = phases[phase_name]["range"]
        self.goal_threshold = phases[phase_name]["threshold"]
        
        print(f"[ENV] Phase: {phase_name}, distance: {self.goal_distance_range}, threshold: {self.goal_threshold}")
    
    def reset(self, seed=None):
        """Reset environment with random terrain variant."""
        super().reset(seed=seed)
        self.timestep = 0
        self.episode_count += 1
        
        # Randomize terrain variant
        if self.use_terrain:
            new_variant = np.random.randint(1, self.num_terrain_variants + 1)
            if new_variant != self.current_terrain_variant:
                self._create_heightfield_terrain(new_variant)
        
        # Reset robots
        for agent_name in self.robot_states.keys():
            robot_body = self.robot_bodies[agent_name]
            
            robot_id = int(agent_name.split('_')[1])
            angle = (2 * np.pi * robot_id) / self.num_robots + np.random.uniform(-0.3, 0.3)
            spawn_radius = self.arena_size / 5
            spawn_x = spawn_radius * np.cos(angle)
            spawn_y = spawn_radius * np.sin(angle)
            spawn_theta = angle
            spawn_z = self._get_terrain_height(spawn_x, spawn_y) + self.wheel_radius + 0.02
            
            p.resetBasePositionAndOrientation(
                robot_body,
                [spawn_x, spawn_y, spawn_z],
                p.getQuaternionFromEuler([0, 0, spawn_theta]),
                physicsClientId=self.client_id
            )
            p.resetBaseVelocity(robot_body, [0, 0, 0], [0, 0, 0],
                               physicsClientId=self.client_id)
            
            # New goal
            goal_dist = np.random.uniform(*self.goal_distance_range)
            goal_angle = np.random.uniform(-np.pi, np.pi)
            goal_x = spawn_x + goal_dist * np.cos(goal_angle)
            goal_y = spawn_y + goal_dist * np.sin(goal_angle)
            
            margin = 2.0
            goal_x = np.clip(goal_x, -self.arena_size/2 + margin, self.arena_size/2 - margin)
            goal_y = np.clip(goal_y, -self.arena_size/2 + margin, self.arena_size/2 - margin)
            
            self.robot_states[agent_name] = {
                'x': spawn_x, 'y': spawn_y, 'theta': spawn_theta,
                'vx': 0, 'vy': 0, 'v': 0, 'omega': 0,
                'goal_x': goal_x, 'goal_y': goal_y,
                'reached': False, 'collided': False
            }
            self.prev_distances[agent_name] = self._compute_distance_to_goal(agent_name)
        
        observations = self._get_observations()
        return observations, {}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one timestep."""
        
        for agent_name, action in actions.items():
            robot_body = self.robot_bodies[agent_name]
            v, omega = action
            
            # Denormalize
            linear_vel = v * self.v_max
            angular_vel = omega * self.omega_max
            
            # Apply friction modifier based on position
            state = self.robot_states[agent_name]
            friction = self._get_friction_at_position(state['x'], state['y'])
            friction_modifier = friction / self.wheel_friction
            
            # Reduce velocity on low-friction surfaces
            linear_vel *= friction_modifier
            
            # Get orientation
            pos, orn = p.getBasePositionAndOrientation(robot_body, physicsClientId=self.client_id)
            theta = p.getEulerFromQuaternion(orn)[2]
            
            vx = linear_vel * np.cos(theta)
            vy = linear_vel * np.sin(theta)
            
            p.resetBaseVelocity(
                robot_body,
                [vx, vy, 0],
                [0, 0, angular_vel],
                physicsClientId=self.client_id
            )
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)
        self.timestep += 1
        
        # Update robot states
        self._update_robot_states()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check terminations
        terminations = self._check_terminations()
        truncations = {name: self.timestep >= self.max_episode_steps for name in self.robot_states}
        
        observations = self._get_observations()
        infos = {name: {} for name in self.robot_states}
        
        return observations, rewards, terminations, truncations, infos
    
    def _update_robot_states(self):
        """Update robot state from physics."""
        for agent_name, robot_body in self.robot_bodies.items():
            pos, orn = p.getBasePositionAndOrientation(robot_body, physicsClientId=self.client_id)
            vel, ang_vel = p.getBaseVelocity(robot_body, physicsClientId=self.client_id)
            
            self.robot_states[agent_name]['x'] = pos[0]
            self.robot_states[agent_name]['y'] = pos[1]
            self.robot_states[agent_name]['theta'] = p.getEulerFromQuaternion(orn)[2]
            self.robot_states[agent_name]['vx'] = vel[0]
            self.robot_states[agent_name]['vy'] = vel[1]
            self.robot_states[agent_name]['v'] = np.sqrt(vel[0]**2 + vel[1]**2)
            self.robot_states[agent_name]['omega'] = ang_vel[2]
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get 8D observation for each robot."""
        observations = {}
        
        for agent_name, state in self.robot_states.items():
            dx_goal = state['goal_x'] - state['x']
            dy_goal = state['goal_y'] - state['y']
            dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
            
            obs = np.array([
                np.clip(dx_goal / 10.0, -1, 1),
                np.clip(dy_goal / 10.0, -1, 1),
                np.clip(dist_goal / 10.0, 0, 1),
                np.cos(state['theta']),
                np.sin(state['theta']),
                np.clip(state['vx'] / self.v_max, -1, 1),
                np.clip(state['vy'] / self.v_max, -1, 1),
                np.clip(state['omega'] / self.omega_max, -1, 1),
            ], dtype=np.float32)
            
            observations[agent_name] = obs
        
        return observations
    
    def _compute_distance_to_goal(self, agent_name: str) -> float:
        """Compute distance to goal for agent."""
        state = self.robot_states[agent_name]
        return np.sqrt(
            (state['goal_x'] - state['x'])**2 + 
            (state['goal_y'] - state['y'])**2
        )
    
    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards based on goal progress."""
        rewards = {}
        
        for agent_name in self.robot_states.keys():
            state = self.robot_states[agent_name]
            
            if state['reached'] or state['collided']:
                rewards[agent_name] = 0.0
                continue
            
            curr_dist = self._compute_distance_to_goal(agent_name)
            prev_dist = self.prev_distances[agent_name]
            
            # Progress reward
            progress = (prev_dist - curr_dist) * 10.0
            
            # Goal bonus
            if curr_dist < self.goal_threshold:
                state['reached'] = True
                progress += 100.0
            
            # Time penalty
            progress -= 0.1
            
            rewards[agent_name] = progress * 0.01  # Scale down
            self.prev_distances[agent_name] = curr_dist
        
        return rewards
    
    def _check_terminations(self) -> Dict[str, bool]:
        """Check termination conditions."""
        terminations = {}
        
        for agent_name in self.robot_states.keys():
            state = self.robot_states[agent_name]
            robot_body = self.robot_bodies[agent_name]
            
            # Check collisions
            contacts = p.getContactPoints(
                bodyA=robot_body,
                physicsClientId=self.client_id
            )
            
            for contact in contacts:
                other_body = contact[2]
                if other_body in self.pillars or other_body in self.obstacles:
                    state['collided'] = True
                    break
            
            terminations[agent_name] = state['reached'] or state['collided']
        
        return terminations
    
    def close(self):
        """Clean up."""
        p.disconnect(self.client_id)


def test_terrain_environment():
    """Test the terrain environment."""
    print("\n" + "="*70)
    print("TESTING TERRAIN MINING ENVIRONMENT")
    print("="*70)
    
    env = MiningEnvironmentTerrain(
        num_robots=3, 
        arena_size=50.0,
        max_episode_steps=300, 
        visualize=False,
        use_terrain=False  # Start with flat terrain for testing
    )
    
    obs, _ = env.reset()
    print(f"\n✓ Environment initialized")
    print(f"  - Observation shape: {obs['robot_0'].shape}")
    print(f"  - Action shape: {env.action_space.shape}")
    print(f"  - Pillars: {len(env.pillars)}")
    print(f"  - Obstacles: {len(env.obstacles)}")
    
    total_reward = {f"robot_{i}": 0 for i in range(3)}
    
    for step in range(100):
        actions = {f"robot_{i}": env.action_space.sample() for i in range(3)}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent_name in rewards.keys():
            total_reward[agent_name] += rewards[agent_name]
        
        if any(terminations.values()):
            break
    
    print(f"\n✓ Ran {step+1} steps")
    print(f"  - Total rewards: {total_reward}")
    
    env.close()
    print("\n✓ Environment closed successfully")


if __name__ == "__main__":
    test_terrain_environment()
