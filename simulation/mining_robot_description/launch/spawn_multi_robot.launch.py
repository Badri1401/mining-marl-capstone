#!/usr/bin/env python3
"""
Multi-Robot Spawn Launch File
==============================
Spawns 3 mining robots in the arena with appropriate namespaces.

Robot positions are arranged in a triangle pattern to avoid collisions.

Usage:
    ros2 launch mining_robot_description spawn_multi_robot.launch.py
"""

import os
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    pkg_robot = get_package_share_directory('mining_robot_description')
    
    # Launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to spawn'
    )
    
    spawn_radius_arg = DeclareLaunchArgument(
        'spawn_radius',
        default_value='5.0',
        description='Radius of spawn circle (meters)'
    )
    
    # Robot spawn positions (arranged in triangle)
    num_robots = 3
    spawn_radius = 5.0
    
    robot_spawns = []
    
    for i in range(num_robots):
        angle = (2 * math.pi * i) / num_robots
        x = spawn_radius * math.cos(angle)
        y = spawn_radius * math.sin(angle)
        yaw = angle + math.pi  # Face center
        
        robot_spawn = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_robot, 'launch', 'spawn_robot.launch.py')
            ),
            launch_arguments={
                'namespace': f'robot_{i}',
                'x_pose': str(x),
                'y_pose': str(y),
                'z_pose': '0.1',
                'yaw': str(yaw),
                'use_rviz': 'false',
            }.items()
        )
        
        robot_spawns.append(robot_spawn)
    
    return LaunchDescription([
        num_robots_arg,
        spawn_radius_arg,
        *robot_spawns,
    ])
