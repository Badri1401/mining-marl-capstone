#!/usr/bin/env python3
"""
Gazebo World Launch File
========================
Launch Gazebo with the mining arena world.

Usage:
    ros2 launch mining_arena_world gazebo_world.launch.py
    ros2 launch mining_arena_world gazebo_world.launch.py terrain_variant:=3
"""

import os
import random
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_arena = get_package_share_directory('mining_arena_world')
    
    # Set Gazebo resource path
    gazebo_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_arena)
    )
    
    # Launch arguments
    terrain_variant_arg = DeclareLaunchArgument(
        'terrain_variant',
        default_value='random',
        description='Terrain variant (1-5 or "random")'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (no GUI)'
    )
    
    # World file path
    world_file = os.path.join(pkg_arena, 'worlds', 'mining_arena.sdf')
    
    # Gazebo server
    gz_server = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', world_file],
        output='screen'
    )
    
    # Gazebo client (GUI)
    gz_client = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo_resource_path,
        terrain_variant_arg,
        headless_arg,
        gz_server,
        gz_client,
    ])
