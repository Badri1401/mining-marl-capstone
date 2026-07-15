#!/usr/bin/env python3
"""
Spawn Robot Launch File
=======================
Launch file for spawning mining robot(s) in Gazebo with namespace support.

Supports:
- Single or multi-robot spawning via namespace parameter
- Terrain variant selection (random or fixed)
- RViz visualization option

Usage:
    ros2 launch mining_robot_description spawn_robot.launch.py
    ros2 launch mining_robot_description spawn_robot.launch.py namespace:=robot_0
    ros2 launch mining_robot_description spawn_robot.launch.py use_rviz:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directories
    pkg_robot = get_package_share_directory('mining_robot_description')
    pkg_arena = get_package_share_directory('mining_arena_world')
    
    # Launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace for multi-robot support (e.g., robot_0, robot_1, robot_2)'
    )
    
    x_pose_arg = DeclareLaunchArgument(
        'x_pose',
        default_value='0.0',
        description='Initial X position of the robot'
    )
    
    y_pose_arg = DeclareLaunchArgument(
        'y_pose',
        default_value='0.0',
        description='Initial Y position of the robot'
    )
    
    z_pose_arg = DeclareLaunchArgument(
        'z_pose',
        default_value='0.1',
        description='Initial Z position of the robot'
    )
    
    yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value='0.0',
        description='Initial yaw orientation of the robot'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )
    
    terrain_variant_arg = DeclareLaunchArgument(
        'terrain_variant',
        default_value='1',
        description='Terrain heightmap variant (1-5, or "random" for random selection)'
    )
    
    # URDF file path
    urdf_file = os.path.join(pkg_robot, 'urdf', 'mining_robot.urdf.xacro')
    
    # Robot description from xacro
    robot_description = Command(['xacro ', urdf_file])
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=LaunchConfiguration('namespace'),
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        output='screen'
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace=LaunchConfiguration('namespace'),
        parameters=[{
            'use_sim_time': True
        }],
        output='screen'
    )
    
    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=LaunchConfiguration('namespace'),
        arguments=[
            '-name', LaunchConfiguration('namespace'),
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x_pose'),
            '-y', LaunchConfiguration('y_pose'),
            '-z', LaunchConfiguration('z_pose'),
            '-Y', LaunchConfiguration('yaw')
        ],
        output='screen'
    )
    
    # RViz
    rviz_config = os.path.join(pkg_robot, 'rviz', 'mining_robot.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        namespace=LaunchConfiguration('namespace'),
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    # ROS-Gazebo bridge for topics
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        namespace=LaunchConfiguration('namespace'),
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan',
            '/camera/depth@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/camera/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        ],
        output='screen'
    )
    
    return LaunchDescription([
        namespace_arg,
        x_pose_arg,
        y_pose_arg,
        z_pose_arg,
        yaw_arg,
        use_rviz_arg,
        terrain_variant_arg,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        rviz,
    ])
