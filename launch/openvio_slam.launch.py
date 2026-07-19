"""
launch/openvio_slam.launch.py
==============================
Launches the full OpenVIO-SLAM pipeline.

Starts:
  1. rosbridge WebSocket (phone -> ROS bridge)
  2. slam_bridge_node (your existing node: re-timestamps phone images)
  3. openvio_slam_node (the SLAM system)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('openvio_slam')

    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{'port': 9090, 'max_message_size': 10000000}],
        output='screen',
    )

    bridge_node = Node(
        package='slam_project',
        executable='slam_bridge_node',
        name='slam_bridge_node',
        output='screen',
    )

    slam_node = Node(
        package='openvio_slam',
        executable='openvio_slam_node',
        name='openvio_slam',
        output='screen',
        parameters=[{
            'window_size': 20,
            'n_features':  300,
            'gyro_noise':  0.005,
            'accel_noise': 0.05,
        }],
    )
    
    rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'openvio_slam.rviz'])],
    condition=IfCondition(LaunchConfiguration('rviz')),
    output='screen',
    )

    return LaunchDescription([
        rviz_arg,
        rosbridge_node,
        TimerAction(period=2.0, actions=[bridge_node]),
        TimerAction(period=4.0, actions=[slam_node]),
        TimerAction(period=8.0, actions=[rviz_node]),
    ])
