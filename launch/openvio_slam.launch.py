"""
launch/openvio_slam.launch.py
==============================
Launches the full OpenVIO-SLAM robot stack.

Starts:
  1. rosbridge WebSocket for phone camera streaming
  2. optional bridge that can publish non-IMU camera data
  3. Arduino IMU bridge node that owns the serial interface
  4. OpenVIO-SLAM visual-inertial estimator
  5. web controller node for browser control and telemetry
  6. RViz visualization when enabled
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
        parameters=[{'publish_imu': False}],
        output='screen',
    )

    imu_node = Node(
        package='openvio_slam',
        executable='imu_publisher_node',
        name='imu_publisher',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'baud_rate': 115200,
            'publish_rate': 100.0,
            'frame_id': 'imu_link',
            'gyro_noise': 0.005,
            'accel_noise': 0.085,
            'port_retry_sec': 3.0,
        }],
    )

    slam_node = Node(
        package='openvio_slam',
        executable='openvio_slam_node',
        name='openvio_slam',
        output='screen',
        parameters=[{
            'window_size': 12,
            'n_features': 150,
            'gyro_noise': 0.005,
            'accel_noise': 0.085,
            'extrinsic_yaml': 'config/extrinsic.yaml',
            'slam_yaml': 'config/slam.yaml',
        }],
    )

    web_node = Node(
        package='openvio_slam',
        executable='web_controller_node',
        name='web_controller',
        output='screen',
        parameters=[{
            'http_port': 8080,
            'ws_port': 8765,
            'max_cam_fps': 10,
        }],
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_to_base_tf',
        arguments=[
            '--roll-pitch-yaw', '0.0', '0.0', '0.0',
            '0.0', '0.0', '0.0',
            'imu_link', 'base_link'
        ],
        output='screen',
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
        TimerAction(period=3.0, actions=[imu_node]),
        TimerAction(period=4.0, actions=[static_tf_node]),
        TimerAction(period=5.0, actions=[slam_node]),
        TimerAction(period=7.0, actions=[web_node]),
        TimerAction(period=9.0, actions=[rviz_node]),
    ])
