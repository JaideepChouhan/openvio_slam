"""
launch/openvio_slam.launch.py — corrected for 9-DOF Arduino protocol
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('openvio_slam')

    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    # ── 1. rosbridge WebSocket (phone camera stream) ──────────────────────────
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{'port': 9090, 'max_message_size': 10000000}],
        output='screen',
    )

    # ── 2. Phone camera bridge (publish_imu=False — Arduino is IMU source) ────
    bridge_node = Node(
        package='slam_project',
        executable='slam_bridge_node',
        name='slam_bridge_node',
        output='screen',
        parameters=[{
            'publish_imu': False,   # Arduino IMU node takes over /imu
        }],
    )

    # ── 3. Arduino IMU publisher ──────────────────────────────────────────────
    # Reads 43-byte binary packets from Arduino (9-DOF: gyro+accel+mag).
    # Publishes:
    #   /imu       sensor_msgs/Imu          (gyro + accel, 100 Hz)
    #   /imu/mag   sensor_msgs/MagneticField (magnetometer, 100 Hz)
    # Subscribes:
    #   /robot_control Int16MultiArray → "M L R\n" ASCII over serial
    #   /imu_recal     Bool            → "CAL\n"   ASCII over serial
    imu_node = Node(
        package='openvio_slam',
        executable='imu_publisher_node',
        name='imu_publisher',
        output='screen',
        parameters=[{
            'serial_port':    '/dev/ttyUSB0',
            'baud_rate':      115200,
            'publish_rate':   100.0,
            'frame_id':       'imu_link',
            'gyro_noise':     0.005,
            'accel_noise':    0.085,
            'port_retry_sec': 3.0,
        }],
    )

    # ── 4. Static TF: imu_link → base_link ───────────────────────────────────
    # Adjust x y z yaw pitch roll to match your physical mounting.
    # Default: identity (IMU at robot center, aligned with robot axes).
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_to_baselink_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'imu_link', 'base_link'],
        output='screen',
    )

    # ── 5. OpenVIO SLAM ───────────────────────────────────────────────────────
    # Subscribes to /imu (from imu_publisher_node, NOT phone).
    # openvio_slam_node itself is unchanged — protocol correction is
    # entirely handled by imu_publisher_node.
    slam_node = Node(
        package='openvio_slam',
        executable='openvio_slam_node',
        name='openvio_slam',
        output='screen',
        parameters=[{
            'window_size':    20,
            'n_features':     500,
            'gyro_noise':     0.005,
            'accel_noise':    0.085,
            'slam_yaml':      PathJoinSubstitution(
                [pkg, 'config', 'slam.yaml']),
            'extrinsic_yaml': PathJoinSubstitution(
                [pkg, 'config', 'extrinsic.yaml']),
        }],
    )

    # ── 6. Web Controller ─────────────────────────────────────────────────────
    # Serves http://0.0.0.0:8080  and  ws://0.0.0.0:8765
    # Now subscribes to /imu/mag in addition to /imu (for compass widget).
    web_node = Node(
        package='openvio_slam',
        executable='web_controller_node',
        name='web_controller',
        output='screen',
        parameters=[{
            'http_port':   8080,
            'ws_port':     8765,
            'max_cam_fps': 10.0,
            'record_path': '~/.ros/bags/',
        }],
    )

    # ── 7. RViz ───────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution(
            [pkg, 'config', 'openvio_slam.rviz'])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        rviz_arg,
        rosbridge_node,                                      # t=0
        TimerAction(period=2.0,  actions=[bridge_node]),     # t=2  camera
        TimerAction(period=3.0,  actions=[imu_node]),        # t=3  Arduino IMU
        TimerAction(period=4.0,  actions=[static_tf_node]),  # t=4  TF
        TimerAction(period=5.0,  actions=[slam_node]),       # t=5  SLAM
        TimerAction(period=7.0,  actions=[web_node]),        # t=7  Web UI
        TimerAction(period=9.0,  actions=[rviz_node]),       # t=9  RViz
    ])


