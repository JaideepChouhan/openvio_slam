"""
launch/openvio_slam.launch.py — OpenVIO-SLAM full stack
========================================================
Data flow:

    Phone  ──► /camera/image_raw/compressed ──► slam_bridge_node
                                                   ├─► /image        (bgr8)
                                                   └─► /camera_info  (plumb_bob)

    Arduino ──► /imu  + /imu/mag              (imu_publisher_node)

    openvio_slam_node consumes /image + /camera_info + /imu → /odom + map_points
    web_controller_node consumes all of the above → HTTP UI on :8080, WS :8765

Orientation policy:
    The phone app now rotates the camera frame in canvas, so
    /camera/image_raw/compressed is already upright landscape (640×480).
    The bridge node is told `rotate_portrait:=False` and just forwards the
    decoded image with no cv2.rotate — the Pi spends zero CPU on rotation.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('openvio_slam')

    args = [
        # ── Hardware ─────────────────────────────────────────────────────
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baud_rate',   default_value='115200'),

        # ── Frames ───────────────────────────────────────────────────────
        DeclareLaunchArgument('imu_frame',    default_value='imu_link'),
        DeclareLaunchArgument('base_frame',   default_value='base_link'),
        DeclareLaunchArgument('camera_frame', default_value='camera_link'),

        # ── Camera mount (base_link → camera_link), metres / radians ─────
        DeclareLaunchArgument('cam_x',     default_value='0.10'),
        DeclareLaunchArgument('cam_y',     default_value='0.00'),
        DeclareLaunchArgument('cam_z',     default_value='0.05'),
        DeclareLaunchArgument('cam_roll',  default_value='0.0'),
        DeclareLaunchArgument('cam_pitch', default_value='0.0'),
        DeclareLaunchArgument('cam_yaw',   default_value='0.0'),

        # ── Camera intrinsics (LANDSCAPE 640×480 calibration) ────────────
        DeclareLaunchArgument('calib_width',  default_value='640'),
        DeclareLaunchArgument('calib_height', default_value='480'),
        DeclareLaunchArgument('fx', default_value='518.533'),
        DeclareLaunchArgument('fy', default_value='519.433'),
        DeclareLaunchArgument('cx', default_value='320.224'),
        DeclareLaunchArgument('cy', default_value='242.973'),

        # ── Network ──────────────────────────────────────────────────────
        DeclareLaunchArgument('rosbridge_port', default_value='9090'),
        DeclareLaunchArgument('http_port',      default_value='8080'),
        DeclareLaunchArgument('ws_port',        default_value='8765'),

        # ── Misc ─────────────────────────────────────────────────────────
        DeclareLaunchArgument('max_cam_fps', default_value='10.0'),
        DeclareLaunchArgument('record_path', default_value='~/.ros/bags/'),
        DeclareLaunchArgument('rviz',        default_value='false'),
        DeclareLaunchArgument('log_level',   default_value='info'),
    ]

    log_level = LaunchConfiguration('log_level')

    # ── 1. rosbridge — phone connects here ───────────────────────────────
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'port':             ParameterValue(
                                    LaunchConfiguration('rosbridge_port'),
                                    value_type=int),
            'max_message_size': 20_000_000,
            'fragment_timeout': 600,
            'delay_between_messages': 0.0,
        }],
    )

    # ── 2. Camera bridge: CompressedImage (phone) → /image + /camera_info ─
    # `rotate_portrait` is disabled because the phone now publishes already-
    # upright landscape frames. The bridge just decodes + republishes.
    bridge_node = Node(
        package='slam_project',
        executable='slam_bridge_node',
        name='slam_bridge_node',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'publish_imu':          False,
            'rotate_portrait':      False,               # phone rotates now
            'frame_id':             LaunchConfiguration('camera_frame'),

            'image_in_topic':       '/camera/image_raw/compressed',
            'image_out_topic':      '/image',
            'camera_info_out_topic':'/camera_info',

            'calib_width':  ParameterValue(LaunchConfiguration('calib_width'),  value_type=int),
            'calib_height': ParameterValue(LaunchConfiguration('calib_height'), value_type=int),
            'fx':           ParameterValue(LaunchConfiguration('fx'),           value_type=float),
            'fy':           ParameterValue(LaunchConfiguration('fy'),           value_type=float),
            'cx':           ParameterValue(LaunchConfiguration('cx'),           value_type=float),
            'cy':           ParameterValue(LaunchConfiguration('cy'),           value_type=float),
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── 3. Arduino IMU publisher ─────────────────────────────────────────
    imu_node = Node(
        package='openvio_slam',
        executable='imu_publisher_node',
        name='imu_publisher',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=3.0,
        parameters=[{
            'serial_port':    LaunchConfiguration('serial_port'),
            'baud_rate':      ParameterValue(LaunchConfiguration('baud_rate'),
                                             value_type=int),
            'publish_rate':   100.0,
            'frame_id':       LaunchConfiguration('imu_frame'),
            'gyro_noise':     0.005,
            'accel_noise':    0.085,
            'port_retry_sec': 3.0,
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── 4. TF: imu_link → base_link (identity) ──────────────────────────
    tf_imu_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_imu_to_base',
        output='log',
        arguments=[
            '--x','0','--y','0','--z','0',
            '--roll','0','--pitch','0','--yaw','0',
            '--frame-id',       LaunchConfiguration('imu_frame'),
            '--child-frame-id', LaunchConfiguration('base_frame'),
        ],
    )

    # ── 5. TF: base_link → camera_link ──────────────────────────────────
    tf_base_to_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_cam',
        output='log',
        arguments=[
            '--x',     LaunchConfiguration('cam_x'),
            '--y',     LaunchConfiguration('cam_y'),
            '--z',     LaunchConfiguration('cam_z'),
            '--roll',  LaunchConfiguration('cam_roll'),
            '--pitch', LaunchConfiguration('cam_pitch'),
            '--yaw',   LaunchConfiguration('cam_yaw'),
            '--frame-id',       LaunchConfiguration('base_frame'),
            '--child-frame-id', LaunchConfiguration('camera_frame'),
        ],
    )

    # ── 6. OpenVIO SLAM ──────────────────────────────────────────────────
    slam_node = Node(
        package='openvio_slam',
        executable='openvio_slam_node',
        name='openvio_slam',
        output='screen',
        emulate_tty=True,
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
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── 7. Web controller ────────────────────────────────────────────────
    web_node = Node(
        package='openvio_slam',
        executable='web_controller_node',
        name='web_controller',
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'http_port':   ParameterValue(LaunchConfiguration('http_port'),   value_type=int),
            'ws_port':     ParameterValue(LaunchConfiguration('ws_port'),     value_type=int),
            'max_cam_fps': ParameterValue(LaunchConfiguration('max_cam_fps'), value_type=float),
            'record_path': LaunchConfiguration('record_path'),
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )

    # ── 8. RViz (opt-in) ─────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        emulate_tty=True,
        arguments=['-d', PathJoinSubstitution(
            [pkg, 'config', 'openvio_slam.rviz'])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        sigterm_timeout='15',
        sigkill_timeout='20',
    )

    # ── Sequencing: everything waits for rosbridge ──────────────────────
    start_after_rosbridge = RegisterEventHandler(OnProcessStart(
        target_action=rosbridge_node,
        on_start=[
            TimerAction(period=0.5, actions=[imu_node]),
            TimerAction(period=0.5, actions=[tf_imu_to_base]),
            TimerAction(period=0.5, actions=[tf_base_to_cam]),
            TimerAction(period=1.0, actions=[bridge_node]),
            TimerAction(period=2.0, actions=[slam_node]),
            TimerAction(period=2.5, actions=[web_node]),
            TimerAction(period=3.5, actions=[rviz_node]),
        ],
    ))

    return LaunchDescription([
        *args,
        LogInfo(msg='OpenVIO-SLAM bringing up…'),
        rosbridge_node,
        start_after_rosbridge,
    ])
