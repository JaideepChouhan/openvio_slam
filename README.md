# OpenVIO-SLAM

OpenVIO-SLAM is a monocular visual-inertial SLAM system built for indoor ground robots running on a Raspberry Pi 5 with ROS 2 Jazzy. The current architecture separates the core SLAM estimator from the robot I/O stack: a dedicated Arduino IMU bridge owns the serial link, the browser controller provides human control and telemetry, and the SLAM node continues to consume the fixed `/imu` topic required by the estimator.

This repository now implements the multi-agent robot workflow described by the architecture prompt:

- phone camera stream via rosbridge
- Arduino MPU-9250/MPU-6500 IMU via USB serial
- ROS 2 `/imu` publisher node
- active web dashboard for telemetry and direct robot control
- launch orchestration for the full pipeline

---

## System architecture

```text
Phone camera --> rosbridge --> slam_bridge_node --> /image, /camera_info
                                               |
Arduino UNO --> USB serial --> imu_publisher_node --> /imu
                                  |
                                  +--> /robot_control
                                  +--> /imu_recal

openvio_slam_node --> /odom, /openvio/map_points, /tf

web_controller_node <-- /image, /imu, /odom, /openvio/map_points, /diagnostics
        |
        +--> HTTP dashboard on :8080
        +--> WebSocket bridge on :8765
        +--> browser UI for keyboard/joystick control
```

Important constraint:

- `openvio_slam_node` still subscribes to `/imu`.
- The legacy phone IMU output is disabled in the bridge setup.
- The Arduino IMU publisher is the single serial owner.

---

## Project layout

```text
openvio_slam/
├── CMakeLists.txt
├── launch/
│   └── openvio_slam.launch.py
├── config/
│   ├── slam.yaml
│   └── extrinsic.yaml
├── openvio_slam/
│   ├── imu/
│   │   └── imu_publisher_node.py
│   ├── ros/
│   │   └── vio_node.py
│   └── ...
├── web_controller/
│   ├── __init__.py
│   ├── web_controller_node.py
│   └── static/
│       ├── index.html
│       ├── controller.js
│       └── style.css
├── package.xml
├── setup.py
├── requirements.txt
├── README.md
└── tests/
```

---

## Core components

### 1. SLAM core
The visual-inertial estimator remains the core algorithmic part of the project.

Relevant files:

- [openvio_slam/ros/vio_node.py](openvio_slam/ros/vio_node.py)
- [openvio_slam/estimator](openvio_slam/estimator)
- [openvio_slam/frontend](openvio_slam/frontend)
- [openvio_slam/map](openvio_slam/map)

This layer continues to provide:

- monocular visual-inertial tracking
- ORB + KLT frontend
- MSCKF estimation
- ZUPT and planar constraints
- loop closure and mapping support

### 2. IMU publisher node
The ROS 2 IMU bridge lives in:

- [openvio_slam/imu/imu_publisher_node.py](openvio_slam/imu/imu_publisher_node.py)

Responsibilities:

- open and monitor the Arduino serial port
- parse 31-byte IMU packets from the Arduino
- validate the XOR checksum over bytes 2 through 29 inclusive (`packet[2:30]`)
- publish `sensor_msgs/Imu` on `/imu`
- publish `diagnostic_msgs/DiagnosticArray` on `/diagnostics`
- subscribe to `/robot_control` and forward motor commands to serial
- subscribe to `/imu_recal` and request a firmware re-calibration

Important protocol notes:

- Arduino `millis()` is a 32-bit counter and will wrap after roughly 49 days. The receiver should treat timestamps as wrap-around-safe values using `uint32_t` arithmetic when computing elapsed time.
- The packet checksum is intentionally validated as a byte-wise XOR over bytes 2 through 29 inclusive, which matches the Arduino transmitter implementation.
- The `imu_publisher_node` publishes only the raw IMU stream; the browser controller is responsible for JPEG conversion before sending images to the client.

### 3. Web controller node
The browser control layer lives in:

- [web_controller/web_controller_node.py](web_controller/web_controller_node.py)

Responsibilities:

- serve the HTTP dashboard on port `8080`
- serve a WebSocket bridge on port `8765`
- forward browser commands to ROS topics:
  - `/robot_control`
  - `/imu_recal`
  - `/slam_reset`
- subscribe to raw `/image` messages, convert them to JPEG in OpenCV, and send the encoded base64 payload to the browser
- relay `/imu`, `/odom`, `/openvio/map_points`, and `/diagnostics` to the browser
- emit system telemetry for CPU, memory, and runtime health

The browser-facing image path is intentionally JPEG-encoded for the UI. If a compressed camera topic is available later, it can replace the raw `/image` subscription, but the node should still deliver JPEG data to the browser rather than raw OpenCV bytes.

### 4. Browser UI
The dashboard is in:

- [web_controller/static/index.html](web_controller/static/index.html)
- [web_controller/static/controller.js](web_controller/static/controller.js)
- [web_controller/static/style.css](web_controller/static/style.css)

Features:

- camera feed with crosshair overlay
- IMU charts using Chart.js
- odometry and map statistics
- keyboard and joystick control hooks
- emergency stop and reset actions
- debug console output

---

## Launch workflow

The default launch file is:

- [launch/openvio_slam.launch.py](launch/openvio_slam.launch.py)

Launch order:

1. `rosbridge_websocket`
2. bridge node for the phone camera stream
3. `imu_publisher_node`
4. `static_transform_publisher` for the `imu_link` to `base_link` relationship
5. `openvio_slam_node`
6. `web_controller_node`
7. RViz when enabled

The static transform is launched explicitly with the ROS 2 Euler-angle form: `--roll-pitch-yaw` followed by `roll pitch yaw` values. This avoids ambiguity when the transform is defined in the launch file.

The launch file also disables legacy phone IMU publishing by keeping the camera bridge from emitting `/imu`.

---

## Configuration

Configuration files:

- [config/slam.yaml](config/slam.yaml)
- [config/extrinsic.yaml](config/extrinsic.yaml)

Key settings include:

- IMU frame: `imu_link`
- Arduino serial port setup
- camera-to-IMU extrinsic calibration placeholders
- web controller ports and dashboard FPS settings

---

## Build and run

From the workspace root:

```bash
cd ~/slam_ws
colcon build --packages-select openvio_slam
source install/setup.bash
ros2 launch openvio_slam openvio_slam.launch.py
```

Open the browser dashboard at:

```text
http://localhost:8080
```

---

## Dependencies

Install the runtime dependencies:

```bash
sudo apt install python3-serial python3-psutil ros-jazzy-diagnostic-msgs -y
pip3 install pyserial psutil websockets --break-system-packages
```

Python package metadata is updated in:

- [package.xml](package.xml)
- [CMakeLists.txt](CMakeLists.txt)
- [requirements.txt](requirements.txt)
- [setup.py](setup.py)

---

## Operational notes

- Keep `use_sim_time` disabled for live streaming.
- Ensure the Arduino serial device is accessible to the ROS user (`dialout` group often needs to be updated).
- The IMU publisher owns the serial port; do not send motor commands directly from the estimator.
- Treat the core SLAM files as the stable estimator layer; do not modify them for the robot control integration.
- The camera-to-IMU vector `p_IC` is measured from the camera optical center to the IMU sensor. If the phone is not rigidly and repeatably mounted, this offset changes and the estimator becomes inconsistent. Fasten the phone mechanically and verify the offset before operation.
- The web controller runs an HTTP server and WebSocket server alongside `rclpy`; keep the ROS node in the main thread and use `asyncio.run_coroutine_threadsafe` for cross-thread client updates as done in the current implementation.

---

## Validation

The repository includes a lightweight regression test for the multi-agent architecture:

- [tests/test_multi_agent_architecture.py](tests/test_multi_agent_architecture.py)

It checks that the IMU bridge node, web controller node, launch file, and static UI are present and wired into the planned architecture.

---

## Summary

This project now follows the intended robotics architecture:

- the estimator remains specialized on state estimation
- the IMU node owns the serial interface and IMU data stream
- the browser controller handles robot control and monitoring
- the launch workflow coordinates the complete system in a clean deployment flow
