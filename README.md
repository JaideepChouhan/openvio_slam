<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:060918,40:0d1f3c,100:091428&height=180&section=header&text=OpenVIO-SLAM&fontSize=64&fontColor=00d4ff&fontAlignY=42&desc=Monocular%20Visual-Inertial%20SLAM%20%E2%80%94%20Built%20From%20First%20Principles&descSize=16&descAlignY=66&animation=fadeIn" width="100%"/>

<br/>

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-0A66C2?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![CPU Only](https://img.shields.io/badge/CPU%20Only-No%20GPU-f97316?style=for-the-badge)]()
[![Pi 5 Ready](https://img.shields.io/badge/Raspberry%20Pi-5%20Ready-A22846?style=for-the-badge&logo=raspberry-pi&logoColor=white)]()

<br/>

[![Estimator](https://img.shields.io/badge/Estimator-MSCKF-7c3aed?style=flat-square)]()
[![Frontend](https://img.shields.io/badge/Frontend-ORB%20%2B%20KLT-7c3aed?style=flat-square)]()
[![IMU](https://img.shields.io/badge/IMU-9--DOF%20MPU--9250-7c3aed?style=flat-square)]()
[![Loop Closure](https://img.shields.io/badge/Loop%20Closure-BoW%20%2B%20Pose%20Graph-7c3aed?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-Pytest-7c3aed?style=flat-square)]()

</div>

---

## What this is

OpenVIO-SLAM is a monocular Visual-Inertial SLAM system built entirely from mathematical first principles — no RTABMap, no g2o, no Ceres. It takes a single camera and an IMU and produces a metrically-scaled sparse 3D map, real-time odometry, loop closures, and re-localization on map reload. All on a laptop CPU or Raspberry Pi 5, no GPU required.

The estimator is a **Multi-State Constraint Kalman Filter (MSCKF)** — the same class of filter used in production VIO systems — implemented in clean, documented Python so every design decision is traceable to a paper and testable in isolation.

**The tracked ground robot described in this repository is the physical testing platform for the SLAM system.** It is not the end product. It exists to generate real-world motion data — varied trajectories, different lighting, real IMU noise — that a phone camera or simulator cannot provide. The robot carries the phone camera (visual input) and a dedicated Arduino IMU (inertial input), and gives an operator full remote control via a browser dashboard. Everything the SLAM estimator needs to be validated on real hardware is here.

---

## Testing platform hardware

<div align="center">

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TESTING PLATFORM                                │
├──────────────────────────┬─────────────────────────────────────────────┤
│  SENSORS                 │  COMPUTE & CONTROL                          │
│                          │                                             │
│  Android phone           │  Raspberry Pi 5  (8 GB RAM)                │
│  ├─ Camera  (visual)     │  ├─ openvio_slam_node   ← SLAM estimator   │
│  └─ Landscape mount      │  ├─ imu_publisher_node  ← Arduino bridge   │
│                          │  ├─ slam_bridge_node    ← phone camera     │
│  Arduino UNO             │  ├─ web_controller_node ← browser UI       │
│  ├─ MPU-9250  (9-DOF)    │  └─ rosbridge           ← phone WebSocket  │
│  │   Gyro + Accel + Mag  │                                             │
│  ├─ Cytron MDD20A driver │  OPERATOR  (any laptop browser)            │
│  └─ USB serial → Pi      │  ├─ Live camera feed + compass             │
│                          │  ├─ Gyro / Accel / Mag real-time charts   │
│  Two 45 RPM DC motors    │  ├─ Trajectory mini-map                   │
│  (tracked, one per side) │  └─ Keyboard / joystick drive control     │
└──────────────────────────┴─────────────────────────────────────────────┘
```

</div>

### Why a tracked ground robot

Ground robots constrain motion to a plane. This unlocks two MSCKF improvements that handheld testing cannot demonstrate cleanly:

**ZUPT (Zero-Velocity Update)** — When the robot stops, the estimator injects a tight `v = 0` pseudo-measurement. This eliminates accelerometer bias drift during pauses — the single highest-impact tuning for slow-moving platforms. A stationary robot is genuinely stationary; a person holding a phone is not.

**Planar motion constraint** — A soft `v_z ≈ 0` constraint that gives the MSCKF the geometric information a depth sensor would otherwise provide, for free. Only meaningful when the platform actually moves on a plane.

Both constraints are specific to ground robots. The test platform is built to exercise them on real hardware with controllable motion.

---

## System architecture

```
Phone (WiFi) ─────────────────────────────────────────► Raspberry Pi 5
  Chrome browser                                                │
  ├─ JPEG frames @ 10 fps ─► rosbridge:9090 ─► slam_bridge_node│
  └─ Phone IMU  DISABLED  (publish_imu: False)                  │
                                                                │
Arduino UNO ──── USB serial 115200 baud ───────────────────────┐│
  MPU-9250 sends 43-byte binary packets @ 100 Hz:              ││
  [AA][55][ts:4][gx:4][gy:4][gz:4]                             ▼│
  [ax:4][ay:4][az:4][mx:4][my:4][mz:4][chk:1]       imu_publisher_node
  Receives ASCII text: "M <L> <R>\n"  or  "CAL\n"    ├─ /imu        (100 Hz)
                                                      ├─ /imu/mag    (100 Hz)
                                                      └─ /robot_control  ◄──┐
                                                                            │
                                          /image, /camera_info              │
                                          /imu,   /imu/mag                  │
                                                  │                         │
                                                  ▼                         │
                                       openvio_slam_node                    │
                                       ├─ ORB + KLT visual frontend         │
                                       ├─ MSCKF state estimator             │
                                       ├─ IMU pre-integration (RK4)         │
                                       ├─ ZUPT + planar constraints         │
                                       ├─ DLT triangulation                 │
                                       └─ BoW loop closure                  │
                                                  │                         │
                                     /odom  /tf  /openvio/map_points        │
                                                  │                         │
                                                  ▼                         │
                                       web_controller_node                  │
                                       ├─ HTTP  :8080  (browser UI)         │
                                       └─ WS    :8765  (telemetry bridge)   │
                                                  │                         │
                                       Laptop browser ───────────────────────┘
                                       ├─ Camera feed + tilt-compensated compass
                                       ├─ Gyro / Accel / Mag live charts
                                       ├─ Odometry readout + trajectory map
                                       └─ Keyboard / joystick drive control
```

> **Critical:** `openvio_slam_node` subscribes to `/imu`. This name is fixed — do not rename it. The phone IMU is disabled at the bridge (`publish_imu: False`). `imu_publisher_node` is the sole serial port owner and sole `/imu` publisher.

---

## IMU protocol — corrected v2

> The original specification (Parts 1–3) described the wrong protocol. The authoritative ground truth is `visualizer.py` — a working Python visualizer tested against the actual Arduino hardware.

| Property | Original spec ❌ | Actual Arduino protocol ✅ |
|:---|:---:|:---:|
| Packet size | 31 bytes | **43 bytes** |
| Sensor axes | 6-DOF | **9-DOF (gyro + accel + mag)** |
| Checksum covers | bytes 2–29 | **bytes 2–41** |
| Checksum byte | index 30 | **index 42** |
| Motor command | Binary 7-byte | **ASCII `"M L R\n"`** |
| Calibration | Binary `0xCC 0x77` | **ASCII `"CAL\n"`** |
| Struct format | `<Iffffff` | **`<Ifffffffff`** |
| Mag ROS topic | not present | **`/imu/mag` MagneticField** |

Full diff with impact analysis: [`PROTOCOL_CORRECTIONS.md`](PROTOCOL_CORRECTIONS.md)

**43-byte packet layout:**

```
Bytes  0– 1:  0xAA 0x55       header
Bytes  2– 5:  uint32           Arduino millis() timestamp
Bytes  6– 9:  float            gyro_x   (rad/s)
Bytes 10–13:  float            gyro_y   (rad/s)
Bytes 14–17:  float            gyro_z   (rad/s)
Bytes 18–21:  float            accel_x  (m/s²)
Bytes 22–25:  float            accel_y  (m/s²)
Bytes 26–29:  float            accel_z  (m/s²)
Bytes 30–33:  float            mag_x    (µT)
Bytes 34–37:  float            mag_y    (µT)
Bytes 38–41:  float            mag_z    (µT)
Byte  42:     uint8            checksum  XOR of bytes 2..41
```

---

## Project structure

```
openvio_slam/
│
├── openvio_slam/                    Python package
│   ├── utils/
│   │   ├── so3.py                   SO(3): hat, vee, exp_map, log_map, Jr
│   │   └── quaternion.py            Hamilton quaternion operations
│   ├── imu/
│   │   ├── preintegrator.py         RK4 pre-integration + bias Jacobians
│   │   ├── initializer.py           Static bias init + scale alignment
│   │   └── imu_publisher_node.py    Arduino serial → /imu + /imu/mag ← NEW
│   ├── frontend/
│   │   ├── undistortion.py          Pinhole camera + undistortion
│   │   ├── orb_extractor.py         Grid-bucketed ORB detection
│   │   ├── klt_tracker.py           Bidirectional KLT + epipolar RANSAC
│   │   └── feature_manager.py       Feature lifetime + marginalization
│   ├── estimator/
│   │   ├── state.py                 Error-state vector + covariance
│   │   └── msckf.py                 Propagate · Augment · Update · ZUPT
│   ├── map/
│   │   └── triangulator.py          DLT triangulation + validity checks
│   ├── constraints/
│   │   ├── zupt.py                  Zero-velocity pseudo-measurement
│   │   └── planar.py                Planar motion (v_z ≈ 0) constraint
│   └── ros/
│       └── vio_node.py              Main ROS 2 node  ← DO NOT MODIFY
│
├── web_controller/                  Browser dashboard ← NEW
│   ├── web_controller_node.py       HTTP :8080 + WebSocket :8765
│   └── static/
│       ├── index.html               Compass · IMU charts · drive UI
│       ├── controller.js            Tilt-compensated compass + WASD
│       └── style.css                Dark industrial theme
│
├── arduino/
│   └── robot_controller/
│       └── robot_controller.ino     9-DOF firmware + ASCII motor parser ← NEW
│
├── launch/
│   └── openvio_slam.launch.py       Full pipeline, 9-node orchestration
│
├── config/
│   ├── slam.yaml                    All tunable estimator parameters
│   ├── extrinsic.yaml               T_IC: IMU-to-camera transform
│   └── openvio_slam.rviz            RViz2 display config
│
├── tools/
│   ├── test_protocol.py             Verify Arduino packets (no ROS needed)
│   ├── imu_noise_calibration.py     Allan variance from stationary log
│   ├── deploy.sh                    One-shot workspace deploy script
│   ├── build_vocabulary.py          Offline BoW vocabulary builder
│   └── evaluate_trajectory.py       ATE + RTE vs ground truth
│
└── tests/
    ├── test_math.py                 SO(3) + quaternion  ← run first
    ├── test_preintegrator.py        IMU pre-integration accuracy
    ├── test_triangulation.py        DLT triangulator edge cases
    ├── test_msckf_propagation.py    MSCKF covariance properties
    └── test_integration.py          End-to-end pipeline smoke test
```

---

## Quick start

```bash
# ── Dependencies ─────────────────────────────────────────────────────────────
sudo apt install python3-serial python3-psutil ros-jazzy-diagnostic-msgs -y
pip3 install pyserial psutil websockets --break-system-packages
sudo usermod -aG dialout $USER && newgrp dialout   # serial port access

# ── Build ─────────────────────────────────────────────────────────────────────
cd ~/slam_ws
colcon build --packages-select openvio_slam
source install/setup.bash

# ── Step 1: verify Arduino protocol (no ROS needed, run this first) ──────────
python3 ~/slam_ws/src/openvio_slam/tools/test_protocol.py /dev/ttyUSB0
# ✅ Pkt #1: ts=1234 G(+0.001,-0.000,+0.002) A(+0.04,+0.01,+9.81) M(+23.4,-5.1,+41.2)
# ✅ Rate: 99.8 Hz — Arduino protocol CORRECT

# ── Step 2: run math tests (ALL must be green before anything else) ───────────
cd ~/slam_ws/src/openvio_slam
python3 -m pytest tests/test_math.py -v

# ── Step 3: launch the full system ───────────────────────────────────────────
# IMPORTANT: never use use_sim_time:=true for live phone streaming
ros2 launch openvio_slam openvio_slam.launch.py

# ── Step 4: open browser controller ──────────────────────────────────────────
# http://<raspberry-pi-ip>:8080
echo "Controller: http://$(hostname -I | awk '{print $1}'):8080"
```

---

## Expected startup sequence

```
[rosbridge]         Rosbridge WebSocket server started on port 9090
[slam_bridge]       SLAM Bridge Node ready — Publishing: /image /camera_info
[slam_bridge]       publish_imu: False — Arduino IMU is primary source
[imu_publisher]     ✅ Serial opened: /dev/ttyUSB0
[imu_publisher]     IMU Publisher ready (9-DOF, 43-byte packets) @ 100 Hz
[openvio_slam]      ✅ Extrinsic loaded: p_IC=[0.110, -0.013, 0.146]
[openvio_slam]      ✅ Camera initialized: fx=518.5, 640×480
[openvio_slam]      Hold robot stationary 2 seconds for gyro bias init...
[web_controller]    HTTP server: http://0.0.0.0:8080
[web_controller]    WebSocket:   ws://0.0.0.0:8765
# --- hold robot still for 2 seconds ---
[openvio_slam]      ✅ VIO initialized! Scale=1.000, |v|=0.008 m/s
[openvio_slam]      Frames:50 | Avg:24ms | Feats:276 | MapPts:0  | ZUPT:no
[openvio_slam]      Frames:100| Avg:25ms | Feats:263 | MapPts:47 | ZUPT:no
[openvio_slam]      ✦ +12 pts → total: 47
```

---

## Estimator performance

| Platform | Features | Window | Frame time | ATE (10 m loop) |
|:---:|:---:|:---:|:---:|:---:|
| i7 Laptop 16 GB | 300 | 20 | ~25 ms | **< 5 cm** |
| Raspberry Pi 5 8 GB | 150 | 12 | ~80 ms | **< 10 cm** |
| Raspberry Pi 4 4 GB | 100 | 8 | ~200 ms | **< 20 cm** |

*ATE measured after one loop closure on a 10 m × 10 m indoor room.*

---

## References

| # | Paper | Role in codebase |
|:---:|:---|:---|
| 1 | Mourikis & Roumeliotis, *MSCKF*, ICRA 2007 · [PDF](https://intra.ece.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf) | Core filter — `estimator/msckf.py` |
| 2 | Forster et al., *On-Manifold Preintegration*, TRO 2017 · [arXiv](https://arxiv.org/abs/1512.02363) | IMU integration — `imu/preintegrator.py` |
| 3 | Sola et al., *Micro Lie Theory*, 2018 · [arXiv](https://arxiv.org/abs/1812.01537) | SO(3) math — `utils/so3.py` |
| 4 | Gálvez-López & Tardos, *DBoW2*, TRO 2012 | Loop closure — `loop_closure/vocabulary.py` |
| 5 | Grisetti et al., *Graph-Based SLAM*, ITSM 2010 | Pose graph — `loop_closure/pose_graph.py` |
| 6 | Rublee et al., *ORB*, ICCV 2011 | Features — `frontend/orb_extractor.py` |
| 7 | Hartley & Zisserman, *MVG* Ch.12 | Triangulation — `map/triangulator.py` |

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:091428,50:0d1f3c,100:060918&height=100&section=footer" width="100%"/>
<br/>
<sub>Built from first principles. Every equation is in the code. Every design decision is in a paper.</sub>
</div>
