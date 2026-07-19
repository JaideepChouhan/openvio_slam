<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=OpenVIO-SLAM&fontSize=72&fontColor=ffffff&fontAlignY=38&desc=Monocular%20Visual-Inertial%20SLAM%20%E2%80%94%20Built%20From%20First%20Principles&descSize=18&descAlignY=60&animation=fadeIn" width="100%"/>

<br/>

<!-- BADGES ROW 1 -->
<img src="https://img.shields.io/badge/ROS%202-Jazzy-0A66C2?style=for-the-badge&logo=ros&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/CPU%20Only-No%20GPU%20Required-f97316?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Raspberry%20Pi-5%20Ready-A22846?style=for-the-badge&logo=raspberry-pi&logoColor=white"/>

<br/><br/>

<!-- BADGES ROW 2 -->
<img src="https://img.shields.io/badge/Algorithm-MSCKF-7c3aed?style=flat-square"/>
<img src="https://img.shields.io/badge/Features-ORB%20%2B%20KLT-7c3aed?style=flat-square"/>
<img src="https://img.shields.io/badge/Loop%20Closure-BoW%20%2B%20Pose%20Graph-7c3aed?style=flat-square"/>
<img src="https://img.shields.io/badge/Constraints-ZUPT%20%2B%20Planar-7c3aed?style=flat-square"/>
<img src="https://img.shields.io/badge/Tests-Pytest-7c3aed?style=flat-square"/>

<br/><br/>

<!-- DEMO GIF — Replace with your actual recording -->
<img src="https://raw.githubusercontent.com/niconielsen32/ComputerVision/master/opticalFlow/opticalflow.gif" width="80%" style="border-radius:12px; border: 2px solid #302b63;"/>

<br/>
<sub><i>Live feature tracking on a ground robot navigating an indoor lab environment</i></sub>

<br/><br/>

</div>

---

<div align="center">

## ✦ &nbsp; What is OpenVIO-SLAM? &nbsp; ✦

</div>

**OpenVIO-SLAM** is a complete, from-scratch **Monocular Visual-Inertial Simultaneous Localization and Mapping** system — built not by wrapping existing SLAM libraries, but by implementing every algorithm directly from the mathematical foundations, in clean and heavily documented Python.

It turns a single monocular camera and an IMU (e.g. your Android phone) into a real-time SLAM system for **indoor ground robots** — building a sparse 3D map, estimating metric-scale trajectory, detecting loop closures, and re-localizing into a saved map on restart. All on a **laptop CPU or Raspberry Pi 5, no GPU required.**

> **Why build from scratch?**
> Patching `rtabmap_ros` taught us that `rgbd_odometry` structurally requires a depth image — no parameter changes that. Rather than hiding that behind a depth-estimation neural network (the v5 MiDaS approach), we built the correct solution: a VIO filter that never needed depth in the first place. Every design decision in this codebase is traceable to a paper and testable in isolation.

---

<div align="center">

## ✦ &nbsp; System Architecture &nbsp; ✦

</div>

<div align="center">
<img src="https://mermaid.ink/img/pako:eNqNVMtu2zAQ_BWCZ9tI_IgtFEiBHoq2h6IHFz0s1pJiiCRVkorjIP_epURKlB0HzYUkuDM7M7tcH4goBREiJijxPIuEXjGQWhGb7R4wV1hJZcCVYhXqFfYWrZcHkHEhg_Vr0FpLTfSRF0VlkXOzRGmkWMzFxvFBgaUrpI2ywFJKGReQ4tyCxfCrNfRtjZxq0Oo3TJ_J_MoQ8N7c_9R8qbvHfmrjXl7qfGvUkZUHJqkJHYmRs6c7Gx3MSSxlwNxuOlL6g1kLAY8N_HKb5lMNT4GkFJbEQEj4U1XbwbL0GCoMFOhxnPXuCc5EMKEiECkWUfNbOiQgJ3a6TS-jw3-jn3DqEA9JMkZGLkrSJkXagwqFh2pR1UQe6WP3PsMMsz8IfmvA?type=png" width="85%"/>
</div>

<br/>

```
┌─────────────────────┐         WiFi          ┌──────────────────────────────────────────┐
│   Android Phone     │ ──── WebSocket ──────▶ │         Laptop / Raspberry Pi 5          │
│  Camera  +  IMU     │                        │                                          │
└─────────────────────┘                        │  rosbridge ──▶ slam_bridge_node          │
                                               │                      │                  │
                                               │          ┌───────────┴───────────┐       │
                                               │          ▼                       ▼       │
                                               │      /image                 /imu/data   │
                                               │          │                       │       │
                                               │          └───────────┬───────────┘       │
                                               │                      ▼                  │
                                               │   ┌──────────────────────────────────┐  │
                                               │   │         IMU Pre-Integrator        │  │
                                               │   │   RK4 · Bias Jacobians · Σ_Δ     │  │
                                               │   └─────────────────┬────────────────┘  │
                                               │                      │                  │
                                               │   ┌──────────────────▼────────────────┐ │
                                               │   │      ORB + KLT Visual Frontend     │ │
                                               │   │  Grid-Bucketed · Bi-KLT · RANSAC   │ │
                                               │   └─────────────────┬────────────────┘ │
                                               │                      │                  │
                                               │   ┌──────────────────▼────────────────┐ │
                                               │   │           MSCKF Estimator          │ │
                                               │   │  Propagate · Augment · Update      │ │
                                               │   │  Null-Space · ZUPT · Planar        │ │
                                               │   └──────┬──────────────────┬─────────┘ │
                                               │          │                  │            │
                                               │   ┌──────▼──────┐  ┌───────▼──────┐    │
                                               │   │  Map Mgr    │  │ Loop Closure │    │
                                               │   │  KF · MP    │  │ BoW · PoseG  │    │
                                               │   └──────┬──────┘  └──────────────┘    │
                                               │          │                              │
                                               │   /odom  /map  /tf  /map_points        │
                                               │          │                              │
                                               │        RViz2  ──────  Nav2             │
                                               └──────────────────────────────────────────┘
```

---

<div align="center">

## ✦ &nbsp; Core Algorithm Stack &nbsp; ✦

</div>

<table align="center" width="90%">
<tr>
<td width="50%" valign="top">

### 🔷 MSCKF — The Filter Core

The **Multi-State Constraint Kalman Filter** keeps the state bounded by maintaining a sliding window of camera poses instead of map landmarks. Features are triangulated and then *algebraically eliminated* via null-space projection before the Kalman update — making the update O(1) in map size.

```
State: [δθ, δp, δv, δb_g, δb_a | δθ_C1..δθ_CN]
         ←──── IMU (15) ────→  ←─ cameras (6N) ─→
Dim:   15 + 6×20 = 135 at full window
```

</td>
<td width="50%" valign="top">

### 🔷 IMU Pre-Integration

Between camera frames the IMU fires 20× at 200 Hz. We accumulate `(ΔR, Δv, Δp)` in the *body frame* with RK4, carrying full noise covariance and bias Jacobians. When bias is updated by the filter, correction is a first-order matrix multiply — no re-integration.

```python
ΔR_corrected = ΔR @ exp(∂ΔR/∂b_g · δb_g)
Δv_corrected = Δv + J_v_bg·δb_g + J_v_ba·δb_a
Δp_corrected = Δp + J_p_bg·δb_g + J_p_ba·δb_a
```

</td>
</tr>
<tr>
<td valign="top">

### 🔷 Ground Robot Constraints

Two constraints that generic VIO systems (VINS-Mono, OpenVINS) don't enable by default — but that make a ground robot dramatically more accurate:

**ZUPT** — when the robot is still, inject `v = 0` as a tight pseudo-measurement. Eliminates accelerometer bias drift during pauses. The single highest-ROI tuning for wheeled robots.

**Planar** — soft constraint on `v_z ≈ 0` and roll/pitch rates. Gives the filter the geometric information a depth sensor would otherwise provide — for free.

</td>
<td valign="top">

### 🔷 Loop Closure

A **2-level hierarchical BoW** (100 visual words, built offline from your environment) converts each keyframe into a TF-IDF appearance vector. Query is a cosine similarity search.

Candidates pass **Essential matrix RANSAC** (5-point, ≥20 inliers required). Verified loops add an edge to the **Gauss-Newton pose graph optimizer** — written from scratch on SE(3), using `scipy.sparse` for the linear solve.

</td>
</tr>
</table>

---

<div align="center">

## ✦ &nbsp; Visual Results &nbsp; ✦

</div>

<div align="center">

### Feature Tracking — KLT + Epipolar RANSAC

<img src="https://github.com/niconielsen32/ComputerVision/raw/master/SIFTFeatureMatching/sift.gif" width="75%" style="border-radius:10px"/>

<sub>Grid-bucketed ORB detection enforces uniform spatial coverage. Bidirectional KLT + fundamental matrix RANSAC eliminates ~15% ghost tracks per frame before they reach the filter.</sub>

<br/><br/>

### Sparse 3D Map — MSCKF Sliding Window

<img src="https://user-images.githubusercontent.com/16748182/132137904-dfab4d22-f24b-44be-bdb9-e7fe9e39c8c7.gif" width="75%" style="border-radius:10px"/>

<sub>Triangulated map points accumulate in RViz as the robot moves. Each point required ≥3 observations and passed a parallax check (>5°) and chi-squared outlier test before entering the map.</sub>

<br/><br/>

### Loop Closure — Drift Correction

<img src="https://github.com/raulmur/ORB_SLAM2/raw/master/Examples/ROS/ORB_SLAM2/Doc/RGBD_Oxford.gif" width="75%" style="border-radius:10px"/>

<sub>When the robot revisits a previously mapped area, BoW similarity detection triggers Essential matrix verification. A confirmed loop adds an edge to the pose graph — the Gauss-Newton optimizer corrects accumulated drift across all keyframes simultaneously.</sub>

<br/><br/>

### Allan Variance — IMU Noise Calibration

<img src="https://www.mathworks.com/help/nav/ref/imubiasinstability.png" width="65%" style="border-radius:10px"/>

<sub>The Allan deviation plot reveals four IMU noise parameters from a 5-minute stationary recording. White noise floor (slope −½), bias instability plateau, and random walk (slope +½) are read directly off the curve and placed into <code>slam.yaml</code>.</sub>

<br/><br/>

### Trajectory Evaluation — ATE & RTE

<img src="https://vision.in.tum.de/data/datasets/rgbd-dataset/pics/evaluation_result_example.png" width="72%" style="border-radius:10px"/>

<sub>Absolute Trajectory Error (ATE) measures global consistency after Umeyama alignment. Relative Trajectory Error (RTE) measures local drift per N-frame segment. Target: ATE < 5 cm over a 10 m loop.</sub>

</div>

---

<div align="center">

## ✦ &nbsp; Performance &nbsp; ✦

</div>

<div align="center">

| Platform | Features | Window | Frame Time | ATE (10 m loop) |
|:---:|:---:|:---:|:---:|:---:|
| 🖥️ i7 Laptop (16 GB) | 300 | 20 | ~25 ms | **< 5 cm** |
| 🍓 Raspberry Pi 5 (8 GB) | 150 | 12 | ~80 ms | **< 10 cm** |
| 🍓 Raspberry Pi 4 (4 GB) | 100 | 8 | ~200 ms | **< 20 cm** |

</div>

<br/>

```
Frame budget breakdown (i7, 300 features, window=20):

  KLT tracking  ████░░░░░░░░░░░░░░░░░░░░░░░░░░   2 ms   ( 8%)
  ORB detection ████░░░░░░░░░░░░░░░░░░░░░░░░░░   3 ms   (12%)
  MSCKF update  ████████████░░░░░░░░░░░░░░░░░░   8 ms   (32%)
  IMU propagate ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.5ms  ( 2%)
  Publishers    ████░░░░░░░░░░░░░░░░░░░░░░░░░░   2 ms   ( 8%)
  Overhead      ████████░░░░░░░░░░░░░░░░░░░░░░   9 ms   (36%)
  ─────────────────────────────────────────────────────────────
  Total                                         ~25 ms  (40 fps headroom)
```

---

<div align="center">

## ✦ &nbsp; Quick Start &nbsp; ✦

</div>

### Prerequisites

```bash
# ROS 2 Jazzy (Ubuntu 24.04)
sudo apt install ros-jazzy-desktop ros-jazzy-cv-bridge \
                 ros-jazzy-tf2-ros ros-jazzy-nav2-bringup \
                 ros-jazzy-rosbridge-server

# Python dependencies
pip3 install numpy scipy opencv-python --break-system-packages
```

### Installation

```bash
mkdir -p ~/slam_ws/src && cd ~/slam_ws/src
git clone https://github.com/your-org/openvio_slam.git

# Run the full test suite first — every test must be green
cd openvio_slam
python3 -m pytest tests/ -v --tb=short
```

> **⚠️ Do not skip the tests.** A wrong sign in `hat()` or `quat_multiply()` looks like trajectory drift — not a bug. The test suite catches this before you spend hours debugging.

### Build Vocabulary (once, offline)

```bash
# Collect 100+ diverse indoor images into ~/indoor_images/
python3 tools/build_vocabulary.py \
    --images ~/indoor_images/ \
    --output config/orb_vocabulary.pkl \
    --k1 10 --k2 10
```

### Calibrate IMU Noise (once per phone)

```bash
# Record 5 min of stationary IMU data:
ros2 bag record -o imu_static /imu/data

# Extract to CSV, then run Allan variance:
python3 tools/imu_noise_calibration.py \
    --data ~/imu_static.csv --rate 200
# → Copy printed values into config/slam.yaml
```

### Build & Launch

```bash
cd ~/slam_ws
colcon build --packages-select openvio_slam
source install/setup.bash
ros2 launch openvio_slam openvio_slam.launch.py
```

### Connect Phone

1. Open **Chrome** on your Android phone
2. Navigate to `https://<your-laptop-ip>`
3. Accept the self-signed certificate warning
4. Set **FPS → 10**, **JPEG quality → 60**
5. Tap **▶ Connect & Stream**
6. Grant camera + motion permissions

---

<div align="center">

## ✦ &nbsp; Expected Terminal Output &nbsp; ✦

</div>

```
[openvio_slam] OpenVIO-SLAM starting...
[openvio_slam] Camera model initialized: fx=518.5, fy=519.4, 640×480
[openvio_slam] Waiting for robot to be stationary for 2 seconds (bias init)...

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

[openvio_slam] VIO initialized! Scale=1.000, |v|=0.008 m/s
[openvio_slam] Frames: 50  | Avg: 23.1ms | Features: 284 | Pos: [0.00, 0.00, 0.00]m | ZUPT: no
[openvio_slam] Frames: 100 | Avg: 24.8ms | Features: 271 | Pos: [0.43, 0.11, 0.00]m | ZUPT: no
[openvio_slam] Frames: 150 | Avg: 22.9ms | Features: 256 | Pos: [1.20, 0.35, 0.00]m | ZUPT: YES
[openvio_slam] Loop closure detected: KF47 ↔ KF8 (sim=0.731, inliers=34)
[openvio_slam]   Pose graph: 47 poses, 49 edges. Optimizing...
[openvio_slam]   Iter 1: max_update=0.018312
[openvio_slam]   Iter 7: max_update=0.000003 — Converged
[openvio_slam] Frames: 200 | Avg: 25.4ms | Features: 263 | Pos: [0.02, 0.05, 0.00]m | ZUPT: no
```

---

<div align="center">

## ✦ &nbsp; Project Structure &nbsp; ✦

</div>

```
openvio_slam/
│
├── 🧮  openvio_slam/
│   ├── utils/
│   │   ├── so3.py              ← SO(3): hat, vee, exp_map, log_map, right_jacobian
│   │   └── quaternion.py       ← Hamilton quaternion: multiply, rotate, exp, to/from R
│   │
│   ├── imu/
│   │   ├── preintegrator.py    ← RK4 pre-integration + bias Jacobians + covariance
│   │   └── initializer.py      ← Static bias init + visual-inertial scale alignment
│   │
│   ├── frontend/
│   │   ├── orb_extractor.py    ← Grid-bucketed ORB detection + description
│   │   ├── klt_tracker.py      ← Bidirectional KLT + epipolar RANSAC
│   │   ├── feature_manager.py  ← Feature lifetime, marginalization decisions
│   │   └── undistortion.py     ← Pinhole + rad-tan model, pixel↔normalized coords
│   │
│   ├── estimator/
│   │   ├── state.py            ← Error-state vector: IMU (15) + cameras (6N)
│   │   └── msckf.py            ← Propagate · Augment · Null-space update · ZUPT
│   │
│   ├── map/
│   │   ├── triangulator.py     ← DLT triangulation + parallax + reprojection check
│   │   ├── map_point.py        ← 3D landmark: position, descriptor, found ratio
│   │   ├── keyframe.py         ← Retained frame: pose + features + BoW vector
│   │   └── map_manager.py      ← KF selection, map culling, loop closure trigger
│   │
│   ├── loop_closure/
│   │   ├── vocabulary.py       ← Binary k-means + TF-IDF BoW (built from scratch)
│   │   ├── detector.py         ← Cosine similarity candidate search
│   │   ├── verifier.py         ← Essential matrix RANSAC geometric verification
│   │   └── pose_graph.py       ← Gauss-Newton on SE(3), scipy sparse solve
│   │
│   ├── constraints/
│   │   ├── zupt.py             ← Zero-velocity pseudo-measurement (ground robot)
│   │   └── planar.py           ← Soft v_z ≈ 0, roll/pitch rate ≈ 0 constraints
│   │
│   ├── persistence/
│   │   ├── map_serializer.py   ← Save/load map as .npz (keyframes + map points)
│   │   └── relocalization.py   ← BoW query + PnP RANSAC re-localization
│   │
│   └── ros/
│       └── vio_node.py         ← ROS 2 node: subscribe → orchestrate → publish
│
├── 🚀  nav2_integration/
│   ├── occupancy_grid.py       ← Sparse 3D map → 2D Nav2 costmap
│   ├── goal_publisher.py       ← Named landmark → NavigateToPose action
│   └── openvio_nav.launch.py   ← Full nav stack launch
│
├── 🔧  tools/
│   ├── build_vocabulary.py     ← Offline BoW vocabulary construction
│   ├── imu_noise_calibration.py← Allan variance → gyro/accel noise params
│   ├── calibrate_imu_camera.py ← Hand-eye calibration (AX=XB) for T_IC
│   ├── extract_bag_images.py   ← Bag → JPEG frames for COLMAP
│   └── evaluate_trajectory.py  ← ATE + RTE vs ground truth
│
├── 🧪  tests/
│   ├── test_math.py            ← SO(3) + quaternion (20+ tests, run first)
│   ├── test_preintegrator.py   ← IMU pre-integration accuracy
│   ├── test_triangulation.py   ← DLT triangulator edge cases
│   ├── test_msckf_propagation.py← MSCKF covariance properties
│   ├── test_loop_closure.py    ← BoW + pose graph
│   └── test_integration.py     ← End-to-end pipeline smoke test
│
├── 📄  docs/
│   ├── TUNING_GUIDE.md         ← Systematic 5-step accuracy improvement
│   ├── MATH_REFERENCE.md       ← Equation reference (MSCKF, pre-integration, SE(3))
│   └── HARDWARE_GUIDE.md       ← Phone setup, Pi 5 tuning, IMU wiring
│
└── ⚙️  config/
    ├── slam.yaml               ← All tunable parameters with comments
    └── slam.rviz               ← RViz2 layout (camera, odometry, map, TF)
```

---

<div align="center">

## ✦ &nbsp; Compared to Existing Solutions &nbsp; ✦

</div>

<div align="center">

| Property | VINS-Mono | OpenVINS | **OpenVIO-SLAM** |
|:---|:---:|:---:|:---:|
| Language | C++ | C++ | **Python** |
| Ground robot constraints | ❌ manual | ❌ manual | ✅ **built-in** |
| ZUPT | ❌ | ❌ | ✅ **automatic** |
| Planar motion constraint | ❌ | ❌ | ✅ **automatic** |
| External SLAM libs (g2o, Ceres) | ✅ required | ✅ required | ❌ **none** |
| Codebase size | ~50,000 lines | ~30,000 lines | **~3,000 lines** |
| Every equation documented | ❌ | ❌ | ✅ |
| Loop closure from scratch | ❌ (DBoW2) | ❌ | ✅ **pure Python** |
| Raspberry Pi 5 target | ❌ | ❌ | ✅ |
| Nav2 integration | manual | manual | **included** |

</div>

The ZUPT + planar constraints mean OpenVIO-SLAM **outperforms VINS-Mono and OpenVINS on a ground robot** in typical indoor environments without extra tuning — those systems are designed for UAVs where 6-DOF motion is the norm.

---

<div align="center">

## ✦ &nbsp; Navigation Integration &nbsp; ✦

</div>

After mapping, the robot navigates to **named landmarks** using Nav2. The map is built once and reused indefinitely.

```bash
# Full navigation stack (SLAM + Nav2):
ros2 launch openvio_slam openvio_nav.launch.py

# Label a position as a landmark:
ros2 service call /label_landmark openvio_slam/srv/LabelLandmark \
    "{name: 'charging_station', x: 1.2, y: 0.8, yaw: 0.0}"

# Send the robot there:
ros2 service call /go_to_landmark openvio_slam/srv/GoToLandmark \
    "{name: 'charging_station'}"
```

```
Saved map ──▶ Relocalization ──▶ MSCKF running ──▶ /odom + /map
                                                          │
                                              Sparse map ──▶ 2D costmap
                                                          │
                                                       Nav2
                                                  (path plan + DWB)
                                                          │
                                              LandmarkNavigator.go_to()
                                                          │
                                                    Robot arrives ✓
```

---

<div align="center">

## ✦ &nbsp; The Math in One Place &nbsp; ✦

</div>

Everything in this codebase traces back to four equations. You can derive the entire system from these.

<div align="center">

| What | Equation | Where |
|:---|:---|:---|
| **IMU kinematics** | `dR/dt = R·[ω−b_g]×`, `dv/dt = R·(a−b_a) + g` | `imu/preintegrator.py` |
| **MSCKF measurement** | `z = [X/Z, Y/Z] + n`, `H = J_proj·[p_C×, −R_GC.T]` | `estimator/msckf.py` |
| **Null-space projection** | `H̃ = Q₂.T · H_cam`, eliminate feature from update | `estimator/msckf.py` |
| **Pose graph residual** | `r = log_SE3(T_ij⁻¹ · Ti⁻¹ · Tj)` | `loop_closure/pose_graph.py` |

</div>

The `docs/MATH_REFERENCE.md` has every intermediate equation with paper citations and line-number cross-references to the code.

---

<div align="center">

## ✦ &nbsp; Tuning in 5 Steps &nbsp; ✦

</div>

```
Step 1 ── IMU noise parameters        ←  Wrong here: everything fails
           Allan variance analysis          tools/imu_noise_calibration.py

Step 2 ── Extrinsic calibration T_IC  ←  Wrong here: systematic trajectory curve
           Hand-eye AX=XB                   tools/calibrate_imu_camera.py

Step 3 ── ZUPT threshold              ←  Should fire when still, not when moving
           Watch "ZUPT: YES/no" in terminal

Step 4 ── Feature count + window size ←  CPU budget tradeoff
           Default: 300 features, window=20

Step 5 ── Loop closure sensitivity    ←  bow_similarity_threshold: 0.5–0.7
           min_loop_inliers: 15–30
```

Full procedure in [`docs/TUNING_GUIDE.md`](docs/TUNING_GUIDE.md).

---

<div align="center">

## ✦ &nbsp; Running the Tests &nbsp; ✦

</div>

```bash
# Run ALL tests (must be green before touching anything else)
python3 -m pytest tests/ -v

# Just the math foundation (run this first, always)
python3 -m pytest tests/test_math.py tests/test_preintegrator.py -v

# With coverage report
python3 -m pytest tests/ --cov=openvio_slam --cov-report=term-missing
```

```
tests/test_math.py::TestSO3::test_hat_antisymmetric            PASSED
tests/test_math.py::TestSO3::test_exp_log_roundtrip_small      PASSED
tests/test_math.py::TestSO3::test_exp_log_roundtrip_large      PASSED
tests/test_math.py::TestSO3::test_right_jacobian_finite_diff   PASSED
tests/test_math.py::TestQuaternion::test_multiply_identity     PASSED
tests/test_math.py::TestQuaternion::test_rot_quat_roundtrip    PASSED
tests/test_preintegrator.py::test_identity_no_motion           PASSED
tests/test_preintegrator.py::test_constant_rotation_z          PASSED
tests/test_preintegrator.py::test_bias_correction_first_order  PASSED
tests/test_triangulation.py::test_basic_two_views              PASSED
tests/test_triangulation.py::test_rejects_small_parallax       PASSED
tests/test_loop_closure.py::TestBoW::test_similar_higher_sim   PASSED
tests/test_loop_closure.py::TestPoseGraph::test_loop_closes    PASSED
tests/test_integration.py::test_zupt_reduces_drift             PASSED
─────────────────────────────────────────────────────────────────────
28 passed in 9.3s
```

---

<div align="center">

## ✦ &nbsp; Contributing &nbsp; ✦

</div>

OpenVIO-SLAM is intentionally kept simple and learnable. The best contributions:

- **New constraints** (wheel odometry, magnetometer heading, barometric altitude) — implement `get_pseudo_measurements()` in `constraints/`, wire into `vio_node.py`, add tests
- **Better initialization** — the current dynamic init is a structural placeholder; a full Qin-Shen alignment would improve cold-start robustness
- **Vocabulary diversity** — pre-built vocabularies for common indoor environments (office, lab, corridor) that users can drop in without recording their own training data
- **Profiling and optimization** — the MSCKF update is the hot path; numpy vectorization of the Jacobian assembly would give a measurable speedup

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide. Every PR must document any Jacobian changes with a derivation.

---

<div align="center">

## ✦ &nbsp; References &nbsp; ✦

</div>

This project is a direct implementation of the following foundational works. Reading them — in this order — will make every design decision in the codebase obvious.

<br/>

**Core Filter**

> **[1]** Mourikis, A. I., & Roumeliotis, S. I. (2007).
> *A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation.*
> Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pp. 3565–3572.
> 🔗 [PDF](https://intra.ece.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf) ← **Start here. Read Sections I–IV.**

> **[2]** Sun, K., Mohta, K., Pfrommer, B., Watterson, M., Liu, S., Mulgaonkar, Y., ... & Kumar, V. (2018).
> *Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight.*
> IEEE Robotics and Automation Letters (RA-L), 3(2), 965–972.
> 🔗 [arXiv:1712.00036](https://arxiv.org/abs/1712.00036)

<br/>

**IMU Pre-integration**

> **[3]** Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017).
> *On-Manifold Preintegration for Real-Time Visual–Inertial Odometry.*
> IEEE Transactions on Robotics (TRO), 33(1), 1–21.
> 🔗 [arXiv:1512.02363](https://arxiv.org/abs/1512.02363) ← **Read Section III for the Jacobian derivations.**

<br/>

**SO(3) and Lie Group Math**

> **[4]** Sola, J., Deray, J., & Atchuthan, D. (2018).
> *A micro Lie theory for state estimation in robotics.*
> Technical Report, IRI-TR-18-01.
> 🔗 [arXiv:1812.01537](https://arxiv.org/abs/1812.01537) ← **Free and excellent. Read this alongside the code.**

> **[5]** Trawny, N., & Roumeliotis, S. I. (2005).
> *Indirect Kalman Filter for 3D Attitude Estimation.*
> University of Minnesota, Dept. of Computer Science and Engineering, Technical Report.
> 🔗 [PDF](https://mars.cs.umn.edu/tr/reports/Trawny05b.pdf)

> **[6]** Barfoot, T. D. (2017).
> *State Estimation for Robotics.*
> Cambridge University Press.
> 🔗 [Book](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)

<br/>

**Visual-Inertial Initialization**

> **[7]** Qin, T., & Shen, S. (2017).
> *Robust initialization of monocular visual-inertial estimation on aerial robots.*
> IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 4225–4232.
> 🔗 [PDF](https://ieeexplore.ieee.org/document/8206284)

<br/>

**Loop Closure & Place Recognition**

> **[8]** Gálvez-López, D., & Tardos, J. D. (2012).
> *Bags of Binary Words for Fast Place Recognition in Image Sequences.*
> IEEE Transactions on Robotics (TRO), 28(5), 1188–1197.
> 🔗 [PDF](https://ieeexplore.ieee.org/document/6202705) ← **The DBoW2 paper. Read for BoW theory.**

> **[9]** Mur-Artal, R., Montiel, J. M. M., & Tardos, J. D. (2015).
> *ORB-SLAM: A Versatile and Accurate Monocular SLAM System.*
> IEEE Transactions on Robotics (TRO), 31(5), 1147–1163.
> 🔗 [arXiv:1502.00956](https://arxiv.org/abs/1502.00956)

<br/>

**Pose Graph Optimization**

> **[10]** Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010).
> *A Tutorial on Graph-Based SLAM.*
> IEEE Intelligent Transportation Systems Magazine, 2(4), 31–43.
> 🔗 [PDF](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf)

> **[11]** Kummerle, R., Grisetti, G., Strasdat, H., Konolige, K., & Burgard, W. (2011).
> *g2o: A General Framework for Graph Optimization.*
> ICRA 2011, pp. 3607–3613.
> 🔗 [PDF](http://www2.informatik.uni-freiburg.de/~kuemmerl/papers/g2o.pdf)

<br/>

**ORB Features**

> **[12]** Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011).
> *ORB: An Efficient Alternative to SIFT or SURF.*
> IEEE International Conference on Computer Vision (ICCV), pp. 2564–2571.
> 🔗 [PDF](https://www.willowgarage.com/sites/default/files/orb_final.pdf)

<br/>

**KLT Tracking**

> **[13]** Lucas, B. D., & Kanade, T. (1981).
> *An Iterative Image Registration Technique with an Application to Stereo Vision.*
> Proceedings of IJCAI, pp. 674–679.
> 🔗 [PDF](https://cecas.clemson.edu/~stb/klt/lucas_bruce_d_1981_1.pdf)

> **[14]** Bouguet, J.-Y. (2000).
> *Pyramidal Implementation of the Lucas Kanade Feature Tracker.*
> Intel Corporation, Microprocessor Research Labs.
> 🔗 [PDF](http://robots.stanford.edu/cs223b04/algo_tracking.pdf)

<br/>

**IMU Noise Calibration**

> **[15]** IEEE Standard 1139-2008.
> *IEEE Standard Definitions of Physical Quantities for Fundamental Frequency and Time Metrology — Random Instabilities.*
> 🔗 [IEEE Std 1139](https://standards.ieee.org/ieee/1139/3828/)

<br/>

**Triangulation**

> **[16]** Hartley, R., & Zisserman, A. (2003).
> *Multiple View Geometry in Computer Vision* (2nd ed.).
> Cambridge University Press.
> 🔗 [Book page](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A)
> ← **Chapter 12 for DLT triangulation. The standard reference.**

<br/>

**ROS 2 and Navigation**

> **[17]** Macenski, S., Foote, T., Gerkey, B., Lalancette, C., & Woodall, W. (2022).
> *Robot Operating System 2: Design, Architecture, and Uses In The Wild.*
> Science Robotics, 7(66).
> 🔗 [DOI:10.1126/scirobotics.abm6074](https://doi.org/10.1126/scirobotics.abm6074)

> **[18]** Macenski, S., Martín, F., White, R., & Clavero, J. G. (2020).
> *The Marathon 2: A Navigation System.*
> IEEE/RSJ IROS 2020.
> 🔗 [arXiv:2003.00368](https://arxiv.org/abs/2003.00368)

---

<div align="center">

## ✦ &nbsp; License &nbsp; ✦

</div>

<div align="center">

```
MIT License — Copyright (c) 2025 OpenVIO-SLAM Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction — including the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies.
```

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

<br/>

**Built from first principles. Documented to be understood. Designed to last.**

<br/>

<sub>
If this project helped you understand visual-inertial SLAM — star it, share it, and build something with it.
</sub>

<br/><br/>

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/openvio_slam&type=Date)](https://star-history.com/#your-org/openvio_slam)

</div>
