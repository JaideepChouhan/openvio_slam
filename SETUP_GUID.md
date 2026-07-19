# OpenVIO-SLAM — Setup Guide

## Prerequisites

- Ubuntu 22.04 (or 24.04)
- ROS 2 Jazzy (already installed)
- Python 3.10+
- Existing `slam_ws` workspace with your `slam_bridge_node`

---

## Step 1 — Install Python Dependencies

```bash
pip3 install numpy scipy opencv-python pytest --break-system-packages
```

---

## Step 2 — Copy Package into Workspace

```bash
cp -r openvio_slam ~/slam_ws/src/
```

---

## Step 3 — Run Math Tests First (MANDATORY)

Before building, verify the math foundation is correct:

```bash
cd ~/slam_ws/src/openvio_slam
python3 -m pytest tests/test_math.py -v
```

**All tests must pass.** Expected output:
```
PASSED tests/test_math.py::TestSO3::test_hat_antisymmetric
PASSED tests/test_math.py::TestSO3::test_hat_vee_roundtrip
PASSED tests/test_math.py::TestSO3::test_hat_cross_product
PASSED tests/test_math.py::TestSO3::test_exp_map_is_SO3
PASSED tests/test_math.py::TestSO3::test_exp_log_roundtrip_small
PASSED tests/test_math.py::TestSO3::test_exp_log_roundtrip_large
PASSED tests/test_math.py::TestSO3::test_exp_identity
PASSED tests/test_math.py::TestSO3::test_exp_known_rotation_z
PASSED tests/test_math.py::TestSO3::test_right_jacobian_finite_difference
PASSED tests/test_math.py::TestQuaternion::test_multiply_identity
... (all green)
```

If any test fails: **do not proceed**. Fix the math first.

---

## Step 4 — Build

```bash
cd ~/slam_ws
colcon build --packages-select openvio_slam
source install/setup.bash
```

---

## Step 5 — Calibrate Your Camera

Edit `config/camera.yaml` with your camera's actual intrinsics.

**Option A — From ROS camera_info topic:**
```bash
ros2 topic echo /camera_info --once
```
Copy `k` matrix values: `fx=k[0], fy=k[4], cx=k[2], cy=k[5]`
Copy `d` array for distortion.

**Option B — Run calibration:**
```bash
ros2 run camera_calibration cameracalibrator \
  --size 9x6 --square 0.025 \
  image:=/image camera:=/camera
```

---

## Step 6 — Set IMU-Camera Extrinsics

The transform between IMU and camera frames (`T_IC`) must be known.

**If IMU and camera are in the same device (phone):**
For most phones, the IMU is approximately aligned with the camera.
Start with identity (already set as default) and tune if drift is observed.

**For precise calibration** (recommended for good accuracy):
Use Kalibr: https://github.com/ethz-asl/kalibr

Copy the resulting `R_IC` and `p_IC` into `config/imu.yaml`.

---

## Step 7 — Tune IMU Noise Parameters

The default values in `config/imu.yaml` are typical Android phone values.
For better accuracy, calibrate them:

```bash
# Record 10 minutes of IMU data with the robot COMPLETELY STILL
# Then run Allan variance analysis
python3 tools/imu_noise_calibration.py /path/to/imu_bag.db3
```

Update `config/imu.yaml` with the calibrated values.

---

## Step 8 — Launch

```bash
ros2 launch openvio_slam openvio_slam.launch.py
```

**Expected startup sequence:**
```
[openvio_slam] OpenVIO-SLAM starting...
[openvio_slam] Waiting for /camera_info...
[openvio_slam] Camera initialized: fx=518.5, 640x480
[openvio_slam] Hold robot stationary 2 seconds for bias init...
# --- hold still for 2 seconds ---
[openvio_slam] VIO initialized! Scale=1.0, |v|=0.01m/s
[openvio_slam] Frames: 10 | Avg: 22.3ms | Features: 247 | Pos: [0.00, 0.00, 0.00]m
```

---

## Step 9 — Verify in RViz2

Topics to add in RViz:
| Topic | Type | What you see |
|---|---|---|
| `/odom` | Odometry | Robot position trail |
| `/openvio/map_points` | PointCloud2 | Sparse keyframe positions |
| `/tf` | TF | Coordinate frames |

**Sanity checks:**
- Features should stay stable on stationary objects (not slide)
- ZUPT should fire (show "YES") when robot pauses
- Position should return near origin after a short loop

---

## Tuning Guide

If trajectory drifts, adjust in this order:

### 1. IMU noise too wrong → propagation error dominates
Symptom: drift during straight-line motion even before features update
Fix: increase `gyro_noise` and `accel_noise` slightly in `config/imu.yaml`

### 2. Pixel noise misconfigured → visual update rejected
Symptom: chi-squared test rejects most features, "Features: 0 marginalized"
Fix: increase `pixel_noise` in `config/slam.yaml` to 1.5 or 2.0

### 3. ZUPT not firing when robot stops
Symptom: drift during pauses
Fix: increase `zupt_gyro_var_threshold` in `config/slam.yaml` (try 0.005)

### 4. ZUPT firing during slow motion (locking position)
Symptom: position freezes during slow movement
Fix: decrease `zupt_gyro_var_threshold` (try 0.0005)

### 5. Too few features tracked
Symptom: "Features: <50" warning, high drift
Fix: decrease `fast_threshold` (try 7), or decrease `min_features` threshold

### 6. CPU too slow (avg > 100ms)
Fix: reduce `n_features` to 150, reduce `window_size` to 10

---

## File Structure Reference

```
openvio_slam/
├── openvio_slam/
│   ├── utils/
│   │   ├── so3.py              # SO(3) Lie group operations
│   │   └── quaternion.py       # Hamilton quaternion math
│   ├── imu/
│   │   ├── preintegrator.py    # IMU pre-integration (RK4)
│   │   └── initializer.py      # Static + dynamic VIO init
│   ├── frontend/
│   │   ├── undistortion.py     # Camera model + normalization
│   │   ├── orb_extractor.py    # Grid-bucketed ORB detection
│   │   ├── klt_tracker.py      # KLT + FB check + RANSAC
│   │   └── feature_manager.py  # Feature lifetime management
│   ├── estimator/
│   │   ├── state.py            # VIO state + covariance
│   │   └── msckf.py            # MSCKF core filter
│   ├── map/
│   │   └── triangulator.py     # DLT triangulation
│   ├── constraints/
│   │   ├── zupt.py             # Zero-velocity updates
│   │   └── planar.py           # Planar motion constraint
│   └── ros/
│       └── vio_node.py         # ROS 2 node (main entry point)
├── tests/
│   └── test_math.py            # SO(3) + quaternion tests (run first!)
├── config/
│   ├── camera.yaml             # Camera intrinsics (EDIT THIS)
│   ├── imu.yaml                # IMU noise + extrinsics (EDIT THIS)
│   └── slam.yaml               # Tuning parameters
├── launch/
│   └── openvio_slam.launch.py
├── CMakeLists.txt
├── package.xml
├── setup.py
└── requirements.txt
```

---

## Common Issues

**"ModuleNotFoundError: No module named 'openvio_slam'"**
```bash
source ~/slam_ws/install/setup.bash
```

**"Camera model initialized" but "VIO initialized" never appears**
The dynamic initialization needs 10 frames of motion with >5° parallax.
Move the robot forward at least 0.5m while the camera moves laterally.

**"chi-squared test rejects all features"**
Your camera intrinsics may be wrong. Double-check `fx, fy, cx, cy` in `camera.yaml`.

**ZUPT fires continuously even when robot is moving**
Threshold is too high. Decrease `zupt_gyro_var_threshold` to `0.0002`.

**Covariance matrix becomes singular (numerical error)**
Call `state.symmetrize_covariance()` more frequently (every 5 frames instead of 10).
