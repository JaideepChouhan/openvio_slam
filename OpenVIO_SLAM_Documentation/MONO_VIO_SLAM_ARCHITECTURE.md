# OpenVIO-SLAM — Architecture & Design Document
### A From-Scratch Monocular Visual-Inertial SLAM System for Indoor Ground Robots
**Version 1.0 — Design Freeze**

---

## Preamble: Why We Are Building This

RTABMap's `rgbd_odometry` requires a depth image — structurally, by design, forever.
Every "fix" from v2 through v5 was fighting that wall. The MiDaS approach in v5 *works*,
but it replaces one dependency (depth sensor) with another (a 63MB neural network doing
inference at 80–150ms/frame). That is the wrong direction for a system we want to run
on a Raspberry Pi.

More importantly: if you are building this to *understand* SLAM, patching someone else's
pipeline teaches you nothing about the thing itself. Building it yourself, with the math
visible and the design choices explicit, teaches you everything.

This document is the complete design of **OpenVIO-SLAM**: a purpose-built, Python-first,
dependency-minimal, pedagogically-documented monocular VIO-SLAM stack for indoor ground
robots. It is designed to be:

- **Understandable** — every module explained from first principles, every design choice justified
- **Correct** — mathematically sound, not cargo-culted from other codebases
- **Efficient enough** — targeting real-time on a laptop, later portable to Raspberry Pi 5
- **Reusable** — clean ROS 2 interfaces so any monocular camera + IMU can plug in

---

## 1. System Overview

### 1.1 What the System Does

```
Phone Camera (10 fps)          Phone IMU (100–200 Hz)
        │                               │
        │   WebSocket (rosbridge)       │
        └──────────────┬────────────────┘
                       ▼
              slam_bridge_node
           (re-timestamp, publish)
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
     /image        /imu/data      /camera_info
         │             │              │
         └──────────────┬─────────────┘
                        ▼
                 ┌─────────────┐
                 │  IMU Pre-   │  ◄── Module 1: IMU Pre-integration
                 │  integrator │      Integrates gyro + accel between
                 └──────┬──────┘      camera frames using RK4
                        │
                        ▼
                 ┌─────────────┐
                 │  Feature    │  ◄── Module 2: ORB Feature Frontend
                 │  Frontend   │      Detect, describe, track ORB features
                 └──────┬──────┘      across frames (KLT + descriptor)
                        │
                        ▼
                 ┌─────────────┐
                 │   MSCKF     │  ◄── Module 3: Core Estimator
                 │  Estimator  │      Multi-State Constraint Kalman Filter
                 └──────┬──────┘      fuses vision + IMU, maintains sliding
                        │             window of camera poses
                        ▼
                 ┌─────────────┐
                 │  Map Point  │  ◄── Module 4: Map Management
                 │  Manager    │      Maintains sparse 3D map, handles
                 └──────┬──────┘      triangulation, marginalization
                        │
                        ▼
                 ┌─────────────┐
                 │ Loop        │  ◄── Module 5: Loop Closure
                 │ Detector    │      Bag-of-Words (ORB vocabulary),
                 └──────┬──────┘      pose graph correction
                        │
                        ▼
                 ┌─────────────┐
                 │  Map Save / │  ◄── Module 6: Persistence
                 │  Load       │      Save map to disk, relocalize
                 └──────┬──────┘      into saved map on restart
                        │
                        ▼
              /odom  /map  /tf  /map_points
                        │
                        ▼
                      RViz2
```

### 1.2 What We Are NOT Building (and Why)

| Excluded | Reason |
|---|---|
| Dense depth map | Not needed for navigation; sparse map is sufficient and 100x cheaper |
| GPU inference | Explicit design goal: CPU-only, Raspberry Pi portable |
| Global map optimization (full BA) | Use pose graph optimization instead — O(N) vs O(N³) |
| Stereo / RGB-D support | Out of scope; monocular VIO is the hard problem we're solving |
| Obstacle avoidance | Navigation stack (Nav2) consumes our map; we don't do navigation |

---

## 2. The Math Foundation (What You Need to Hold in Your Head)

This section is not optional. Every module below is an implementation of something here.

### 2.1 State Representation

The MSCKF state vector at time k is:

```
X = [ x_IMU | x_cam_1 | x_cam_2 | ... | x_cam_N ]
```

Where the IMU state is:

```
x_IMU = [ q_GI,  p_GI,  v_GI,  b_g,  b_a ]
         rotation position velocity gyro  accel
          (4)     (3)     (3)    (3)   (3)
         = 16 elements total (but we use error-state formulation: 15-dim)
```

And each camera state in the sliding window is:

```
x_cam_i = [ q_GCi,  p_GCi ]
            rotation position
             (4)     (3)
            = 7 elements (error-state: 6-dim)
```

With a window of N=20 camera poses, total error-state dimension ≈ 15 + 20×6 = 135.
This is the core computational cost. Every MSCKF operation is O(dim²) at worst.

### 2.2 Rotation Conventions

We use **active rotation** and **Hamilton quaternion** convention throughout:
- `q_AB` rotates a vector from frame B into frame A: `v_A = R(q_AB) * v_B`
- Quaternion stored as `[qw, qx, qy, qz]`
- SO(3) Lie algebra: perturbations via `exp_map(δθ)` and `log_map(R)`

The **IMU-to-camera extrinsic** `T_IC = [R_IC | p_IC]` is calibrated once and fixed.
This is the transform that takes a point from camera frame into IMU frame.

### 2.3 The MSCKF Core Idea

Classical EKF-SLAM augments state with map landmarks → state grows unboundedly → O(N³).

MSCKF insight (Mourikis & Roumeliotis, 2007):
> Do not put landmarks in the state. Instead, accumulate observations of a feature
> across multiple camera poses. When the feature is about to leave the window,
> derive a **measurement constraint directly between the camera poses** that observed
> it — eliminating the feature via the **null-space projection trick**.

This keeps the state bounded (sliding window), makes updates O(1) in map size,
and is the reason MSCKF is the filter of choice for resource-constrained VIO.

The null-space projection: given feature j observed in cameras {i₁, i₂, ..., iₘ},
the measurement Jacobian `H_fj` has a component w.r.t. the feature position.
We project onto the left null space of that component to get `H̃_fj` — a constraint
purely between camera poses with no feature state involved.

### 2.4 IMU Pre-integration

Between camera frames at times tₖ and tₖ₊₁, the IMU fires M times (M ≈ 10–20).
Naively, we'd re-integrate from the start on every state update — O(M×N) per step.

Pre-integration (Forster et al., 2015): integrate IMU increments in the **body frame**,
producing `ΔR, Δv, Δp` that are independent of the linearization point (the current
state estimate). Only a small Jacobian correction is needed when the bias estimate
changes. This reduces re-integration to a cheap first-order correction.

### 2.5 Zero-Velocity & Planar Constraints (Our Advantage Over Generic VIO)

Since this is a **ground robot**:

1. **Zero-Velocity Update (ZUPT)**: when wheel encoders (or IMU magnitude) indicate
   the robot is stationary, inject a pseudo-measurement `v = 0` with small noise.
   This dramatically reduces accelerometer bias drift — the #1 failure mode of VIO
   on slow-moving platforms.

2. **Planar Motion Constraint**: the robot's z-velocity and roll/pitch rates are
   near-zero during normal motion. Inject these as soft constraints at ~10Hz.
   This effectively gives us "fake floor" information without a depth sensor.

These two constraints are the key reason our system will outperform a generic
monocular VIO (like OpenVINS or VINS-Mono) on a ground robot without extra tuning.

### 2.6 Scale Observability

This is the fundamental limitation of monocular VIO and must be understood clearly:

- **Pure monocular vision**: scale is **unobservable**. A scene 1m away moving at
  1m/s is indistinguishable from 2m away at 2m/s. The map has no metric scale.
- **IMU + vision**: scale becomes **observable** because the IMU measures acceleration
  in metric units (m/s²). When the camera translates, the IMU measures that translation
  in real metres. The VIO filter fuses these to recover metric scale.
- **Required condition**: sufficient acceleration (not pure rotation). During
  initialization, the robot *must* translate (not just rotate) for scale to become
  observable. We handle this with an explicit **initialization module**.

---

## 3. Module-by-Module Design

---

### Module 0: Project Structure

```
openvio_slam/
├── README.md
├── setup.py
├── requirements.txt                   # numpy, opencv-python, scipy, rclpy
│
├── openvio_slam/                      # Python package (ROS 2 node)
│   ├── __init__.py
│   │
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── orb_extractor.py           # ORB detection + description
│   │   ├── klt_tracker.py             # KLT optical flow tracker
│   │   ├── feature_manager.py         # Tracks feature lifetime, decides when to extract new
│   │   └── undistortion.py            # Camera model + undistortion
│   │
│   ├── imu/
│   │   ├── __init__.py
│   │   ├── preintegrator.py           # IMU pre-integration (RK4, bias Jacobians)
│   │   ├── noise_model.py             # IMU noise parameters (σ_g, σ_a, τ_bg, τ_ba)
│   │   └── initializer.py             # Static + dynamic VIO initialization
│   │
│   ├── estimator/
│   │   ├── __init__.py
│   │   ├── msckf.py                   # THE CORE: MSCKF filter (propagate + update)
│   │   ├── state.py                   # State vector, covariance, sliding window
│   │   ├── measurement.py             # Feature measurement model + Jacobians
│   │   ├── null_space.py              # Null-space projection (the MSCKF trick)
│   │   └── marginalization.py         # Remove old camera states from window
│   │
│   ├── map/
│   │   ├── __init__.py
│   │   ├── map_point.py               # 3D landmark: position, descriptor, observations
│   │   ├── map_manager.py             # Add/remove/triangulate landmarks
│   │   ├── triangulator.py            # DLT triangulation + RANSAC
│   │   └── keyframe.py                # Keyframe: pose + features + map point refs
│   │
│   ├── loop_closure/
│   │   ├── __init__.py
│   │   ├── vocabulary.py              # ORB Bag-of-Words vocabulary (build + query)
│   │   ├── detector.py                # Candidate detection (cosine similarity)
│   │   ├── verifier.py                # Geometric verification (homography RANSAC)
│   │   └── pose_graph.py              # Pose graph optimizer (Gauss-Newton, from scratch)
│   │
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── map_serializer.py          # Save/load map (numpy .npz format)
│   │   └── relocalization.py          # Localize into saved map
│   │
│   ├── constraints/
│   │   ├── __init__.py
│   │   ├── zupt.py                    # Zero-velocity pseudo-measurement
│   │   └── planar.py                  # Planar motion constraint
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── so3.py                     # SO(3) operations: exp, log, hat, vee
│   │   ├── se3.py                     # SE(3) operations
│   │   └── quaternion.py              # Quaternion math (Hamilton convention)
│   │
│   └── ros/
│       ├── __init__.py
│       ├── vio_node.py                # Main ROS 2 node (subscribes + orchestrates)
│       └── publishers.py              # Publish odom, map, tf, rviz markers
│
├── launch/
│   └── openvio_slam.launch.py
├── config/
│   ├── camera.yaml                    # Intrinsics + distortion
│   ├── imu.yaml                       # Noise params, IMU-camera extrinsics
│   └── slam.yaml                      # Tuning parameters
├── tools/
│   ├── build_vocabulary.py            # Offline ORB vocabulary construction
│   ├── calibrate_imu_camera.py        # Estimate T_IC from a calibration sequence
│   └── evaluate_trajectory.py         # ATE/RTE against ground truth
└── tests/
    ├── test_so3.py
    ├── test_msckf_propagation.py
    └── test_triangulation.py
```

---

### Module 1: Utility Math (`utils/`)

**Build this first. Test it first. Everything else depends on it.**

#### `so3.py` — SO(3) Lie Group Operations

```python
# Key functions to implement (with tests):

def hat(v):
    """R³ → so(3): skew-symmetric matrix from 3-vector.
    hat(v) @ w == cross(v, w) for all w.
    This is the Lie algebra of SO(3)."""

def vee(M):
    """so(3) → R³: inverse of hat. Extract vector from skew matrix."""

def exp_map(phi):
    """so(3) → SO(3): Rodrigues formula.
    phi: rotation vector (axis * angle), shape (3,)
    Returns: 3×3 rotation matrix.
    Uses small-angle approximation for |phi| < 1e-7 to avoid divide-by-zero."""

def log_map(R):
    """SO(3) → so(3): matrix logarithm.
    R: 3×3 rotation matrix
    Returns: rotation vector phi, shape (3,)
    Handles edge cases: R ≈ I and R ≈ -I separately."""

def right_jacobian(phi):
    """Right Jacobian of SO(3) — needed for IMU pre-integration Jacobians.
    J_r(phi) such that exp(phi + δ) ≈ exp(phi) @ exp(J_r(phi) @ δ)"""

def quat_to_rot(q):
    """Hamilton quaternion [qw,qx,qy,qz] → 3×3 rotation matrix."""

def rot_to_quat(R):
    """3×3 rotation matrix → Hamilton quaternion [qw,qx,qy,qz].
    Shepperd's method — numerically stable for all orientations."""

def quat_multiply(q1, q2):
    """Hamilton quaternion product. q1 ⊗ q2."""

def quat_normalize(q):
    """Normalize quaternion to unit length."""

def quat_inverse(q):
    """Quaternion inverse: [qw, -qx, -qy, -qz] for unit quaternion."""
```

Design note: we implement SO(3) from scratch (using only numpy) rather than using
Sophus because (a) educational goal, (b) Python Sophus bindings are fragile across
ROS distros, (c) we only need a small subset of Sophus functionality.

---

### Module 2: IMU Pre-integrator (`imu/preintegrator.py`)

#### Why Pre-integration Matters

The IMU fires at 200 Hz; the camera fires at 10 Hz. Between two camera frames,
we have ~20 IMU measurements. The pre-integrator accumulates these into three
quantities that summarize the motion between camera frames:

```
ΔR ∈ SO(3)   — rotation increment
Δv ∈ R³      — velocity increment
Δp ∈ R³      — position increment
```

All expressed in the IMU body frame at time tₖ (the start of the interval).

#### Integration Method: RK4 on SO(3) × R³ × R³

We use Runge-Kutta 4th order (not Euler) because:
- IMU noise amplifies with Euler integration at ~200Hz → large orientation error
- RK4 costs 4× more per step but reduces error by ~O(h⁴) vs O(h²) for Euler

#### Bias Jacobians

When the bias estimate changes by `δb_g` (gyro) or `δb_a` (accel), we don't
re-integrate from scratch. We maintain first-order Jacobians:

```
∂ΔR/∂b_g,  ∂Δv/∂b_g,  ∂Δv/∂b_a,  ∂Δp/∂b_g,  ∂Δp/∂b_a
```

These are integrated alongside the main quantities and are updated at each
IMU step using the chain rule and the right Jacobian of SO(3).

#### Noise Propagation

We also propagate the **pre-integration covariance** `Σ_Δ` forward at each step.
This covariance encodes how uncertain we are about `(ΔR, Δv, Δp)` due to
IMU measurement noise and random walk in the biases.

The process noise matrix Q at each step:
```
Q = diag(σ_g², σ_g², σ_g², σ_a², σ_a², σ_a²,
         σ_bg², σ_bg², σ_bg², σ_ba², σ_ba², σ_ba²)
```

`σ_g` (gyro noise), `σ_a` (accel noise): from IMU datasheet or calibration.
`σ_bg` (gyro random walk), `σ_ba` (accel random walk): from Allan variance.

#### Data Flow

```python
class IMUPreintegrator:
    def reset(self, bias_g, bias_a):
        """Start a new pre-integration interval. Called after each camera frame."""

    def integrate(self, gyro, accel, dt):
        """Add one IMU measurement. Called at IMU rate (~200 Hz).
        Updates ΔR, Δv, Δp, Jacobians, covariance."""

    def get_delta(self):
        """Return (ΔR, Δv, Δp, Σ_Δ, Jacobians) for MSCKF propagation."""

    def correct_for_bias_change(self, δb_g, δb_a):
        """First-order correction when bias estimate is updated.
        Avoids full re-integration."""
```

---

### Module 3: ORB Feature Frontend (`frontend/`)

#### Design Philosophy

The frontend does two jobs and they are deliberately separated:

1. **Tracking** (every frame, KLT optical flow): fast, sub-pixel precise, no descriptor
2. **Description** (new features only, ORB): slower, needed only for loop closure + matching

This separation is critical for real-time performance. KLT on 300 features costs ~2ms.
ORB description of 300 features costs ~15ms. By only describing when needed, we keep
the frontend at ~5ms average per frame.

#### `orb_extractor.py`

We use OpenCV's `cv2.ORB_create()` for detection and description. The only thing
we build ourselves is the **spatial grid bucketing** to enforce uniform feature
distribution:

```python
class ORBExtractor:
    """
    Detects ORB features with spatial bucketing.
    
    The image is divided into a GRID_ROWS × GRID_COLS grid.
    Up to MAX_FEATURES_PER_CELL features are kept per cell.
    This prevents feature clustering in high-texture regions
    (e.g., all features on a single poster, nothing in the rest of the room).
    
    Parameters (tunable):
        n_features: total target feature count (default: 300)
        grid_rows, grid_cols: spatial grid size (default: 4×5)
        scale_factor: ORB pyramid scale (default: 1.2)
        n_levels: pyramid levels (default: 8)
    """
```

#### `klt_tracker.py`

Lucas-Kanade optical flow via `cv2.calcOpticalFlowPyrLK()`.

Key design choices:
- **Bidirectional consistency check**: track forward t→t+1, then backward t+1→t.
  Features where backward track deviates > 1px from original are rejected.
  This kills ~15% of tracks but eliminates the "ghost feature" problem where
  KLT converges to a wrong location silently.
- **Fundamental matrix RANSAC**: after KLT, run 8-point algorithm + RANSAC on
  the tracked correspondences. Outlier tracks (violate epipolar geometry) are
  removed. Threshold: Sampson distance < 1.0 px.

```python
class KLTTracker:
    def track(self, img_prev, img_curr, pts_prev):
        """
        Track features from previous to current frame.
        Returns: (pts_curr, status_mask)
        status_mask[i] = True if feature i was successfully tracked.
        
        Applies: forward-backward check, then epipolar RANSAC.
        """
```

#### `feature_manager.py`

Manages the **lifetime** of features across frames:

```python
class FeatureManager:
    """
    Maintains the active feature set across frames.
    
    Each feature has:
        id: unique integer (never reused)
        observations: list of (frame_id, uv_undistorted) pairs
        descriptor: ORB descriptor (256-bit) — set once, at creation
        status: ACTIVE | LOST | MARGINALIZED
    
    Policy:
        - A feature is LOST if KLT fails to track it (or epipolar check rejects it)
        - A feature is MARGINALIZED when its oldest observation falls out of the
          MSCKF sliding window — at this point it's fed to the MSCKF update step
        - New features are extracted when the active count drops below MIN_FEATURES (150)
        - Features with < 3 observations at marginalization are discarded (can't triangulate)
    """
```

---

### Module 4: MSCKF Core Estimator (`estimator/`)

This is the heart of the system. It is also the most mathematically involved.
We build it in sub-modules with explicit test coverage for each.

#### `state.py` — State and Covariance

```python
class VIOState:
    """
    The full MSCKF state.
    
    IMU state (error-state dimension: 15):
        q_GI: quaternion, world←IMU rotation     (stored as [qw,qx,qy,qz])
        p_GI: position in world frame            (metres)
        v_GI: velocity in world frame            (m/s)
        b_g:  gyroscope bias                     (rad/s)
        b_a:  accelerometer bias                 (m/s²)
    
    Camera window (up to N_MAX=20 entries):
        Each entry: (q_GC, p_GC) — pose of camera at that keyframe instant
        Error-state per camera: 6-dim (δθ, δp)
    
    Full error-state covariance P: (15 + 6*N) × (15 + 6*N)
    
    NOTE: We store the *nominal* state separately from the error-state covariance.
    The MSCKF only propagates the covariance; the nominal state is propagated
    by direct integration of IMU measurements. This is the error-state Kalman
    filter formulation — cleaner and more numerically stable than propagating
    the full state through the nonlinear dynamics.
    """
```

#### `msckf.py` — Propagation Step

Called at IMU rate (~200 Hz):

```
1. Get new IMU measurement (gyro ω, accel a), dt
2. Nominal state propagation (integrate directly on SO(3)):
   q_{k+1} = q_k ⊗ exp_map((ω - b_g) * dt)     # orientation
   v_{k+1} = v_k + (R_k @ (a - b_a) + g) * dt   # velocity (g = [0,0,-9.81])
   p_{k+1} = p_k + v_k*dt + 0.5*(R_k@(a-b_a)+g)*dt²  # position

3. Error-state covariance propagation (linearized):
   P_{k+1} = Φ_k @ P_k @ Φ_k.T + G_k @ Q @ G_k.T
   
   Φ_k: state transition matrix (15×15 for IMU-only block)
   G_k: noise input matrix
   Q:   continuous-time IMU noise PSD (from imu.yaml)

4. Update pre-integrator (for later use in camera augmentation)
```

Key: the `Φ_k` matrix is derived analytically from the IMU kinematics.
We do NOT use a numerical Jacobian here — analytical derivatives are critical
for correctness and give us a good test: `Φ` must satisfy `Φ @ Φ.T > 0`
(positive definite) and degrade gracefully as dt grows.

#### `msckf.py` — Camera Augmentation

Called at camera rate (~10 Hz), when a new image arrives:

```
1. Compute camera pose from current IMU state + known extrinsic T_IC:
   q_GC = q_GI ⊗ q_IC
   p_GC = p_GI + R_GI @ p_IC

2. Augment state: append (q_GC, p_GC) to camera window

3. Augment covariance P:
   P_aug = J_aug @ P @ J_aug.T
   
   J_aug: Jacobian of new camera state w.r.t. full state
          (this is just the identity + T_IC Jacobian; easy to derive)

4. If window is full (N = N_MAX), trigger marginalization (remove oldest camera)
```

#### `null_space.py` + `msckf.py` — MSCKF Update Step

Called when a feature is **marginalized** (leaves the sliding window):

```
For each marginalized feature j that was observed in cameras {i₁...iₘ} (m≥3):

1. Initialize feature position by triangulation (DLT, see triangulator.py)
   If triangulation fails (degenerate geometry), discard feature.

2. Compute measurement Jacobian H_fj:
   - For each observation (camera iₖ, pixel uv_k):
     Linearize projection model around current state + feature estimate
     → partial derivatives w.r.t. camera pose iₖ and feature position
   - Stack into H_fj: (2m × (6N + 3)) matrix
     [left block: w.r.t. camera states] [right block: w.r.t. feature position]

3. NULL-SPACE PROJECTION (the core MSCKF trick):
   A = left null space of right block of H_fj
   H̃_fj = A.T @ left_block(H_fj)     # (2m-3) × 6N — feature eliminated!
   r̃_fj = A.T @ r_fj                  # projected residual

4. Chi-squared test (outlier rejection):
   if r̃_fj.T @ (H̃_fj @ P_cam @ H̃_fj.T + noise).inv() @ r̃_fj > χ²_threshold:
       discard (visual outlier, likely dynamic object or bad track)

5. Kalman update:
   S = H̃ @ P @ H̃.T + R_noise
   K = P @ H̃.T @ inv(S)
   δX = K @ r̃
   P = (I - K @ H̃) @ P   # Joseph form for numerical stability

6. Apply error-state correction to nominal state:
   q ← q ⊗ exp_map(δθ/2)   (quaternion update on manifold)
   p ← p + δp
   v ← v + δv
   b_g ← b_g + δb_g
   b_a ← b_a + δb_a
```

This is the entire MSCKF in one place. The math looks dense but each step
maps directly to a function with a test. We build it incrementally:
propagation first, augmentation second, update last.

---

### Module 5: Map Management (`map/`)

#### Triangulation (`triangulator.py`)

Given a feature observed in ≥2 camera poses, recover 3D position via DLT:

```python
def triangulate_dlt(poses, observations):
    """
    Direct Linear Transform triangulation.
    
    poses: list of (R, t) camera poses (world←camera)
    observations: list of (u, v) normalized image coordinates
    
    For each observation, form:
        [u*P[2] - P[0]] @ X = 0
        [v*P[2] - P[1]] @ X = 0
    
    Stack into A @ X = 0, solve via SVD (smallest singular vector).
    
    Returns: X_world (3,) — 3D point in world frame
             quality score (inverse condition number of A)
    
    Reject if:
        - Point is behind any camera (z < 0)
        - Parallax angle < 5° (insufficient baseline — scale unobservable)
        - Reprojection error > 2px in any view
    """
```

#### Map Points (`map_point.py`)

```python
class MapPoint:
    """
    A triangulated 3D landmark.
    
    Fields:
        id:          unique integer
        position:    np.array(3,) in world frame
        descriptor:  np.array(32,) uint8 — ORB descriptor (256-bit = 32 bytes)
                     Stored as the *median* descriptor over all observations
                     (more robust than any single observation's descriptor)
        observations: dict {keyframe_id: feature_id}
        n_observed:  total observation count
        n_visible:   times point was in view (for culling ratio)
    
    Culling policy (remove if):
        - Observed < 3 times AND existed > 3 keyframes (never triangulated well)
        - Found ratio (n_observed / n_visible) < 0.25 (consistently not found)
    """
```

---

### Module 6: Ground Robot Constraints (`constraints/`)

#### Zero-Velocity Update (`zupt.py`)

```python
class ZUPTDetector:
    """
    Detects zero-velocity intervals from IMU signal.
    
    Method: compute variance of gyro magnitude over a sliding window.
    If var(|ω|) < threshold AND |a| ≈ 9.81 (static accel), robot is stationary.
    
    When stationary, inject pseudo-measurement:
        z = v_GI = 0
        H = [0 | I₃ | 0 | 0 | 0]  (selects velocity from state)
        R_zupt = diag(0.001, 0.001, 0.001)  m²/s² (tight — we're confident)
    
    This is one of the most impactful features for ground robots.
    Without it, velocity drift during slow periods accumulates into large
    position errors. With it, the estimator "resets" velocity drift
    every time the robot pauses.
    """
```

#### Planar Motion Constraint (`planar.py`)

```python
class PlanarConstraint:
    """
    Soft constraint: ground robot moves on a plane.
    
    Injected as pseudo-measurement at camera rate:
        z_z   = v_z   ≈ 0   (no vertical velocity)
        z_roll  = ω_roll  ≈ 0   (no roll rate — wheels keep robot level)
        z_pitch = ω_pitch ≈ 0   (no pitch rate)
    
    Noise: R_planar = diag(0.05², 0.01², 0.01²)  — soft, not hard constraint
           (wheels can go over small bumps, so we don't clamp too tightly)
    
    Implementation: same Kalman update as ZUPT, different H matrix.
    
    Combined with ZUPT, these constraints effectively give our monocular
    system the geometric information that a depth sensor would otherwise provide
    for the vertical dimension — because a ground robot's motion is inherently
    2.5D, not full 6-DOF.
    """
```

---

### Module 7: Loop Closure (`loop_closure/`)

#### Bag-of-Words Vocabulary (`vocabulary.py`)

Loop closure requires recognizing a previously-seen place from visual appearance.
We use a **Bag-of-Words (BoW)** model over ORB descriptors:

1. **Offline: build vocabulary** — run k-means clustering on a large set of ORB
   descriptors extracted from representative indoor images. Produces a tree
   of visual "words" (cluster centers). This is done once with `tools/build_vocabulary.py`
   and saved to `config/orb_vocabulary.npz`.

   We use a 2-level tree: 10 branches at level 1, 10 at level 2 → 100 visual words.
   This is intentionally small (DBoW2 uses 10⁶ words) — sufficient for indoor rooms,
   much faster to query.

2. **Online: represent each keyframe** as a normalized histogram over the 100 words.
   This is the keyframe's "appearance vector" — a 100-dim float vector.

3. **Query: find loop candidates** — compute cosine similarity between current
   keyframe's BoW vector and all stored keyframe vectors. Top-K candidates
   (where similarity > threshold) are loop closure proposals.

#### Geometric Verification (`verifier.py`)

BoW gives us appearance-similar candidates, but appearance similarity ≠ same place
(two offices with the same posters look similar but may be different rooms).

Geometric verification:
```
1. Match ORB descriptors between current keyframe and candidate keyframe
   (brute-force Hamming distance, ratio test: best_match / second_best < 0.8)
2. Compute homography H (or essential matrix E) via RANSAC
3. Accept loop closure only if RANSAC inlier count > MIN_LOOP_INLIERS (20)
4. Recover relative pose from E (or H for planar scenes) + cheirality check
```

#### Pose Graph Optimizer (`pose_graph.py`)

When a loop closure is accepted, we correct accumulated drift by optimizing
the pose graph — not a full bundle adjustment (too slow), but a graph where:
- **Nodes**: keyframe poses (SE(3))
- **Edges**: relative pose constraints (from consecutive odometry + loop closures)
- **Cost**: sum of squared pose residuals, weighted by covariance

We implement **Gauss-Newton on SE(3)** from scratch:

```
for iter in range(MAX_ITER):
    H = 0,  b = 0
    for each edge (i, j, T_ij, Ω_ij):
        r_ij = log_map(T_ij.inv() @ T_i.inv() @ T_j)   # 6-dim residual
        J_i, J_j = ∂r_ij/∂T_i, ∂r_ij/∂T_j             # 6×6 Jacobians
        H += J.T @ Ω_ij @ J
        b += J.T @ Ω_ij @ r_ij
    
    δX = solve(H, -b)   # sparse linear system
    apply δX to all poses (SE(3) retraction)
    if |δX| < 1e-5: break
```

Sparse solver: `scipy.sparse.linalg.spsolve` (the one external math dependency
we use here — implementing a sparse Cholesky from scratch is not educational,
it is just tedious).

After optimization, the MSCKF state is corrected to match the optimized poses.

---

### Module 8: Map Persistence & Re-localization (`persistence/`)

#### Save Format

Map saved as `~/.openvio_slam/map_TIMESTAMP.npz`:
```python
{
    'keyframe_poses':  np.array (N, 7),   # [qw,qx,qy,qz, px,py,pz]
    'keyframe_ids':    np.array (N,),
    'map_points':      np.array (M, 3),   # 3D positions
    'descriptors':     np.array (M, 32),  # ORB descriptors
    'observations':    sparse matrix,     # keyframe↔map_point links
    'vocabulary':      BoW tree,          # embedded for self-contained file
    'metadata':        dict               # calibration, timestamp, stats
}
```

#### Re-localization

On startup with an existing map:
1. Load map, build BoW index over all keyframe descriptors
2. For first N_INIT=5 frames, match descriptors against map → PnP RANSAC to get pose
3. If PnP succeeds (> 20 inliers), initialize VIO state from this pose
4. Resume MSCKF as normal — now localized in the saved map's coordinate frame

This is the **"map once, navigate forever"** property requested in the design brief.

---

### Module 9: ROS 2 Interface (`ros/`)

#### `vio_node.py` — Main Node

```python
class OpenVIOSLAMNode(Node):
    """
    Subscriptions:
        /image          sensor_msgs/Image       — camera frames (bgr8, 10 Hz)
        /imu/data       sensor_msgs/Imu         — IMU measurements (200 Hz)
        /camera_info    sensor_msgs/CameraInfo  — (latched, once)
    
    Publications:
        /odom           nav_msgs/Odometry       — VIO odometry (10 Hz)
        /map_points     sensor_msgs/PointCloud2 — sparse 3D map (1 Hz)
        /keyframes      visualization_msgs/MarkerArray  — camera trajectory
        /loop_closures  visualization_msgs/MarkerArray  — loop closure edges
        /tf             odom→base_link, map→odom
    
    Services:
        /save_map       std_srvs/Trigger        — save current map to disk
        /load_map       std_srvs/SetString      — load map from path
        /reset          std_srvs/Trigger        — reset to uninitialized state
    
    Thread model:
        - IMU callback: high-priority, runs pre-integration synchronously
        - Image callback: spawns processing thread (avoids blocking ROS executor)
        - Loop closure: runs in separate thread, posts result to main thread queue
        - Map save/load: separate thread (disk I/O must not block estimation)
    """
```

---

## 4. Implementation Phases

### Phase 1 — Foundation (Weeks 1–2)
**Goal: IMU pre-integration working, verifiable against numerical integration**

Deliverables:
- `utils/so3.py` with full test suite (10+ tests covering edge cases)
- `utils/quaternion.py` with roundtrip tests (quat → matrix → quat)
- `imu/preintegrator.py` with synthetic IMU data test
- Verify: pre-integrated trajectory matches direct integration to < 0.1% error

**How to know it works:**
Generate synthetic IMU data for a known trajectory (circular motion).
Pre-integrate. Compare to ground truth. Error should be < 1cm over 10 seconds.

### Phase 2 — Visual Frontend (Week 3)
**Goal: Stable feature tracking visible in RViz**

Deliverables:
- `frontend/orb_extractor.py` — grid-bucketed ORB
- `frontend/klt_tracker.py` — with bidirectional check + epipolar RANSAC
- `frontend/feature_manager.py` — lifetime management
- ROS node that subscribes to `/image` and publishes tracked features as markers

**How to know it works:**
Stream phone camera. In RViz, see colored dots tracking across frames.
Dots should stay on the same physical points for 15–30 frames on average.
Tracks should not "slide" or "jump" on a stationary scene.

### Phase 3 — MSCKF Core (Weeks 4–6)
**Goal: Odometry publishing, no loop closure yet**

Deliverables:
- `estimator/state.py`, `msckf.py` (propagation + augmentation + update)
- `estimator/null_space.py`
- `map/triangulator.py`
- `constraints/zupt.py` + `constraints/planar.py`
- `/odom` publishing, TF tree: `map → odom → base_link`

**How to know it works:**
Walk the robot in a 2m × 2m square. Trajectory in RViz should close the square
within ~10cm (before loop closure). IMU-only (no vision) should drift ~50cm.
Vision should make it much better.

### Phase 4 — Loop Closure (Weeks 7–9)
**Goal: Drift corrected, map consistent across revisits**

Deliverables:
- `loop_closure/vocabulary.py` (offline vocabulary builder)
- `loop_closure/detector.py` + `verifier.py`
- `loop_closure/pose_graph.py`
- Visible loop closure edges in RViz (red lines between matched keyframes)

**How to know it works:**
Walk a loop (leave a room, come back). Watch RViz snap the trajectory into
alignment when the loop is detected. Drift before: ~30cm. After closure: < 5cm.

### Phase 5 — Persistence & Re-localization (Week 10)
**Goal: Save map, restart, re-localize within 5 seconds**

Deliverables:
- `persistence/map_serializer.py`
- `persistence/relocalization.py`
- `/save_map` and `/load_map` services
- Documentation: "map once, navigate forever" demo procedure

---

## 5. Configuration Reference (`config/slam.yaml`)

```yaml
# --- Feature Frontend ---
frontend:
  n_features: 300           # target feature count per frame
  grid_rows: 4              # spatial distribution grid
  grid_cols: 5
  min_features: 150         # extract new features below this threshold
  min_track_length: 3       # minimum observations before MSCKF update

# --- MSCKF Estimator ---
estimator:
  window_size: 20           # maximum camera states in sliding window
  min_triangulation_parallax: 5.0  # degrees — reject near-degenerate triangulations
  chi2_threshold: 7.815     # χ² test, 3 DoF, 95% confidence

# --- IMU Noise (tune to your phone — these are typical Android values) ---
imu:
  gyro_noise:       0.005   # rad/s/√Hz
  accel_noise:      0.05    # m/s²/√Hz
  gyro_bias_walk:   0.0001  # rad/s²/√Hz
  accel_bias_walk:  0.001   # m/s³/√Hz
  gravity:          9.81    # m/s²

# --- Ground Robot Constraints ---
constraints:
  zupt_gyro_var_threshold: 0.001   # rad²/s² — stationary detection
  zupt_noise: 0.001                 # m²/s² — velocity measurement noise when ZUPT
  planar_vz_noise: 0.05             # m²/s² — soft vertical velocity constraint
  planar_roll_noise: 0.01           # rad²/s² — roll rate constraint
  planar_pitch_noise: 0.01          # pitch rate constraint

# --- Loop Closure ---
loop_closure:
  bow_similarity_threshold: 0.6    # cosine similarity for candidate detection
  min_loop_inliers: 20             # geometric verification minimum
  min_keyframe_spacing: 10         # minimum keyframes between loop candidates
  pose_graph_max_iter: 50

# --- Map Management ---
map:
  max_map_points: 5000             # cull beyond this (keep highest-quality)
  cull_found_ratio: 0.25           # remove points found < 25% of visible times

# --- Initialization ---
init:
  min_parallax_deg: 5.0            # need this much baseline to initialize scale
  min_imu_excitation: 0.5          # m/s² — need acceleration to recover scale
  static_init_duration: 2.0        # seconds of static data for bias init
```

---

## 6. What Makes This Different From VINS-Mono / OpenVINS

| Property | VINS-Mono / OpenVINS | OpenVIO-SLAM (this project) |
|---|---|---|
| Language | C++ | Python (portable, readable) |
| Target hardware | General (UAV, car, phone) | Ground robot (specialized constraints) |
| Planar constraint | Optional, manual | First-class, automatic |
| ZUPT | Not by default | Built-in, automatic |
| Codebase size | 15,000–50,000 lines | Target: < 3,000 lines |
| Comments | Minimal | Pedagogical (every equation referenced) |
| Dependencies | Eigen, Ceres, OpenCV, ROS | numpy, scipy, OpenCV, ROS |
| Loop closure | DBoW2 (external) | BoW from scratch |
| Pose graph | g2o (external) | Gauss-Newton from scratch |
| Documentation | Research paper | Inline + this document |

The planar + ZUPT constraints alone mean this system will **outperform VINS-Mono
on a ground robot** without any extra tuning — those systems are designed for UAVs
where 6-DOF motion is the norm.

---

## 7. First Code to Write

Start here, in this exact order:

```bash
mkdir -p ~/slam_ws/src/openvio_slam/openvio_slam/utils
mkdir -p ~/slam_ws/src/openvio_slam/tests

# File 1: utils/so3.py
# File 2: tests/test_so3.py  (write tests BEFORE moving on)
# File 3: utils/quaternion.py
# File 4: tests/test_quaternion.py
# File 5: imu/preintegrator.py
# File 6: tests/test_preintegrator.py
```

The test suite is not optional. The SO(3) and quaternion code will be used in
every single other module. A bug here — an off-by-one in the quaternion multiply,
a wrong sign in the skew matrix — will produce wrong results everywhere, but in
ways that look plausible (the system runs but drifts). Tests catch this early.

**Suggested first test (put this in `test_so3.py`):**
```python
def test_exp_log_roundtrip():
    """exp(log(R)) == R for random rotation matrices."""
    for _ in range(100):
        phi = np.random.randn(3) * 0.5   # random small rotation
        R = exp_map(phi)
        phi_recovered = log_map(R)
        assert np.allclose(phi, phi_recovered, atol=1e-10)

def test_hat_vee_roundtrip():
    v = np.random.randn(3)
    assert np.allclose(vee(hat(v)), v)

def test_rotation_is_SO3():
    """R @ R.T == I and det(R) == 1."""
    for _ in range(100):
        phi = np.random.randn(3)
        R = exp_map(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
```

When these pass: you have correct SO(3). Every module above it can be trusted.

---

## 8. Questions to Answer Before Starting Each Module

Before writing each module, ask and answer:

**IMU Preintegrator:**
- What is my IMU noise density? (from phone spec or Allan variance)
- What is the IMU-camera time offset? (calibrate with `tools/calibrate_imu_camera.py`)
- What is `T_IC`? (the rigid transform between IMU and camera frames)

**MSCKF:**
- What is `N_MAX` (window size)? 20 is right for a laptop; 10 for Pi 4
- What chi² threshold? 7.815 (3-DoF, 95%) is standard
- What is minimum parallax for triangulation? 5° is standard

**Loop Closure:**
- Do I have a vocabulary file? If not, build one first offline.
- What environments will the robot operate in? (affects vocabulary diversity)

**Persistence:**
- Is the map coordinate frame stable? (yes, after loop closure)
- Does the robot always start in the same place? (affects re-localization strategy)

---

*This document is the architecture contract. Every module built should be traceable
to a section here. If implementation reveals a design error, update this document
before updating the code — the document is the source of truth.*
