#!/usr/bin/env python3
"""
ros/vio_node.py — OpenVIO‑SLAM (Pi‑5 Optimised)
================================================

Performance‑oriented rewrite for Raspberry Pi 5 CPU‑only deployment.

Revision 3 — fixes found in second test run:
  BUG: ZUPT and planar corrections were applied inside _process_frame(),
  which only runs when a camera frame arrives (~10 fps). Between frames,
  _imu_callback() kept propagating the state with zero corrections. If the
  camera was blocked, dropped, or lagging, IMU propagation ran completely
  unconstrained — any residual gravity / bias error integrated freely and
  the position blew up to hundreds or thousands of meters (observed: -Z
  explosion while camera blocked, then random teleport on reconnect).
  FIX: ZUPT and planar are now applied inside _imu_callback() every IMU
  sample (~40-100 Hz depending on phone), completely independent of whether
  a camera frame has arrived. The visual-flow gate state persists between
  IMU samples — it's updated each time a frame IS processed and reused for
  all subsequent IMU samples until the next frame. This is the standard
  design used in OpenVINS, MSCKF-VIO, and ROVIO.

Revision 2 — fixes from first test run:
  1. IMU integration used a hardcoded dt=0.005 (assumed perfect 200 Hz).
     Phone-over-WiFi IMU streams are not evenly spaced — real per-sample dt
     is now computed from message timestamps with clamping and guards.
  2. ZUPT fired during constant-velocity straight-line motion because
     gyro+accel variance cannot detect this case (no rotation, no net
     acceleration looks identical to "parked"). Now additionally gated on
     tracked-feature optical flow.
  3. Map-point ring buffer wiped the whole map on overflow — fixed to a
     proper circular buffer.
  4. Startup noise-parameter sanity check (catches accel_bias_walk ~100x
     too high, which was the root source of noisy trajectory).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, Imu, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster

import numpy as np
import threading
import time
import cv2
import yaml
from pathlib import Path as FilePath
import queue  # used for the image worker

from openvio_slam.estimator.msckf import MSCKF
from openvio_slam.frontend.orb_extractor import ORBExtractor
from openvio_slam.frontend.klt_tracker import KLTTracker
from openvio_slam.frontend.feature_manager import FeatureManager
from openvio_slam.frontend.undistortion import PinholeCameraModel
from openvio_slam.imu.preintegrator import IMUPreintegrator
from openvio_slam.imu.initializer import VIOInitializer
from openvio_slam.constraints.zupt import ZUPTDetector
from openvio_slam.constraints.planar import PlanarConstraint
from openvio_slam.utils.quaternion import rot_to_quat, quat_to_rot

class OpenVIOSLAMNode(Node):

    def __init__(self):
        super().__init__('openvio_slam')
        self.get_logger().info("OpenVIO‑SLAM starting (Pi‑5 optimised)...")

        self._declare_parameters()
        config = self._load_config()
        self._sanity_check_noise(config['estimator'])

        self._initialized = False
        self._camera_ready = False
        self._prev_img = None
        self._frame_id = 0
        self._lock = threading.Lock()

        # Visual odometry accumulators (init phase only)
        self._R_vis = np.eye(3)
        self._p_vis = np.zeros(3)
        self._prev_kps = None

        # Map point storage – numpy ring buffer for zero‑copy publishing
        self._MAX_MAP_PTS = int(config.get('map', {}).get('max_map_points', 8000))
        self._map_points = np.empty((self._MAX_MAP_PTS, 3), dtype=np.float32)
        self._map_pts_count = 0
        self._map_lock = threading.Lock()  # separate lock for point cloud

        # Path for RViz
        self._path = Path()
        self._path.header.frame_id = 'odom'

        # Core SLAM components (initialised when camera info arrives)
        self._camera_model = None
        self._orb = ORBExtractor(config['frontend'])
        self._klt = KLTTracker()
        self._feature_mgr = None
        self._preintegrator = None
        self._msckf = None
        self._initializer = VIOInitializer(config['init'])
        self._zupt = ZUPTDetector(config['constraints'])
        self._planar = PlanarConstraint(config['constraints'])
        self._config = config

        # QoS: reliable communication
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Subscribers
        self._sub_img = self.create_subscription(
            Image, '/image', self._image_callback, reliable_qos)
        self._sub_imu = self.create_subscription(
            Imu, '/imu', self._imu_callback, reliable_qos)
        self._sub_info = self.create_subscription(
            CameraInfo, '/camera_info', self._camera_info_callback, reliable_qos)

        # Publishers
        self._pub_odom = self.create_publisher(Odometry, '/odom', reliable_qos)
        self._pub_pts = self.create_publisher(PointCloud2, '/openvio/map_points', reliable_qos)
        self._pub_path = self.create_publisher(Path, '/openvio/path', reliable_qos)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Image worker with blocking queue (no busy‑wait)
        self._img_queue = queue.Queue(maxsize=2)  # avoid unbounded memory
        self._img_worker = threading.Thread(target=self._image_worker, daemon=True)
        self._img_worker.start()

        # Timers
        self._stats_timer = self.create_timer(5.0, self._print_stats)
        self._map_pub_timer = self.create_timer(0.5, self._timer_publish_map_points)

        self._frame_times = []
        self._n_imu = 0
        self._init_accel_buffer = []  # for gravity alignment during static init

        # Real IMU dt tracking (replaces the old fixed dt=0.005 assumption)
        self._last_imu_t = None
        self._imu_dt_window = []   # rolling window for rate diagnostics

        # Visual-motion gate for ZUPT (fixes false-positive ZUPT during
        # constant-velocity straight-line motion, which gyro+accel alone
        # can never distinguish from being stationary)
        self._last_visual_flow_px = 0.0
        self._visual_flow_valid = False
        self._zupt_visual_flow_thresh = config['constraints']['zupt_visual_flow_px_threshold']
        self._zupt_min_features = config['constraints']['zupt_min_features']
        self._last_zupt_applied = False   # for diagnostic logging in _print_stats

        # Circular-buffer write pointer for map points (replaces the old
        # "wipe to front" overflow behaviour)
        self._map_write_ptr = 0

        self.get_logger().info("Waiting for /camera_info...")

    # ── Parameter handling ──────────────────────────────────────────────────
    def _declare_parameters(self):
        self.declare_parameter('window_size', 12)        # smaller for Pi 5
        self.declare_parameter('n_features', 400)        # balanced for speed
        self.declare_parameter('gyro_noise', 0.005)
        self.declare_parameter('accel_noise', 0.085)
        self.declare_parameter('extrinsic_yaml', 'config/extrinsic.yaml')
        self.declare_parameter('slam_yaml', 'config/slam.yaml')

    def _load_config(self) -> dict:
        """Load slam.yaml, fall back to ROS parameters if file missing."""
        yaml_path = self.get_parameter('slam_yaml').value
        search_paths = [
            yaml_path,
            FilePath.home() / 'slam_ws/src/openvio_slam/config/slam.yaml',
            'config/slam.yaml',
        ]
        slam_cfg = {}
        for p in search_paths:
            try:
                with open(str(p), 'r') as f:
                    slam_cfg = yaml.safe_load(f) or {}
                self.get_logger().info(f"Loaded slam.yaml from {p}")
                break
            except Exception:
                continue

        fe = slam_cfg.get('frontend', {})
        es = slam_cfg.get('estimator', {})
        im = slam_cfg.get('imu', {})
        co = slam_cfg.get('constraints', {})
        it = slam_cfg.get('init', {})
        ma = slam_cfg.get('map', {})

        return {
            'frontend': {
                'n_features':     fe.get('n_features', self.get_parameter('n_features').value),
                'grid_rows':      fe.get('grid_rows', 4),
                'grid_cols':      fe.get('grid_cols', 5),
                'min_features':   fe.get('min_features', 200),
                'min_track_length': fe.get('min_track_length', 2),
                'fast_threshold': fe.get('fast_threshold', 20),  # higher = faster
                'scale_factor':   fe.get('scale_factor', 1.2),
                'n_levels':       fe.get('n_levels', 8),
            },
            'estimator': {
                'window_size':       es.get('window_size', self.get_parameter('window_size').value),
                'pixel_noise':       es.get('pixel_noise', 1.0),
                'chi2_threshold':    es.get('chi2_threshold', 0.95),
                'gyro_noise':        im.get('gyro_noise', self.get_parameter('gyro_noise').value),
                'accel_noise':       im.get('accel_noise', self.get_parameter('accel_noise').value),
                'gyro_bias_walk':    im.get('gyro_bias_walk', 0.0002),   # was 0.0001 - typical phone IMU
                'accel_bias_walk':   im.get('accel_bias_walk', 0.0015), # was 0.221 - that default was ~100x too high
                'gravity':           im.get('gravity', 9.81),
            },
            'constraints': {
                'zupt_gyro_var_threshold':      co.get('zupt_gyro_var_threshold', 0.01),
                'zupt_noise':                   co.get('zupt_noise', 0.01),
                'planar_enabled':               co.get('planar_enabled', True),
                'planar_vz_noise':               co.get('planar_vz_noise', 0.05),
                'planar_roll_noise':             co.get('planar_roll_noise', 0.01),
                'planar_pitch_noise':            co.get('planar_pitch_noise', 0.01),
                # NEW: visual-motion gate. ZUPT (gyro+accel only) is physically
                # blind to constant-velocity straight-line motion, so we require
                # tracked-feature optical flow to also be near-zero before we
                # trust "stationary". This is the actual fix for ZUPT firing
                # while the robot is moving at constant speed.
                'zupt_visual_flow_px_threshold': co.get('zupt_visual_flow_px_threshold', 1.0),
                'zupt_min_features':             co.get('zupt_min_features', 15),
            },
            'init': {
                'static_init_duration': it.get('static_init_duration', 2.0),
                'min_parallax_deg':     it.get('min_parallax_deg', 1.5),  # relaxed
                'min_imu_excitation':   it.get('min_imu_excitation', 0.1),
            },
            'map': {
                'max_map_points':  ma.get('max_map_points', 8000),
                'cull_found_ratio': ma.get('cull_found_ratio', 0.15),
            },
            'loop_closure': slam_cfg.get('loop_closure', {}),
            'extrinsic': {'R_IC': np.eye(3), 'p_IC': np.zeros(3)},
        }

    def _load_extrinsic_yaml(self) -> dict:
        """Load calibrated extrinsic.yaml, fallback to identity."""
        yaml_path = self.get_parameter('extrinsic_yaml').value
        search_paths = [
            yaml_path,
            str(FilePath.home() / 'slam_ws/src/openvio_slam/config/extrinsic.yaml'),
            'config/extrinsic.yaml',
        ]
        for p in search_paths:
            try:
                with open(p, 'r') as f:
                    data = yaml.safe_load(f)
                ext = data.get('extrinsic', data)
                R_IC = np.array(ext['R_IC'], dtype=np.float64).reshape(3, 3)
                p_IC = np.array(ext['p_IC'], dtype=np.float64).ravel()
                self.get_logger().info(f"✅ Extrinsic loaded: {p}")
                return {'R_IC': R_IC, 'p_IC': p_IC}
            except Exception:
                continue
        self.get_logger().warn("⚠️  extrinsic.yaml not found — using identity.")
        return {'R_IC': np.eye(3), 'p_IC': np.zeros(3)}

    def _sanity_check_noise(self, est_cfg: dict):
        """Warn loudly if loaded IMU noise parameters look physically wrong.
        These are common copy-paste / units bugs (e.g. an Allan-variance
        script outputting variance instead of std-dev, or a stray 100x
        scale factor) and they are a leading cause of 'noisy trajectory'
        symptoms, since they let the filter's bias states wander freely."""
        warnings = []
        if est_cfg['accel_bias_walk'] > 0.01:
            warnings.append(
                f"accel_bias_walk={est_cfg['accel_bias_walk']:.6f} is unusually high "
                f"(typical phone/MEMS grade is ~1e-3 to 1e-4 m/s^3/sqrt(Hz)). This lets "
                f"the accelerometer bias estimate swing freely and is a likely source of "
                f"a noisy, jittery trajectory. Re-run imu_noise_calibration.py and check units.")
        if est_cfg['gyro_bias_walk'] > 0.001:
            warnings.append(
                f"gyro_bias_walk={est_cfg['gyro_bias_walk']:.6f} is unusually high "
                f"(typical is ~1e-4 to 1e-5 rad/s^2/sqrt(Hz)).")
        if est_cfg['accel_noise'] > 0.5:
            warnings.append(f"accel_noise={est_cfg['accel_noise']:.6f} looks high for a phone IMU.")
        if est_cfg['gyro_noise'] > 0.05:
            warnings.append(f"gyro_noise={est_cfg['gyro_noise']:.6f} looks high for a phone IMU.")
        for w in warnings:
            self.get_logger().warn(f"⚠️  CONFIG SANITY CHECK: {w}")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        if self._camera_ready:
            return

        cam_config = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5],
            'dist': list(msg.d) if msg.d else [0, 0, 0, 0, 0],
            'width': msg.width, 'height': msg.height,
        }
        extrinsic = self._load_extrinsic_yaml()

        self._camera_model = PinholeCameraModel(cam_config)
        self._feature_mgr = FeatureManager(self._config['frontend'], self._camera_model)
        self._preintegrator = IMUPreintegrator(self._config['estimator'])
        self._msckf = MSCKF(self._config['estimator'], self._camera_model, extrinsic)

        self._camera_ready = True
        self.get_logger().info(
            f"✅ Camera initialised: {cam_config['width']}×{cam_config['height']} "
            f"fx={cam_config['fx']:.1f} fy={cam_config['fy']:.1f}")
        self.get_logger().info("Hold robot stationary for 2s for gyro bias init...")

    def _imu_callback(self, msg: Imu):
        if not self._camera_ready:
            return
        gyro = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z])

        # ── Real dt from message timestamps ─────────────────────────────
        # The old code assumed a fixed 200 Hz stream (dt=0.005s) regardless
        # of actual arrival timing. Phone IMU streams over a network are
        # NOT evenly spaced (browser Sensor API rate varies by device,
        # DeviceMotion fallback is often ~50-60 Hz, and WiFi jitter adds
        # more irregularity). Integrating with a wrong/fixed dt directly
        # injects error into velocity and position — this was a real
        # contributor to your noisy trajectory.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_imu_t is None:
            dt = 1.0 / 100.0  # first sample: no prior delta yet, use a conservative nominal value
        else:
            dt = t - self._last_imu_t
            if dt <= 0.0:
                # Non-monotonic timestamp (out-of-order delivery, clock jump
                # on bridge fallback, or duplicate). Don't integrate garbage
                # dt — just record this sample for ZUPT and bail.
                self._last_imu_t = t
                self._zupt.add_measurement(gyro, accel)
                return
            # Clamp to guard against reconnect gaps / dropped packets
            # producing a huge single-step dt that would blow up the filter.
            dt = float(np.clip(dt, 1e-4, 0.05))
        self._last_imu_t = t

        self._imu_dt_window.append(dt)
        if len(self._imu_dt_window) > 100:
            self._imu_dt_window.pop(0)

        self._n_imu += 1
        self._zupt.add_measurement(gyro, accel)

        if not self._initialized:
            # store accel for gravity alignment
            self._init_accel_buffer.append(accel.copy())
            if len(self._init_accel_buffer) > 200:
                self._init_accel_buffer.pop(0)
            self._initializer.add_imu_measurement(gyro, accel, dt)
            return

        # IMU propagation – short locked section
        with self._lock:
            self._preintegrator.integrate(gyro, accel, dt)
            self._msckf.propagate(gyro, accel, dt)

            # ── Apply ZUPT / planar constraints HERE, at IMU rate ─────────────
            #
            # CRITICAL FIX: In the previous version these corrections lived
            # inside _process_frame(), which only runs when a camera frame
            # arrives (10-15 fps). That means:
            #   • If the camera is blocked/dropped/lagging, IMU propagation
            #     runs completely unconstrained — no ZUPT, no planar — so any
            #     residual gravity or bias error integrates freely and the
            #     position explodes in -Z (or whichever gravity-error direction).
            #   • Even when frames DO arrive, the correction only fires once
            #     per frame instead of once per IMU sample, so the filter has
            #     ~10 uncorrected IMU steps between each correction.
            #
            # Moving them here means they run every IMU sample, regardless of
            # whether a camera frame has arrived. This is the standard design
            # used in production VIO systems (ROVIO, MSCKF-VIO, OpenVINS).
            #
            # Visual-flow gate for ZUPT still applies — _last_visual_flow_px
            # and _visual_flow_valid are updated by _track_features() each
            # time a camera frame IS processed, so the gate state persists
            # across IMU callbacks between frames. This is correct: if the
            # last frame we saw showed 0.1px of flow, we trust that signal
            # for the next ~10 IMU samples until the next frame updates it.
            # Only if flow_valid is False (not enough features, no frames yet)
            # do we conservatively skip the update.
            zupt_imu_says_still = self._zupt.is_stationary()
            zupt_visually_confirmed = (
                self._visual_flow_valid and
                self._last_visual_flow_px < self._zupt_visual_flow_thresh
            )
            if zupt_imu_says_still and zupt_visually_confirmed:
                H_z, _, R_z = self._zupt.get_pseudo_measurement(
                    self._msckf.state.state_dim)
                self._msckf._kalman_update(H_z, -self._msckf.state.v_GI, R_z)
                self._last_zupt_applied = True
            else:
                self._last_zupt_applied = False

            for H_p, r_p, R_p in self._planar.get_pseudo_measurements(
                    self._msckf.state.state_dim, self._msckf.state.R_GI):
                r_p[0] = -self._msckf.state.v_GI[2]
                self._msckf._kalman_update(H_p, r_p, R_p)

            # ── Position sanity clamp ─────────────────────────────────────────
            # If the filter has blown up (common symptom: position jumps to
            # hundreds of meters while stationary), log loudly and reset
            # velocity to zero. We do NOT reset position — a large position
            # offset is still useful for debugging; zeroing it silently would
            # hide the problem. The right fix is always to address the root
            # cause (bad noise params, bad dt, bad gravity alignment), not to
            # quietly clamp.
            pos_norm = float(np.linalg.norm(self._msckf.state.p_GI))
            vel_norm = float(np.linalg.norm(self._msckf.state.v_GI))
            if pos_norm > 200.0 or vel_norm > 20.0:
                self.get_logger().warn(
                    f"⚠️  RUNAWAY STATE DETECTED: |p|={pos_norm:.1f}m |v|={vel_norm:.1f}m/s. "
                    f"Clamping velocity to zero. Check accel_bias_walk, gravity alignment, "
                    f"and IMU dt. This is NOT a TF/coordinate-frame bug.",
                    throttle_duration_sec=2.0)
                self._msckf.state.v_GI = np.zeros(3)

    def _image_callback(self, msg: Image):
        if not self._camera_ready:
            return
        try:
            h, w = msg.height, msg.width
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            # Fast NEON-optimised grayscale conversion
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Non-blocking push into the queue (blocks only if full)
            gray_undist = self._camera_model.undistort_image(gray)
            self._img_queue.put((msg.header.stamp, gray_undist))
        except queue.Full:
            self.get_logger().warn("Image queue full – dropping frame", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    # ── Image worker (blocking) ──────────────────────────────────────────────
    def _image_worker(self):
        """Blocking loop that processes frames as soon as they arrive."""
        while True:
            stamp, gray = self._img_queue.get()  # blocks until next frame
            try:
                self._process_frame(stamp, gray)
            except Exception as e:
                import traceback
                self.get_logger().error(
                    f"Frame error: {e}\n{traceback.format_exc()}",
                    throttle_duration_sec=2.0)

    # ── Core frame processing ────────────────────────────────────────────────
    def _process_frame(self, stamp, gray: np.ndarray):
        t0 = time.time()

        if not self._initialized:
            self._process_frame_init(stamp, gray)
            return

        self._frame_id += 1

        # ─ 1. Compute visual data (no lock) ──────────────────────────────
        pts_curr, valid = self._track_features(gray)
        new_pts, new_descs = self._extract_new_features(gray, valid)

        # ─ 2. MSCKF update (locked) ─────────────────────────────────────
        with self._lock:
            marginalized, _ = self._feature_mgr.process_frame(
                gray, pts_curr, valid, new_pts, new_descs,
                window_size=self._msckf.state.max_window)

            if marginalized:
                self._msckf.update_with_features(marginalized)

            self._msckf.augment_state(self._frame_id)
            self._preintegrator.reset(self._msckf.state.b_g, self._msckf.state.b_a)

            # Copy camera window for thread‑safe triangulation
            cam_window = list(self._msckf.state.camera_window)

        # ─ 3. Triangulation outside lock (read‑only, uses copy) ─────────
        if marginalized:
            self._triangulate_and_store(marginalized, cam_window)

        # ─ 4. Publish odometry ──────────────────────────────────────────
        self._publish_odometry(stamp)
        self._prev_img = gray.copy()
        self._frame_times.append(time.time() - t0)

    def _track_features(self, gray):
        """KLT tracking – returns (pts_curr, valid) or empty arrays.
        Also records mean optical-flow magnitude of matched points, used
        as the visual-motion gate for ZUPT (see _process_frame)."""
        if self._prev_img is not None and self._feature_mgr.n_active > 0:
            prev_pts = self._feature_mgr.current_pts
            pts_curr, valid = self._klt.track(self._prev_img, gray, prev_pts)
            n_valid = int(np.count_nonzero(valid))
            if n_valid > 0:
                flow = np.linalg.norm(pts_curr[valid] - prev_pts[valid], axis=1)
                self._last_visual_flow_px = float(np.mean(flow))
            else:
                self._last_visual_flow_px = 0.0
            # Only trust this as a "no motion" signal if we actually have
            # enough tracked features to be confident — in a feature-poor
            # scene (low texture, dark) we'd rather skip a ZUPT update than
            # risk falsely zeroing velocity during real motion.
            self._visual_flow_valid = n_valid >= self._zupt_min_features
            return pts_curr, valid
        self._last_visual_flow_px = 0.0
        self._visual_flow_valid = False
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    def _extract_new_features(self, gray, valid):
        """ORB extraction – returns (new_pts, new_descs) or empty arrays."""
        if self._feature_mgr.needs_new_features or self._prev_img is None:
            excl_mask = self._orb.create_exclusion_mask(
                gray.shape,
                self._feature_mgr.current_pts[valid] if valid.any() else None)
            return self._orb.detect_and_describe(gray, excl_mask)
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

    # ── Initialisation logic ────────────────────────────────────────────────
    def _process_frame_init(self, stamp, gray: np.ndarray):
        if self._prev_img is None:
            pts, _ = self._orb.detect_and_describe(gray)
            self._prev_kps = pts if len(pts) > 0 else np.zeros((0, 2), dtype=np.float32)
            self._prev_img = gray.copy()
            self._initializer.add_visual_frame(
                float(stamp.sec) + stamp.nanosec * 1e-9,
                self._R_vis.copy(), self._p_vis.copy())
            return

        prev_matched = np.zeros((0, 2), dtype=np.float32)
        curr_matched = np.zeros((0, 2), dtype=np.float32)
        if self._prev_kps is not None and len(self._prev_kps) >= 8:
            pts_curr, valid = self._klt.track(self._prev_img, gray, self._prev_kps)
            prev_matched = self._prev_kps[valid]
            curr_matched = pts_curr[valid]

        if len(prev_matched) >= 8:
            K = self._camera_model.K
            try:
                E, inlier_mask = cv2.findEssentialMat(
                    prev_matched, curr_matched, K, cv2.RANSAC, 0.999, 2.0)
                if E is not None and inlier_mask is not None:
                    inlier_mask = inlier_mask.ravel().astype(bool)
                    if inlier_mask.sum() >= 8:
                        _, R_rel, t_rel, _ = cv2.recoverPose(
                            E, prev_matched[inlier_mask], curr_matched[inlier_mask], K)
                        self._R_vis = self._R_vis @ R_rel
                        self._p_vis = self._p_vis + self._R_vis @ t_rel.flatten()
            except Exception as e:
                self.get_logger().warn(f"Init pose estimation: {e}", throttle_duration_sec=1.0)

        # Refresh keypoint pool
        new_pts, _ = self._orb.detect_and_describe(gray)
        if len(curr_matched) > 0 and len(new_pts) > 0:
            self._prev_kps = np.vstack([curr_matched, new_pts])
        elif len(curr_matched) > 0:
            self._prev_kps = curr_matched
        elif len(new_pts) > 0:
            self._prev_kps = new_pts
        else:
            self._prev_kps = np.zeros((0, 2), dtype=np.float32)

        max_pts = self._config['frontend']['n_features']
        if len(self._prev_kps) > max_pts:
            self._prev_kps = self._prev_kps[:max_pts]

        self._prev_img = gray.copy()

        self._initializer.add_visual_frame(
            float(stamp.sec) + stamp.nanosec * 1e-9,
            self._R_vis.copy(), self._p_vis.copy())

        if self._initializer.is_initialized():
            self._apply_initialization()

    def _apply_initialization(self):
        init_state = self._initializer.get_initial_state()
        with self._lock:
            self._msckf.state.initialize_from(init_state)
            self._preintegrator.reset(init_state['b_g'], init_state['b_a'])

        # Align gravity before setting origin (ensures correct orientation)
        self._align_gravity_with_buffer()

        # Force starting at origin and stationary
        with self._lock:
            self._msckf.state.p_GI = np.zeros(3)
            self._msckf.state.v_GI = np.zeros(3)

        self._initialized = True
        self.get_logger().info(
            f"✅ VIO initialised at origin. Scale={init_state['scale']:.3f} "
            f"|v|={np.linalg.norm(init_state['v_GI']):.2f} m/s")

    def _align_gravity_with_buffer(self):
        """Rotate the current state so that the measured gravity (from buffer)
        aligns with the world's [0,0,-g] direction."""
        if len(self._init_accel_buffer) < 10:
            self.get_logger().warn("Not enough static IMU samples; skipping gravity alignment.")
            return

        mean_accel = np.mean(np.array(self._init_accel_buffer), axis=0)
        g = np.linalg.norm(mean_accel)
        if g < 1e-3:
            return

        gravity_dir = mean_accel / g   # unit vector in IMU frame
        world_gravity = np.array([0.0, 0.0, -1.0])  # we want gravity to point down

        # Compute rotation R that maps gravity_dir → world_gravity
        v = np.cross(gravity_dir, world_gravity)
        s = np.linalg.norm(v)
        c = np.dot(gravity_dir, world_gravity)
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            R = np.eye(3)   # already aligned or exactly opposite

        # Get current quaternion, rotate it, and write back
        old_q = self._msckf.state.q_GI   # quaternion (w, x, y, z)
        old_R = quat_to_rot(old_q)       # convert to rotation matrix
        new_R = R @ old_R                # apply alignment rotation
        new_q = rot_to_quat(new_R)       # convert back to quaternion
        self._msckf.state.q_GI = new_q   # set the new orientation

        self.get_logger().info(f"Gravity aligned using {len(self._init_accel_buffer)} static samples.")
    
    # ── Triangulation ───────────────────────────────────────────────────────
    def _triangulate_and_store(self, marginalized_features: list, cam_window: list):
        """Triangulate new map points. Uses a pre-built camera dictionary for speed."""
        candidates = 0
        reject_no_obs   = 0
        reject_dlt_fail = 0
        reject_behind   = 0
        reject_parallax = 0
        reject_reproj   = 0
        reject_depth    = 0
        accepted        = 0

        cam_dict = {cam.frame_id: cam for cam in cam_window}
        new_pts = []

        for feat in marginalized_features:
            candidates += 1
            if feat.n_observations < 3:
                reject_no_obs += 1
                continue

            poses = []       # list of (R_GC, p_GC)
            obs_norm = []    # list of (u, v)
            for frame_id, uv_norm in feat.observations:
                cam = cam_dict.get(frame_id)
                if cam is not None:
                    poses.append((cam.R_GC, cam.p_GC))
                    obs_norm.append(np.asarray(uv_norm, dtype=np.float64))

            if len(poses) < 3:
                reject_no_obs += 1
                continue

            p3d = self._triangulate_dlt(poses, obs_norm)
            if p3d is None:
                reject_dlt_fail += 1
                continue

            # Sanity checks (relaxed for indoor Pi‑5 usage)
            if not np.all(np.isfinite(p3d)):
                continue

            behind = False
            for (R_GC, p_GC) in poses:
                p_cam = R_GC.T @ (p3d - p_GC)
                if p_cam[2] < 0.05:
                    behind = True
                    break

            if behind:
                reject_behind += 1
                continue

            # Parallax check (relaxed)
            bearing_0 = poses[0][0] @ np.array([obs_norm[0][0], obs_norm[0][1], 1.0])
            bearing_0 /= np.linalg.norm(bearing_0)
            bearing_n = poses[-1][0] @ np.array([obs_norm[-1][0], obs_norm[-1][1], 1.0])
            bearing_n /= np.linalg.norm(bearing_n)
            cos_ang = float(np.clip(bearing_0 @ bearing_n, -1.0, 1.0))
            parallax_deg = np.degrees(np.arccos(cos_ang))

            if parallax_deg < 2.0: # was 0.5
                reject_parallax += 1
                continue

            fx = self._camera_model.fx
            reproj_ok = True
            for (R_GC, p_GC), uv in zip(poses, obs_norm):
                p_cam = R_GC.T @ (p3d - p_GC)
                if p_cam[2] < 1e-6:
                    reproj_ok = False
                    break
                proj = p_cam[:2] / p_cam[2]
                if np.linalg.norm(proj - uv) * fx > 12.0:   # was 15.0
                    reproj_ok = False
                    break
            if not reproj_ok:
                reject_reproj += 1
                continue

            first_p_cam = poses[0][0].T @ (p3d - poses[0][1])
            depth = float(first_p_cam[2])
            
            if depth < 0.1 or depth > 30.0: # was 0.05 and 50.0
                reject_depth += 1
                continue
            
            new_pts.append(p3d.astype(np.float32))
            accepted += 1

        # Add to circular buffer.
        #
        # OLD BEHAVIOUR (bug): when the buffer was full, new points
        # overwrote starting from index 0, which on every subsequent
        # overflow effectively threw away almost the entire accumulated
        # map and replaced it with just the latest batch. That contradicts
        # the "persistent map" behaviour described in the project report.
        # NEW BEHAVIOUR: a real wraparound circular buffer — oldest points
        # are overwritten in order, newest points are always kept, and nothing
        # is bulk-discarded.
        if new_pts:
            with self._map_lock:
                new_arr = np.asarray(new_pts, dtype=np.float32)
                n_new = new_arr.shape[0]
                if n_new >= self._MAX_MAP_PTS:
                    self._map_points[:] = new_arr[-self._MAX_MAP_PTS:]
                    self._map_pts_count = self._MAX_MAP_PTS
                    self._map_write_ptr = 0
                else:
                    end = self._map_write_ptr + n_new
                    if end <= self._MAX_MAP_PTS:
                        self._map_points[self._map_write_ptr:end] = new_arr
                    else:
                        first = self._MAX_MAP_PTS - self._map_write_ptr
                        self._map_points[self._map_write_ptr:] = new_arr[:first]
                        self._map_points[:n_new - first] = new_arr[first:]
                    self._map_write_ptr = end % self._MAX_MAP_PTS
                    self._map_pts_count = min(self._MAX_MAP_PTS, self._map_pts_count + n_new)

            self.get_logger().info(
                f"Triangulation: {candidates} candidates → {accepted} accepted "
                f"(obs:{reject_no_obs} dlt:{reject_dlt_fail} behind:{reject_behind} "
                f"parallax:{reject_parallax} reproj:{reject_reproj} depth:{reject_depth})",
                throttle_duration_sec=2.0)

    @staticmethod
    def _triangulate_dlt(poses, obs_norm):
        """Direct Linear Transform triangulation."""
        A_mat = []
        for (R_GC, p_GC), uv in zip(poses, obs_norm):
            R = R_GC.T
            t = -(R @ p_GC)
            P = np.hstack([R, t.reshape(3, 1)])
            u, v = float(uv[0]), float(uv[1])
            A_mat.append(u * P[2] - P[0])
            A_mat.append(v * P[2] - P[1])

        A_np = np.array(A_mat, dtype=np.float64)
        if A_np.shape[0] < 4:
            return None
        try:
            _, s, Vt = np.linalg.svd(A_np)
        except np.linalg.LinAlgError:
            return None
        if s[-1] < 1e-10 or s[0] / s[-1] > 1e5:
            return None

        X_h = Vt[-1]
        if abs(X_h[3]) < 1e-9:
            return None
        return (X_h[:3] / X_h[3]).astype(np.float64)

    # ── Publishers ────────────────────────────────────────────────────────────
    def _publish_odometry(self, stamp):
        s = self._msckf.state
        q, p, v = s.q_GI, s.p_GI, s.v_GI

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(p[0])
        odom.pose.pose.position.y = float(p[1])
        odom.pose.pose.position.z = float(p[2])
        odom.pose.pose.orientation.w = float(q[0])
        odom.pose.pose.orientation.x = float(q[1])
        odom.pose.pose.orientation.y = float(q[2])
        odom.pose.pose.orientation.z = float(q[3])
        odom.twist.twist.linear.x = float(v[0])
        odom.twist.twist.linear.y = float(v[1])
        odom.twist.twist.linear.z = float(v[2])
        self._pub_odom.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = float(p[0])
        tf_msg.transform.translation.y = float(p[1])
        tf_msg.transform.translation.z = float(p[2])
        tf_msg.transform.rotation.w = float(q[0])
        tf_msg.transform.rotation.x = float(q[1])
        tf_msg.transform.rotation.y = float(q[2])
        tf_msg.transform.rotation.z = float(q[3])
        self._tf_broadcaster.sendTransform(tf_msg)

        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self._path.header.stamp = stamp
        self._path.poses.append(ps)
        if len(self._path.poses) > 500:
            self._path.poses = self._path.poses[-500:]
        self._pub_path.publish(self._path)

    def _timer_publish_map_points(self):
        if not self._initialized:
            return
        with self._map_lock:
            count = self._map_pts_count
            if count == 0:
                return
            pts_arr = self._map_points[:count]

        N = pts_arr.shape[0]
        if not np.all(np.isfinite(pts_arr)):
            pts_arr = pts_arr[np.all(np.isfinite(pts_arr), axis=1)]
            N = pts_arr.shape[0]
            if N == 0:
                return

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = PointCloud2()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = 'odom'
        cloud.height = 1
        cloud.width = N
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * N
        cloud.is_dense = True
        cloud.data = pts_arr.tobytes()
        self._pub_pts.publish(cloud)

    def _print_stats(self):
        n_map = self._map_pts_count
        if not self._frame_times:
            status = "UNINITIALIZED" if not self._initialized else "RUNNING"
            self.get_logger().info(
                f"[{status}] IMU: {self._n_imu}  Frames: {self._frame_id}  MapPts: {n_map}")
            return

        avg_ms = 1000 * np.mean(self._frame_times[-50:])
        n_feats = self._feature_mgr.n_active if self._feature_mgr else 0
        p = self._msckf.state.p_GI if self._initialized else np.zeros(3)

        imu_hz = (1.0 / np.mean(self._imu_dt_window)) if self._imu_dt_window else 0.0
        zupt_imu = self._zupt.is_stationary()

        self.get_logger().info(
            f"  Frames: {self._frame_id} | Avg: {avg_ms:.1f}ms | "
            f"Features: {n_feats} | MapPts: {n_map} | "
            f"Pos: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]m | "
            f"IMU: {self._n_imu} ({imu_hz:.0f}Hz) | "
            f"ZUPT(imu={'Y' if zupt_imu else 'n'},"
            f"flow={self._last_visual_flow_px:.1f}px,"
            f"applied={'YES' if self._last_zupt_applied else 'no'})",
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = OpenVIOSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()