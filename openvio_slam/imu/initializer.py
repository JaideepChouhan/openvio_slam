"""
imu/initializer.py
==================
VIO initialization: recover initial orientation, velocity, biases, and scale.

Stage 1 - Static initialization (robot is still, 2 seconds):
    - gravity direction gives roll and pitch (not yaw)
    - gyro mean gives initial gyro bias estimate
    - yaw set to 0 (unknown without magnetometer)

Stage 2 - Dynamic initialization (robot moves):
    - visual-inertial alignment recovers metric scale
    - velocity and accelerometer bias solved simultaneously
"""

import numpy as np
from ..utils.so3 import exp_map, hat
from ..utils.quaternion import quat_to_rot, rot_to_quat

GRAVITY_MAGNITUDE = 9.81


class VIOInitializer:
    """
    Manages the two-stage VIO initialization.

    Usage:
        init = VIOInitializer(config)
        while not init.is_initialized():
            init.add_imu_measurement(gyro, accel, dt)
            if new_frame:
                init.add_visual_frame(timestamp, R_visual, p_visual)
        state = init.get_initial_state()
    """

    def __init__(self, config: dict):
        self.static_duration = config.get('static_init_duration', 2.0)
        self.min_parallax    = np.deg2rad(config.get('min_parallax_deg', 5.0))
        self.min_excitation  = config.get('min_imu_excitation', 0.5)

        self._imu_buffer    = []
        self._static_done   = False
        self._init_done     = False
        self._init_state    = None

        self._static_gyro_sum    = np.zeros(3)
        self._static_accel_sum   = np.zeros(3)
        self._static_count       = 0
        self._static_time        = 0.0

        self._visual_frames = []
        self._init_R        = np.eye(3)
        self._init_bias_g   = np.zeros(3)
        
        self._latest_R_vis = np.eye(3)   # store latest visual rotation
        self._latest_p_vis = np.zeros(3) # store latest visual position
        self._fallback_used = False      # flag to know if we fell back to default scale

    def add_imu_measurement(self, gyro, accel, dt):
        gyro  = np.asarray(gyro,  dtype=float)
        accel = np.asarray(accel, dtype=float)

        self._imu_buffer.append((gyro.copy(), accel.copy(), dt))

        if not self._static_done:
            self._static_gyro_sum  += gyro
            self._static_accel_sum += accel
            self._static_count     += 1
            self._static_time      += dt

            if self._static_time >= self.static_duration:
                self._finish_static_init()

    def add_visual_frame(self, timestamp, R_visual, p_visual):
        if not self._static_done:
            return
        # Store the latest visual pose (world frame)
        self._latest_R_vis = np.asarray(R_visual, dtype=float)
        self._latest_p_vis = np.asarray(p_visual, dtype=float)
        self._visual_frames.append({
            'timestamp': timestamp,
            'R': self._latest_R_vis.copy(),
            'p': self._latest_p_vis.copy(),
        })
        if len(self._visual_frames) >= 10:
            self._attempt_dynamic_init()

    def is_initialized(self) -> bool:
        return self._init_done

    def get_initial_state(self) -> dict:
        if not self._init_done:
            raise RuntimeError("Initialization not complete yet")
        return self._init_state

    def _finish_static_init(self):
        n = self._static_count
        mean_gyro  = self._static_gyro_sum  / n
        mean_accel = self._static_accel_sum / n

        accel_mag = np.linalg.norm(mean_accel)
        if abs(accel_mag - GRAVITY_MAGNITUDE) > 0.3:
            self._static_gyro_sum  = np.zeros(3)
            self._static_accel_sum = np.zeros(3)
            self._static_count     = 0
            self._static_time      = 0.0
            return

        self._init_bias_g = mean_gyro.copy()

        g_body = mean_accel / accel_mag
        roll  = np.arctan2(g_body[1], -g_body[2])
        pitch = np.arctan2(-g_body[0], np.sqrt(g_body[1]**2 + g_body[2]**2))

        Rx = exp_map(np.array([roll, 0, 0]))
        Ry = exp_map(np.array([0, pitch, 0]))
        self._init_R  = Ry @ Rx
        self._static_done = True
    def _attempt_dynamic_init(self):
        N = len(self._visual_frames) - 1
        if N < 5:
            return

        accels = np.array([a for (_, a, _) in self._imu_buffer[-200:]])
        accel_var = np.var(accels, axis=0).sum()
        if accel_var < self.min_excitation**2:
            # Not enough excitation -> use visual pose fallback
            self._fallback_used = True
            self._init_state = {
                'q_GI': rot_to_quat(self._latest_R_vis),
                'p_GI': self._latest_p_vis.copy(),
                'v_GI': np.zeros(3),
                'b_g':  self._init_bias_g,
                'b_a':  np.zeros(3),
                'scale': 1.0,
            }
            self._init_done = True
            print("⚠️ Dynamic init: not enough IMU excitation, using visual pose with scale=1.0")
            return

        # Build linear system (existing code)...
        # ... if H, b, solve fail, fallback:
        try:
            result, _, rank, sv = np.linalg.lstsq(H, b, rcond=None)
        except:
            self._fallback_used = True
            self._init_state = {
                'q_GI': rot_to_quat(self._latest_R_vis),
                'p_GI': self._latest_p_vis.copy(),
                'v_GI': np.zeros(3),
                'b_g':  self._init_bias_g,
                'b_a':  np.zeros(3),
                'scale': 1.0,
            }
            self._init_done = True
            print("⚠️ Dynamic init: linear solve failed, using visual pose with scale=1.0")
            return

        # Check condition number and scale
        if sv[-1] < 1e-6 or sv[0]/sv[-1] > 1e4 or scale <= 0 or scale > 100:
            self._fallback_used = True
            self._init_state = {
                'q_GI': rot_to_quat(self._latest_R_vis),
                'p_GI': self._latest_p_vis.copy(),
                'v_GI': np.zeros(3),
                'b_g':  self._init_bias_g,
                'b_a':  np.zeros(3),
                'scale': 1.0,
            }
            self._init_done = True
            print("⚠️ Dynamic init: scale invalid or ill-conditioned, using visual pose with scale=1.0")
            return

        # Success path (same as before)
        b_a_init = result[3*(N+1)+1:3*(N+1)+4]
        v_0_init = result[0:3]
        self._init_state = {
            'q_GI':  rot_to_quat(self._init_R),
            'p_GI':  np.zeros(3),
            'v_GI':  v_0_init,
            'b_g':   self._init_bias_g,
            'b_a':   b_a_init,
            'scale': scale,
        }
        self._init_done = True
        print("✅ Dynamic init: scale={:.3f}, velocity norm={:.3f} m/s".format(scale, np.linalg.norm(v_0_init)))