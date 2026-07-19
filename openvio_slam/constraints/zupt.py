"""
constraints/zupt.py
===================
Zero-Velocity Update (ZUPT) for ground robots.

When robot is stationary, inject a tight pseudo-measurement v = 0.
This dramatically reduces accelerometer bias drift.
"""

import numpy as np
from collections import deque


class ZUPTDetector:
    """Detects zero-velocity conditions from IMU signal."""

    def __init__(self, config: dict):
        self.gyro_var_threshold = config.get('zupt_gyro_var_threshold', 0.001)
        self.zupt_noise         = config.get('zupt_noise', 0.001)
        self.window_size        = 20
        self.gravity            = config.get('gravity', 9.81)

        self._gyro_window  = deque(maxlen=self.window_size)
        self._accel_window = deque(maxlen=self.window_size)

    def add_measurement(self, gyro: np.ndarray, accel: np.ndarray):
        self._gyro_window.append(np.linalg.norm(gyro))
        self._accel_window.append(np.linalg.norm(accel))

    def is_stationary(self) -> bool:
        if len(self._gyro_window) < self.window_size:
            return False

        gyro_var   = np.var(np.array(self._gyro_window))
        accel_mean = np.mean(np.array(self._accel_window))

        return bool(gyro_var < self.gyro_var_threshold and
                    abs(accel_mean - self.gravity) < 0.3)

    def get_pseudo_measurement(self, state_dim: int, imu_dim: int = 15):
        """Build ZUPT pseudo-measurement: v_GI = 0"""
        H = np.zeros((3, state_dim))
        H[0:3, 6:9] = np.eye(3)
        r = np.zeros(3)
        R_noise = np.eye(3) * self.zupt_noise**2
        return H, r, R_noise
