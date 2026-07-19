"""
imu/preintegrator.py
====================
IMU Pre-integration following Forster et al. (2015).

Reference:
    Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
    Odometry." IEEE TRO. arxiv.org/abs/1512.02363

The core idea:
    Between two camera frames, IMU fires M times. We summarize this as:
        DeltaR: rotation increment (SO(3))
        Deltav: velocity increment (R3)
        Deltap: position increment (R3)
    expressed in the body frame at t_k, independent of current state estimate.

Integration method: RK4 (Runge-Kutta 4th order) for accuracy.

Bias model:
    Gyroscope:     omega_meas = omega_true + b_g + n_g
    Accelerometer: a_meas = a_true + b_a + n_a
"""

import numpy as np
from ..utils.so3 import exp_map, hat, right_jacobian
from ..utils.quaternion import quat_to_rot, rot_to_quat, quat_multiply, quat_normalize


class IMUPreintegrator:
    """
    Accumulates IMU measurements between two camera frames.

    Usage:
        preint = IMUPreintegrator(noise_params)
        preint.reset(bias_g, bias_a)
        # In IMU callback:
        preint.integrate(gyro, accel, dt)
        # When camera frame arrives:
        delta = preint.get_delta()
        preint.reset(updated_bias_g, updated_bias_a)
    """

    def __init__(self, noise_params: dict):
        self.sigma_g  = noise_params.get('gyro_noise', 0.005)
        self.sigma_a  = noise_params.get('accel_noise', 0.05)
        self.sigma_bg = noise_params.get('gyro_bias_walk', 0.0001)
        self.sigma_ba = noise_params.get('accel_bias_walk', 0.001)
        self._bias_g  = np.zeros(3)
        self._bias_a  = np.zeros(3)
        self._reset_state()

    def _reset_state(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)

        self.dR_dbg = np.zeros((3, 3))
        self.dv_dbg = np.zeros((3, 3))
        self.dv_dba = np.zeros((3, 3))
        self.dp_dbg = np.zeros((3, 3))
        self.dp_dba = np.zeros((3, 3))

        self.covariance = np.zeros((9, 9))
        self.dt_sum = 0.0
        self.n_measurements = 0

    def reset(self, bias_g: np.ndarray, bias_a: np.ndarray):
        """Start a new pre-integration interval."""
        self._bias_g = np.asarray(bias_g, dtype=float).copy()
        self._bias_a = np.asarray(bias_a, dtype=float).copy()
        self._reset_state()

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """Add one IMU measurement. Called at IMU rate (~200 Hz)."""
        gyro  = np.asarray(gyro,  dtype=float)
        accel = np.asarray(accel, dtype=float)

        omega = gyro  - self._bias_g
        a_hat = accel - self._bias_a

        # RK4 integration on SO(3) x R3 x R3
        k1_v = self.delta_R @ a_hat
        k1_p = self.delta_v

        R_mid_1 = self.delta_R @ exp_map(omega * dt / 2)
        k2_v = R_mid_1 @ a_hat
        k2_p = self.delta_v + k1_v * dt / 2

        k3_v = R_mid_1 @ a_hat
        k3_p = self.delta_v + k2_v * dt / 2

        R_end = self.delta_R @ exp_map(omega * dt)
        k4_v = R_end @ a_hat
        k4_p = self.delta_v + k3_v * dt

        delta_v_update = (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6

        self.delta_p = self.delta_p + self.delta_v * dt + 0.5 * (self.delta_R @ a_hat) * dt**2
        self.delta_v = self.delta_v + delta_v_update
        self.delta_R = self.delta_R @ exp_map(omega * dt)

        # Re-orthogonalize periodically
        if self.n_measurements % 100 == 99:
            U, _, Vt = np.linalg.svd(self.delta_R)
            self.delta_R = U @ Vt

        # Jacobian propagation
        Jr = right_jacobian(omega * dt)
        R_dt = exp_map(omega * dt)

        self.dR_dbg = R_dt.T @ self.dR_dbg - Jr * dt
        self.dv_dbg = self.dv_dbg + (-self.delta_R @ hat(a_hat) @ self.dR_dbg) * dt
        self.dv_dba = self.dv_dba + (-self.delta_R) * dt
        self.dp_dbg = self.dp_dbg + self.dv_dbg * dt + \
                      (-self.delta_R @ hat(a_hat) @ self.dR_dbg) * dt**2 / 2
        self.dp_dba = self.dp_dba + self.dv_dba * dt + (-self.delta_R) * dt**2 / 2

        # Covariance propagation
        F = np.eye(9)
        F[3:6, 0:3] = -self.delta_R @ hat(a_hat) * dt
        F[6:9, 0:3] = -self.delta_R @ hat(a_hat) * dt**2 / 2
        F[6:9, 3:6] = np.eye(3) * dt

        G = np.zeros((9, 6))
        G[0:3, 0:3] = -self.delta_R * dt
        G[3:6, 3:6] = -self.delta_R * dt
        G[6:9, 3:6] = -self.delta_R * dt**2 / 2

        Q_c = np.diag([self.sigma_g**2]*3 + [self.sigma_a**2]*3)
        self.covariance = F @ self.covariance @ F.T + G @ Q_c @ G.T

        self.dt_sum += dt
        self.n_measurements += 1

    def get_delta(self) -> dict:
        """Return the accumulated pre-integration result."""
        return {
            'delta_R':    self.delta_R.copy(),
            'delta_v':    self.delta_v.copy(),
            'delta_p':    self.delta_p.copy(),
            'covariance': self.covariance.copy(),
            'jacobians': {
                'dR_dbg': self.dR_dbg.copy(),
                'dv_dbg': self.dv_dbg.copy(),
                'dv_dba': self.dv_dba.copy(),
                'dp_dbg': self.dp_dbg.copy(),
                'dp_dba': self.dp_dba.copy(),
            },
            'dt':     self.dt_sum,
            'n_meas': self.n_measurements,
        }

    def correct_for_bias_change(self, delta_bg: np.ndarray, delta_ba: np.ndarray) -> dict:
        """First-order correction when bias estimate changes (avoids re-integration)."""
        delta_bg = np.asarray(delta_bg, dtype=float)
        delta_ba = np.asarray(delta_ba, dtype=float)
        d = self.get_delta()
        j = d['jacobians']

        d['delta_R'] = d['delta_R'] @ exp_map(j['dR_dbg'] @ delta_bg)
        d['delta_v'] = d['delta_v'] + j['dv_dbg'] @ delta_bg + j['dv_dba'] @ delta_ba
        d['delta_p'] = d['delta_p'] + j['dp_dbg'] @ delta_bg + j['dp_dba'] @ delta_ba
        return d
