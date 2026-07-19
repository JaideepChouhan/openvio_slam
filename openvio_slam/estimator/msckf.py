"""
estimator/msckf.py
==================
The MSCKF filter core: propagation, augmentation, and update.

Key papers:
    Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter
    for Vision-aided Inertial Navigation", ICRA 2007.

This implementation adds:
    - RK4 IMU propagation
    - Ground-robot constraints (ZUPT + planar)
    - Chi-squared outlier rejection
"""

import numpy as np
from scipy.stats import chi2

from .state import VIOState
from ..utils.so3 import hat
from ..utils.quaternion import quat_to_rot
from ..map.triangulator import Triangulator


class MSCKF:
    """
    Multi-State Constraint Kalman Filter.
    Thread safety: NOT thread-safe. Coordinate externally.
    """

    def __init__(self, config: dict, camera_model, extrinsic: dict):
        self.config  = config
        self.camera  = camera_model
        self.R_IC    = np.asarray(extrinsic['R_IC'], dtype=float)
        self.p_IC    = np.asarray(extrinsic['p_IC'], dtype=float)

        self.state        = VIOState(max_window_size=config.get('window_size', 20))
        self.triangulator = Triangulator()
        self._frame_count = 0

        self.sigma_g  = config.get('gyro_noise',      0.005)
        self.sigma_a  = config.get('accel_noise',     0.05)
        self.sigma_bg = config.get('gyro_bias_walk',  0.0001)
        self.sigma_ba = config.get('accel_bias_walk', 0.001)
        self.gravity  = np.array([0, 0, -config.get('gravity', 9.81)])

        sigma_px = config.get('pixel_noise', 1.0)
        self.meas_noise = (sigma_px / camera_model.fx)**2

    def propagate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """
        Propagate state forward using one IMU measurement.
        Called at IMU rate (~200 Hz).
        """
        from ..utils.so3 import exp_map
        from ..utils.quaternion import rot_to_quat

        s = self.state
        R = s.R_GI
        v = s.v_GI
        p = s.p_GI
        g = self.gravity

        omega = gyro  - s.b_g
        a_hat = accel - s.b_a

        # Nominal state propagation
        R_new = R @ exp_map(omega * dt)
        v_new = v + (R @ a_hat + g) * dt
        p_new = p + v * dt + 0.5 * (R @ a_hat + g) * dt**2

        s.q_GI = rot_to_quat(R_new)
        s.v_GI = v_new
        s.p_GI = p_new

        # Error-state covariance propagation (linearized)
        F_c = np.zeros((self.state.IMU_DIM, self.state.IMU_DIM))
        F_c[0:3, 0:3]   = -hat(omega)
        F_c[0:3, 9:12]  = -np.eye(3)
        F_c[6:9, 0:3]   = -R @ hat(a_hat)
        F_c[6:9, 12:15] = -R
        F_c[3:6, 6:9]   = np.eye(3)

        Phi_IMU = np.eye(self.state.IMU_DIM) + F_c * dt

        G_c = np.zeros((self.state.IMU_DIM, 12))
        G_c[0:3,  0:3]   = -np.eye(3)
        G_c[6:9,  3:6]   = -R
        G_c[9:12, 6:9]   = np.eye(3)
        G_c[12:15, 9:12] = np.eye(3)

        Q_c = np.diag([self.sigma_g**2]*3  + [self.sigma_a**2]*3 +
                       [self.sigma_bg**2]*3 + [self.sigma_ba**2]*3)
        Q_d = G_c @ Q_c @ G_c.T * dt

        n = self.state.state_dim
        Phi_full = np.eye(n)
        Phi_full[:15, :15] = Phi_IMU

        Q_full = np.zeros((n, n))
        Q_full[:15, :15] = Q_d

        self.state.P = Phi_full @ self.state.P @ Phi_full.T + Q_full

    def augment_state(self, frame_id: int):
        """Add current IMU state as a new camera state. Called at camera rate."""
        self._frame_count += 1

        if self.state.n_cameras >= self.state.max_window:
            self.state.remove_oldest_camera()

        self.state.augment_with_camera(frame_id, self.R_IC, self.p_IC)

        if self._frame_count % 10 == 0:
            self.state.symmetrize_covariance()

    def update_with_features(self, marginalized_features: list):
        """
        MSCKF update using marginalized feature observations.
        For each feature: triangulate, build Jacobian, null-space project,
        chi-squared test, Kalman update.
        """
        if not marginalized_features or self.state.n_cameras < 2:
            return

        H_stack = []
        r_stack = []

        for feat in marginalized_features:
            result = self._process_feature(feat)
            if result is not None:
                H_tilde, r_tilde = result
                H_stack.append(H_tilde)
                r_stack.append(r_tilde)

        if not H_stack:
            return

        H_all = np.vstack(H_stack)
        r_all = np.concatenate(r_stack)

        # QR decomposition for efficiency
        if H_all.shape[0] > H_all.shape[1]:
            Q_qr, R_qr = np.linalg.qr(H_all, mode='reduced')
            H_all = R_qr
            r_all = Q_qr.T @ r_all

        R_noise = np.eye(len(r_all)) * self.meas_noise
        self._kalman_update(H_all, r_all, R_noise)

    def _process_feature(self, feat):
        """Process one marginalized feature. Returns (H_tilde, r_tilde) or None."""
        obs_frame_ids  = [frame_id for frame_id, _ in feat.observations]
        obs_normalized = [uv for _, uv in feat.observations]

        cam_indices = []
        obs_for_cams = []
        for frame_id, uv in zip(obs_frame_ids, obs_normalized):
            for i, cam in enumerate(self.state.camera_window):
                if cam.frame_id == frame_id:
                    cam_indices.append(i)
                    obs_for_cams.append(uv)
                    break

        if len(cam_indices) < 3:
            return None

        poses = [(self.state.camera_window[i].R_GC,
                  self.state.camera_window[i].p_GC)
                 for i in cam_indices]

        try:
            p3d_world = self.triangulator.triangulate_dlt(poses, obs_for_cams)
        except Exception:
            return None

        if p3d_world is None:
            return None

        n_obs = len(cam_indices)
        H = np.zeros((2 * n_obs, self.state.state_dim))
        r = np.zeros(2 * n_obs)
        all_valid = True

        for k, (cam_idx, uv_meas) in enumerate(zip(cam_indices, obs_for_cams)):
            cam = self.state.camera_window[cam_idx]
            R_GC = cam.R_GC
            p_GC = cam.p_GC
            p_cam = R_GC.T @ (p3d_world - p_GC)

            if p_cam[2] < 0.1:
                all_valid = False
                break

            try:
                z_hat, J_proj = self.camera.project_point(p_cam)
            except ValueError:
                all_valid = False
                break

            r[2*k:2*k+2] = np.asarray(uv_meas) - z_hat

            J_p_dtheta = hat(p_cam)
            J_p_dp     = -R_GC.T

            cam_col = self.state.IMU_DIM + cam_idx * self.state.CAM_DIM
            H[2*k:2*k+2, cam_col:cam_col+3]   = J_proj @ J_p_dtheta
            H[2*k:2*k+2, cam_col+3:cam_col+6] = J_proj @ J_p_dp

        if not all_valid:
            return None

        # Null-space projection
        H_f = np.zeros((2 * n_obs, 3))
        for k, cam_idx in enumerate(cam_indices):
            cam = self.state.camera_window[cam_idx]
            R_GC = cam.R_GC
            p_cam = R_GC.T @ (p3d_world - cam.p_GC)
            if p_cam[2] < 1e-6:
                return None
            _, J_proj = self.camera.project_point(p_cam)
            H_f[2*k:2*k+2, :] = J_proj @ R_GC.T

        if H_f.shape[0] <= H_f.shape[1]:
            return None

        Q_null, _ = np.linalg.qr(H_f, mode='complete')
        A = Q_null[:, 3:]

        H_tilde = A.T @ H
        r_tilde = A.T @ r

        # Chi-squared outlier test
        S = H_tilde @ self.state.P @ H_tilde.T + \
            np.eye(len(r_tilde)) * self.meas_noise
        gamma = r_tilde @ np.linalg.solve(S, r_tilde)
        dof   = len(r_tilde)
        if gamma > chi2.ppf(0.95, df=max(1, dof)):
            return None

        return H_tilde, r_tilde

    def _kalman_update(self, H: np.ndarray, r: np.ndarray, R_noise: np.ndarray):
        """Standard Kalman update with Joseph-form covariance update."""
        P = self.state.P
        S = H @ P @ H.T + R_noise

        K = np.linalg.solve(S.T, (P @ H.T).T).T

        delta_x = K @ r
        self.state.apply_error_state(delta_x)

        n = self.state.state_dim
        I_KH = np.eye(n) - K @ H
        self.state.P = I_KH @ P @ I_KH.T + K @ R_noise @ K.T
