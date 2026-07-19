"""
estimator/state.py
==================
The MSCKF state vector and covariance matrix.

Error-state formulation: separate nominal state from error-state covariance.
The Kalman filter only touches the covariance and computes corrections.
The nominal state is corrected in the "apply error-state" step.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from ..utils.quaternion import quat_to_rot, quat_normalize, quat_multiply, quat_exp
from ..utils.so3 import hat


@dataclass
class CameraState:
    """One camera pose in the MSCKF sliding window."""
    frame_id: int
    q_GC: np.ndarray   # shape (4,) world<-camera rotation
    p_GC: np.ndarray   # shape (3,) position in world frame

    @property
    def R_GC(self) -> np.ndarray:
        return quat_to_rot(self.q_GC)


class VIOState:
    """
    Complete MSCKF state: IMU state + sliding window of camera poses.

    Error-state dimension:
        IMU:    15  (delta_theta:3, delta_p:3, delta_v:3, delta_b_g:3, delta_b_a:3)
        Camera: 6 x N_cameras
    """

    IMU_DIM = 15
    CAM_DIM = 6

    def __init__(self, max_window_size: int = 20):
        self.max_window = max_window_size

        self.q_GI = np.array([1, 0, 0, 0], dtype=float)
        self.p_GI = np.zeros(3, dtype=float)
        self.v_GI = np.zeros(3, dtype=float)
        self.b_g  = np.zeros(3, dtype=float)
        self.b_a  = np.zeros(3, dtype=float)

        self.camera_window: List[CameraState] = []

        P_init = np.zeros((self.IMU_DIM, self.IMU_DIM))
        P_init[0:3,   0:3]   = np.eye(3) * (0.01)**2
        P_init[3:6,   3:6]   = np.eye(3) * (0.01)**2
        P_init[6:9,   6:9]   = np.eye(3) * (0.1)**2
        P_init[9:12,  9:12]  = np.eye(3) * (0.01)**2
        P_init[12:15, 12:15] = np.eye(3) * (0.1)**2
        self.P = P_init

    @property
    def n_cameras(self) -> int:
        return len(self.camera_window)

    @property
    def state_dim(self) -> int:
        return self.IMU_DIM + self.CAM_DIM * self.n_cameras

    @property
    def R_GI(self) -> np.ndarray:
        return quat_to_rot(self.q_GI)

    def get_camera_state(self, idx: int) -> CameraState:
        return self.camera_window[idx]

    def augment_with_camera(self, frame_id: int, R_IC: np.ndarray, p_IC: np.ndarray):
        """Add current IMU pose as a new camera state."""
        from ..utils.quaternion import rot_to_quat

        R_GI = self.R_GI
        R_GC = R_GI @ R_IC
        p_GC = self.p_GI + R_GI @ p_IC
        q_GC = rot_to_quat(R_GC)

        new_cam = CameraState(frame_id=frame_id, q_GC=q_GC, p_GC=p_GC)
        self.camera_window.append(new_cam)

        old_dim = self.IMU_DIM + self.CAM_DIM * (self.n_cameras - 1)
        new_dim = old_dim + self.CAM_DIM

        J = np.zeros((self.CAM_DIM, old_dim))
        J[0:3, 0:3] = np.eye(3)
        J[3:6, 0:3] = -hat(R_GI @ p_IC)
        J[3:6, 3:6] = np.eye(3)

        P_old = self.P
        P_new = np.zeros((new_dim, new_dim))
        P_new[:old_dim, :old_dim] = P_old
        P_new[:old_dim, old_dim:] = P_old @ J.T
        P_new[old_dim:, :old_dim] = J @ P_old
        P_new[old_dim:, old_dim:] = J @ P_old @ J.T
        self.P = P_new

    def remove_oldest_camera(self) -> CameraState:
        """Remove the oldest camera state and corresponding rows/cols from P."""
        if not self.camera_window:
            raise RuntimeError("No cameras to remove")

        removed = self.camera_window.pop(0)
        idx = self.IMU_DIM
        keep = list(range(idx)) + list(range(idx + self.CAM_DIM,
                                              self.state_dim + self.CAM_DIM))
        self.P = self.P[np.ix_(keep, keep)]
        return removed

    def apply_error_state(self, delta_x: np.ndarray):
        """Apply error-state correction to the nominal state."""
        self.q_GI = quat_normalize(quat_multiply(self.q_GI, quat_exp(delta_x[0:3])))
        self.p_GI = self.p_GI + delta_x[3:6]
        self.v_GI = self.v_GI + delta_x[6:9]
        self.b_g  = self.b_g  + delta_x[9:12]
        self.b_a  = self.b_a  + delta_x[12:15]

        for i, cam in enumerate(self.camera_window):
            base = self.IMU_DIM + i * self.CAM_DIM
            cam.q_GC = quat_normalize(
                quat_multiply(cam.q_GC, quat_exp(delta_x[base:base+3])))
            cam.p_GC = cam.p_GC + delta_x[base+3:base+6]

    def symmetrize_covariance(self):
        """Force P to be symmetric. Call every ~10 frames."""
        self.P = (self.P + self.P.T) / 2.0

    def initialize_from(self, init_state: dict):
        """Set state from initialization result."""
        self.q_GI = init_state['q_GI']
        self.p_GI = init_state.get('p_GI', np.zeros(3))
        self.v_GI = init_state.get('v_GI', np.zeros(3))
        self.b_g  = init_state.get('b_g',  np.zeros(3))
        self.b_a  = init_state.get('b_a',  np.zeros(3))
