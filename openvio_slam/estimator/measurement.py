"""
estimator/measurement.py
========================
Feature measurement model and Jacobian computation for the MSCKF update step.

This module handles everything between "here is a feature's observations" and
"here is the stacked Jacobian matrix H ready for null-space projection."

Architecture position:
    FeatureManager  →  [marginalized Feature]
                                ↓
                     measurement.py (this file)
                       - triangulate 3D position
                       - compute per-observation residuals
                       - compute per-camera Jacobians
                                ↓
                     null_space.py
                       - project out feature position dependence
                                ↓
                     msckf.py (chi² test + Kalman update)

The Measurement Model:
    A feature j observed in camera frame i at pixel (u, v) gives:
        z_ij = π(R_GCi.T @ (p_j - p_GCi)) + noise
    where:
        π(X) = [X/Z, Y/Z]   (perspective projection into normalized coords)
        noise ~ N(0, σ_pix² I₂)

    The residual is:
        r_ij = z_ij_measured - z_ij_predicted

    Jacobians needed for MSCKF:
        ∂r_ij / ∂δθ_Ci  (2×3)  — w.r.t. camera i orientation error
        ∂r_ij / ∂δp_Ci  (2×3)  — w.r.t. camera i position error
        ∂r_ij / ∂p_j    (2×3)  — w.r.t. feature position (to be projected out)

References:
    Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter for
    Vision-aided Inertial Navigation", ICRA 2007 — Equations (14)–(19).
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple

from ..utils.so3 import hat
from ..utils.quaternion import quat_to_rot
from ..map.triangulator import Triangulator


class PerCameraJacobian(NamedTuple):
    """
    Jacobian contributions from one feature observation in one camera.

    J_cam:    (2, 6) — derivative of residual w.r.t. [δθ_C, δp_C]
    J_feat:   (2, 3) — derivative of residual w.r.t. feature position p_j (world frame)
    residual: (2,)   — measured - predicted normalized image coordinate
    cam_idx:  int    — index into the MSCKF camera window
    """
    J_cam:    np.ndarray   # (2, 6)
    J_feat:   np.ndarray   # (2, 3)
    residual: np.ndarray   # (2,)
    cam_idx:  int


class FeatureMeasurementModel:
    """
    Computes measurement residuals and Jacobians for one marginalized feature.

    Usage (called from msckf.py):
        model = FeatureMeasurementModel(camera_model, state)
        result = model.process(feature)
        if result is not None:
            H_full, r_full = result   # ready for null_space.py
    """

    def __init__(self, camera_model, state):
        """
        Args:
            camera_model: PinholeCameraModel instance (for project_point, fx, fy)
            state:        VIOState instance (for camera window poses)
        """
        self.camera      = camera_model
        self.state       = state
        self.triangulator = Triangulator()

    def process(
        self,
        feature,
        min_observations: int = 3,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process one marginalized feature: triangulate + compute Jacobians.

        Args:
            feature:          Feature object with .observations list of (frame_id, uv_norm)
            min_observations: Minimum valid observations needed (default 3)

        Returns:
            (H_full, r_full) where:
                H_full: (2*M, 6*N + 3)  stacked Jacobian; last 3 cols = ∂/∂p_j
                r_full: (2*M,)           stacked residuals
            Or None if triangulation fails or too few valid cameras.
        """
        # ── Step 1: match observations to current camera window ──────────────
        obs_matched = self._match_observations_to_window(feature)
        if len(obs_matched) < min_observations:
            return None

        # ── Step 2: triangulate feature position ──────────────────────────────
        poses = [(self.state.camera_window[i].R_GC,
                  self.state.camera_window[i].p_GC)
                 for i, _ in obs_matched]
        obs_norms = [uv for _, uv in obs_matched]
        cam_indices = [i for i, _ in obs_matched]

        try:
            p3d_world = self.triangulator.triangulate_dlt(poses, obs_norms)
        except Exception:
            return None

        if p3d_world is None:
            return None

        # ── Step 3: compute per-camera Jacobians ──────────────────────────────
        per_cam = self._compute_jacobians(p3d_world, cam_indices, obs_norms)
        if len(per_cam) < min_observations:
            return None

        # ── Step 4: assemble full stacked Jacobian ────────────────────────────
        H_full, r_full = self._assemble(per_cam)
        return H_full, r_full

    # ─────────────────────────────────────────────────────
    # Step 1: observation-to-window matching
    # ─────────────────────────────────────────────────────

    def _match_observations_to_window(
        self, feature
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Match feature observations to active camera window entries by frame_id.

        Returns list of (window_index, uv_normalized) pairs, one per matched obs.
        Observations for frames no longer in the window are silently dropped.
        """
        window_frame_ids = {cam.frame_id: idx
                            for idx, cam in enumerate(self.state.camera_window)}
        matched = []
        for frame_id, uv_norm in feature.observations:
            if frame_id in window_frame_ids:
                matched.append((window_frame_ids[frame_id],
                                np.asarray(uv_norm, dtype=float)))
        return matched

    # ─────────────────────────────────────────────────────
    # Step 2+3: per-camera residual and Jacobian
    # ─────────────────────────────────────────────────────

    def _compute_jacobians(
        self,
        p3d_world:   np.ndarray,
        cam_indices: List[int],
        obs_norms:   List[np.ndarray],
    ) -> List[PerCameraJacobian]:
        """
        For each (camera_index, observation) pair, compute:
            - Residual r = z_measured - z_predicted  (2,)
            - Jacobian J_cam w.r.t. camera pose error  (2, 6)
            - Jacobian J_feat w.r.t. feature position  (2, 3)

        The full 2×9 Jacobian per observation is:
            [J_cam | J_feat]

        Derivation (chain rule through perspective projection):
            z = π(p_cam)     where p_cam = R_GC.T @ (p_j - p_GC)

            ∂z/∂p_cam = J_proj  (2×3, from camera.project_point)

            ∂p_cam/∂δθ_C = hat(p_cam)         (3×3, from rotation perturbation)
            ∂p_cam/∂δp_C = -R_GC.T            (3×3)
            ∂p_cam/∂p_j  = R_GC.T             (3×3)

        Therefore:
            J_cam[:, 0:3] = J_proj @ hat(p_cam)    # ∂z/∂δθ_C
            J_cam[:, 3:6] = J_proj @ (-R_GC.T)     # ∂z/∂δp_C
            J_feat        = J_proj @ R_GC.T         # ∂z/∂p_j
        """
        results = []
        for cam_idx, uv_meas in zip(cam_indices, obs_norms):
            cam  = self.state.camera_window[cam_idx]
            R_GC = cam.R_GC
            p_GC = cam.p_GC

            # Transform feature into this camera's frame
            p_cam = R_GC.T @ (p3d_world - p_GC)   # (3,)

            # Reject points behind camera (z must be positive)
            if p_cam[2] < 0.05:
                continue

            # Perspective projection + projection Jacobian
            try:
                z_hat, J_proj = self.camera.project_point(p_cam)  # z_hat:(2,), J_proj:(2,3)
            except ValueError:
                continue

            # Residual
            residual = uv_meas - z_hat   # (2,)

            # Jacobians via chain rule
            J_cam = np.zeros((2, 6))
            J_cam[:, 0:3] = J_proj @ hat(p_cam)    # ∂z/∂δθ_C: rotation perturbation
            J_cam[:, 3:6] = J_proj @ (-R_GC.T)     # ∂z/∂δp_C: translation perturbation

            J_feat = J_proj @ R_GC.T               # ∂z/∂p_j: feature position (2×3)

            results.append(PerCameraJacobian(
                J_cam=J_cam, J_feat=J_feat,
                residual=residual, cam_idx=cam_idx,
            ))
        return results

    # ─────────────────────────────────────────────────────
    # Step 4: assemble stacked H and r
    # ─────────────────────────────────────────────────────

    def _assemble(
        self,
        per_cam: List[PerCameraJacobian],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack per-camera Jacobians into the full measurement Jacobian H_full.

        H_full layout:  (2*M) × (state_dim + 3)
                         ↑              ↑    ↑
                     residual rows    MSCKF  feature
                                      state  position
                                      cols   cols

        Specifically: columns 0..state_dim-1 are MSCKF state (camera block only,
        non-camera blocks are zero). Columns state_dim..state_dim+2 are ∂/∂p_j.

        Only the camera-state columns for the cameras that observed this feature
        are non-zero. All other columns are zero. This sparsity is preserved by
        null_space.py.
        """
        M         = len(per_cam)
        state_dim = self.state.state_dim
        H_full    = np.zeros((2 * M, state_dim + 3))
        r_full    = np.zeros(2 * M)

        for k, jac in enumerate(per_cam):
            row = 2 * k
            # Camera state block: columns for camera jac.cam_idx
            col = self.state.IMU_DIM + jac.cam_idx * self.state.CAM_DIM
            H_full[row:row+2, col:col+6] = jac.J_cam    # ∂z/∂[δθ_Ci, δp_Ci]
            # Feature position block: last 3 columns
            H_full[row:row+2, state_dim:] = jac.J_feat  # ∂z/∂p_j
            # Residual
            r_full[row:row+2] = jac.residual

        return H_full, r_full

    # ─────────────────────────────────────────────────────
    # Utility: single-observation residual (for debug / chi² pre-check)
    # ─────────────────────────────────────────────────────

    def compute_reprojection_error(
        self,
        p3d_world: np.ndarray,
        cam_idx:   int,
        uv_meas:   np.ndarray,
    ) -> Optional[float]:
        """
        Reprojection error in normalized image coordinates (not pixels).
        Returns None if the point is behind the camera.
        """
        cam  = self.state.camera_window[cam_idx]
        p_cam = cam.R_GC.T @ (p3d_world - cam.p_GC)
        if p_cam[2] < 0.05:
            return None
        try:
            z_hat, _ = self.camera.project_point(p_cam)
        except ValueError:
            return None
        return float(np.linalg.norm(uv_meas - z_hat))

    def compute_mean_reprojection_error(
        self,
        p3d_world:   np.ndarray,
        cam_indices: List[int],
        obs_norms:   List[np.ndarray],
    ) -> Optional[float]:
        """
        Mean reprojection error over all observations of a feature.
        Useful for map point quality assessment.
        """
        errors = []
        for cam_idx, uv in zip(cam_indices, obs_norms):
            e = self.compute_reprojection_error(p3d_world, cam_idx, uv)
            if e is not None:
                errors.append(e)
        return float(np.mean(errors)) if errors else None
