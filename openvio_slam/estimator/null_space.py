"""
estimator/null_space.py
=======================
Null-space projection: the mathematical heart of the MSCKF.

The Problem:
    After computing H_full from measurement.py, we have:
        H_full = [H_x | H_f]     shape: (2M) × (state_dim + 3)
                                                ↑ camera states  ↑ feature pos

    We CANNOT use H_full directly in a Kalman update because:
        - The feature position p_j is NOT in our state vector
        - Adding it would make the state grow unboundedly → SLAM, not MSCKF

The Solution (Mourikis & Roumeliotis, 2007, §IV-B):
    Project the measurement onto the LEFT NULL SPACE of H_f.
    That is, find matrix A such that:
        A.T @ H_f = 0   (A spans the null space of H_f.T)

    Then:
        H̃ = A.T @ H_x     shape: (2M-3) × state_dim
        r̃ = A.T @ r        shape: (2M-3,)

    These projected quantities H̃ and r̃ are completely free of p_j.
    The Kalman update uses only these projected quantities:
        S = H̃ @ P @ H̃.T + A.T @ R_noise @ A
        K = P @ H̃.T @ inv(S)
        δX = K @ r̃

Why Does This Work?
    H_f has shape (2M, 3). Its column space has dimension ≤ 3.
    Therefore its null space (the rows of A.T) has dimension 2M - 3.
    Projecting onto this null space eliminates p_j from the equations
    while retaining 2M - 3 independent constraints on the camera poses.

    Geometrically: we're constraining the camera poses WITHOUT
    triangulating and storing the map point in the state — the core
    insight that keeps the MSCKF state bounded.

QR Decomposition:
    A = left null space of H_f = Q_2 in the QR decomposition:
        H_f = [Q_1 | Q_2] @ [R_1; 0]
    where Q_1 has 3 columns (spanning col space of H_f)
    and Q_2 has 2M-3 columns (spanning null space of H_f.T).

    We compute this via numpy's QR with mode='complete'.

References:
    [1] Mourikis & Roumeliotis, ICRA 2007, §IV-B "Measurement Model"
    [2] Sun, Liu, Huang, "Efficient and Consistent Visual-Inertial Estimator
        with Line Features", IEEE T-RO 2020 — for generalization to lines
    [3] Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial
        Estimation", ICRA 2020 — for implementation notes
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import chi2


class NullSpaceProjector:
    """
    Projects measurement Jacobians and residuals onto the null space of H_f.

    This class is intentionally stateless (pure functions as methods) so it
    can be instantiated once and used for every feature without side effects.
    """

    def __init__(self, measurement_noise_var: float = 1.0,
                 chi2_confidence: float = 0.95):
        """
        Args:
            measurement_noise_var: Pixel measurement noise variance in
                                   NORMALIZED coordinates squared.
                                   = (σ_pixel / fx)²
                                   Typical: (1.0 / 500)² ≈ 4e-6
            chi2_confidence:       Chi-squared test confidence level.
                                   0.95 (95%) is standard; 0.99 is more permissive.
        """
        self.noise_var       = measurement_noise_var
        self.chi2_confidence = chi2_confidence

    # ─────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────

    def project(
        self,
        H_full: np.ndarray,
        r_full: np.ndarray,
        state_dim: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Full null-space projection pipeline:
            1. Split H_full into H_x (camera states) and H_f (feature position)
            2. Compute left null space of H_f
            3. Project H_x and r_full
            4. Run chi-squared outlier test on projected residual
            5. Optionally QR-compress (for efficiency with many features)

        Args:
            H_full:    (2M, state_dim+3) stacked Jacobian from measurement.py
            r_full:    (2M,) stacked residual
            state_dim: MSCKF state dimension (to split H_full into H_x | H_f)

        Returns:
            (H_tilde, r_tilde) if inlier, None if outlier or projection fails.

        Dimensions after projection:
            H_tilde: (2M-3, state_dim)
            r_tilde: (2M-3,)
        """
        n_obs_2 = H_full.shape[0]   # = 2M
        M       = n_obs_2 // 2

        if M < 2 or n_obs_2 < 4:
            # Need at least 2 observations to have rank-3 H_f and a non-trivial null space
            return None

        # Split into camera-state block and feature-position block
        H_x = H_full[:, :state_dim]       # (2M, state_dim)
        H_f = H_full[:, state_dim:]        # (2M, 3)

        # Compute null space
        A = self._left_null_space(H_f)     # (2M, 2M-3)

        if A is None or A.shape[1] < 1:
            return None

        # Project
        H_tilde = A.T @ H_x               # (2M-3, state_dim)
        r_tilde = A.T @ r_full             # (2M-3,)

        # Chi-squared outlier rejection
        if not self._chi2_test(H_tilde, r_tilde):
            return None

        # Optional QR compression: if 2M-3 > state_dim, compress to state_dim rows
        # This reduces the cost of the Kalman update from O(state_dim²·(2M-3)) to O(state_dim³)
        H_tilde, r_tilde = self._qr_compress(H_tilde, r_tilde)

        return H_tilde, r_tilde

    # ─────────────────────────────────────────────────────
    # Step 1: left null space of H_f
    # ─────────────────────────────────────────────────────

    def _left_null_space(self, H_f: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the left null space of H_f using QR decomposition.

        H_f has shape (2M, 3). We perform:
            QR decomposition: H_f = Q @ R
            Q is (2M, 2M) orthogonal (mode='complete')
            R is (2M, 3) upper triangular

            Q[:,  :3] = Q_1  spans the column space of H_f
            Q[:, 3:]  = Q_2  spans the left null space of H_f (what we want)

        Returns Q_2 of shape (2M, 2M-3), or None if H_f is degenerate.

        Degeneracy check: if rank(H_f) < 3, the null space has dimension > 2M-3.
        This happens when all cameras are co-linear (zero baseline) — a degenerate
        configuration where we cannot triangulate. We detect this via the
        condition number of the upper triangular R.
        """
        n_rows = H_f.shape[0]
        n_cols = H_f.shape[1]   # should be 3

        if n_rows <= n_cols:
            # More feature columns than rows — cannot project
            return None

        try:
            Q, R_upper = np.linalg.qr(H_f, mode='complete')
        except np.linalg.LinAlgError:
            return None

        # Rank check: inspect diagonal of R_upper
        diag_R = np.abs(np.diag(R_upper[:n_cols, :n_cols]))
        if diag_R[-1] < 1e-8 * diag_R[0]:
            # H_f is rank-deficient → degenerate triangulation
            return None

        # The left null space: Q columns from index n_cols onward
        A = Q[:, n_cols:]   # shape (2M, 2M-3)
        return A

    # ─────────────────────────────────────────────────────
    # Step 2: chi-squared outlier test
    # ─────────────────────────────────────────────────────

    def _chi2_test(self, H_tilde: np.ndarray, r_tilde: np.ndarray) -> bool:
        """
        Mahalanobis-distance chi-squared test for outlier rejection.

        Test statistic: γ = r̃.T @ S⁻¹ @ r̃
        where S = H̃ @ P_cam @ H̃.T + R_noise_projected

        For efficiency, we use only the noise term (conservative — rejects more):
            γ_approx = r̃.T @ r̃ / noise_var

        A feature is an outlier if γ > χ²(dof, confidence).
        Typical outlier sources:
            - Dynamic objects (moving people/cars)
            - Bad KLT tracks that slipped through epipolar RANSAC
            - Wrong ORB descriptor matches in loop-closure back-propagation

        Note: we use a simplified test here (no P term) because:
            1. We don't have easy access to the projected covariance here
            2. The full chi² test is done in msckf.py using P
            3. This simple test still catches gross outliers efficiently
        """
        n_dof = len(r_tilde)
        if n_dof < 1:
            return False

        gamma = float(np.dot(r_tilde, r_tilde)) / max(self.noise_var, 1e-15)
        threshold = chi2.ppf(self.chi2_confidence, df=n_dof)

        return gamma <= threshold

    # ─────────────────────────────────────────────────────
    # Step 3: QR compression (optional efficiency step)
    # ─────────────────────────────────────────────────────

    def _qr_compress(
        self,
        H_tilde: np.ndarray,
        r_tilde: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition of H_tilde to compress the system before Kalman update.

        When we have M observations, H_tilde has 2M-3 rows. The Kalman gain
        computation requires inverting the (2M-3) × (2M-3) innovation covariance S,
        costing O((2M-3)²·state_dim + (2M-3)³) — expensive for large M.

        The trick: if H_tilde = Q_h @ [T_h; 0] (QR decomp), then:
            r̃_compressed = Q_h.T @ r̃     (first state_dim entries)
            H_compressed  = T_h            (state_dim × state_dim upper triangular)

        The Kalman update with (H_compressed, r̃_compressed) is equivalent to
        the update with (H_tilde, r̃) but costs only O(state_dim³).

        This is the "QR accelerated MSCKF update" from:
            Huang et al., "Observability-based Rules for Designing Consistent
            EKF SLAM Estimators", IJRR 2010

        We apply this only when 2M-3 > state_dim (otherwise it's no cheaper).
        """
        n_rows, n_cols = H_tilde.shape

        if n_rows <= n_cols:
            # Already small — no benefit from compression
            return H_tilde, r_tilde

        try:
            Q_h, T_h = np.linalg.qr(H_tilde, mode='reduced')
        except np.linalg.LinAlgError:
            return H_tilde, r_tilde

        r_compressed = Q_h.T @ r_tilde   # (n_cols,)
        # T_h is (n_cols, n_cols) upper triangular — compact representation
        return T_h, r_compressed


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function (used in msckf.py)
# ─────────────────────────────────────────────────────────────────────────────

def project_and_compress(
    H_full:        np.ndarray,
    r_full:        np.ndarray,
    state_dim:     int,
    noise_var:     float = 4e-6,
    chi2_conf:     float = 0.95,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper: null-space project + chi² test + QR compress.

    Args:
        H_full:    (2M, state_dim+3) measurement Jacobian from measurement.py
        r_full:    (2M,) residual vector
        state_dim: MSCKF full state dimension
        noise_var: Measurement noise variance in normalized coords (σ_px/fx)²
        chi2_conf: Chi-squared test confidence (0.95 = 95%)

    Returns:
        (H_tilde, r_tilde) compressed and projected, or None if rejected.
    """
    projector = NullSpaceProjector(
        measurement_noise_var=noise_var,
        chi2_confidence=chi2_conf,
    )
    return projector.project(H_full, r_full, state_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic / educational helper
# ─────────────────────────────────────────────────────────────────────────────

def explain_null_space(H_f: np.ndarray) -> dict:
    """
    Inspect the null space properties of H_f for debugging.
    Returns a dict with rank, singular values, null space dimension.

    Usage:
        from openvio_slam.estimator.null_space import explain_null_space
        info = explain_null_space(H_f)
        print(info)
    """
    _, sv, _ = np.linalg.svd(H_f)
    rank   = int(np.sum(sv > 1e-8 * sv[0]))
    n_rows = H_f.shape[0]

    return {
        'H_f_shape':         H_f.shape,
        'singular_values':   sv.tolist(),
        'rank':              rank,
        'null_space_dim':    n_rows - rank,
        'condition_number':  float(sv[0] / max(sv[-1], 1e-20)),
        'is_rank_3':         rank == 3,
        'projected_rows':    n_rows - 3,
    }
