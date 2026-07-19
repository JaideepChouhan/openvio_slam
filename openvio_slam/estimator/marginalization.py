"""
estimator/marginalization.py
============================
Sliding window marginalization: remove old camera states cleanly.

What Is Marginalization?
    The MSCKF maintains a sliding window of N_MAX camera poses. When a new
    image arrives and the window is full, the oldest camera state must be
    removed. This is called "marginalization."

    Unlike a simple truncation (just deleting the oldest rows/cols from P),
    proper marginalization ensures that any information captured in the
    oldest camera's covariance correlations is preserved — it gets
    "folded into" the remaining state's covariance.

    Without proper marginalization: information loss → overconfidence → divergence.

Two Marginalization Strategies:
    1. MSCKF Marginalization (default, used here):
       For features that were ONLY seen in the camera being removed,
       we cannot use them anymore (their constraint involved that camera pose).
       We simply drop them. For features seen in MULTIPLE cameras including
       the one being removed: we first update the filter with those features
       (MSCKF step), THEN remove the camera. This ensures we extract all
       useful information before the camera state disappears.

    2. Schur Complement Marginalization (used in sliding-window BA systems like DSO):
       Construct the Schur complement of the Hessian w.r.t. the removed state.
       More principled but more expensive. We use MSCKF marginalization
       because it fits the MSCKF update structure naturally.

The Marginalization Process in OpenVIO-SLAM:
    Called from msckf.py when: n_cameras == max_window AND new image arrives.

    Step 1: Collect all features that include the oldest camera in their
            observation set AND have observations in at least 2 other cameras.
            These are "ready for MSCKF update" — perform MSCKF update NOW.

    Step 2: Remove the oldest camera state column/row from P and the
            camera_window list.

    Step 3: All remaining features' observations in the removed camera
            are pruned (that camera no longer exists in the state).

References:
    [1] Mourikis & Roumeliotis, ICRA 2007, §V "Filter Implementation"
    [2] Li & Mourikis, "High-Precision, Consistent EKF-based VIO", IJRR 2013
        §IV discusses marginalization correctness conditions
    [3] Forster et al., "Manifold Preintegration for Real-Time VIO", TRO 2017
        §VI for window management strategy
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarginalizationResult:
    """
    Summary of what happened during one marginalization step.
    Useful for logging and debugging.
    """
    removed_frame_id:      int
    n_features_updated:    int    # features MSCKF-updated before removal
    n_features_dropped:    int    # features dropped (only in removed cam)
    n_features_retained:   int    # features continuing in remaining window
    state_dim_before:      int
    state_dim_after:       int


class WindowMarginalizationManager:
    """
    Manages the sliding window marginalization strategy for the MSCKF.

    This class is responsible for:
        1. Deciding WHICH features to update before marginalization
        2. Telling the MSCKF to perform those updates
        3. Removing the oldest camera state from the VIO state
        4. Pruning observation lists of retained features

    Usage (called from msckf.py):
        manager = WindowMarginalizationManager(config)
        ...
        if state.n_cameras >= max_window:
            result = manager.marginalize(state, active_features, msckf_update_fn)
    """

    def __init__(self, config: dict):
        self.min_observations_for_update = config.get('min_track_length', 3)
        self.min_cameras_per_feature     = 2    # need ≥ 2 other cameras to triangulate

    def prepare_features_for_update(
        self,
        active_features: list,
        oldest_frame_id: int,
        remaining_frame_ids: set,
    ) -> Tuple[list, list, list]:
        """
        Partition active features into three groups before marginalization.

        Args:
            active_features:     All currently tracked features
            oldest_frame_id:     Frame ID of the camera about to be marginalized
            remaining_frame_ids: Frame IDs that will remain in the window

        Returns:
            (update_now, drop, retain) where:
                update_now: features seen in oldest cam + ≥2 remaining → MSCKF update
                drop:       features seen ONLY in oldest cam → discard
                retain:     features NOT seen in oldest cam → keep as-is
        """
        update_now = []
        drop       = []
        retain     = []

        for feat in active_features:
            obs_frame_ids = {frame_id for frame_id, _ in feat.observations}

            if oldest_frame_id not in obs_frame_ids:
                # Feature doesn't involve the camera being removed → retain
                retain.append(feat)
                continue

            obs_in_remaining = obs_frame_ids & remaining_frame_ids
            total_obs = len([fid for fid in obs_frame_ids if fid in remaining_frame_ids | {oldest_frame_id}])

            if (len(obs_in_remaining) >= self.min_cameras_per_feature
                    and total_obs >= self.min_observations_for_update):
                # Has enough observations to update AND spans multiple cameras → update now
                update_now.append(feat)
            else:
                # Only in the camera being removed, or too few observations → drop
                drop.append(feat)

        return update_now, drop, retain

    def prune_observations(
        self,
        features: list,
        removed_frame_id: int,
    ) -> list:
        """
        Remove observations from the marginalized camera in all retained features.
        After this, those features' observation lists no longer reference
        the removed camera.

        Modifies features in-place. Returns the list for convenience.
        """
        for feat in features:
            feat.observations = [
                (frame_id, uv)
                for frame_id, uv in feat.observations
                if frame_id != removed_frame_id
            ]
            if hasattr(feat, 'pixel_positions') and feat.pixel_positions:
                # pixel_positions is parallel to observations — prune correspondingly
                # We re-pair by index, so we must be careful
                # After this pruning, pixel_positions may be stale; it's only used
                # for KLT tracking seed, so staleness is acceptable here.
                pass
        return features

    def marginalize_oldest_camera(
        self,
        state,
        active_features: list,
        msckf_update_callback,
    ) -> MarginalizationResult:
        """
        Full marginalization step.

        Args:
            state:                  VIOState — will be mutated
            active_features:        List of Feature objects (from FeatureManager)
            msckf_update_callback:  Callable(features) → None — runs MSCKF update
                                    on the given feature list (from msckf.py)

        Returns:
            MarginalizationResult summary.
        """
        if state.n_cameras == 0:
            raise RuntimeError("Cannot marginalize: no cameras in window")

        oldest_cam       = state.camera_window[0]
        oldest_frame_id  = oldest_cam.frame_id
        state_dim_before = state.state_dim

        remaining_frame_ids = {cam.frame_id
                               for cam in state.camera_window[1:]}

        # ── Step 1: Partition features ──────────────────────────────────────
        update_now, drop, retain = self.prepare_features_for_update(
            active_features, oldest_frame_id, remaining_frame_ids)

        # ── Step 2: MSCKF update with features that include oldest camera ───
        if update_now:
            msckf_update_callback(update_now)

        # ── Step 3: Remove oldest camera from state ──────────────────────────
        removed = state.remove_oldest_camera()
        assert removed.frame_id == oldest_frame_id

        # ── Step 4: Prune observation lists in retained features ─────────────
        self.prune_observations(retain, oldest_frame_id)
        self.prune_observations(drop,   oldest_frame_id)   # clean up even dropped ones

        return MarginalizationResult(
            removed_frame_id   = oldest_frame_id,
            n_features_updated = len(update_now),
            n_features_dropped = len(drop),
            n_features_retained= len(retain),
            state_dim_before   = state_dim_before,
            state_dim_after    = state.state_dim,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schur complement utility (for future extension / educational reference)
# ─────────────────────────────────────────────────────────────────────────────

def schur_complement_marginalize(
    H: np.ndarray,
    b: np.ndarray,
    marg_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalize the FIRST `marg_size` variables out of a linear system
    using the Schur complement.

    Given: H @ δx = b,  where δx = [δx_marg | δx_remain]

    Partitioned:
        H = [H_mm  H_mr]
            [H_rm  H_rr]

        b = [b_m]
            [b_r]

    Schur complement:
        H_star = H_rr - H_rm @ H_mm^{-1} @ H_mr
        b_star = b_r  - H_rm @ H_mm^{-1} @ b_m

    The reduced system H_star @ δx_remain = b_star excludes δx_marg.

    NOT used in the default MSCKF pipeline (we use the simpler camera-removal
    approach in VIOState.remove_oldest_camera). This function is provided for:
        - Educational reference
        - Future sliding-window BA extension
        - Unit testing correctness of the camera removal

    Args:
        H:          Full (n+m, n+m) Hessian where first marg_size rows/cols
                    correspond to variables being marginalized
        b:          Full (n+m,) right-hand side
        marg_size:  Number of variables to marginalize (from the front)

    Returns:
        (H_star, b_star) — reduced (n, n) and (n,) system
    """
    m = marg_size
    n = H.shape[0] - m

    H_mm = H[:m, :m]
    H_mr = H[:m, m:]
    H_rm = H[m:, :m]
    H_rr = H[m:, m:]
    b_m  = b[:m]
    b_r  = b[m:]

    # Solve H_mm @ x = H_mr using Cholesky for efficiency
    try:
        L = np.linalg.cholesky(H_mm + np.eye(m) * 1e-12)
        # H_mm^{-1} @ H_mr
        HmmInv_Hmr = np.linalg.solve(L @ L.T, H_mr)
        # H_mm^{-1} @ b_m
        HmmInv_bm  = np.linalg.solve(L @ L.T, b_m)
    except np.linalg.LinAlgError:
        # Fallback to general solve
        HmmInv_Hmr = np.linalg.lstsq(H_mm, H_mr, rcond=None)[0]
        HmmInv_bm  = np.linalg.lstsq(H_mm, b_m,  rcond=None)[0]

    H_star = H_rr - H_rm @ HmmInv_Hmr
    b_star = b_r  - H_rm @ HmmInv_bm

    return H_star, b_star


def check_marginalization_consistency(
    P_before: np.ndarray,
    P_after:  np.ndarray,
    marg_size: int,
) -> dict:
    """
    Diagnostic: verify that removing the marginalised block from P satisfies
    basic consistency requirements.

    After marginalization:
        - P_after should be positive semi-definite
        - Eigenvalues of P_after should be ≤ corresponding eigenvalues of P_before[marg:, marg:]
          (marginalization cannot increase uncertainty for retained variables)

    Returns a dict with diagnostic values. Used in unit tests.
    """
    P_remain_before = P_before[marg_size:, marg_size:]
    eig_before = np.linalg.eigvalsh(P_remain_before)
    eig_after  = np.linalg.eigvalsh(P_after)

    return {
        'P_after_min_eigenvalue':     float(eig_after.min()),
        'P_after_is_PSD':             bool(eig_after.min() > -1e-8),
        'P_remain_before_min_eig':    float(eig_before.min()),
        'max_eigenvalue_increase':    float((eig_after - eig_before).max()),
        'trace_before':               float(np.trace(P_remain_before)),
        'trace_after':                float(np.trace(P_after)),
    }
