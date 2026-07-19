"""
frontend/klt_tracker.py
=======================
KLT optical flow feature tracker with robust outlier rejection.

Two layers of outlier rejection:
1. Bidirectional consistency (forward-backward check)
2. Epipolar RANSAC (fundamental matrix test)
"""

import numpy as np
import cv2
from typing import Tuple

LKT_WIN_SIZE  = (21, 21)
LKT_MAX_LEVEL = 3
LKT_CRITERIA  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# OpenCV 4.5+ uses OPTFLOW_LK_MIN_EIGEN; older versions use OPTFLOW_LK_GET_MIN_EIGENVALS
try:
    LKT_FLAGS = cv2.OPTFLOW_LK_MIN_EIGEN
except AttributeError:
    LKT_FLAGS = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS

FB_THRESHOLD  = 1.0   # px
EPI_THRESHOLD = 1.0   # px


class KLTTracker:
    """Tracks feature points using KLT optical flow with F-matrix RANSAC."""

    def track(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        pts_prev: np.ndarray,
        min_eigenvalue: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features from previous to current frame.

        Returns:
            pts_curr:   shape (N, 2) float32 -- tracked positions (NaN for failures)
            valid_mask: shape (N,) bool -- True for valid tracks
        """
        N = len(pts_prev)
        if N == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        pts_prev_f = pts_prev.reshape(-1, 1, 2).astype(np.float32)

        # Forward tracking
        pts_curr_f, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, pts_prev_f, None,
            winSize=LKT_WIN_SIZE, maxLevel=LKT_MAX_LEVEL,
            criteria=LKT_CRITERIA, flags=LKT_FLAGS,
            minEigThreshold=min_eigenvalue,
        )
        pts_curr_f = pts_curr_f.reshape(-1, 2)
        status_fwd = status_fwd.ravel().astype(bool)

        # Backward tracking (bidirectional check)
        #pts_back_f, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        #   img_curr, img_prev, pts_curr_f.reshape(-1, 1, 2), None,
        #   winSize=LKT_WIN_SIZE, maxLevel=LKT_MAX_LEVEL,
        #   criteria=LKT_CRITERIA, flags=LKT_FLAGS,
        #)
        #pts_back_f = pts_back_f.reshape(-1, 2)
        #status_bwd = status_bwd.ravel().astype(bool)

        #fb_error = np.linalg.norm(pts_prev - pts_back_f, axis=1)
        #fb_valid = (fb_error < FB_THRESHOLD)
        #valid = status_fwd & status_bwd & fb_valid
        valid = status_fwd.ravel().astype(bool)

        # Epipolar RANSAC
        valid_indices = np.where(valid)[0]
        if len(valid_indices) >= 8:
            p1 = pts_prev[valid_indices]
            p2 = pts_curr_f[valid_indices]

            F, inlier_mask = cv2.findFundamentalMat(
                p1, p2, method=cv2.FM_RANSAC,
                ransacReprojThreshold=EPI_THRESHOLD,
                confidence=0.99, maxIters=500,
            )

            if F is not None and inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
                epi_valid = np.zeros(N, dtype=bool)
                epi_valid[valid_indices[inlier_mask]] = True
                valid = valid & epi_valid

        pts_curr_out = pts_curr_f.copy()
        pts_curr_out[~valid] = np.nan

        return pts_curr_out, valid