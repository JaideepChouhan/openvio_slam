"""
loop_closure/verifier.py
========================
Geometric verification of loop closure candidates via Essential matrix RANSAC.

BoW similarity = appearance match; Essential matrix RANSAC = geometric proof.
Returns relative pose (R_rel, t_rel) for pose graph edge if verified.
"""

import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass


@dataclass
class LoopClosureResult:
    query_id:    int
    match_id:    int
    R_rel:       np.ndarray
    t_rel:       np.ndarray
    n_inliers:   int
    inlier_pts1: np.ndarray
    inlier_pts2: np.ndarray


class GeometricVerifier:
    def __init__(self, config: dict, camera_model):
        self.min_inliers   = config.get('min_loop_inliers', 20)
        self.ransac_thresh = 1.0
        self.camera        = camera_model

    def verify(self, query_id: int, match_id: int,
               query_descs: np.ndarray, match_descs: np.ndarray,
               query_pts: np.ndarray, match_pts: np.ndarray
               ) -> Optional[LoopClosureResult]:
        if len(query_descs) < 8 or len(match_descs) < 8:
            return None

        # BF matching with Lowe's ratio test
        matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_raw = matcher.knnMatch(query_descs, match_descs, k=2)
        good = [m for pair in matches_raw
                if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

        if len(good) < self.min_inliers:
            return None

        pts1 = query_pts[[m.queryIdx for m in good]].astype(np.float64)
        pts2 = match_pts[[m.trainIdx for m in good]].astype(np.float64)

        pts1_px = pts1 * self.camera.fx + np.array([self.camera.cx, self.camera.cy])
        pts2_px = pts2 * self.camera.fx + np.array([self.camera.cx, self.camera.cy])

        E, mask = cv2.findEssentialMat(pts1_px, pts2_px, self.camera.K,
                                        method=cv2.RANSAC, prob=0.999,
                                        threshold=self.ransac_thresh)
        if E is None or mask is None:
            return None

        mask = mask.ravel().astype(bool)
        if mask.sum() < self.min_inliers:
            return None

        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1_px[mask], pts2_px[mask], self.camera.K)
        if R_rel is None:
            return None

        return LoopClosureResult(
            query_id=query_id, match_id=match_id,
            R_rel=R_rel, t_rel=t_rel.ravel(),
            n_inliers=int(mask.sum()),
            inlier_pts1=pts1[mask], inlier_pts2=pts2[mask],
        )
