"""
persistence/relocalization.py
==============================
Re-localize the robot into a previously saved map via BoW + PnP RANSAC.

"Map once, navigate forever": on startup, load saved map, stream camera frames,
find pose via PnP RANSAC, initialize MSCKF at that pose in the map's coordinate frame.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from ..loop_closure.vocabulary import ORBVocabulary
from ..loop_closure.detector import LoopCandidateDatabase
from ..utils.quaternion import rot_to_quat


class Relocalizator:
    def __init__(self, vocabulary: ORBVocabulary, camera_model, config: dict):
        self.camera          = camera_model
        self.min_pnp_inliers = config.get('min_pnp_inliers', 20)
        self._bow_db = LoopCandidateDatabase(
            vocabulary, similarity_threshold=0.4, min_temporal_gap=0)
        self._kf_data: Dict[int, dict] = {}
        self._mp_data: Dict[int, dict] = {}
        self._loaded = False

    def load_map_data(self, keyframes_data: list, map_points_data: list):
        for kf in keyframes_data:
            self._bow_db.add_keyframe(kf['id'], kf['descriptors'])
            self._kf_data[kf['id']] = kf

        for mp in map_points_data:
            self._mp_data[mp['id']] = mp

        self._loaded = True
        print(f"Relocalizator ready: {len(self._kf_data)} keyframes, "
              f"{len(self._mp_data)} map points")

    def attempt_relocalization(
        self,
        query_descriptors: np.ndarray,
        query_norm_pts:    np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._loaded or len(query_descriptors) < 20:
            return None

        query_bow  = self._bow_db.vocab.compute_bow_vector(query_descriptors)
        candidates = self._bow_db.query(query_bow, query_keyframe_id=-1, top_k=5)

        for cand_kf_id, sim in candidates:
            result = self._try_pnp(cand_kf_id, query_descriptors, query_norm_pts)
            if result is not None:
                q_GC, p_GC, n_in = result
                print(f"  Relocalization succeeded: KF{cand_kf_id} "
                      f"(sim={sim:.3f}, inliers={n_in})")
                return q_GC, p_GC

        return None

    def _try_pnp(self, kf_id: int, q_descs: np.ndarray,
                 q_norm_pts: np.ndarray):
        kf = self._kf_data.get(kf_id)
        if kf is None:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(q_descs, kf['descriptors'])
        if len(matches) < self.min_pnp_inliers:
            return None

        pts_3d, pts_2d = [], []
        for m in matches:
            kf_norm = kf['norm_pts'][m.trainIdx]
            pts_3d.append([kf_norm[0], kf_norm[1], 1.0])
            pts_2d.append(q_norm_pts[m.queryIdx])

        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d_px = (np.array(pts_2d, dtype=np.float64) * self.camera.fx
                     + np.array([self.camera.cx, self.camera.cy]))

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d_px, self.camera.K, np.zeros(4),
            reprojectionError=2.0, confidence=0.999, iterationsCount=1000)

        if not success or inliers is None or len(inliers) < self.min_pnp_inliers:
            return None

        R_CW, _ = cv2.Rodrigues(rvec)
        t_CW    = tvec.ravel()
        R_WC    = R_CW.T
        t_WC    = -R_CW.T @ t_CW

        return rot_to_quat(R_WC), t_WC, len(inliers)
