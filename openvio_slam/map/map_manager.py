"""
map/map_manager.py
==================
Full map lifecycle: keyframe selection, map point management, loop closure integration.

Sits between MSCKF (produces odometry) and loop closure (corrects drift).
Loop closure search runs in a background thread to avoid blocking odometry.
"""

import numpy as np
import threading
from typing import List, Optional, Dict, Tuple
from .keyframe import Keyframe
from .map_point import MapPoint
from .triangulator import Triangulator
from ..loop_closure.vocabulary import ORBVocabulary
from ..loop_closure.detector import LoopCandidateDatabase
from ..loop_closure.verifier import GeometricVerifier
from ..loop_closure.pose_graph import PoseGraphOptimizer, PoseGraphEdge
from ..utils.quaternion import quat_to_rot, rot_to_quat
from ..utils.so3 import log_map


class MapManager:
    def __init__(self, config: dict, vocabulary: ORBVocabulary, camera_model):
        self.config     = config
        self.camera     = camera_model
        self.max_points = config.get('max_map_points', 5000)
        self.cull_ratio = config.get('cull_found_ratio', 0.25)

        self.kf_min_trans  = 0.1
        self.kf_min_rot    = np.deg2rad(10.0)
        self.kf_min_feats  = 100
        self.kf_min_frames = 3

        self._keyframes:   Dict[int, Keyframe] = {}
        self._map_points:  Dict[int, MapPoint] = {}
        self._kf_order:    List[int] = []
        self._next_kf_id   = 0
        self._next_mp_id   = 0
        self._frames_since_kf = 0
        self._last_kf_R: Optional[np.ndarray] = None
        self._last_kf_t: Optional[np.ndarray] = None

        self._loop_db     = LoopCandidateDatabase(
            vocabulary,
            similarity_threshold=config.get('bow_similarity_threshold', 0.6),
            min_temporal_gap=config.get('min_keyframe_spacing', 10),
        )
        self._verifier    = GeometricVerifier(config, camera_model)
        self._pose_graph  = PoseGraphOptimizer(
            max_iter=config.get('pose_graph_max_iter', 50))
        self._triangulator = Triangulator()

        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock    = threading.Lock()
        self._pending_correction: Optional[Dict] = None
        self._n_loop_closures = 0

    def should_insert_keyframe(self, R_curr: np.ndarray, t_curr: np.ndarray,
                                n_tracked: int) -> bool:
        self._frames_since_kf += 1
        if self._frames_since_kf < self.kf_min_frames:
            return False
        if self._last_kf_R is None:
            return True
        trans = np.linalg.norm(t_curr - self._last_kf_t)
        R_rel = self._last_kf_R.T @ R_curr
        rot   = np.linalg.norm(log_map(R_rel))
        return (trans >= self.kf_min_trans
                or rot  >= self.kf_min_rot
                or n_tracked < self.kf_min_feats)

    def insert_keyframe(self, timestamp: float, q_GC: np.ndarray,
                         p_GC: np.ndarray, descriptors: np.ndarray,
                         norm_pts: np.ndarray) -> Keyframe:
        kf = Keyframe(
            id=self._next_kf_id, timestamp=timestamp,
            q_GC=q_GC.copy(), p_GC=p_GC.copy(),
            descriptors=descriptors.copy(), norm_pts=norm_pts.copy(),
            is_fixed=(self._next_kf_id == 0),
        )
        kf.bow_vector = self._loop_db.add_keyframe(kf.id, descriptors)
        self._keyframes[kf.id] = kf
        self._kf_order.append(kf.id)
        self._next_kf_id   += 1
        self._frames_since_kf = 0
        self._last_kf_R = quat_to_rot(q_GC)
        self._last_kf_t = p_GC.copy()

        R = quat_to_rot(q_GC)
        self._pose_graph.add_pose(kf.id, R, p_GC)

        if len(self._kf_order) >= 2:
            prev_kf  = self._keyframes[self._kf_order[-2]]
            R_prev   = quat_to_rot(prev_kf.q_GC)
            R_ij     = R_prev.T @ R
            t_ij     = R_prev.T @ (p_GC - prev_kf.p_GC)
            self._pose_graph.add_edge(PoseGraphEdge(
                id_i=prev_kf.id, id_j=kf.id, R_ij=R_ij, t_ij=t_ij,
                information=np.diag([100.]*3 + [400.]*3)))

        with self._loop_lock:
            if self._pending_correction is not None:
                self._apply_loop_correction(self._pending_correction)
                self._pending_correction = None

        self._trigger_loop_closure_search(kf)
        return kf

    def add_map_point(self, position: np.ndarray, descriptor: np.ndarray) -> MapPoint:
        mp = MapPoint(id=self._next_mp_id,
                      position=position.copy(), descriptor=descriptor.copy())
        self._map_points[self._next_mp_id] = mp
        self._next_mp_id += 1
        if len(self._map_points) > self.max_points:
            self._cull_map_points()
        return mp

    def get_all_map_points(self) -> List[MapPoint]:
        return list(self._map_points.values())

    def _cull_map_points(self):
        scored = sorted([(mp.found_ratio, mp.id) for mp in self._map_points.values()])
        n_remove = len(self._map_points) - self.max_points
        for _, mp_id in scored[:n_remove]:
            del self._map_points[mp_id]

    def _trigger_loop_closure_search(self, query_kf: Keyframe):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(
            target=self._run_loop_closure, args=(query_kf,), daemon=True)
        self._loop_thread.start()

    def _run_loop_closure(self, query_kf: Keyframe):
        candidates = self._loop_db.query(query_kf.bow_vector, query_kf.id, top_k=3)
        for match_id, sim_score in candidates:
            if match_id not in self._keyframes:
                continue
            match_kf = self._keyframes[match_id]
            result = self._verifier.verify(
                query_kf.id, match_id,
                query_kf.descriptors, match_kf.descriptors,
                query_kf.norm_pts, match_kf.norm_pts)
            if result is not None:
                print(f"  Loop closure: KF{query_kf.id} ↔ KF{match_id} "
                      f"(sim={sim_score:.3f}, inliers={result.n_inliers})")
                self._pose_graph.add_edge(PoseGraphEdge(
                    id_i=match_id, id_j=query_kf.id,
                    R_ij=result.R_rel, t_ij=result.t_rel,
                    information=np.diag([400.]*3 + [50.]*3)))
                corrected = self._pose_graph.optimize()
                with self._loop_lock:
                    self._pending_correction = corrected
                    self._n_loop_closures += 1
                break

    def _apply_loop_correction(self, corrected_poses: Dict):
        for kf_id, (R_new, t_new) in corrected_poses.items():
            if kf_id in self._keyframes:
                kf = self._keyframes[kf_id]
                kf.q_GC = rot_to_quat(R_new)
                kf.p_GC = t_new.copy()

    @property
    def n_keyframes(self) -> int:
        return len(self._keyframes)

    @property
    def n_map_points(self) -> int:
        return len(self._map_points)

    @property
    def n_loop_closures(self) -> int:
        return self._n_loop_closures
