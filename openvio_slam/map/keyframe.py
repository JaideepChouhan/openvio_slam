"""
map/keyframe.py
===============
Keyframe data structure: a retained camera frame with visual data and BoW vector.

Keyframe selection policy (implemented in MapManager):
  - Minimum translation:  0.1m
  - Minimum rotation:     10°
  - Feature-based:        if tracked < 100, always insert
  - Temporal:             max 1 per 3 frames
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Keyframe:
    id:            int
    timestamp:     float
    q_GC:          np.ndarray   # (4,) [qw,qx,qy,qz]
    p_GC:          np.ndarray   # (3,) world frame
    descriptors:   np.ndarray   # (N,32) uint8
    norm_pts:      np.ndarray   # (N,2) normalized image coords
    bow_vector:    Optional[np.ndarray] = None
    map_point_ids: Optional[List[int]]  = None
    is_fixed:      bool = False

    @property
    def R_GC(self) -> np.ndarray:
        from ..utils.quaternion import quat_to_rot
        return quat_to_rot(self.q_GC)

    @property
    def n_features(self) -> int:
        return len(self.descriptors)
