"""
map/map_point.py
================
3D landmark in the sparse map.
Quality tracked via found_ratio = n_observed / n_visible.
Low found_ratio → cull this point.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MapPoint:
    id:         int
    position:   np.ndarray   # (3,) world frame
    descriptor: np.ndarray   # (32,) uint8 ORB
    n_observed: int = 0
    n_visible:  int = 0

    def mark_observed(self):
        self.n_observed += 1
        self.n_visible  += 1

    def mark_visible_not_found(self):
        self.n_visible += 1

    @property
    def found_ratio(self) -> float:
        return self.n_observed / max(1, self.n_visible)
