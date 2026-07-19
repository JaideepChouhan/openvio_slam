"""
frontend/feature_manager.py
===========================
Manages the lifetime and state of all visual features across frames.

Feature lifecycle:
    ACTIVE -> tracked across frames, accumulating observations
    LOST   -> KLT could not track
    MARGINALIZED -> oldest observation exits sliding window -> fed to MSCKF
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FeatureStatus(Enum):
    ACTIVE       = "active"
    LOST         = "lost"
    MARGINALIZED = "marginalized"


@dataclass
class Feature:
    """
    One tracked feature across multiple frames.
    observations: list of (frame_id, uv_normalized) pairs
    """
    id: int
    status: FeatureStatus
    descriptor: np.ndarray
    observations: List[Tuple]
    pixel_positions: List[np.ndarray]

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def first_frame(self) -> int:
        return self.observations[0][0] if self.observations else -1

    @property
    def last_frame(self) -> int:
        return self.observations[-1][0] if self.observations else -1


class FeatureManager:
    """
    Tracks all active features across frames.
    Decides when to marginalize features for MSCKF update.
    """

    def __init__(self, config: dict, camera_model):
        self.min_track_length = config.get('min_track_length', 3)
        self.min_features     = config.get('min_features', 150)
        self.max_features     = config.get('n_features', 300)
        self.camera = camera_model

        self._features: Dict[int, Feature] = {}
        self._next_id  = 0
        self._frame_id = 0
        self._current_pts: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._current_ids: List[int]  = []

    def process_frame(
        self,
        img_curr: np.ndarray,
        pts_tracked: np.ndarray,
        valid_mask: np.ndarray,
        new_pts: np.ndarray,
        new_descs: np.ndarray,
        window_size: int,
    ) -> Tuple[List[Feature], List[Feature]]:
        """Process one frame: update existing tracks, add new features."""
        self._frame_id += 1
        fid = self._frame_id
        marginalized = []

        still_active_pts = []
        still_active_ids = []

        for i, feat_id in enumerate(self._current_ids):
            feat = self._features[feat_id]

            if i < len(valid_mask) and valid_mask[i]:
                pt_px = pts_tracked[i]
                pt_norm = self.camera.pixel_to_normalized(pt_px)
                feat.observations.append((fid, pt_norm))
                feat.pixel_positions.append(pt_px)
                still_active_pts.append(pt_px)
                still_active_ids.append(feat_id)
            else:
                feat.status = FeatureStatus.LOST
                if feat.n_observations >= self.min_track_length:
                    feat.status = FeatureStatus.MARGINALIZED
                    marginalized.append(feat)

        # Check for window-exited features
        for feat_id in list(still_active_ids):
            feat = self._features[feat_id]
            if fid - feat.first_frame + 1 >= window_size:
                feat.status = FeatureStatus.MARGINALIZED
                marginalized.append(feat)
                idx = still_active_ids.index(feat_id)
                still_active_pts.pop(idx)
                still_active_ids.pop(idx)

        # Add new features
        for pt_px, desc in zip(new_pts, new_descs):
            pt_norm = self.camera.pixel_to_normalized(pt_px)
            feat = Feature(
                id=self._next_id,
                status=FeatureStatus.ACTIVE,
                descriptor=desc.copy(),
                observations=[(fid, pt_norm)],
                pixel_positions=[pt_px],
            )
            self._features[self._next_id] = feat
            still_active_pts.append(pt_px)
            still_active_ids.append(self._next_id)
            self._next_id += 1

        self._current_pts = (np.array(still_active_pts, dtype=np.float32)
                             if still_active_pts
                             else np.zeros((0, 2), dtype=np.float32))
        self._current_ids = still_active_ids

        active = [self._features[i] for i in still_active_ids]
        return marginalized, active

    @property
    def n_active(self) -> int:
        return len(self._current_ids)

    @property
    def needs_new_features(self) -> bool:
        return self.n_active < self.min_features

    @property
    def current_pts(self) -> np.ndarray:
        return self._current_pts

    @property
    def current_ids(self) -> List[int]:
        return self._current_ids
