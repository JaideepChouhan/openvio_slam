"""
frontend/orb_extractor.py
=========================
ORB feature detection with spatial grid bucketing.

Grid bucketing enforces uniform coverage across the image.
Without it, ORB clusters features in high-texture regions,
causing tracking failure when the camera pans away from those regions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class ORBExtractor:
    """
    Grid-bucketed ORB feature extractor.
    Detects features uniformly by enforcing a maximum per grid cell.
    """

    def __init__(self, config: dict):
        self.n_features   = config.get('n_features', 300)
        self.grid_rows    = config.get('grid_rows', 4)
        self.grid_cols    = config.get('grid_cols', 5)
        n_cells           = self.grid_rows * self.grid_cols
        self.max_per_cell = max(1, self.n_features // n_cells + 2)

        self._orb = cv2.ORB_create(
            nfeatures=self.n_features,   # not *2 – we only need enough to fill the grid
            scaleFactor=config.get('scale_factor', 1.2),
            nlevels=config.get('n_levels', 8),
            fastThreshold=config.get('fast_threshold', 10),
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
        )

    def detect_and_describe(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors with grid bucketing.

        Returns:
            pts:   shape (N, 2) float32 -- [u, v] pixel coordinates
            descs: shape (N, 32) uint8  -- ORB binary descriptors
        """
        assert len(image.shape) == 2, "Expected grayscale image"
        H, W = image.shape

        kps, descs_raw = self._orb.detectAndCompute(image, mask)

        if kps is None or len(kps) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

        cell_h = H / self.grid_rows
        cell_w = W / self.grid_cols

        cells: dict = {}
        for idx, kp in enumerate(kps):
            row = min(int(kp.pt[1] / cell_h), self.grid_rows - 1)
            col = min(int(kp.pt[0] / cell_w), self.grid_cols - 1)
            key = (row, col)
            if key not in cells:
                cells[key] = []
            cells[key].append((kp.response, idx))

        kept_indices = []
        for cell_kps in cells.values():
            cell_kps.sort(key=lambda x: x[0], reverse=True)
            for _, idx in cell_kps[:self.max_per_cell]:
                kept_indices.append(idx)

        if not kept_indices:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

        kept_kps   = [kps[i] for i in kept_indices]
        kept_descs = descs_raw[kept_indices]

        pts = np.array([kp.pt for kp in kept_kps], dtype=np.float32)
        return pts, kept_descs

    def create_exclusion_mask(
        self,
        image_shape: Tuple[int, int],
        existing_pts: np.ndarray,
        radius: int = 10
    ) -> np.ndarray:
        """Create a mask excluding regions near existing tracked points."""
        mask = np.full(image_shape, 255, dtype=np.uint8)
        if existing_pts is None or len(existing_pts) == 0:
            return mask
        for pt in existing_pts:
            u, v = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(mask, (u, v), radius, 0, -1)
        return mask
