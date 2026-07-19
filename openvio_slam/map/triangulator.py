"""
map/triangulator.py
===================
3D point triangulation from multiple views using DLT.

Working in normalized coordinates (no K needed).
"""

import numpy as np
from typing import List, Tuple, Optional


class Triangulator:

    def triangulate_dlt(
        self,
        poses: List[Tuple[np.ndarray, np.ndarray]],
        observations: List[np.ndarray],
        min_parallax_deg: float = 2.0,
        max_reproj_error: float = 0.05,
    ) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from multiple observations.

        Args:
            poses:        list of (R_GC, p_GC)
            observations: list of (x_n, y_n) normalized image coordinates

        Returns:
            p3d: shape (3,) or None if failed
        """
        assert len(poses) == len(observations) >= 2

        n = len(poses)

        # Parallax check
        bearings = []
        for (R_GC, _), obs in zip(poses, observations):
            ray_cam = np.array([obs[0], obs[1], 1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = R_GC @ ray_cam
            bearings.append(ray_world)

        max_parallax = 0.0
        for i in range(n):
            for j in range(i+1, n):
                cos_angle = np.clip(np.dot(bearings[i], bearings[j]), -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                max_parallax = max(max_parallax, angle_deg)

        if max_parallax < min_parallax_deg:
            return None

        # DLT system
        A = np.zeros((2 * n, 4))
        for i, ((R_GC, p_GC), obs) in enumerate(zip(poses, observations)):
            P = np.hstack([R_GC.T, -(R_GC.T @ p_GC).reshape(3, 1)])
            x, y = obs[0], obs[1]
            A[2*i]   = x * P[2] - P[0]
            A[2*i+1] = y * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A)
        X_homog = Vt[-1]

        if abs(X_homog[3]) < 1e-8:
            return None

        p3d = X_homog[:3] / X_homog[3]

        # Validity checks
        for R_GC, p_GC in poses:
            p_cam = R_GC.T @ (p3d - p_GC)
            if p_cam[2] < 0.1:
                return None

        for (R_GC, p_GC), obs in zip(poses, observations):
            p_cam = R_GC.T @ (p3d - p_GC)
            z_hat = p_cam[:2] / p_cam[2]
            err = np.linalg.norm(np.asarray(obs) - z_hat)
            if err > max_reproj_error:
                return None

        return p3d
