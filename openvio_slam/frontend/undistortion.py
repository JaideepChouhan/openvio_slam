"""
frontend/undistortion.py
========================
Camera model: undistortion and coordinate normalization.

Pinhole + radial-tangential distortion model:
    Intrinsics: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Distortion: [k1, k2, p1, p2, k3]  (OpenCV convention)

All MSCKF Jacobians work with NORMALIZED coordinates:
    x_norm = (u_undistorted - cx) / fx
    y_norm = (v_undistorted - cy) / fy
"""

import numpy as np
import cv2
from typing import Tuple


class PinholeCameraModel:
    """
    Pinhole camera with radial-tangential distortion.

    Config keys: fx, fy, cx, cy, dist, width, height
    """

    def __init__(self, config: dict):
        cam = config.get('cam', config)
        self.fx = float(cam['fx'])
        self.fy = float(cam['fy'])
        self.cx = float(cam['cx'])
        self.cy = float(cam['cy'])
        self.dist = np.array(cam.get('dist', [0, 0, 0, 0, 0]), dtype=float)
        self.width  = int(cam.get('width',  640))
        self.height = int(cam.get('height', 480))

        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]], dtype=float)

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K,
            (self.width, self.height), cv2.CV_32FC1
        )

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
        pts_undist = cv2.undistortPoints(pts_cv, self.K, self.dist, P=self.K)
        return pts_undist.reshape(-1, 2)

    def pixel_to_normalized(self, pt: np.ndarray) -> np.ndarray:
        pts = self.undistort_points(pt.reshape(1, 2))
        u, v = pts[0]
        return np.array([(u - self.cx) / self.fx,
                         (v - self.cy) / self.fy], dtype=float)

    def normalized_to_pixel(self, pt_n: np.ndarray) -> np.ndarray:
        return np.array([pt_n[0] * self.fx + self.cx,
                         pt_n[1] * self.fy + self.cy], dtype=float)

    def project_point(self, p_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D point in camera frame to normalized image coordinates.
        Returns (z_norm, J_p) where J_p is the 2x3 Jacobian d[x,y]/dp_cam.
        """
        X, Y, Z = p_cam
        if Z < 1e-6:
            raise ValueError(f"Point is behind camera: Z={Z}")

        z_norm = np.array([X / Z, Y / Z])
        J_p = np.array([
            [1/Z,  0,   -X/Z**2],
            [0,    1/Z, -Y/Z**2]
        ])
        return z_norm, J_p
