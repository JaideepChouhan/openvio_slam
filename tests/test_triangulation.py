"""
tests/test_triangulation.py
============================
Tests for the DLT triangulator.
Run: python3 -m pytest tests/test_triangulation.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openvio_slam.map.triangulator import Triangulator
from openvio_slam.utils.so3 import exp_map


class TestTriangulator:
    def setup_method(self):
        self.tri = Triangulator()

    def _obs(self, R_GC, p_GC, p3d):
        p_cam = R_GC.T @ (p3d - p_GC)
        return p_cam[:2] / p_cam[2]

    def test_basic_two_views(self):
        p3d = np.array([1.0, 0.5, 3.0])
        R1, t1 = np.eye(3), np.zeros(3)
        R2, t2 = np.eye(3), np.array([0.5, 0, 0])
        result = self.tri.triangulate_dlt(
            [(R1,t1),(R2,t2)], [self._obs(R1,t1,p3d), self._obs(R2,t2,p3d)],
            min_parallax_deg=2.0)
        assert result is not None
        assert np.allclose(result, p3d, atol=1e-6), f"{result} vs {p3d}"

    def test_rejects_insufficient_parallax(self):
        p3d = np.array([0., 0., 10.])
        R1, t1 = np.eye(3), np.zeros(3)
        R2, t2 = np.eye(3), np.array([0.001, 0, 0])
        result = self.tri.triangulate_dlt(
            [(R1,t1),(R2,t2)], [self._obs(R1,t1,p3d), self._obs(R2,t2,p3d)],
            min_parallax_deg=5.0)
        assert result is None

    def test_multiple_views_accuracy(self):
        p3d = np.array([0.5, -0.3, 4.0])
        poses, obs = [], []
        for i in range(5):
            R = exp_map(np.array([0, i*0.2, 0]))
            t = np.array([i*0.3, 0., 0.])
            poses.append((R, t)); obs.append(self._obs(R, t, p3d))

        r2 = self.tri.triangulate_dlt(poses[:2], obs[:2], min_parallax_deg=2.0)
        r5 = self.tri.triangulate_dlt(poses,     obs,     min_parallax_deg=2.0)
        assert r2 is not None and r5 is not None
        assert np.linalg.norm(r5-p3d) <= np.linalg.norm(r2-p3d) + 1e-4

    def test_noisy_observations(self):
        p3d = np.array([0., 0., 3.0])
        R1, t1 = np.eye(3), np.zeros(3)
        R2, t2 = np.eye(3), np.array([0.3, 0, 0])
        np.random.seed(42)
        obs1 = self._obs(R1,t1,p3d) + np.random.randn(2)*0.002
        obs2 = self._obs(R2,t2,p3d) + np.random.randn(2)*0.002
        result = self.tri.triangulate_dlt(
            [(R1,t1),(R2,t2)], [obs1,obs2], min_parallax_deg=2.0, max_reproj_error=0.1)
        assert result is not None
        assert np.linalg.norm(result - p3d) < 0.1
