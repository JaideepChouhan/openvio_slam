"""
tests/test_math.py
==================
Comprehensive tests for SO(3) and quaternion utilities.

Run with: python3 -m pytest tests/test_math.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openvio_slam.utils.so3 import hat, vee, exp_map, log_map, right_jacobian
from openvio_slam.utils.quaternion import (
    quat_multiply, quat_normalize, quat_inverse,
    quat_to_rot, rot_to_quat, quat_rotate, quat_exp
)


class TestSO3:

    def test_hat_antisymmetric(self):
        for _ in range(20):
            v = np.random.randn(3)
            M = hat(v)
            assert np.allclose(M + M.T, 0)

    def test_hat_vee_roundtrip(self):
        for _ in range(50):
            v = np.random.randn(3)
            assert np.allclose(vee(hat(v)), v, atol=1e-14)

    def test_hat_cross_product(self):
        for _ in range(50):
            a, b = np.random.randn(3), np.random.randn(3)
            assert np.allclose(hat(a) @ b, np.cross(a, b), atol=1e-14)

    def test_exp_map_is_SO3(self):
        for _ in range(100):
            phi = np.random.randn(3) * np.pi
            R = exp_map(phi)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_exp_log_roundtrip_small(self):
        for _ in range(100):
            phi = np.random.randn(3) * 0.3
            R = exp_map(phi)
            phi_back = log_map(R)
            assert np.allclose(phi, phi_back, atol=1e-9)

    def test_exp_log_roundtrip_large(self):
        for _ in range(100):
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0.01, 0.95 * np.pi)
            phi = axis * angle
            R = exp_map(phi)
            phi_back = log_map(R)
            R_back = exp_map(phi_back)
            assert np.allclose(R, R_back, atol=1e-8)

    def test_exp_identity(self):
        assert np.allclose(exp_map(np.zeros(3)), np.eye(3))

    def test_exp_known_rotation_z(self):
        R = exp_map(np.array([0, 0, np.pi / 2]))
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_right_jacobian_finite_difference(self):
        phi = np.array([0.1, 0.2, -0.15])
        eps = 1e-5
        Jr = right_jacobian(phi)
        Jr_num = np.zeros((3, 3))
        R0 = exp_map(phi)
        for i in range(3):
            e = np.zeros(3); e[i] = eps
            R1 = exp_map(phi + e)
            Jr_num[:, i] = log_map(R0.T @ R1) / eps
        assert np.allclose(Jr, Jr_num, atol=1e-4)


class TestQuaternion:

    def test_multiply_identity(self):
        q_id = np.array([1, 0, 0, 0], dtype=float)
        for _ in range(20):
            q = quat_normalize(np.random.randn(4))
            assert np.allclose(quat_multiply(q, q_id), q, atol=1e-14)

    def test_multiply_inverse(self):
        for _ in range(20):
            q = quat_normalize(np.random.randn(4))
            prod = quat_multiply(q, quat_inverse(q))
            assert np.allclose(abs(prod[0]), 1.0, atol=1e-10)
            assert np.allclose(prod[1:], 0.0, atol=1e-10)

    def test_to_rot_is_SO3(self):
        for _ in range(50):
            q = quat_normalize(np.random.randn(4))
            R = quat_to_rot(q)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rot_quat_roundtrip(self):
        for _ in range(100):
            q = quat_normalize(np.random.randn(4))
            if q[0] < 0:
                q = -q
            R = quat_to_rot(q)
            q_back = rot_to_quat(R)
            assert np.allclose(q, q_back, atol=1e-9) or np.allclose(q, -q_back, atol=1e-9)

    def test_rotate_preserves_norm(self):
        for _ in range(50):
            q = quat_normalize(np.random.randn(4))
            v = np.random.randn(3)
            assert np.allclose(np.linalg.norm(quat_rotate(q, v)), np.linalg.norm(v), atol=1e-10)

    def test_known_rotation(self):
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        v_rot = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        assert np.allclose(v_rot, [0, 1, 0], atol=1e-10)
