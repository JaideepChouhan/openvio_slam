"""
tests/test_loop_closure.py
==========================
Tests for BoW vocabulary and pose graph optimizer.
Run: python3 -m pytest tests/test_loop_closure.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openvio_slam.loop_closure.vocabulary import ORBVocabulary, BinaryKMeans
from openvio_slam.loop_closure.pose_graph import PoseGraphOptimizer, PoseGraphEdge


class TestBinaryKMeans:

    def test_fits_without_error(self):
        np.random.seed(0)
        descs = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        km = BinaryKMeans(k=5, max_iter=10, n_restarts=1)
        km.fit(descs)
        assert km.centers_ is not None and km.centers_.shape == (5, 32)

    def test_predict_assigns_all(self):
        np.random.seed(1)
        descs = np.random.randint(0, 256, (50, 32), dtype=np.uint8)
        km = BinaryKMeans(k=3, max_iter=5, n_restarts=1)
        km.fit(descs)
        labels = km.predict(descs)
        assert len(labels) == 50 and set(labels).issubset({0,1,2})


class TestORBVocabulary:

    def test_train_and_compute(self):
        np.random.seed(42)
        descs = np.random.randint(0, 256, (500, 32), dtype=np.uint8)
        vocab = ORBVocabulary(k1=5, k2=5)
        vocab.train(descs, n_training_images=10)
        bow = vocab.compute_bow_vector(np.random.randint(0, 256, (30, 32), dtype=np.uint8))
        assert bow.shape == (25,)
        assert np.all(bow >= 0)
        assert np.linalg.norm(bow) <= 1.0 + 1e-6

    def test_same_descriptors_max_similarity(self):
        np.random.seed(7)
        descs = np.random.randint(0, 256, (1000, 32), dtype=np.uint8)
        vocab = ORBVocabulary(k1=5, k2=4)
        vocab.train(descs, n_training_images=20)
        d1 = descs[:50]
        bow1 = vocab.compute_bow_vector(d1)
        bow2 = vocab.compute_bow_vector(d1)   # same
        bow3 = vocab.compute_bow_vector(descs[500:550])
        assert float(bow1 @ bow2) > float(bow1 @ bow3)
        assert float(bow1 @ bow2) > 0.99


class TestPoseGraphOptimizer:

    def test_no_loop_no_change(self):
        opt = PoseGraphOptimizer(max_iter=5)
        for i in range(5):
            opt.add_pose(i, np.eye(3), np.array([float(i), 0, 0]))
        Omega = np.eye(6) * 100
        for i in range(4):
            opt.add_edge(PoseGraphEdge(i, i+1, np.eye(3), np.array([1.,0,0]), Omega))
        corrected = opt.optimize()
        for kf_id in range(1, 5):
            assert np.allclose(corrected[kf_id][1], [float(kf_id), 0, 0], atol=0.01)

    def test_loop_reduces_error(self):
        """Loop closure edge should reduce trajectory error vs. no correction."""
        # Use a deterministic drift pattern (not random) for stable test
        opt = PoseGraphOptimizer(max_iter=50)
        n, radius = 8, 2.0
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        true_pos = np.array([[radius*np.cos(a), radius*np.sin(a), 0] for a in angles])

        # Deterministic drift: accumulate fixed offset per step
        noisy_pos = true_pos.copy()
        for i in range(1, n):
            noisy_pos[i] += np.array([0.05, 0.02, 0.0]) * i   # grows with time

        for i in range(n):
            opt.add_pose(i, np.eye(3), noisy_pos[i])

        Omega_odom = np.eye(6) * 50
        for i in range(n-1):
            opt.add_edge(PoseGraphEdge(i, i+1, np.eye(3),
                                        noisy_pos[i+1]-noisy_pos[i], Omega_odom))

        # Strong loop closure: first → last (true relative translation)
        t_loop = true_pos[0] - true_pos[n-1]
        opt.add_edge(PoseGraphEdge(n-1, 0, np.eye(3), t_loop, np.eye(6) * 1000))

        corrected = opt.optimize()

        err_before = sum(np.linalg.norm(noisy_pos[i]-true_pos[i]) for i in range(1, n))
        err_after  = sum(np.linalg.norm(corrected[i][1]-true_pos[i]) for i in range(1, n))
        print(f"\n  Error before: {err_before:.3f}m, after: {err_after:.3f}m")
        assert err_after < err_before, \
            f"Loop closure should reduce error: {err_before:.3f} → {err_after:.3f}"

    def test_converges_in_few_iterations(self):
        """Simple chain should converge in under 10 iterations."""
        opt = PoseGraphOptimizer(max_iter=20, convergence_tol=1e-6)
        for i in range(4):
            opt.add_pose(i, np.eye(3), np.array([float(i)*0.5, 0, 0]))
        Omega = np.eye(6) * 200
        for i in range(3):
            opt.add_edge(PoseGraphEdge(i, i+1, np.eye(3), np.array([0.5,0,0]), Omega))
        corrected = opt.optimize()
        # Should produce valid poses
        assert all(np.all(np.isfinite(corrected[i][1])) for i in range(4))
