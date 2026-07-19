"""
tests/test_msckf_propagation.py
================================
Tests for MSCKF IMU propagation invariants.
Run: python3 -m pytest tests/test_msckf_propagation.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openvio_slam.estimator.msckf import MSCKF
from openvio_slam.estimator.state import VIOState
from openvio_slam.frontend.undistortion import PinholeCameraModel


def make_msckf():
    config = {'window_size': 5, 'gyro_noise': 1e-4, 'accel_noise': 1e-3,
              'gyro_bias_walk': 1e-6, 'accel_bias_walk': 1e-5,
              'gravity': 9.81, 'pixel_noise': 1.0}
    cam = PinholeCameraModel({'fx':500.,'fy':500.,'cx':320.,'cy':240.,
                              'dist':[0,0,0,0,0],'width':640,'height':480})
    return MSCKF(config, cam, {'R_IC': np.eye(3), 'p_IC': np.zeros(3)})


class TestMSCKFPropagation:

    def test_covariance_grows(self):
        msckf = make_msckf()
        trace_init = np.trace(msckf.state.P)
        g = 9.81
        for _ in range(100):
            msckf.propagate(np.zeros(3), np.array([0,0,g]), 0.005)
        assert np.trace(msckf.state.P) > trace_init

    def test_stationary_stays_at_origin(self):
        msckf = make_msckf()
        g = 9.81
        for _ in range(400):
            msckf.propagate(np.zeros(3), np.array([0,0,g]), 0.005)
        assert np.linalg.norm(msckf.state.p_GI) < 0.001, \
            f"Drifted {np.linalg.norm(msckf.state.p_GI)*100:.2f}cm"

    def test_covariance_stays_positive_definite(self):
        msckf = make_msckf()
        np.random.seed(0)
        for step in range(200):
            msckf.propagate(np.random.randn(3)*0.1,
                            np.array([0,0,9.81])+np.random.randn(3)*0.1, 0.005)
            if step % 20 == 0:
                P = msckf.state.P
                assert np.allclose(P, P.T, atol=1e-8)
                assert np.all(np.linalg.eigvalsh(P) > -1e-10)

    def test_augmentation_expands_state(self):
        msckf = make_msckf()
        dim_before = msckf.state.state_dim
        msckf.augment_state(frame_id=1)
        assert msckf.state.state_dim == dim_before + VIOState.CAM_DIM

    def test_window_size_limit(self):
        msckf = make_msckf()   # window_size=5
        for i in range(10):
            msckf.augment_state(frame_id=i)
        assert msckf.state.n_cameras <= msckf.state.max_window
