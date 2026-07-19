# tests/test_preintegrator.py
import numpy as np
import pytest
from openvio_slam.imu.preintegrator import IMUPreintegrator

class TestIMUPreintegrator:
    def test_zero_imu(self):
        """No motion should keep deltas identity/zero."""
        params = {'gyro_noise': 0.001, 'accel_noise': 0.01,
                  'gyro_bias_walk': 0.0001, 'accel_bias_walk': 0.001}
        preint = IMUPreintegrator(params)
        preint.reset(np.zeros(3), np.zeros(3))
        for _ in range(100):
            preint.integrate(np.zeros(3), np.zeros(3), 0.005)
        d = preint.get_delta()
        assert np.allclose(d['delta_R'], np.eye(3), atol=1e-6)
        assert np.allclose(d['delta_v'], 0, atol=1e-6)
        assert np.allclose(d['delta_p'], 0, atol=1e-6)

    def test_bias_correction(self):
        """Bias change should update deltas correctly."""
        params = {'gyro_noise': 0.001, 'accel_noise': 0.01,
                  'gyro_bias_walk': 0.0001, 'accel_bias_walk': 0.001}
        preint = IMUPreintegrator(params)
        preint.reset(np.zeros(3), np.zeros(3))
        # Integrate some motion
        preint.integrate(np.array([0.1,0,0]), np.array([0,0,9.81]), 0.01)
        preint.integrate(np.array([0,0.1,0]), np.array([0,0,9.81]), 0.01)
        d_old = preint.get_delta()
        # Correct bias
        d_corr = preint.correct_for_bias_change(np.array([0.001,0,0]), np.zeros(3))
        # Should be different
        assert not np.allclose(d_corr['delta_R'], d_old['delta_R'], atol=1e-6)
