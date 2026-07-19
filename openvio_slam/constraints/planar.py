"""
constraints/planar.py
=====================
Planar motion constraint for ground robots.

Soft constraints on vertical velocity and roll/pitch rates.
"""

import numpy as np


class PlanarConstraint:
    """Generates planar motion pseudo-measurements."""

    def __init__(self, config: dict):
        self.vz_noise    = config.get('planar_vz_noise',    0.05)
        self.roll_noise  = config.get('planar_roll_noise',  0.01)
        self.pitch_noise = config.get('planar_pitch_noise', 0.01)
        self.enabled     = config.get('planar_enabled',     True)

    def get_pseudo_measurements(self, state_dim: int, R_GI: np.ndarray):
        """
        Build planar constraint pseudo-measurements.
        Constrains world-frame vertical velocity v_z ~= 0.
        """
        if not self.enabled:
            return []

        measurements = []

        # Constraint: near-zero vertical velocity
        H_vz = np.zeros((1, state_dim))
        H_vz[0, 8] = 1.0   # v_z at index 8
        R_vz = np.array([[self.vz_noise**2]])
        r_vz = np.zeros(1)
        measurements.append((H_vz, r_vz, R_vz))

        return measurements
