"""
imu/noise_model.py
==================
IMU noise parameter model: a typed, validated container for all noise
densities and bias random-walk parameters used throughout the system.

Why a dedicated module?
    IMU noise parameters flow into three separate places:
        1. IMUPreintegrator  — per-step covariance propagation
        2. MSCKF.propagate() — process noise matrix Q (continuous-time PSD)
        3. VIOInitializer    — determines when excitation is sufficient

    Without a single authoritative source, each module would either
    duplicate the parameters (divergence risk) or accept raw dicts with
    no validation (silent bugs from typos like 'gyro_noize': 0.005).

    This module provides:
        - A validated dataclass (catches missing / out-of-range params at startup)
        - Derived quantities (continuous-time PSD matrices Q_c, Q_imu)
        - Allan-variance interpretation helpers
        - A factory that loads from the slam.yaml config dict

Noise model (standard continuous-time IMU model):
    Gyroscope:
        ω_meas(t) = ω_true(t) + b_g(t) + n_g(t)
        n_g ~ N(0, σ_g² I)          white noise,    units: rad/s / √Hz
        ḃ_g  ~ N(0, σ_bg² I)        random walk,    units: rad/s² / √Hz

    Accelerometer:
        a_meas(t) = a_true(t) + b_a(t) + n_a(t)
        n_a ~ N(0, σ_a² I)          white noise,    units: m/s² / √Hz
        ḃ_a  ~ N(0, σ_ba² I)        random walk,    units: m/s³ / √Hz

Discrete-time equivalents (at sample rate f_s = 1/dt):
    σ_g_d  = σ_g  / √dt     [rad/s]
    σ_a_d  = σ_a  / √dt     [m/s²]
    σ_bg_d = σ_bg * √dt     [rad/s]
    σ_ba_d = σ_ba * √dt     [m/s²]

References:
    [1] Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation"
        MARS Lab Technical Report 2005-002 — Appendix A
    [2] Woodman, "An Introduction to Inertial Navigation"
        UCAM Technical Report UCAM-CL-TR-696 (2007)
    [3] IEEE Std 647-2006 — Specification Format Guide for Inertial Sensor Packages
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
# Physical bounds for sanity-checking loaded parameters
# ─────────────────────────────────────────────────────────

# Gyroscope white noise floor (MEMS grade = ~0.0001 rad/s/√Hz minimum)
_GYRO_NOISE_MIN  = 1e-5    # rad/s/√Hz
_GYRO_NOISE_MAX  = 1.0     # rad/s/√Hz  (very noisy consumer IMU)

# Accelerometer white noise floor
_ACCEL_NOISE_MIN = 1e-4    # m/s²/√Hz
_ACCEL_NOISE_MAX = 5.0     # m/s²/√Hz

# Gyro bias random walk
_GYRO_BW_MIN  = 1e-7       # rad/s²/√Hz
_GYRO_BW_MAX  = 0.1        # rad/s²/√Hz

# Accel bias random walk
_ACCEL_BW_MIN = 1e-6       # m/s³/√Hz
_ACCEL_BW_MAX = 0.5        # m/s³/√Hz

# Gravity magnitude range (accounting for altitude variation)
_GRAVITY_MIN = 9.70        # m/s²  (high altitude)
_GRAVITY_MAX = 9.83        # m/s²  (sea level poles)


@dataclass
class IMUNoiseModel:
    """
    Validated IMU noise parameter container.

    All values are in SI units, continuous-time noise density convention
    (multiply by 1/√dt to get discrete-time std dev per sample).

    Attributes:
        gyro_noise     (σ_g):  Gyroscope white noise density  [rad/s/√Hz]
        accel_noise    (σ_a):  Accelerometer white noise density [m/s²/√Hz]
        gyro_bias_walk (σ_bg): Gyroscope bias random walk density [rad/s²/√Hz]
        accel_bias_walk(σ_ba): Accelerometer bias random walk density [m/s³/√Hz]
        gravity        (g):    Local gravitational acceleration [m/s²]
        imu_rate       (f_s):  Expected IMU sample rate [Hz] — for discrete conversion
    """

    gyro_noise:      float  = 0.005
    accel_noise:     float  = 0.05
    gyro_bias_walk:  float  = 0.0001
    accel_bias_walk: float  = 0.001
    gravity:         float  = 9.81
    imu_rate:        float  = 200.0   # Hz

    def __post_init__(self):
        """Validate all parameters on construction. Raises ValueError on bad input."""
        self._validate('gyro_noise',      self.gyro_noise,      _GYRO_NOISE_MIN,  _GYRO_NOISE_MAX)
        self._validate('accel_noise',     self.accel_noise,     _ACCEL_NOISE_MIN, _ACCEL_NOISE_MAX)
        self._validate('gyro_bias_walk',  self.gyro_bias_walk,  _GYRO_BW_MIN,     _GYRO_BW_MAX)
        self._validate('accel_bias_walk', self.accel_bias_walk, _ACCEL_BW_MIN,    _ACCEL_BW_MAX)
        if not (_GRAVITY_MIN <= self.gravity <= _GRAVITY_MAX):
            raise ValueError(
                f"gravity={self.gravity} is outside expected range "
                f"[{_GRAVITY_MIN}, {_GRAVITY_MAX}] m/s². "
                f"Did you accidentally set it in feet/s²?")
        if self.imu_rate < 10.0 or self.imu_rate > 10000.0:
            raise ValueError(f"imu_rate={self.imu_rate} Hz seems wrong. Expected 10–10000 Hz.")

    @staticmethod
    def _validate(name: str, value: float, lo: float, hi: float):
        if not np.isfinite(value) or not (lo <= value <= hi):
            raise ValueError(
                f"IMU parameter '{name}' = {value} is out of expected range "
                f"[{lo}, {hi}]. Check your config/imu.yaml.")

    # ─────────────────────────────────────────────────────
    # Derived: continuous-time process noise PSD matrices
    # ─────────────────────────────────────────────────────

    @property
    def Q_gyro(self) -> np.ndarray:
        """3×3 continuous-time gyroscope white noise PSD matrix. Units: (rad/s)²/Hz"""
        return np.eye(3) * self.gyro_noise ** 2

    @property
    def Q_accel(self) -> np.ndarray:
        """3×3 continuous-time accelerometer white noise PSD matrix. Units: (m/s²)²/Hz"""
        return np.eye(3) * self.accel_noise ** 2

    @property
    def Q_gyro_bias(self) -> np.ndarray:
        """3×3 gyroscope bias random walk PSD matrix. Units: (rad/s²)²/Hz"""
        return np.eye(3) * self.gyro_bias_walk ** 2

    @property
    def Q_accel_bias(self) -> np.ndarray:
        """3×3 accelerometer bias random walk PSD matrix. Units: (m/s³)²/Hz"""
        return np.eye(3) * self.accel_bias_walk ** 2

    @property
    def Q_continuous(self) -> np.ndarray:
        """
        12×12 full continuous-time process noise PSD matrix.
        Order: [n_g (3), n_a (3), n_bg (3), n_ba (3)]

        Used in MSCKF propagation:
            Q_d = G @ Q_continuous @ G.T * dt
        where G is the noise input matrix and dt is the IMU time step.
        """
        return np.diag([
            self.gyro_noise      ** 2,  self.gyro_noise      ** 2,  self.gyro_noise      ** 2,
            self.accel_noise     ** 2,  self.accel_noise     ** 2,  self.accel_noise     ** 2,
            self.gyro_bias_walk  ** 2,  self.gyro_bias_walk  ** 2,  self.gyro_bias_walk  ** 2,
            self.accel_bias_walk ** 2,  self.accel_bias_walk ** 2,  self.accel_bias_walk ** 2,
        ])

    # ─────────────────────────────────────────────────────
    # Derived: discrete-time equivalents
    # ─────────────────────────────────────────────────────

    @property
    def dt_nominal(self) -> float:
        """Nominal time step at the expected IMU rate. dt = 1/f_s"""
        return 1.0 / self.imu_rate

    def gyro_noise_discrete(self, dt: Optional[float] = None) -> float:
        """
        Discrete-time gyroscope noise std dev for a given dt.
        σ_g_d = σ_g / √dt   [rad/s per sample]
        """
        dt = dt if dt is not None else self.dt_nominal
        return self.gyro_noise / np.sqrt(dt)

    def accel_noise_discrete(self, dt: Optional[float] = None) -> float:
        """
        Discrete-time accelerometer noise std dev for a given dt.
        σ_a_d = σ_a / √dt   [m/s² per sample]
        """
        dt = dt if dt is not None else self.dt_nominal
        return self.accel_noise / np.sqrt(dt)

    def gyro_bias_walk_discrete(self, dt: Optional[float] = None) -> float:
        """
        Discrete-time gyroscope bias random walk std dev per step.
        σ_bg_d = σ_bg * √dt   [rad/s per step]
        """
        dt = dt if dt is not None else self.dt_nominal
        return self.gyro_bias_walk * np.sqrt(dt)

    def accel_bias_walk_discrete(self, dt: Optional[float] = None) -> float:
        """
        Discrete-time accelerometer bias random walk std dev per step.
        σ_ba_d = σ_ba * √dt   [m/s² per step]
        """
        dt = dt if dt is not None else self.dt_nominal
        return self.accel_bias_walk * np.sqrt(dt)

    # ─────────────────────────────────────────────────────
    # Allan Variance interpretation
    # ─────────────────────────────────────────────────────

    def allan_angle_random_walk(self) -> float:
        """
        Angle Random Walk (ARW) from Allan variance curve.
        ARW = σ_g  [deg/√hr]    (read from slope=-½ region at τ=1s)
        Typical MEMS: 0.1–5 deg/√hr
        """
        return self.gyro_noise * np.degrees(1.0) * 60.0  # convert rad/s/√Hz → deg/√hr

    def allan_velocity_random_walk(self) -> float:
        """
        Velocity Random Walk (VRW) from Allan variance curve.
        VRW = σ_a  [m/s/√hr]
        Typical MEMS: 0.01–0.5 m/s/√hr
        """
        return self.accel_noise * 3600.0 ** 0.5   # convert m/s²/√Hz → m/s/√hr

    def bias_instability_gyro(self) -> float:
        """
        Approximate gyroscope bias instability (floor of Allan deviation curve).
        Estimated from bias random walk: B ≈ σ_bg * √(τ_corr) at minimum.
        This is an approximation; true bias instability needs full Allan curve.
        """
        # Rule of thumb: correlation time ~100s for MEMS gyros
        tau_corr = 100.0
        return self.gyro_bias_walk * np.sqrt(tau_corr)

    # ─────────────────────────────────────────────────────
    # Factory methods
    # ─────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> 'IMUNoiseModel':
        """
        Build from a config dictionary (from slam.yaml 'imu' section).

        Example config:
            {
              'gyro_noise': 0.005,
              'accel_noise': 0.05,
              'gyro_bias_walk': 0.0001,
              'accel_bias_walk': 0.001,
              'gravity': 9.81,
              'imu_rate': 200.0
            }
        """
        # Support both nested ('imu': {...}) and flat dicts
        imu_cfg = config.get('imu', config)
        try:
            return cls(
                gyro_noise      = float(imu_cfg.get('gyro_noise',      0.005)),
                accel_noise     = float(imu_cfg.get('accel_noise',     0.05)),
                gyro_bias_walk  = float(imu_cfg.get('gyro_bias_walk',  0.0001)),
                accel_bias_walk = float(imu_cfg.get('accel_bias_walk', 0.001)),
                gravity         = float(imu_cfg.get('gravity',         9.81)),
                imu_rate        = float(imu_cfg.get('imu_rate',        200.0)),
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse IMU noise parameters from config: {e}\n"
                f"Config was: {imu_cfg}") from e

    @classmethod
    def default_phone(cls) -> 'IMUNoiseModel':
        """
        Default parameters for a typical Android smartphone IMU.
        Good starting point; calibrate with tools/imu_noise_calibration.py.
        """
        return cls(
            gyro_noise      = 0.005,
            accel_noise     = 0.05,
            gyro_bias_walk  = 0.0001,
            accel_bias_walk = 0.001,
            gravity         = 9.81,
            imu_rate        = 200.0,
        )

    @classmethod
    def default_rpi_imu(cls) -> 'IMUNoiseModel':
        """
        Default parameters for MPU-9250 or ICM-42688 (common Raspberry Pi IMUs).
        These are better-quality sensors than phone IMUs.
        """
        return cls(
            gyro_noise      = 0.002,
            accel_noise     = 0.03,
            gyro_bias_walk  = 0.00005,
            accel_bias_walk = 0.0005,
            gravity         = 9.81,
            imu_rate        = 400.0,
        )

    # ─────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of all parameters and derived quantities."""
        return (
            f"IMUNoiseModel:\n"
            f"  Gyro white noise:       {self.gyro_noise:.6f} rad/s/√Hz"
            f"  ({self.allan_angle_random_walk():.3f} deg/√hr ARW)\n"
            f"  Accel white noise:      {self.accel_noise:.6f} m/s²/√Hz"
            f"  ({self.allan_velocity_random_walk():.4f} m/s/√hr VRW)\n"
            f"  Gyro bias walk:         {self.gyro_bias_walk:.8f} rad/s²/√Hz\n"
            f"  Accel bias walk:        {self.accel_bias_walk:.8f} m/s³/√Hz\n"
            f"  Gravity:                {self.gravity:.4f} m/s²\n"
            f"  IMU rate:               {self.imu_rate:.1f} Hz "
            f"(dt = {self.dt_nominal*1000:.3f} ms)\n"
            f"  Discrete gyro noise:    {self.gyro_noise_discrete():.6f} rad/s per sample\n"
            f"  Discrete accel noise:   {self.accel_noise_discrete():.6f} m/s² per sample\n"
        )

    def __repr__(self) -> str:
        return (f"IMUNoiseModel("
                f"σ_g={self.gyro_noise:.5f}, σ_a={self.accel_noise:.4f}, "
                f"σ_bg={self.gyro_bias_walk:.7f}, σ_ba={self.accel_bias_walk:.6f}, "
                f"g={self.gravity:.3f}, fs={self.imu_rate:.0f}Hz)")
