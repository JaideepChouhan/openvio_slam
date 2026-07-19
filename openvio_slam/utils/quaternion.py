"""
utils/quaternion.py
===================
Hamilton quaternion operations.

Convention (CRITICAL):
  We use HAMILTON convention: q = [qw, qx, qy, qz]
  where qw is the scalar part.

  Rotation: v' = q tensor [0, v] tensor q*   (active rotation)
  This is DIFFERENT from JPL convention.
  Mixing conventions silently produces wrong rotations.
"""

import numpy as np
from .so3 import exp_map, log_map


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product: q1 tensor q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length. Raises if near-zero norm."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-10:
        raise ValueError(f"Quaternion has near-zero norm: {n}")
    return q / n


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (= conjugate for unit quaternion)."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert Hamilton quaternion to 3x3 rotation matrix."""
    q = quat_normalize(np.asarray(q, dtype=float))
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)    ],
        [2*(x*y + w*z),          1 - 2*(x**2 + z**2), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),          2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Hamilton quaternion.
    Uses Shepperd's method - numerically stable for all orientations.
    Returns: [qw, qx, qy, qz], unit norm, qw >= 0 (canonical form)
    """
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q. v' = R(q) @ v"""
    return quat_to_rot(q) @ v


def quat_exp(delta_theta: np.ndarray) -> np.ndarray:
    """
    Quaternion exponential map from rotation vector.
    q = [cos(|delta_theta|/2), sin(|delta_theta|/2) * delta_theta/|delta_theta|]
    Used in MSCKF error-state update.
    """
    delta_theta = np.asarray(delta_theta, dtype=float).ravel()
    angle = np.linalg.norm(delta_theta)

    if angle < 1e-7:
        q = np.array([1.0, delta_theta[0]/2,
                           delta_theta[1]/2,
                           delta_theta[2]/2])
    else:
        half = angle / 2.0
        q = np.array([np.cos(half),
                      *(np.sin(half) / angle * delta_theta)])

    return quat_normalize(q)
