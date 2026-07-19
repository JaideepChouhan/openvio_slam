"""
utils/se3.py
============
SE(3) Lie group operations for rigid body transformations.

SE(3) = { (R, t) | R ∈ SO(3), t ∈ R³ } with composition:
    (R1, t1) ∘ (R2, t2) = (R1 R2, R1 t2 + t1)
Inverse: (R, t)⁻¹ = (R.T, -R.T t)

We represent a transform as a 4×4 homogeneous matrix:
    T = [R  t; 0 0 0 1]

We also provide 'exp' and 'log' for SE(3) using the standard
twist coordinates (ω, v) ∈ R⁶.
"""

import numpy as np
from .so3 import hat, vee, exp_map, log_map


def se3_hat(twist: np.ndarray) -> np.ndarray:
    """
    se3_hat: R⁶ → se(3) (4×4 matrix)
    twist = [ω, v] where ω is rotation vector, v is translation vector.
    Returns the 4×4 Lie algebra element:
        [hat(ω)   v]
        [ 0       0 ]
    """
    omega = twist[:3]
    v = twist[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = hat(omega)
    M[:3, 3] = v
    return M


def se3_vee(M: np.ndarray) -> np.ndarray:
    """Inverse of se3_hat: extract (ω, v) from a 4×4 se(3) matrix."""
    omega = vee(M[:3, :3])
    v = M[:3, 3]
    return np.concatenate([omega, v])


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """
    Exponential map for SE(3): twist → homogeneous 4×4 matrix.
    Uses the closed‑form formula from:
        Murray, Li, Sastry "A Mathematical Introduction to Robotic Manipulation".
    """
    omega = twist[:3]
    v = twist[3:]
    theta = np.linalg.norm(omega)
    R = exp_map(omega)

    if theta < 1e-7:
        t = v
    else:
        omega_hat = hat(omega)
        omega_hat2 = omega_hat @ omega_hat
        V = (np.eye(3)
             + (1 - np.cos(theta)) / (theta**2) * omega_hat
             + (theta - np.sin(theta)) / (theta**3) * omega_hat2)
        t = V @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithmic map for SE(3): homogeneous 4×4 matrix → twist (ω, v).
    Returns (ω, v) with ||ω|| ∈ [0, π].
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = log_map(R)
    theta = np.linalg.norm(omega)

    if theta < 1e-7:
        v = t
    else:
        omega_hat = hat(omega)
        omega_hat2 = omega_hat @ omega_hat
        # Inverse of V: V⁻¹ = I - ½ hat(ω) + (1/θ² - (1+cosθ)/(2θ sinθ)) hat(ω)²
        coeff = (1.0 / theta**2
                 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
        V_inv = (np.eye(3)
                 - 0.5 * omega_hat
                 + coeff * omega_hat2)
        v = V_inv @ t

    return np.concatenate([omega, v])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transformations: T1 ∘ T2."""
    return T1 @ T2


def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from R (3×3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
