"""
utils/so3.py
============
SO(3) Lie Group operations for rotation math.

SO(3) is the group of 3x3 rotation matrices {R : R^T R = I, det(R) = 1}.
Its Lie algebra so(3) is the space of 3x3 skew-symmetric matrices,
isomorphic to R3 via the hat/vee operators.

Why Lie groups for rotations?
  Rotations are NOT a vector space - you cannot add two rotation matrices
  and get a valid rotation. But you CAN:
    - Compose them (matrix multiply): R1 @ R2
    - Invert them: R.T
    - Perturb them smoothly: R @ exp_map(delta_theta) for small delta_theta in R3

  The exp_map and log_map connect the group (rotations) to its algebra
  (rotation vectors), enabling us to do calculus (Jacobians, covariance
  propagation) on a curved manifold using flat-space tools.

Convention: active rotation. R_AB rotates vectors from frame B into frame A.
  v_A = R_AB @ v_B

References:
  [1] Sola et al., "A micro Lie theory for state estimation in robotics" (2018)
      arxiv.org/abs/1812.01537
  [2] Barfoot, "State Estimation for Robotics" (2017) - Chapter 7
"""

import numpy as np


def hat(v: np.ndarray) -> np.ndarray:
    """
    hat: R3 -> so(3)
    Converts a 3-vector into its 3x3 skew-symmetric matrix representation.

    Property: hat(v) @ w == np.cross(v, w) for all w in R3
    """
    v = np.asarray(v, dtype=float).ravel()
    assert v.shape == (3,), f"hat() requires a 3-vector, got shape {v.shape}"
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])


def vee(M: np.ndarray) -> np.ndarray:
    """
    vee: so(3) -> R3
    Inverse of hat(). Extracts the 3-vector from a skew-symmetric matrix.
    """
    M = np.asarray(M, dtype=float)
    assert M.shape == (3, 3), f"vee() requires a 3x3 matrix, got {M.shape}"
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def exp_map(phi: np.ndarray) -> np.ndarray:
    """
    exp_map: so(3) -> SO(3)
    Converts a rotation vector (axis x angle) into a rotation matrix.
    Rodrigues' rotation formula with small-angle stabilization.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    assert phi.shape == (3,)
    theta = np.linalg.norm(phi)

    if theta < 1e-7:
        return np.eye(3) + hat(phi)

    axis = phi / theta
    K = hat(axis)
    R = (np.eye(3)
         + np.sin(theta) * K
         + (1.0 - np.cos(theta)) * (K @ K))
    return R


def log_map(R: np.ndarray) -> np.ndarray:
    """
    log_map: SO(3) -> so(3)
    Converts a rotation matrix into a rotation vector.
    Inverse of exp_map. Handles near-identity and near-180deg edge cases.
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3)

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-7:
        return vee(R - R.T) * 0.5

    if abs(theta - np.pi) < 1e-7:
        M = (R + np.eye(3)) / 2.0
        col = np.argmax(np.diag(M))
        n = M[:, col]
        n = n / np.linalg.norm(n)
        phi_unsigned = n * theta
        phi_candidate = vee(R - R.T)
        if np.dot(phi_candidate, phi_unsigned) < 0:
            phi_unsigned = -phi_unsigned
        return phi_unsigned

    return vee(R - R.T) * theta / (2.0 * np.sin(theta))


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """
    Right Jacobian of SO(3), J_r(phi).
    Satisfies: exp(phi + delta) ~= exp(phi) @ exp(J_r(phi) @ delta) for small delta
    Used in IMU pre-integration noise propagation.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    K = hat(phi)

    if theta < 1e-7:
        return np.eye(3) - 0.5 * K

    return (np.eye(3)
            - (1.0 - np.cos(theta)) / (theta ** 2) * K
            + (theta - np.sin(theta)) / (theta ** 3) * (K @ K))


def right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian, J_r_inv(phi)."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    K = hat(phi)

    if theta < 1e-7:
        return np.eye(3) + 0.5 * K

    coeff = (1.0 / theta ** 2
             - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) + 0.5 * K + coeff * (K @ K)
