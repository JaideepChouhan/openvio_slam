#!/usr/bin/env python3
"""
tools/evaluate_trajectory.py
=============================
Evaluate trajectory accuracy against ground truth.

Metrics:
  ATE (Absolute Trajectory Error): global consistency after Umeyama alignment.
      < 5cm = excellent | < 20cm = good | < 50cm = acceptable | > 50cm = poor
  RTE (Relative Trajectory Error): drift rate over fixed-length segments.

Usage:
    python3 tools/evaluate_trajectory.py \
        --estimated traj_est.txt \
        --groundtruth traj_gt.txt

TUM format: timestamp tx ty tz qx qy qz qw
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_tum(path: str):
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:,0], data[:,1:8]


def to_matrices(poses: np.ndarray) -> np.ndarray:
    N  = len(poses)
    Ms = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),  1-2*(qx**2+qy**2)],
        ])
        Ms[i,:3,:3] = R; Ms[i,:3,3] = [tx, ty, tz]
    return Ms


def align_umeyama(P: np.ndarray, Q: np.ndarray):
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P_c, Q_c   = P - mu_P, Q - mu_Q
    sigma_P    = np.mean(np.sum(P_c**2, axis=1))
    cov        = (Q_c.T @ P_c) / len(P)
    U, S, Vt   = np.linalg.svd(cov)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = np.sum(S * D.diagonal()) / max(sigma_P, 1e-10)
    t = mu_Q - s * R @ mu_P
    return R, t, s


def compute_ate(T_est, T_gt):
    p_est, p_gt = T_est[:,:3,3], T_gt[:,:3,3]
    R, t, s = align_umeyama(p_est, p_gt)
    p_aligned = s * (R @ p_est.T).T + t
    errors = np.linalg.norm(p_aligned - p_gt, axis=1)
    return {'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'max': float(np.max(errors))}


def compute_rte(T_est, T_gt, delta: int = 10):
    from openvio_slam.utils.so3 import log_map
    N = len(T_est)
    trans_errors, rot_errors = [], []
    for i in range(0, N - delta, delta):
        j = i + delta
        Q_est = np.linalg.inv(T_est[i]) @ T_est[j]
        Q_gt  = np.linalg.inv(T_gt[i])  @ T_gt[j]
        Q_err = np.linalg.inv(Q_gt) @ Q_est
        trans_errors.append(np.linalg.norm(Q_err[:3,3]))
        try:
            rot_errors.append(np.linalg.norm(log_map(Q_err[:3,:3])))
        except Exception:
            rot_errors.append(0.0)
    return {'rmse_trans': float(np.sqrt(np.mean(np.array(trans_errors)**2))),
            'rmse_rot':   float(np.sqrt(np.mean(np.array(rot_errors)**2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimated',   required=True)
    parser.add_argument('--groundtruth', required=True)
    parser.add_argument('--delta', type=int, default=10)
    args = parser.parse_args()

    ts_est, poses_est = load_tum(args.estimated)
    ts_gt,  poses_gt  = load_tum(args.groundtruth)

    idx_est, idx_gt = [], []
    for i, t_e in enumerate(ts_est):
        j = np.argmin(np.abs(ts_gt - t_e))
        if abs(ts_gt[j] - t_e) < 0.05:
            idx_est.append(i); idx_gt.append(j)

    if len(idx_est) < 10:
        print(f"ERROR: Only {len(idx_est)} matching timestamps"); return

    T_est = to_matrices(poses_est[idx_est])
    T_gt  = to_matrices(poses_gt[idx_gt])
    print(f"Associated {len(idx_est)} pose pairs")

    ate = compute_ate(T_est, T_gt)
    rte = compute_rte(T_est, T_gt, delta=args.delta)

    print("\n" + "="*50)
    print("  ABSOLUTE TRAJECTORY ERROR (ATE)")
    print("="*50)
    print(f"  RMSE:   {ate['rmse']*100:.2f} cm")
    print(f"  Mean:   {ate['mean']*100:.2f} cm")
    print(f"  Median: {ate['median']*100:.2f} cm")
    print(f"  Max:    {ate['max']*100:.2f} cm")
    rating = ("EXCELLENT" if ate['rmse'] < 0.05 else
              "GOOD"      if ate['rmse'] < 0.20 else
              "ACCEPTABLE" if ate['rmse'] < 0.50 else "POOR")
    print(f"  Rating: {rating}")
    print(f"\n  RTE Trans RMSE: {rte['rmse_trans']*100:.2f} cm/{args.delta}f")
    print(f"  RTE Rot RMSE:   {np.degrees(rte['rmse_rot']):.3f} deg/{args.delta}f")


if __name__ == '__main__':
    main()
