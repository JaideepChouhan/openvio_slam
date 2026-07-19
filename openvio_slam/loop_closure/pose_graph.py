"""
loop_closure/pose_graph.py
==========================
Gauss-Newton pose graph optimizer on SE(3) — implemented from scratch.

Corrects accumulated VIO drift when a loop closure is detected.
Gauge fixed by holding the first pose constant.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import scipy.sparse
import scipy.sparse.linalg

from ..utils.so3 import hat, exp_map, log_map, right_jacobian_inverse


@dataclass
class PoseGraphEdge:
    id_i:        int
    id_j:        int
    R_ij:        np.ndarray
    t_ij:        np.ndarray
    information: np.ndarray   # 6×6


class PoseGraphOptimizer:
    def __init__(self, max_iter: int = 50, convergence_tol: float = 1e-5):
        self.max_iter        = max_iter
        self.convergence_tol = convergence_tol
        self._poses:      Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._pose_order: List[int] = []
        self._edges:      List[PoseGraphEdge] = []

    def add_pose(self, keyframe_id: int, R: np.ndarray, t: np.ndarray):
        self._poses[keyframe_id] = (R.copy(), t.copy())
        self._pose_order.append(keyframe_id)

    def add_edge(self, edge: PoseGraphEdge):
        self._edges.append(edge)

    def update_pose(self, keyframe_id: int, R: np.ndarray, t: np.ndarray):
        self._poses[keyframe_id] = (R.copy(), t.copy())

    def optimize(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        n_poses = len(self._poses)
        if n_poses < 2 or not self._edges:
            return dict(self._poses)

        id_to_idx = {kf_id: i for i, kf_id in enumerate(self._pose_order)}
        dim = n_poses * 6

        print(f"Pose graph: {n_poses} poses, {len(self._edges)} edges. Optimizing...")

        for iteration in range(self.max_iter):
            H_rows, H_cols, H_vals = [], [], []
            b = np.zeros(dim)

            for edge in self._edges:
                if edge.id_i not in id_to_idx or edge.id_j not in id_to_idx:
                    continue

                idx_i = id_to_idx[edge.id_i]
                idx_j = id_to_idx[edge.id_j]
                R_i, t_i = self._poses[edge.id_i]
                R_j, t_j = self._poses[edge.id_j]

                # Residual: r = log(R_ij^T R_i^T R_j), R_ij^T(R_i^T(t_j-t_i)-t_ij)
                R_res = edge.R_ij.T @ R_i.T @ R_j
                r_rot = log_map(R_res)   # (3,)
                t_res = edge.R_ij.T @ (R_i.T @ (t_j - t_i) - edge.t_ij)  # (3,)
                r     = np.concatenate([r_rot, t_res])   # (6,)

                # Jacobians w.r.t. pose i: [dθ_i, dt_i]
                Jr_inv   = right_jacobian_inverse(r_rot)
                J_i = np.zeros((6, 6))
                J_i[0:3, 0:3] = -Jr_inv @ R_j.T @ R_i          # ∂r_rot/∂dθ_i
                J_i[3:6, 0:3] =  edge.R_ij.T @ hat(R_i.T @ (t_j - t_i))  # ∂r_t/∂dθ_i
                J_i[3:6, 3:6] = -edge.R_ij.T @ R_i.T           # ∂r_t/∂dt_i

                # Jacobians w.r.t. pose j: [dθ_j, dt_j]
                J_j = np.zeros((6, 6))
                J_j[0:3, 0:3] =  Jr_inv @ R_j.T @ R_i          # ∂r_rot/∂dθ_j
                J_j[3:6, 3:6] =  edge.R_ij.T @ R_i.T           # ∂r_t/∂dt_j

                Omega = edge.information
                si, sj = idx_i*6, idx_j*6

                for (ro, co, M) in [(si, si, J_i.T @ Omega @ J_i),
                                    (si, sj, J_i.T @ Omega @ J_j),
                                    (sj, si, J_j.T @ Omega @ J_i),
                                    (sj, sj, J_j.T @ Omega @ J_j)]:
                    for rr in range(6):
                        for cc in range(6):
                            H_rows.append(ro+rr); H_cols.append(co+cc)
                            H_vals.append(float(M[rr, cc]))

                b[si:si+6] += J_i.T @ Omega @ r
                b[sj:sj+6] += J_j.T @ Omega @ r

            H_sp = scipy.sparse.csr_matrix((H_vals, (H_rows, H_cols)), shape=(dim, dim))

            # Gauge fixing: hold pose 0 fixed
            H_lil = H_sp.tolil()
            for row in range(6):
                H_lil[row, :] = 0; H_lil[:, row] = 0
                H_lil[row, row] = 1.0
            H_sp = H_lil.tocsr()
            b[:6] = 0.0

            try:
                delta_x = scipy.sparse.linalg.spsolve(H_sp, -b)
            except Exception as e:
                print(f"  Solve failed: {e}"); break

            if not np.all(np.isfinite(delta_x)):
                print("  Non-finite delta_x — stopping"); break

            for kf_id, idx in id_to_idx.items():
                dθ = delta_x[idx*6:idx*6+3]
                dt = delta_x[idx*6+3:idx*6+6]
                R_old, t_old = self._poses[kf_id]
                self._poses[kf_id] = (R_old @ exp_map(dθ), t_old + dt)

            max_update = np.max(np.abs(delta_x))
            print(f"  Iter {iteration+1}: max_update={max_update:.6f}")
            if max_update < self.convergence_tol:
                print(f"  Converged at iteration {iteration+1}"); break

        return dict(self._poses)
