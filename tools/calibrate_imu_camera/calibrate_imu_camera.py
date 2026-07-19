#!/usr/bin/env python3
"""
tools/calibrate_imu_camera.py
==============================
Estimate the IMU-to-Camera extrinsic transform T_IC from a calibration sequence.

T_IC is the rigid body transform that takes a vector expressed in the Camera
frame and expresses it in the IMU frame:
    v_I = R_IC @ v_C + p_IC

Why this matters:
    The MSCKF fuses IMU measurements (in the IMU frame) with visual features
    (in the Camera frame). To do this correctly, we need to know exactly how
    the camera is mounted relative to the IMU. Even a 1° rotational error in
    T_IC causes significant drift over long trajectories.

Method: Hand-eye calibration via checkerboard.
    1. Record a sequence: move the camera+IMU rig in front of a checkerboard,
       making sure to include enough rotation to excite all 3 rotation axes.
    2. For each frame:
       - Detect the checkerboard corners → estimate camera pose T_CW (camera←world)
       - Integrate IMU data between frames → estimate IMU motion T_II' (IMU progression)
    3. The hand-eye equation: A @ T_IC = T_IC @ B
       where A = IMU relative motion, B = camera relative motion (as SE(3) matrices)
    4. Solve for T_IC using the Tsai-Lenz algorithm.

Tsai-Lenz Hand-Eye Algorithm (simplified):
    For each pair of consecutive frames (i, i+1):
        A_i = T_I_{i+1} @ inv(T_I_i)   — IMU-to-IMU relative motion
        B_i = T_C_{i+1} @ inv(T_C_i)   — Cam-to-Cam relative motion

    The constraint is: A_i @ T_IC = T_IC @ B_i

    Rotation part:   R_A @ R_IC = R_IC @ R_B
        → Solved as: (R_A - I) @ q_IC = (I - R_B) @ q_IC  (Kronecker form)
        → This is a least-squares system; solved via SVD.

    Translation part: t_A + R_A @ p_IC = R_IC @ t_B + p_IC
        → Linear system in p_IC; solved after R_IC is known.

Calibration sequence recommendations:
    - Minimum 30 frames with good checkerboard detection
    - Include rotations about all 3 axes (pitch, roll, yaw)
    - Move at moderate speed (not too slow — IMU integration drifts; not too fast — blur)
    - Keep checkerboard fully in frame with all corners visible
    - 5-10 minutes recording at 10fps is ideal

Input CSV format (one row per IMU measurement):
    timestamp, gx, gy, gz, ax, ay, az
    (all SI units: rad/s and m/s²)

Input image directory:
    Frame images named by timestamp (microseconds): 1234567890.png
    Must be synchronized to IMU timestamps within ~5ms.

Usage:
    python3 tools/calibrate_imu_camera.py \
        --images ~/calibration/frames/ \
        --imu    ~/calibration/imu.csv \
        --board-size 9x6 \
        --square-size 0.025 \
        --output config/extrinsic.yaml
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Checkerboard detection ────────────────────────────────────────────────────

def detect_checkerboard(
    image_path: str,
    board_size: Tuple[int, int],
    square_size: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect checkerboard corners in an image and solve PnP to get camera pose.

    Args:
        image_path:  path to image file
        board_size:  (cols, rows) of interior corners — e.g. (9, 6) for a 10×7 board
        square_size: side length of one square in metres

    Returns:
        (rvec, tvec) OpenCV rotation/translation of camera w.r.t. board,
        or None if detection failed.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sub-pixel criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not found:
        return None

    # Refine to sub-pixel accuracy
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Generate 3D object points (board is at z=0)
    obj_pts = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float64)
    obj_pts[:, :2] = np.mgrid[
        0:board_size[0], 0:board_size[1]
    ].T.reshape(-1, 2) * square_size

    # Solve PnP: we need camera intrinsics — use default if not provided
    # In a production calibration, pass the actual camera matrix K here
    # For now we estimate from image size (rough approximation)
    h, w = gray.shape
    f_est = max(w, h)
    K_est = np.array([
        [f_est, 0,     w / 2],
        [0,     f_est, h / 2],
        [0,     0,     1    ],
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_refined, K_est, np.zeros(5))

    if not success:
        return None

    return rvec.ravel(), tvec.ravel()


def rvec_tvec_to_SE3(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV (rvec, tvec) to a 4×4 SE(3) matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec
    return T


# ─── IMU integration ───────────────────────────────────────────────────────────

def integrate_imu_between(
    imu_data: np.ndarray,   # (N, 7): timestamp, gx, gy, gz, ax, ay, az
    t_start:  float,
    t_end:    float,
) -> np.ndarray:
    """
    Integrate IMU gyro measurements between t_start and t_end to get
    a rotation matrix (pure rotation, ignoring translation from double integration).

    Uses simple Euler integration (sufficient for calibration where we
    only need approximate relative rotations between frames).

    Returns:
        R: (3, 3) rotation matrix representing the rotation accumulated
           between t_start and t_end.
    """
    mask = (imu_data[:, 0] >= t_start) & (imu_data[:, 0] <= t_end)
    segment = imu_data[mask]

    if len(segment) < 2:
        return np.eye(3)

    R = np.eye(3)
    for i in range(len(segment) - 1):
        dt = segment[i+1, 0] - segment[i, 0]
        omega = segment[i, 1:4]   # gx, gy, gz (rad/s)
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-10:
            continue
        axis = omega / np.linalg.norm(omega)
        K = np.array([
            [ 0,        -axis[2],  axis[1]],
            [ axis[2],   0,       -axis[0]],
            [-axis[1],   axis[0],  0      ],
        ])
        dR = (np.eye(3)
              + np.sin(angle) * K
              + (1.0 - np.cos(angle)) * (K @ K))
        R = R @ dR

    return R


def build_imu_SE3(R: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """Build SE(3) from rotation and (optional) translation."""
    T = np.eye(4)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


# ─── Tsai-Lenz hand-eye calibration ───────────────────────────────────────────

def tsai_lenz_rotation(
    R_A_list: List[np.ndarray],
    R_B_list: List[np.ndarray],
) -> np.ndarray:
    """
    Solve for R_IC using the Tsai-Lenz formulation for the rotation part.

    Equation: R_A @ R_IC = R_IC @ R_B
    Rewritten: (I - R_A) @ axis(R_IC) ≈ 0 (linearised for each pair)
    Full linear system solved via SVD.

    Returns:
        R_IC: (3, 3) rotation matrix
    """
    M = np.zeros((3 * len(R_A_list), 3))

    for i, (R_A, R_B) in enumerate(zip(R_A_list, R_B_list)):
        # Convert rotations to rotation vectors (axis-angle)
        rvec_A, _ = cv2.Rodrigues(R_A)
        rvec_B, _ = cv2.Rodrigues(R_B)
        rvec_A = rvec_A.ravel()
        rvec_B = rvec_B.ravel()

        # Skew-symmetric cross product matrices
        def skew(v):
            return np.array([
                [ 0,     -v[2],  v[1]],
                [ v[2],   0,    -v[0]],
                [-v[1],   v[0],  0   ],
            ])

        # Tsai-Lenz: (skew(rvec_A + rvec_B)) @ r_IC = rvec_A - rvec_B
        M[3*i:3*i+3, :] = skew(rvec_A + rvec_B)

    # RHS
    b = np.array([
        (R_A_list[i] @ np.ones(3))[j] - (R_B_list[i] @ np.ones(3))[j]
        for i in range(len(R_A_list)) for j in range(3)
    ])
    b = np.concatenate([
        (rvec_A.ravel() - rvec_B.ravel())
        for (rvec_A, rvec_B) in [
            (cv2.Rodrigues(R_A)[0].ravel(), cv2.Rodrigues(R_B)[0].ravel())
            for R_A, R_B in zip(R_A_list, R_B_list)
        ]
    ])

    # Solve M @ r_IC = b in least-squares sense
    r_IC, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    # Normalize and convert back to rotation matrix
    r_IC_norm = r_IC / np.linalg.norm(r_IC)   # unit vector = axis
    angle = np.linalg.norm(r_IC)
    # Renormalize using actual angle (norm of solution IS the rotation angle
    # when using the Tsai-Lenz parametrization in axis-angle space)
    r_IC_full = r_IC_norm * 2.0 * np.arctan(angle)

    R_IC, _ = cv2.Rodrigues(r_IC_full)
    return R_IC


def tsai_lenz_translation(
    R_A_list:  List[np.ndarray],
    t_A_list:  List[np.ndarray],
    R_B_list:  List[np.ndarray],
    t_B_list:  List[np.ndarray],
    R_IC:      np.ndarray,
) -> np.ndarray:
    """
    Solve for the translation p_IC given known R_IC.

    Equation: t_A + R_A @ p_IC = R_IC @ t_B + p_IC
    Rewritten: (R_A - I) @ p_IC = R_IC @ t_B - t_A
    Solved as linear least-squares system.
    """
    A_rows = []
    b_rows = []

    for R_A, t_A, R_B, t_B in zip(R_A_list, t_A_list, R_B_list, t_B_list):
        A_rows.append(R_A - np.eye(3))
        b_rows.append(R_IC @ t_B - t_A)

    A = np.vstack(A_rows)   # (3N, 3)
    b = np.concatenate(b_rows)   # (3N,)

    p_IC, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return p_IC


def calibrate_extrinsic(
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],   # list of (rvec, tvec)
    imu_rotations: List[np.ndarray],                      # list of R (3×3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Tsai-Lenz hand-eye calibration.

    Args:
        camera_poses:  list of (rvec, tvec) for each frame
        imu_rotations: list of rotation matrices from IMU integration

    Returns:
        (R_IC, p_IC)
    """
    n = len(camera_poses)
    assert n == len(imu_rotations), "Must have same number of camera poses and IMU rotations"
    assert n >= 3, "Need at least 3 frames for calibration"

    R_A_list, t_A_list = [], []
    R_B_list, t_B_list = [], []

    T_cam_list = [rvec_tvec_to_SE3(rv, tv) for rv, tv in camera_poses]
    T_imu_list = [build_imu_SE3(R) for R in imu_rotations]

    for i in range(n - 1):
        # Relative motion in camera frame: B = T_{i+1} @ inv(T_i)
        T_B = T_cam_list[i+1] @ np.linalg.inv(T_cam_list[i])
        R_B_list.append(T_B[:3, :3])
        t_B_list.append(T_B[:3, 3])

        # Relative motion in IMU frame
        T_A = T_imu_list[i+1] @ np.linalg.inv(T_imu_list[i])
        R_A_list.append(T_A[:3, :3])
        t_A_list.append(T_A[:3, 3])

    # Solve rotation
    R_IC = tsai_lenz_rotation(R_A_list, R_B_list)

    # Solve translation
    p_IC = tsai_lenz_translation(R_A_list, t_A_list, R_B_list, t_B_list, R_IC)

    return R_IC, p_IC


# ─── Main ─────────────────────────────────────────────────────────────────────

def rotation_to_euler_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to ZYX Euler angles in degrees (for display)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def main():
    parser = argparse.ArgumentParser(
        description="Estimate IMU-Camera extrinsic transform T_IC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--images',      required=True,
                        help='Directory of calibration frame images (PNG/JPG)')
    parser.add_argument('--imu',         required=True,
                        help='CSV file: timestamp,gx,gy,gz,ax,ay,az')
    parser.add_argument('--board-size',  default='9x6',
                        help='Checkerboard interior corners WxH (default: 9x6)')
    parser.add_argument('--square-size', type=float, default=0.025,
                        help='Checkerboard square size in metres (default: 0.025)')
    parser.add_argument('--output',      default='config/extrinsic.yaml',
                        help='Output YAML file path')
    parser.add_argument('--subsample',   type=int, default=1,
                        help='Use every N-th frame (default: 1 = all frames)')
    args = parser.parse_args()

    # Parse board size
    cols, rows = [int(x) for x in args.board_size.lower().split('x')]
    board_size  = (cols, rows)
    square_size = args.square_size

    print(f"OpenVIO-SLAM IMU-Camera Extrinsic Calibration")
    print(f"  Board:        {cols}×{rows} interior corners")
    print(f"  Square size:  {square_size*100:.1f} cm")
    print(f"  Image dir:    {args.images}")
    print(f"  IMU CSV:      {args.imu}")
    print()

    # ── Load IMU data ──────────────────────────────────────────────────────────
    print("Loading IMU data...")
    imu_raw = np.loadtxt(args.imu, delimiter=',', skiprows=1)
    if imu_raw.ndim == 1:
        imu_raw = imu_raw.reshape(1, -1)
    print(f"  {len(imu_raw)} IMU samples, "
          f"{imu_raw[-1,0] - imu_raw[0,0]:.1f}s duration")

    # ── Load and process images ────────────────────────────────────────────────
    img_dir  = Path(args.images)
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    img_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
    )

    if not img_paths:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    # Subsample
    img_paths = img_paths[::args.subsample]
    print(f"Processing {len(img_paths)} images...")

    camera_poses  = []   # (rvec, tvec) per detected frame
    frame_times   = []   # timestamp per detected frame
    n_detected    = 0
    n_failed      = 0

    for i, img_path in enumerate(img_paths):
        # Parse timestamp from filename (microseconds → seconds)
        try:
            ts = float(img_path.stem) / 1e6
        except ValueError:
            # Fall back to sequence index (assumes constant frame rate — imprecise)
            ts = float(i) / 10.0   # assume 10fps

        result = detect_checkerboard(str(img_path), board_size, square_size)

        if result is not None:
            camera_poses.append(result)
            frame_times.append(ts)
            n_detected += 1
        else:
            n_failed += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(img_paths)}: "
                  f"{n_detected} detected, {n_failed} failed")

    print(f"\nDetection: {n_detected}/{len(img_paths)} frames "
          f"({100*n_detected/max(1,len(img_paths)):.0f}%)")

    if n_detected < 10:
        print(f"ERROR: Only {n_detected} valid frames. Need at least 10.")
        print("Tips: check board_size matches your checkerboard, improve lighting.")
        sys.exit(1)

    # ── Integrate IMU between frames ───────────────────────────────────────────
    print("\nIntegrating IMU between frames...")
    imu_rotations = []

    R_cumulative = np.eye(3)
    for i in range(len(frame_times)):
        if i == 0:
            imu_rotations.append(np.eye(3))
        else:
            R_delta = integrate_imu_between(
                imu_raw, frame_times[i-1], frame_times[i])
            R_cumulative = R_cumulative @ R_delta
            imu_rotations.append(R_cumulative.copy())

    # ── Hand-eye calibration ───────────────────────────────────────────────────
    print("Running Tsai-Lenz hand-eye calibration...")
    try:
        R_IC, p_IC = calibrate_extrinsic(camera_poses, imu_rotations)
    except Exception as e:
        print(f"ERROR: Calibration failed: {e}")
        print("Ensure enough rotational diversity in the calibration sequence.")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────────
    roll, pitch, yaw = rotation_to_euler_deg(R_IC)

    print("\n" + "="*60)
    print("  CALIBRATION RESULT")
    print("="*60)
    print(f"\n  R_IC (IMU ← Camera rotation):")
    for row in R_IC:
        print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    print(f"\n  Euler angles (ZYX): roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
    print(f"\n  p_IC (camera origin in IMU frame):")
    print(f"    [{p_IC[0]:+.4f}, {p_IC[1]:+.4f}, {p_IC[2]:+.4f}] metres")
    print()

    # Sanity checks
    det_R = np.linalg.det(R_IC)
    ortho_err = np.max(np.abs(R_IC @ R_IC.T - np.eye(3)))
    print(f"  Sanity: det(R_IC)={det_R:.6f} (should be 1.0), "
          f"ortho_err={ortho_err:.2e} (should be ~0)")
    if abs(det_R - 1.0) > 0.01 or ortho_err > 0.01:
        print("  WARNING: R_IC is not a clean rotation matrix. "
              "More calibration data or better excitation needed.")

    # ── Save to YAML ───────────────────────────────────────────────────────────
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Convert R_IC to quaternion [qw, qx, qy, qz] for YAML
    import cv2
    rvec_ic, _ = cv2.Rodrigues(R_IC)
    angle = float(np.linalg.norm(rvec_ic))
    if angle > 1e-8:
        axis = (rvec_ic / angle).ravel()
        qw = float(np.cos(angle / 2))
        qx, qy, qz = (axis * np.sin(angle / 2)).tolist()
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    result_yaml = {
        'extrinsic': {
            'R_IC': R_IC.tolist(),
            'p_IC': p_IC.tolist(),
            'q_IC': [qw, qx, qy, qz],
            '_comment': (
                'R_IC rotates vectors from Camera to IMU frame. '
                'p_IC is the position of the camera origin in the IMU frame.'
            ),
        },
        'calibration_info': {
            'timestamp':      time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_frames_used':  n_detected,
            'board_size':     f'{cols}x{rows}',
            'square_size_m':  square_size,
            'euler_zyx_deg':  [float(yaw), float(pitch), float(roll)],
        },
    }

    with open(output_path, 'w') as f:
        yaml.dump(result_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to: {output_path}")
    print()
    print("Copy these values into config/slam.yaml under 'extrinsic:':")
    print(f"""
extrinsic:
  # R_IC: rotation matrix — camera frame → IMU frame
  R_IC:
    - [{R_IC[0,0]:+.6f}, {R_IC[0,1]:+.6f}, {R_IC[0,2]:+.6f}]
    - [{R_IC[1,0]:+.6f}, {R_IC[1,1]:+.6f}, {R_IC[1,2]:+.6f}]
    - [{R_IC[2,0]:+.6f}, {R_IC[2,1]:+.6f}, {R_IC[2,2]:+.6f}]
  # p_IC: camera origin expressed in IMU frame (metres)
  p_IC: [{p_IC[0]:+.4f}, {p_IC[1]:+.4f}, {p_IC[2]:+.4f}]
""")


if __name__ == '__main__':
    main()