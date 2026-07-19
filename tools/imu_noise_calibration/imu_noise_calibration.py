#!/usr/bin/env python3
"""
tools/imu_noise_calibration.py
================================
Estimate IMU noise parameters via Allan Variance from a stationary recording.

Usage:
    # Record 5-10 min stationary:
    ros2 bag record -o imu_static /imu/data
    # Extract to CSV (time,gx,gy,gz,ax,ay,az), then:
    python3 tools/imu_noise_calibration.py --data imu_static.csv --rate 200

Output: recommended values for config/slam.yaml
"""

import argparse
import numpy as np


def compute_allan_variance(data: np.ndarray, fs: float):
    N, dt = len(data), 1.0 / fs
    theta = np.cumsum(data) * dt
    max_n = N // 10
    n_vals = np.unique(np.logspace(0, np.log10(max(max_n, 2)), 100).astype(int))

    taus, adevs = [], []
    for n in n_vals:
        tau = n * dt
        n_clusters = N // n
        if n_clusters < 2:
            break
        c     = theta[:n_clusters * n].reshape(n_clusters, n)
        means = c.mean(axis=1)
        diff  = np.diff(means)
        avar  = np.mean(diff**2) / (2 * tau**2)
        taus.append(tau)
        adevs.append(np.sqrt(max(avar, 1e-20)))

    return np.array(taus), np.array(adevs)


def estimate_noise_params(taus, adev, label: str) -> dict:
    n_short = max(1, len(taus) // 5)
    white_noise = float(np.median(adev[:n_short] * np.sqrt(taus[:n_short])))
    n_long  = max(1, len(taus) // 5)
    bias_walk   = float(np.median(adev[-n_long:] / np.sqrt(taus[-n_long:])))
    print(f"\n  {label}:")
    print(f"    White noise (σ):       {white_noise:.6f}")
    print(f"    Bias random walk (σb): {bias_walk:.6f}")
    return {'white_noise': white_noise, 'bias_walk': bias_walk}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV: time,gx,gy,gz,ax,ay,az')
    parser.add_argument('--rate', type=float, default=200.0)
    args = parser.parse_args()

    data  = np.loadtxt(args.data, delimiter=',', skiprows=1)
    times = data[:,0]; gyro = data[:,1:4]; accel = data[:,4:7]
    print(f"Loaded {len(times)} samples, {(times[-1]-times[0]):.1f}s")

    results = {}
    for ax_idx, ax in enumerate(['X','Y','Z']):
        taus, adev = compute_allan_variance(gyro[:,ax_idx], args.rate)
        results[f'gyro_{ax}'] = estimate_noise_params(taus, adev, f"Gyro {ax}")
        taus, adev = compute_allan_variance(accel[:,ax_idx], args.rate)
        results[f'accel_{ax}'] = estimate_noise_params(taus, adev, f"Accel {ax}")

    gyro_noise  = np.median([results[f'gyro_{a}']['white_noise']  for a in 'XYZ'])
    gyro_walk   = np.median([results[f'gyro_{a}']['bias_walk']    for a in 'XYZ'])
    accel_noise = np.median([results[f'accel_{a}']['white_noise'] for a in 'XYZ'])
    accel_walk  = np.median([results[f'accel_{a}']['bias_walk']   for a in 'XYZ'])

    print("\n" + "="*60)
    print("  RECOMMENDED CONFIG (copy into config/slam.yaml):")
    print("="*60)
    print(f"""
imu:
  gyro_noise:       {gyro_noise:.6f}   # rad/s/sqrt(Hz)
  accel_noise:      {accel_noise:.6f}   # m/s^2/sqrt(Hz)
  gyro_bias_walk:   {gyro_walk:.6f}   # rad/s^2/sqrt(Hz)
  accel_bias_walk:  {accel_walk:.6f}   # m/s^3/sqrt(Hz)
""")


if __name__ == '__main__':
    main()
