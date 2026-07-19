#!/usr/bin/env python3
"""
IMU data analysis with calibration noise parameters.
Usage: python3 imu_analysis_with_calib.py imu_180.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------------------
#  CALIBRATED NOISE VALUES (from your latest calibration)
# -------------------------------------------------------------------
GYRO_NOISE      = 0.000050   # rad/s/√Hz  (white noise density N)
ACCEL_NOISE     = 0.020000   # m/s²/√Hz
GYRO_BIAS_WALK  = 0.000135   # rad/s²/√Hz (rate random walk K)
ACCEL_BIAS_WALK = 0.002000   # m/s³/√Hz
GRAVITY         = 9.81
# -------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 imu_analysis_with_calib.py <imu_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Load and deduplicate
    df = pd.read_csv(csv_file)
    df = df.groupby('timestamp', as_index=False).mean()

    t = df['timestamp'].values
    gz = df['gz'].values      # yaw axis (rotation around Z)
    acc_x = df['ax'].values
    acc_y = df['ay'].values
    acc_z = df['az'].values

    # --- Time step (median dt) ---
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    N = len(t)

    # --- Gyro white noise per sample ---
    sigma_gyro_sample = GYRO_NOISE / np.sqrt(dt)   # rad/s per sample

    # --- Integrate yaw (Euler) ---
    dtheta = gz[:-1] * np.diff(t)
    # Handle potential NaN
    dtheta = np.nan_to_num(dtheta)
    yaw = np.cumsum(dtheta)          # rad
    yaw_deg = np.degrees(yaw)

    # --- Uncertainty growth over time (integrated angle) ---
    # T = cumulative time vector (since start)
    T = t[1:] - t[0]
    # Angle error due to white noise (angle random walk)
    sigma_white = GYRO_NOISE * np.sqrt(T)          # rad
    # Angle error due to bias walk (rate random walk)
    sigma_bias  = GYRO_BIAS_WALK * np.sqrt(T**3 / 3.0)  # rad
    sigma_total = np.sqrt(sigma_white**2 + sigma_bias**2)

    # --- Accelerometer norm ---
    acc_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    sigma_accel_sample = ACCEL_NOISE / np.sqrt(dt)

    # --- Plotting -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Gyro Z with noise envelope
    ax = axes[0, 0]
    ax.plot(t, gz, 'k.', markersize=1, label='gz (raw)')
    # Fill ±2σ noise band (individual sample scatter)
    ax.fill_between(t, -2*sigma_gyro_sample, 2*sigma_gyro_sample,
                    alpha=0.2, color='blue',
                    label=f'±2σ sample noise ({sigma_gyro_sample:.4f} rad/s)')
    ax.set_ylabel('rad/s')
    ax.set_title('Gyro Z (Yaw rate) with noise envelope')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2. Integrated yaw with uncertainty
    ax = axes[0, 1]
    ax.plot(t[1:], yaw_deg, 'b-', label='Integrated yaw')
    # Shade the 2σ uncertainty band
    ax.fill_between(t[1:],
                    (yaw - 2*sigma_total) * 180/np.pi,
                    (yaw + 2*sigma_total) * 180/np.pi,
                    alpha=0.2, color='red',
                    label='±2σ predicted drift')
    # Optional: draw horizontal line at nominal 180°
    ax.axhline(y=180, color='green', linestyle='--', label='Ideal 180°')
    ax.set_ylabel('Degrees')
    ax.set_title('Integrated Yaw (Z axis)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 3. Accelerometer norm
    ax = axes[1, 0]
    ax.plot(t, acc_norm, 'k.', markersize=1, label='|acc|')
    ax.axhline(y=GRAVITY, color='green', linestyle='--', label='g (9.81)')
    ax.fill_between(t,
                    GRAVITY - 2*sigma_accel_sample,
                    GRAVITY + 2*sigma_accel_sample,
                    alpha=0.2, color='cyan',
                    label=f'±2σ sample noise ({sigma_accel_sample:.3f} m/s²)')
    ax.set_ylabel('m/s²')
    ax.set_title('Accelerometer Norm')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 4. Text summary
    ax = axes[1, 1]
    ax.axis('off')
    final_yaw = yaw_deg[-1] if len(yaw_deg) > 0 else 0
    drift_2s = 2 * sigma_total[-1] * 180/np.pi if len(sigma_total) > 0 else 0
    summary = (
        f"Samples: {N}\n"
        f"Duration: {t[-1]-t[0]:.1f} s\n"
        f"Median dt: {dt*1000:.1f} ms  (fs ≈ {fs:.0f} Hz)\n"
        f"Final integrated yaw: {final_yaw:.1f}°\n"
        f"2σ predicted drift: ±{drift_2s:.1f}°\n"
        f"Gyro sample noise (1σ): {sigma_gyro_sample:.4f} rad/s\n"
        f"Accel sample noise (1σ): {sigma_accel_sample:.3f} m/s²"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('imu_calibrated_analysis.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()
