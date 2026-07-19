#!/usr/bin/env python3
"""
Visualize IMU data from a recorded rosbag (CSV format).
Usage: python3 imu_180_viz.py imu_data.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 imu_180_viz.py <imu_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Load CSV
    df = pd.read_csv(csv_file)

    # Ensure columns exist
    required = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    for col in required:
        if col not in df.columns:
            print(f"Error: missing column '{col}' in CSV.")
            sys.exit(1)

    # Remove duplicates by averaging over same timestamps
    df = df.groupby('timestamp', as_index=False).mean()

    t = df['timestamp'].values
    gx = df['gx'].values
    gy = df['gy'].values
    gz = df['gz'].values
    acc_x = df['ax'].values
    acc_y = df['ay'].values
    acc_z = df['az'].values

    # Compute time differences (dt)
    dt = np.diff(t)
    dt[dt > 0.5] = np.nan

    # Integrated orientation from gyro (simple Euler integration)
    dtheta_x = gx[:-1] * dt
    dtheta_y = gy[:-1] * dt
    dtheta_z = gz[:-1] * dt

    theta_x = np.cumsum(np.nan_to_num(dtheta_x))
    theta_y = np.cumsum(np.nan_to_num(dtheta_y))
    theta_z = np.cumsum(np.nan_to_num(dtheta_z))

    theta_x_deg = np.degrees(theta_x)
    theta_y_deg = np.degrees(theta_y)
    theta_z_deg = np.degrees(theta_z)

    # Gyro variance over a moving window (size = 20 samples ≈ 0.1 sec)
    window = 20
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_var = pd.Series(gyro_mag).rolling(window=window, min_periods=1).var().values

    # Accelerometer norm
    accel_norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # 1. Gyro raw
    ax = axes[0, 0]
    ax.plot(t, gx, label='gx')
    ax.plot(t, gy, label='gy')
    ax.plot(t, gz, label='gz')
    ax.set_ylabel('rad/s')
    ax.set_title('Gyroscope (raw)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 2. Accel raw
    ax = axes[0, 1]
    ax.plot(t, acc_x, label='ax')
    ax.plot(t, acc_y, label='ay')
    ax.plot(t, acc_z, label='az')
    ax.set_ylabel('m/s²')
    ax.set_title('Accelerometer (raw)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 3. Integrated yaw (z-axis rotation)
    ax = axes[1, 0]
    ax.plot(t[1:], theta_z_deg, 'b-', label='Yaw (Z)')
    ax.set_ylabel('Degrees')
    ax.set_title('Integrated Yaw (Z)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 4. Integrated pitch/roll
    ax = axes[1, 1]
    ax.plot(t[1:], theta_x_deg, label='Roll (X)')
    ax.plot(t[1:], theta_y_deg, label='Pitch (Y)')
    ax.set_ylabel('Degrees')
    ax.set_title('Integrated Roll / Pitch')
    ax.legend(loc='upper right')
    ax.grid(True)

    # 5. Gyro variance (moving window)
    ax = axes[2, 0]
    ax.plot(t, gyro_var, 'm-')
    ax.axhline(y=0.01, color='r', linestyle='--', label='Example ZUPT threshold')
    ax.set_ylabel('Variance (rad/s)²')
    ax.set_title('Gyro Variance (window size {})'.format(window))
    ax.legend()
    ax.grid(True)

    # 6. Accelerometer norm
    ax = axes[2, 1]
    ax.plot(t, accel_norm, 'c-')
    ax.axhline(y=9.81, color='k', linestyle='--', label='g (9.81)')
    ax.set_ylabel('m/s²')
    ax.set_title('Accelerometer Norm')
    ax.legend()
    ax.grid(True)

    # 7. Gyro magnitude
    ax = axes[3, 0]
    ax.plot(t, gyro_mag, 'k-')
    ax.set_ylabel('rad/s')
    ax.set_title('Gyro Magnitude')
    ax.grid(True)

    # 8. Text summary
    ax = axes[3, 1]
    ax.axis('off')
    summary_text = (
        f"Data points: {len(t)}\n"
        f"Duration: {t[-1]-t[0]:.1f} s\n"
        f"Max gyro |ω|: {np.max(gyro_mag):.3f} rad/s\n"
        f"Max accel deviation from g: {np.max(np.abs(accel_norm-9.81)):.2f} m/s²\n"
        f"Integrated yaw: {theta_z_deg[-1]:.1f}°\n"
        f"Gyro var threshold example: 0.01 rad²/s²"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('imu_analysis.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()