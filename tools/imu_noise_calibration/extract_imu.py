#!/usr/bin/env python3
"""
Extract IMU data from a rosbag into a CSV with columns:
timestamp (seconds), gx, gy, gz, ax, ay, az
All values in rad/s and m/s².

Usage:
    python3 extract_imu_csv.py <bag_directory> [output_csv]
Example:
    python3 extract_imu_csv.py ./imu_static imu_static.csv
"""

import sys
import os
import csv
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_imu_csv.py <bag_dir> [output.csv]")
        sys.exit(1)

    bag_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "imu_static.csv"

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    imu_data = []   # list of (timestamp, gx, gy, gz, ax, ay, az)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/imu':
            imu = deserialize_message(data, Imu)
            # timestamp is nanoseconds, convert to seconds
            t_sec = timestamp * 1e-9
            gx = imu.angular_velocity.x
            gy = imu.angular_velocity.y
            gz = imu.angular_velocity.z
            ax = imu.linear_acceleration.x
            ay = imu.linear_acceleration.y
            az = imu.linear_acceleration.z
            imu_data.append([t_sec, gx, gy, gz, ax, ay, az])

    if not imu_data:
        print("No IMU messages found. Check that the bag contains /imu.")
        sys.exit(1)

    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        writer.writerows(imu_data)

    print(f"Extracted {len(imu_data)} IMU samples → {output}")

if __name__ == '__main__':
    main()
