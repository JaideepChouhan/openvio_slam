#!/usr/bin/env python3
"""
Extract IMU data and images from a calibration bag.
Renames images to their timestamp (microseconds) and creates imu.csv.
"""

import rosbag2_py
import sys
import csv
import os
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, Image
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python extract_calib_data.py <bag_directory>")
    sys.exit(1)

bag_dir = sys.argv[1]
out_dir = "calib_images_ts"   # images with timestamp names
imu_csv = "imu.csv"

os.makedirs(out_dir, exist_ok=True)

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='')
converter_options = rosbag2_py.ConverterOptions('', '')
reader.open(storage_options, converter_options)

# We'll accumulate IMU data and image timestamps
imu_data = []      # list of (timestamp, gx, gy, gz, ax, ay, az)
image_timestamps = []  # list of (timestamp, image_data)

while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == '/imu':
        imu = deserialize_message(data, Imu)
        imu_data.append((timestamp,
                         imu.angular_velocity.x,
                         imu.angular_velocity.y,
                         imu.angular_velocity.z,
                         imu.linear_acceleration.x,
                         imu.linear_acceleration.y,
                         imu.linear_acceleration.z))
    elif topic == '/image':
        img_msg = deserialize_message(data, Image)
        # Convert to BGR for saving
        h, w = img_msg.height, img_msg.width
        raw = np.frombuffer(img_msg.data, dtype=np.uint8)
        if img_msg.encoding == 'bgr8':
            bgr = raw.reshape(h, w, 3)
        elif img_msg.encoding == 'rgb8':
            rgb = raw.reshape(h, w, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding in ('mono8', '8UC1'):
            gray = raw.reshape(h, w)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            print(f"Unknown encoding: {img_msg.encoding}, skipping")
            continue
        # Save image with timestamp (microseconds) as filename
        ts_us = int(timestamp / 1e3)   # nanoseconds → microseconds
        fname = os.path.join(out_dir, f"{ts_us}.jpg")
        cv2.imwrite(fname, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_timestamps.append(timestamp)

print(f"Extracted {len(imu_data)} IMU samples, {len(image_timestamps)} images")

# Write IMU CSV
with open(imu_csv, 'w') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
    for ts, gx, gy, gz, ax, ay, az in imu_data:
        w.writerow([ts*1e-9, gx, gy, gz, ax, ay, az])

print(f"IMU CSV written to {imu_csv}")
print(f"Images with timestamp names saved in {out_dir}/")
