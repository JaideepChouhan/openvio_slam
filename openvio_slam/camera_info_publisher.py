#!/usr/bin/env python3
"""
camera_info_publisher.py — publish a static sensor_msgs/CameraInfo.

The phone does NOT publish camera_info, but OpenVIO-SLAM (and any standard
monocular pipeline) needs intrinsics to triangulate depth. Until you run a
real calibration, this node publishes a reasonable default so SLAM can boot.

Calibrate for real with:
    ros2 run camera_calibration cameracalibrator \\
        --size 8x6 --square 0.025 image:=/image camera:=/camera
then copy the resulting YAML into the parameters below (or load it — see
the `calib_file` parameter, which overrides the numeric params if set).
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CameraInfo


class CameraInfoPublisher(Node):

    def __init__(self):
        super().__init__('camera_info_publisher')

        # ---- parameters ----
        self.declare_parameter('topic',        '/camera_info')
        self.declare_parameter('frame_id',     'camera_link')
        self.declare_parameter('width',        640)
        self.declare_parameter('height',       480)
        self.declare_parameter('fx',           500.0)
        self.declare_parameter('fy',           500.0)
        self.declare_parameter('cx',           320.0)
        self.declare_parameter('cy',           240.0)
        self.declare_parameter('publish_hz',   10.0)
        self.declare_parameter('calib_file',   '')   # optional path to YAML

        self._topic   = self.get_parameter('topic').value
        self._frame   = self.get_parameter('frame_id').value
        self._w       = int(self.get_parameter('width').value)
        self._h       = int(self.get_parameter('height').value)
        self._fx      = float(self.get_parameter('fx').value)
        self._fy      = float(self.get_parameter('fy').value)
        self._cx      = float(self.get_parameter('cx').value)
        self._cy      = float(self.get_parameter('cy').value)
        self._hz      = float(self.get_parameter('publish_hz').value)
        self._calib   = self.get_parameter('calib_file').value

        # If a real calibration YAML is provided, prefer it.
        self._maybe_load_calibration()

        # RELIABLE + KEEP_LAST so late subscribers still get the next tick.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(CameraInfo, self._topic, qos)

        period = 1.0 / max(self._hz, 0.1)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"CameraInfo publisher → {self._topic} @ {self._hz:.1f} Hz  "
            f"[{self._w}x{self._h}, fx={self._fx:.1f}, fy={self._fy:.1f}, "
            f"cx={self._cx:.1f}, cy={self._cy:.1f}, frame='{self._frame}']"
        )

    def _maybe_load_calibration(self):
        if not self._calib:
            return
        if not os.path.isfile(self._calib):
            self.get_logger().warn(
                f"calib_file '{self._calib}' not found — using parameters.")
            return
        try:
            import yaml
            with open(self._calib, 'r') as f:
                data = yaml.safe_load(f) or {}
            self._w  = int(data.get('image_width',  self._w))
            self._h  = int(data.get('image_height', self._h))
            K = data.get('camera_matrix', {}).get('data', None)
            if K and len(K) == 9:
                self._fx, self._fy = float(K[0]), float(K[4])
                self._cx, self._cy = float(K[2]), float(K[5])
            self.get_logger().info(f"Loaded calibration from {self._calib}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load calibration: {e}")

    def _tick(self):
        msg = CameraInfo()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.width           = self._w
        msg.height          = self._h
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [
            self._fx, 0.0,       self._cx,
            0.0,      self._fy,  self._cy,
            0.0,      0.0,       1.0,
        ]
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [
            self._fx, 0.0,       self._cx, 0.0,
            0.0,      self._fy,  self._cy, 0.0,
            0.0,      0.0,       1.0,      0.0,
        ]
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
