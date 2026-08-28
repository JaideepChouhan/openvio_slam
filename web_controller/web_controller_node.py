#!/usr/bin/env python3
import asyncio
import base64
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import psutil
import rclpy
import websockets
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import Bool, Int16MultiArray
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class WebControllerNode(Node):
    """ROS-to-browser controller with HTTP + WebSocket bridge."""

    def __init__(self):
        super().__init__('web_controller')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('http_port', 8080),
                ('ws_port', 8765),
                ('static_dir', ''),
                ('max_cam_fps', 10),
            ],
        )

        self.http_port = int(self.get_parameter('http_port').value)
        self.ws_port = int(self.get_parameter('ws_port').value)
        self.static_dir = self.get_parameter('static_dir').value or str(Path(__file__).resolve().parent / 'static')
        self.max_cam_fps = float(self.get_parameter('max_cam_fps').value)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._bridge = CvBridge()
        self._clients = set()
        self._state_lock = threading.Lock()
        self._last_image_time = 0.0
        self._last_cam_stamp = 0.0
        self._last_imu = {'gx': 0.0, 'gy': 0.0, 'gz': 0.0, 'ax': 0.0, 'ay': 0.0, 'az': 0.0, 'stamp': 0.0}
        self._last_odom = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'vx': 0.0, 'vy': 0.0, 'stamp': 0.0}
        self._map_points_n = 0
        self._diag_payload = []

        self.pub_motor = self.create_publisher(Int16MultiArray, '/robot_control', qos)
        self.pub_imu_recal = self.create_publisher(Bool, '/imu_recal', qos)
        self.pub_slam_reset = self.create_publisher(Bool, '/slam_reset', qos)

        self.create_subscription(Image, '/image', self._image_callback, qos)
        self.create_subscription(Imu, '/imu', self._imu_callback, qos)
        self.create_subscription(Odometry, '/odom', self._odom_callback, qos)
        self.create_subscription(PointCloud2, '/openvio/map_points', self._map_callback, qos)
        self.create_subscription(DiagnosticArray, '/diagnostics', self._diag_callback, qos)

        self._http_server = None
        self._ws_server = None
        self._loop = None
        self._async_thread = threading.Thread(target=self._run_async_servers, daemon=True)
        self._async_thread.start()
        self.create_timer(1.0, self._emit_system_stats)

    def _run_async_servers(self):
        asyncio.run(self._start_servers())

    async def _start_servers(self):
        self._loop = asyncio.get_running_loop()
        self._start_http_server()
        self._ws_server = await websockets.serve(self._ws_handler, '0.0.0.0', self.ws_port)
        self.get_logger().info(f'WebSocket server listening on ws://0.0.0.0:{self.ws_port}')
        try:
            while True:
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            pass

    def _start_http_server(self):
        handler = self._make_http_handler()
        self._http_server = ThreadingHTTPServer(('0.0.0.0', self.http_port), handler)
        self.get_logger().info(f'HTTP server listening on http://0.0.0.0:{self.http_port}')
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

    def _make_http_handler(self):
        static_dir = self.static_dir

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=static_dir, **kwargs)

            def log_message(self, format, *args):
                return

        return Handler

    async def _ws_handler(self, websocket):
        self._clients.add(websocket)
        self.get_logger().info('Browser connected over websocket.')
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle_browser_message(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    def _handle_browser_message(self, msg):
        msg_type = msg.get('type')
        if msg_type == 'motor':
            left = int(msg.get('left', 0))
            right = int(msg.get('right', 0))
            arr = Int16MultiArray(data=[left, right])
            self.pub_motor.publish(arr)
        elif msg_type == 'recal':
            self.pub_imu_recal.publish(Bool(data=True))
        elif msg_type == 'reset_slam':
            self.pub_slam_reset.publish(Bool(data=True))
        elif msg_type == 'stop':
            arr = Int16MultiArray(data=[0, 0])
            self.pub_motor.publish(arr)
        elif msg_type == 'set_speed':
            value = float(msg.get('value', 0.0))
            left = int(255 * value)
            right = int(255 * value)
            arr = Int16MultiArray(data=[left, right])
            self.pub_motor.publish(arr)

    def _image_callback(self, msg: Image):
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self._last_image_time) < (1.0 / max(self.max_cam_fps, 0.1)):
            return
        self._last_image_time = now
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, buffer = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return
            payload = base64.b64encode(buffer.tobytes()).decode('ascii')
            self._broadcast({'type': 'image', 'data': payload, 'stamp': now})
        except Exception as exc:
            self.get_logger().warn(f'Failed to encode image: {exc}')

    def _imu_callback(self, msg: Imu):
        self._last_imu = {
            'gx': float(msg.angular_velocity.x),
            'gy': float(msg.angular_velocity.y),
            'gz': float(msg.angular_velocity.z),
            'ax': float(msg.linear_acceleration.x),
            'ay': float(msg.linear_acceleration.y),
            'az': float(msg.linear_acceleration.z),
            'stamp': float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
        }
        self._broadcast({
            'type': 'imu',
            **self._last_imu,
        })

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = 2.0 * np.arctan2(q.z, q.w)
        self._last_odom = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'z': float(msg.pose.pose.position.z),
            'yaw': float(yaw),
            'vx': float(msg.twist.twist.linear.x),
            'vy': float(msg.twist.twist.linear.y),
            'stamp': float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
        }
        self._broadcast({'type': 'odom', **self._last_odom})

    def _map_callback(self, msg: PointCloud2):
        n_points = int(getattr(msg, 'width', 0) * getattr(msg, 'height', 1))
        self._map_points_n = n_points
        self._broadcast({'type': 'mapstats', 'n_points': n_points, 'stamp': self.get_clock().now().nanoseconds / 1e9})

    def _diag_callback(self, msg: DiagnosticArray):
        self._diag_payload = []
        for status in msg.status:
            values = []
            for val in status.values:
                values.append({'key': val.key, 'value': val.value})
            self._diag_payload.append({'name': status.name, 'level': status.level, 'message': status.message, 'values': values})
        self._broadcast({'type': 'diag', 'status': self._diag_payload})

    def _emit_system_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        mem_mb = psutil.virtual_memory().used // (1024 * 1024)
        sys_msg = {
            'type': 'sys',
            'cpu': float(cpu),
            'mem_mb': int(mem_mb),
            'ros_hz_imu': 99.9,
            'ros_hz_img': 9.8,
            'stamp': self.get_clock().now().nanoseconds / 1e9,
        }
        self._broadcast(sys_msg)

    def _broadcast(self, payload):
        if not self._clients or self._loop is None:
            return
        data = json.dumps(payload)
        dead = set()
        for ws in list(self._clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(data), self._loop)
            except Exception:
                dead.add(ws)
        self._clients.difference_update(dead)


def main(args=None):
    rclpy.init(args=args)
    node = WebControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
