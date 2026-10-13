#!/usr/bin/env python3
"""
web_controller_node.py  —  OpenVIO-SLAM Web Controller  (Protocol-Corrected v3)
=================================================================================

CHANGES (v3):
  • FIX: AttributeError on self._use_https — initialised before the asyncio
         thread is started, and cert availability is probed synchronously in
         __init__ so the "ready" log line is always accurate.
  • FIX: __init__ now waits on an Event that the async thread sets only after
         the HTTP + WebSocket servers are actually up (no more guessing).
  • Cleaner separation between "is HTTPS possible?" (sync check) and
         "set up HTTPS" (async thread).

CHANGES (v2):
  • Subscribes to /imu/mag and forwards magnetometer data.
  • Motor commands via /robot_control.
  • Robust bytes→str conversion in _cb_diag to prevent JSON serialisation errors.
  • Try/except around json.dumps in diagnostic callback.
  • HTTPS support (self‑signed cert) – enables camera access on mobile browsers.
  • Serves static/ directory: dashboard at /, phone app at /phone/.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg     import Image, Imu, PointCloud2, MagneticField
from nav_msgs.msg        import Odometry
from std_msgs.msg        import Int16MultiArray, Bool
from diagnostic_msgs.msg import DiagnosticArray

import asyncio
import threading
import json
import base64
import time
import subprocess
import os
import math
import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools   import partial

try:
    import websockets
    import psutil
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

import numpy as np


# ── HTTPS certificate paths (module-level constants) ─────────────────────────
CERT_PATH = os.path.expanduser('~/certs/cert.pem')
KEY_PATH  = os.path.expanduser('~/certs/key.pem')


def _to_str(obj):
    """Safely convert any bytes-like object to a string for JSON serialisation."""
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, bytearray):
        return bytes(obj).decode('utf-8', errors='replace')
    if isinstance(obj, str):
        return obj
    return str(obj)


class WebControllerNode(Node):

    def __init__(self):
        super().__init__('web_controller')

        if not HAS_DEPS:
            self.get_logger().error(
                "Missing deps: pip3 install websockets psutil --break-system-packages")
            raise RuntimeError("Missing dependencies")

        self.declare_parameter('http_port',    8080)
        self.declare_parameter('ws_port',      8765)
        self.declare_parameter('max_cam_fps',  10.0)
        self.declare_parameter('record_path',  '~/.ros/bags/')

        self._http_port   = self.get_parameter('http_port').value
        self._ws_port     = self.get_parameter('ws_port').value
        self._max_cam_fps = self.get_parameter('max_cam_fps').value
        self._record_path = os.path.expanduser(
            self.get_parameter('record_path').value)

        # ── FIX: decide HTTPS availability synchronously, before the thread ──
        # This makes the "ready" log line truthful and avoids the race.
        self._use_https = (
            os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH)
        )

        # ── Shared state ──────────────────────────────────────────────────────
        self._ws_clients  = set()
        self._ws_lock     = threading.Lock()
        self._loop        = None          # asyncio event loop (set in thread 2)
        self._servers_up  = threading.Event()

        self._last_cam_time = 0.0
        self._cam_interval  = 1.0 / self._max_cam_fps
        self._bag_proc      = None
        self._odom_path     = []          # [(x, y), ...] for path export

        # ── QoS ──────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Image,          '/image',              self._cb_image,  sensor_qos)
        self.create_subscription(
            Imu,            '/imu',                self._cb_imu,    sensor_qos)
        self.create_subscription(
            MagneticField,  '/imu/mag',            self._cb_mag,    sensor_qos)
        self.create_subscription(
            Odometry,       '/odom',               self._cb_odom,   sensor_qos)
        self.create_subscription(
            PointCloud2,    '/openvio/map_points', self._cb_pts,    sensor_qos)
        self.create_subscription(
            DiagnosticArray, '/diagnostics',       self._cb_diag,   sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_ctrl  = self.create_publisher(
            Int16MultiArray, '/robot_control', reliable_qos)
        self._pub_recal = self.create_publisher(
            Bool, '/imu_recal', reliable_qos)
        self._pub_reset = self.create_publisher(
            Bool, '/slam_reset', reliable_qos)

        # ── System stats timer ────────────────────────────────────────────────
        self.create_timer(1.0, self._publish_sys_stats)

        # ── Start asyncio thread ──────────────────────────────────────────────
        self._async_thread = threading.Thread(
            target=self._run_async_loop, daemon=True)
        self._async_thread.start()

        # Wait up to 5 s for the HTTP + WS servers to actually bind.
        if not self._servers_up.wait(timeout=5.0):
            self.get_logger().warn(
                "HTTP/WS servers did not report ready within 5 s — "
                "continuing anyway.")

        # ── Ready log (safe: _use_https always exists) ────────────────────────
        scheme = 'https' if self._use_https else 'http'
        if self._use_https:
            https_note = "(self-signed cert)"
        else:
            https_note = "(cert not found — camera access on mobile may fail)"

        self.get_logger().info(
            f"Web Controller ready\n"
            f"  UI:  {scheme}://0.0.0.0:{self._http_port} {https_note}\n"
            f"  WS:  ws://0.0.0.0:{self._ws_port}\n"
            f"  Topics: /imu /imu/mag /image /odom /openvio/map_points")

    # ── Asyncio thread ────────────────────────────────────────────────────────

    def _run_async_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as e:
            self.get_logger().error(f"Asyncio loop crashed: {e}")
        finally:
            # Always release __init__ even on failure, so it doesn't hang.
            self._servers_up.set()

    async def _async_main(self):
        static_dir = self._find_static_dir()
        handler    = partial(StaticFileHandler, directory=static_dir)

        try:
            http_srv = HTTPServer(('0.0.0.0', self._http_port), handler)
        except OSError as e:
            self.get_logger().error(
                f"Cannot bind HTTP port {self._http_port}: {e}")
            return

        # --- HTTPS support (self‑signed certificate) ---
        # _use_https was already decided in __init__; here we just try to apply it.
        if self._use_https:
            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(CERT_PATH, KEY_PATH)
                http_srv.socket = context.wrap_socket(
                    http_srv.socket, server_side=True)
                self.get_logger().info(
                    f"HTTPS enabled on port {self._http_port}")
            except Exception as e:
                # Cert existed at __init__ but failed to load — fall back.
                self._use_https = False
                self.get_logger().warn(
                    f"Failed to load certificate: {e} – falling back to HTTP")
        else:
            self.get_logger().warn(
                "Certificates not found, falling back to HTTP")

        threading.Thread(target=http_srv.serve_forever, daemon=True).start()

        try:
            async with websockets.serve(
                self._ws_handler, '0.0.0.0', self._ws_port):
                # Servers are up — release __init__.
                self._servers_up.set()
                await asyncio.Future()   # run forever
        except OSError as e:
            self.get_logger().error(
                f"Cannot bind WS port {self._ws_port}: {e}")
            self._servers_up.set()

    def _find_static_dir(self) -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), 'static'),
            os.path.expanduser(
                '~/slam_ws/src/openvio_slam/web_controller/static'),
            os.path.expanduser(
                '~/slam_ws/install/openvio_slam/share/'
                'openvio_slam/web_controller/static'),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        self.get_logger().warn("static/ not found, serving from cwd")
        return '.'

    # ── WebSocket ─────────────────────────────────────────────────────────────

    async def _ws_handler(self, ws):
        with self._ws_lock:
            self._ws_clients.add(ws)
        self.get_logger().info(f"Browser connected: {ws.remote_address}")
        try:
            async for raw in ws:
                await self._handle_browser_msg(raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._ws_lock:
                self._ws_clients.discard(ws)
            self.get_logger().info(f"Browser disconnected: {ws.remote_address}")

    async def _handle_browser_msg(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        t = msg.get('type', '')

        if t == 'motor':
            L = int(max(-255, min(255, msg.get('left',  0))))
            R = int(max(-255, min(255, msg.get('right', 0))))
            m = Int16MultiArray()
            m.data = [L, R]
            self._pub_ctrl.publish(m)

        elif t == 'stop':
            m = Int16MultiArray()
            m.data = [0, 0]
            self._pub_ctrl.publish(m)

        elif t == 'recal':
            self._pub_recal.publish(Bool(data=True))

        elif t == 'reset_slam':
            self._pub_reset.publish(Bool(data=True))

        elif t == 'start_record':
            self._start_bag_record()

        elif t == 'stop_record':
            self._stop_bag_record()

        elif t == 'export_path':
            await self._send_path_export()

        elif t == 'shutdown':
            self.get_logger().warn("Shutdown requested from browser")
            subprocess.Popen(['sudo', 'shutdown', '-h', 'now'])

    async def _send_path_export(self):
        csv_lines = ['index,x,y']
        for i, (x, y) in enumerate(self._odom_path):
            csv_lines.append(f"{i},{x:.4f},{y:.4f}")
        await self._broadcast(json.dumps({
            'type': 'path_export',
            'csv':  '\n'.join(csv_lines),
        }))

    # ── Broadcast helper ──────────────────────────────────────────────────────

    def _push(self, payload: str):
        """Thread-safe: queue a broadcast from the ROS thread."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                self._loop.create_task,
                self._broadcast(payload))

    async def _broadcast(self, payload: str):
        with self._ws_lock:
            clients = set(self._ws_clients)
        if not clients:
            return
        await asyncio.gather(
            *[c.send(payload) for c in clients],
            return_exceptions=True)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        now = time.time()
        if now - self._last_cam_time < self._cam_interval:
            return
        self._last_cam_time = now

        try:
            h, w = msg.height, msg.width
            raw  = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == 'bgr8':
                import cv2
                bgr = raw.reshape(h, w, 3)
                _, buf = cv2.imencode(
                    '.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64 = base64.b64encode(buf.tobytes()).decode()
            else:
                b64 = base64.b64encode(bytes(msg.data)).decode()

            self._push(json.dumps({
                'type': 'image',
                'data': b64,
                'stamp': now,
                'w': w,
                'h': h,
            }))
        except Exception as e:
            self.get_logger().error(
                f"Image encode: {e}", throttle_duration_sec=5.0)

    def _cb_imu(self, msg: Imu):
        self._push(json.dumps({
            'type': 'imu',
            'gx':   round(msg.angular_velocity.x,    4),
            'gy':   round(msg.angular_velocity.y,    4),
            'gz':   round(msg.angular_velocity.z,    4),
            'ax':   round(msg.linear_acceleration.x, 4),
            'ay':   round(msg.linear_acceleration.y, 4),
            'az':   round(msg.linear_acceleration.z, 4),
            'stamp': time.time(),
        }))

    def _cb_mag(self, msg: MagneticField):
        mx = msg.magnetic_field.x * 1e6
        my = msg.magnetic_field.y * 1e6
        mz = msg.magnetic_field.z * 1e6
        heading_deg = math.degrees(math.atan2(my, mx)) % 360.0
        self._push(json.dumps({
            'type':    'mag',
            'mx':      round(mx, 3),
            'my':      round(my, 3),
            'mz':      round(mz, 3),
            'heading': round(heading_deg, 2),
            'stamp':   time.time(),
        }))

    def _cb_odom(self, msg: Odometry):
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        v   = msg.twist.twist.linear
        yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)

        self._odom_path.append((p.x, p.y))
        if len(self._odom_path) > 10000:
            self._odom_path = self._odom_path[-10000:]

        self._push(json.dumps({
            'type': 'odom',
            'x':    round(p.x, 4),
            'y':    round(p.y, 4),
            'z':    round(p.z, 4),
            'yaw':  round(math.degrees(yaw), 2),
            'vx':   round(v.x, 4),
            'vy':   round(v.y, 4),
        }))

    def _cb_pts(self, msg: PointCloud2):
        self._push(json.dumps({
            'type':     'mapstats',
            'n_points': msg.width * msg.height,
            'stamp':    time.time(),
        }))

    def _cb_diag(self, msg: DiagnosticArray):
        """
        Robust diagnostic forwarding with safe bytes→str conversion.
        If serialisation fails, log error and send an empty diagnostic list.
        """
        diag_list = []
        for s in msg.status:
            name    = _to_str(s.name)
            message = _to_str(s.message)

            try:
                level_str = _to_str(s.level)
                level = int(level_str) if level_str.strip() != '' else 0
            except (ValueError, TypeError):
                level = 0

            values = {}
            for kv in s.values:
                key = _to_str(kv.key)
                val = _to_str(kv.value)
                values[key] = val

            diag_list.append({
                'name':    name,
                'level':   level,
                'message': message,
                'values':  values,
            })

        try:
            payload = json.dumps({'type': 'diag', 'status': diag_list})
            self._push(payload)
        except Exception as e:
            self.get_logger().error(
                f"Failed to serialise diagnostics: {e} – sending empty diag",
                throttle_duration_sec=5.0)
            self._push(json.dumps({'type': 'diag', 'status': []}))

    def _publish_sys_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        self._push(json.dumps({
            'type':      'sys',
            'cpu':       round(cpu, 1),
            'mem_mb':    mem.used // (1024 * 1024),
            'mem_pct':   round(mem.percent, 1),
            'stamp':     time.time(),
            'recording': (self._bag_proc is not None
                          and self._bag_proc.poll() is None),
        }))

    # ── Bag recording ─────────────────────────────────────────────────────────

    def _start_bag_record(self):
        if self._bag_proc and self._bag_proc.poll() is None:
            return
        os.makedirs(self._record_path, exist_ok=True)
        ts    = time.strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(self._record_path, f'slam_{ts}')
        self._bag_proc = subprocess.Popen([
            'ros2', 'bag', 'record', '-o', fname,
            '/image', '/imu', '/imu/mag', '/odom',
            '/openvio/map_points',
        ])
        self.get_logger().info(f"Bag recording: {fname}")

    def _stop_bag_record(self):
        if self._bag_proc and self._bag_proc.poll() is None:
            self._bag_proc.terminate()
            self._bag_proc = None
            self.get_logger().info("Bag recording stopped.")

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(qx, qy, qz, qw) -> float:
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))


class StaticFileHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory='.', **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass   # suppress HTTP logs


def main(args=None):
    rclpy.init(args=args)
    node = WebControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
