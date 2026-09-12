#!/usr/bin/env python3
"""
imu_publisher_node.py  —  OpenVIO-SLAM  (Protocol-Corrected v2)
================================================================

PROTOCOL CORRECTION NOTICE
===========================
Parts 1-3 of the specification described a protocol that does NOT match
your actual Arduino firmware. This file is rewritten from your working
visualizer.py (visualizer.py is the ground truth).

ACTUAL ARDUINO PROTOCOL (derived from visualizer.py):
------------------------------------------------------

RECEIVING from Arduino (binary, 43 bytes):
  Byte  0:     0xAA  (header 1)
  Byte  1:     0x55  (header 2)
  Bytes 2–5:   uint32_t  millis()           (little-endian)
  Bytes 6–9:   float     gyro_x  (rad/s)    (little-endian)
  Bytes 10–13: float     gyro_y  (rad/s)
  Bytes 14–17: float     gyro_z  (rad/s)
  Bytes 18–21: float     accel_x (m/s²)
  Bytes 22–25: float     accel_y (m/s²)
  Bytes 26–29: float     accel_z (m/s²)
  Bytes 30–33: float     mag_x   (µT)       ← 9-DOF: magnetometer present
  Bytes 34–37: float     mag_y   (µT)
  Bytes 38–41: float     mag_z   (µT)
  Byte  42:    uint8_t   checksum           (XOR of bytes 2..41 inclusive)

  Total: 43 bytes per packet (NOT 31 bytes as Parts 1-3 said)
  Checksum range: bytes 2 to 41 (NOT 2 to 29)

SENDING to Arduino (ASCII text, NOT binary):
  Motor command:     "M <left> <right>\n"   (e.g. "M 200 -150\n")
  Recalibrate:       "CAL\n"
  Values: integers, range -255 to 255

WHY THE ORIGINAL SPEC WAS WRONG:
  Parts 1-3 specified a 31-byte binary packet (6-DOF only) and binary motor
  commands (0xBB 0x66 ...). Your Arduino uses 43-byte packets (9-DOF with
  magnetometer) and ASCII text motor commands. This file matches reality.

WHAT THIS NODE DOES:
  1. Opens /dev/ttyUSB0 at 115200 baud (configurable via ROS params)
  2. Reads 43-byte binary packets, validates checksum, extracts all 9 sensor axes
  3. Publishes sensor_msgs/Imu on /imu (gyro + accel; orientation_covariance = -1)
  4. Publishes sensor_msgs/MagneticField on /imu/mag (magnetometer)
  5. Forwards /robot_control (Int16MultiArray[left, right]) → "M L R\n" over serial
  6. Forwards /imu_recal (Bool True) → "CAL\n" over serial
  7. Publishes /diagnostics with packet rate and loss statistics

Dependencies:
  pip3 install pyserial --break-system-packages
  sudo apt install ros-jazzy-diagnostic-msgs -y
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg     import Imu, MagneticField
from std_msgs.msg        import Int16MultiArray, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import serial
import struct
import threading
import time
from functools import reduce
from operator  import xor


# ── Protocol constants (from visualizer.py — ground truth) ───────────────────
PKT_HDR1    = 0xAA
PKT_HDR2    = 0x55
PKT_LEN     = 43          # total bytes per IMU packet
PKT_CHK_START = 2         # checksum XOR range start (inclusive)
PKT_CHK_END   = 41        # checksum XOR range end   (inclusive)
#                           checksum byte index = 42

# Layout within 43-byte packet:
#   [0:2]   header
#   [2:6]   uint32 timestamp (ms)
#   [6:10]  float  gx  (rad/s)
#   [10:14] float  gy
#   [14:18] float  gz
#   [18:22] float  ax  (m/s²)
#   [22:26] float  ay
#   [26:30] float  az
#   [30:34] float  mx  (µT)
#   [34:38] float  my
#   [38:42] float  mz
#   [42]    uint8  checksum


class IMUPublisherNode(Node):

    def __init__(self):
        super().__init__('imu_publisher')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('serial_port',    '/dev/ttyUSB0')
        self.declare_parameter('baud_rate',      115200)
        self.declare_parameter('publish_rate',   100.0)
        self.declare_parameter('frame_id',       'imu_link')
        self.declare_parameter('gyro_noise',     0.005)
        self.declare_parameter('accel_noise',    0.085)
        self.declare_parameter('port_retry_sec', 3.0)

        self._port       = self.get_parameter('serial_port').value
        self._baud       = self.get_parameter('baud_rate').value
        self._rate       = self.get_parameter('publish_rate').value
        self._frame_id   = self.get_parameter('frame_id').value
        self._gyro_noise = self.get_parameter('gyro_noise').value
        self._accel_noise= self.get_parameter('accel_noise').value
        self._retry_sec  = self.get_parameter('port_retry_sec').value

        # Covariance diagonals: σ² = noise_density² × sample_rate
        sg2 = (self._gyro_noise  ** 2) * self._rate
        sa2 = (self._accel_noise ** 2) * self._rate
        self._gyro_cov  = [sg2, 0, 0, 0, sg2, 0, 0, 0, sg2]
        self._accel_cov = [sa2, 0, 0, 0, sa2, 0, 0, 0, sa2]

        # ── QoS ──────────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_imu  = self.create_publisher(Imu,            '/imu',         reliable_qos)
        self._pub_mag  = self.create_publisher(MagneticField,  '/imu/mag',     reliable_qos)
        self._pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', reliable_qos)

        # ── Subscriptions ─────────────────────────────────────────────────────
        # /robot_control: [left_speed, right_speed] → ASCII "M L R\n"
        self._sub_ctrl  = self.create_subscription(
            Int16MultiArray, '/robot_control',
            self._motor_callback, reliable_qos)

        # /imu_recal: Bool True → ASCII "CAL\n"
        self._sub_recal = self.create_subscription(
            Bool, '/imu_recal',
            self._recal_callback, reliable_qos)

        # ── Serial state ──────────────────────────────────────────────────────
        self._ser       = None
        self._ser_lock  = threading.Lock()
        self._rx_buf    = bytearray()

        # Statistics
        self._pkt_total  = 0
        self._pkt_bad    = 0   # bad checksum
        self._pkt_lost   = 0   # header found but couldn't parse
        self._last_recv  = time.time()
        self._last_arduino_ts = 0

        # ── Timers ────────────────────────────────────────────────────────────
        self._read_timer = self.create_timer(
            1.0 / self._rate, self._read_timer_cb)
        self._diag_timer = self.create_timer(
            2.0, self._publish_diag)

        # ── Open serial ──────────────────────────────────────────────────────
        self._try_open_port()

        self.get_logger().info(
            f"IMU Publisher ready (9-DOF protocol, 43-byte packets)\n"
            f"  Port:     {self._port} @ {self._baud} baud\n"
            f"  Rate:     {self._rate} Hz\n"
            f"  Frame:    {self._frame_id}\n"
            f"  Topics:   /imu  /imu/mag  /robot_control (sub)  /imu_recal (sub)")

    # ── Serial connection ─────────────────────────────────────────────────────

    def _try_open_port(self):
        try:
            with self._ser_lock:
                self._ser = serial.Serial(
                    self._port, self._baud, timeout=0.01)
            self.get_logger().info(f"✅ Serial opened: {self._port}")
        except serial.SerialException as e:
            self.get_logger().error(
                f"Cannot open {self._port}: {e}\n"
                f"  Check: ls /dev/ttyUSB* /dev/ttyACM*\n"
                f"  Fix:   sudo usermod -aG dialout $USER && newgrp dialout\n"
                f"  Retrying in {self._retry_sec}s…")
            self._ser = None
            self.create_timer(self._retry_sec, self._retry_open_once)

    def _retry_open_once(self):
        if self._ser is None:
            self._try_open_port()

    # ── Read timer ────────────────────────────────────────────────────────────

    def _read_timer_cb(self):
        if self._ser is None:
            return

        # Non-blocking read: grab all available bytes
        try:
            with self._ser_lock:
                n = self._ser.in_waiting
                if n > 0:
                    chunk = self._ser.read(n)
                    self._rx_buf.extend(chunk)
        except serial.SerialException as e:
            self.get_logger().error(
                f"Serial read error: {e}. Closing and retrying…")
            with self._ser_lock:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
            self.create_timer(self._retry_sec, self._retry_open_once)
            return

        # Parse all complete packets in buffer
        self._process_buffer()

        # Warn if silent for >150 ms (1.5 × expected 100ms interval)
        if time.time() - self._last_recv > 0.15:
            self.get_logger().warn(
                "No valid IMU packet for >150ms. "
                "Check Arduino firmware and serial port.",
                throttle_duration_sec=2.0)

    # ── Packet parser ─────────────────────────────────────────────────────────

    def _process_buffer(self):
        """
        Scan rx_buf for valid 43-byte packets.

        Strategy on bad checksum: discard ONE byte and re-scan.
        This means a single corrupted byte costs one resync step,
        not the loss of a full packet window.
        """
        while len(self._rx_buf) >= PKT_LEN:

            # Find header [0xAA, 0x55]
            try:
                i = self._rx_buf.index(PKT_HDR1)
            except ValueError:
                self._rx_buf.clear()
                return

            if i > 0:
                del self._rx_buf[:i]   # discard garbage before header

            if len(self._rx_buf) < PKT_LEN:
                break   # need more data

            if self._rx_buf[1] != PKT_HDR2:
                # Not our header — could be a status byte like 0xAA 0xFF
                del self._rx_buf[:1]
                continue

            # Candidate packet: first PKT_LEN bytes
            pkt = bytes(self._rx_buf[:PKT_LEN])

            # Validate checksum: XOR of bytes 2..41
            expected_chk = reduce(xor,
                                   pkt[PKT_CHK_START:PKT_CHK_END + 1], 0)
            if expected_chk != pkt[42]:
                self._pkt_bad += 1
                del self._rx_buf[:1]   # discard one byte, re-scan
                continue

            # Valid packet — parse
            result = self._parse_packet(pkt)
            if result is not None:
                self._publish_imu_msg(**result)
                self._last_recv = time.time()
                self._pkt_total += 1

            del self._rx_buf[:PKT_LEN]

    def _parse_packet(self, data: bytes) -> dict:
        """
        Unpack 43-byte packet.

        Layout (matches visualizer.py exactly):
          [0:2]   header (already validated)
          [2:6]   uint32  millis timestamp
          [6:18]  3× float gyro xyz (rad/s)
          [18:30] 3× float accel xyz (m/s²)
          [30:42] 3× float mag xyz (µT)
          [42]    uint8 checksum (already validated)
        """
        try:
            ts_ms, gx, gy, gz, ax, ay, az, mx, my, mz = \
                struct.unpack_from('<Ifffffffff', data, 2)
            self._last_arduino_ts = ts_ms
            return {
                'gx': gx, 'gy': gy, 'gz': gz,
                'ax': ax, 'ay': ay, 'az': az,
                'mx': mx, 'my': my, 'mz': mz,
            }
        except struct.error as e:
            self.get_logger().error(
                f"Packet unpack error: {e}", throttle_duration_sec=5.0)
            return None

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish_imu_msg(self, gx, gy, gz, ax, ay, az, mx, my, mz):
        """Publish Imu and MagneticField messages with ROS system time."""
        now = self.get_clock().now().to_msg()

        # ── sensor_msgs/Imu ──────────────────────────────────────────────────
        imu = Imu()
        imu.header.stamp    = now
        imu.header.frame_id = self._frame_id

        imu.angular_velocity.x = float(gx)
        imu.angular_velocity.y = float(gy)
        imu.angular_velocity.z = float(gz)

        imu.linear_acceleration.x = float(ax)
        imu.linear_acceleration.y = float(ay)
        imu.linear_acceleration.z = float(az)

        # Orientation unknown (MSCKF will estimate it)
        imu.orientation_covariance[0] = -1.0

        imu.angular_velocity_covariance    = self._gyro_cov
        imu.linear_acceleration_covariance = self._accel_cov

        self._pub_imu.publish(imu)

        # ── sensor_msgs/MagneticField ────────────────────────────────────────
        # Useful for heading reference / tilt-compensated compass
        # Convert µT → Tesla (ROS uses Tesla)
        mag = MagneticField()
        mag.header.stamp    = now
        mag.header.frame_id = self._frame_id
        mag.magnetic_field.x = float(mx) * 1e-6
        mag.magnetic_field.y = float(my) * 1e-6
        mag.magnetic_field.z = float(mz) * 1e-6
        # Covariance unknown
        mag.magnetic_field_covariance[0] = -1.0

        self._pub_mag.publish(mag)

    # ── Motor command forwarding ──────────────────────────────────────────────

    def _motor_callback(self, msg: Int16MultiArray):
        """
        Forward /robot_control as ASCII text to Arduino.
        Protocol (from visualizer.py): "M <left> <right>\n"
        Values clamped to [-255, 255].
        """
        if len(msg.data) < 2:
            self.get_logger().warn(
                "/robot_control must have 2 elements [left, right]")
            return

        left  = int(max(-255, min(255, msg.data[0])))
        right = int(max(-255, min(255, msg.data[1])))
        cmd   = f"M {left} {right}\n"

        with self._ser_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(cmd.encode())
                except serial.SerialException as e:
                    self.get_logger().error(
                        f"Motor send error: {e}", throttle_duration_sec=2.0)

    def _recal_callback(self, msg: Bool):
        """
        Forward recalibration request as ASCII text to Arduino.
        Protocol (from visualizer.py): "CAL\n"
        """
        if not msg.data:
            return
        self.get_logger().info("Sending calibration command: CAL\\n")
        self.get_logger().info("⚠️  Keep robot completely stationary during calibration!")

        with self._ser_lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"CAL\n")
                except serial.SerialException as e:
                    self.get_logger().error(f"Calibration send error: {e}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _publish_diag(self):
        dt = max(1e-3, time.time() - self._last_recv + 2.0)
        rate = self._pkt_total / dt if dt > 0 else 0.0
        total_seen = self._pkt_total + self._pkt_bad
        loss_pct   = (self._pkt_bad / max(1, total_seen)) * 100.0

        if rate > 90:
            level = DiagnosticStatus.OK
            msg   = f"OK  {rate:.1f} Hz"
        elif rate > 50:
            level = DiagnosticStatus.WARN
            msg   = f"Low rate: {rate:.1f} Hz"
        else:
            level = DiagnosticStatus.ERROR
            msg   = f"No data: {rate:.1f} Hz"

        status = DiagnosticStatus()
        status.name        = 'imu_publisher (9-DOF Arduino)'
        status.hardware_id = self._port
        status.level       = level
        status.message     = msg
        status.values      = [
            KeyValue(key='rate_hz',         value=f'{rate:.1f}'),
            KeyValue(key='packets_total',   value=str(self._pkt_total)),
            KeyValue(key='packets_bad_chk', value=str(self._pkt_bad)),
            KeyValue(key='loss_pct',        value=f'{loss_pct:.1f}'),
            KeyValue(key='arduino_ts_ms',   value=str(self._last_arduino_ts)),
            KeyValue(key='serial_port',     value=self._port),
            KeyValue(key='connected',
                     value='yes' if (self._ser and self._ser.is_open) else 'no'),
            KeyValue(key='protocol',        value='43-byte binary + ASCII motor'),
        ]

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status        = [status]
        self._pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
