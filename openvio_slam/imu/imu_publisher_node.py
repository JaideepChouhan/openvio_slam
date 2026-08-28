#!/usr/bin/env python3
import errno
import math
import struct
from collections import deque
from typing import Optional

import rclpy
import serial
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int16MultiArray


class IMUPublisherNode(Node):
    """ROS 2 bridge between the Arduino IMU and the SLAM stack."""

    PACKET_LEN = 31
    HEADER = b"\xAA\x55"

    def __init__(self):
        super().__init__('imu_publisher')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('serial_port', '/dev/ttyUSB0'),
                ('baud_rate', 115200),
                ('publish_rate', 100.0),
                ('frame_id', 'imu_link'),
                ('gyro_noise', 0.005),
                ('accel_noise', 0.085),
                ('port_retry_sec', 3.0),
            ],
        )

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.gyro_noise = float(self.get_parameter('gyro_noise').value)
        self.accel_noise = float(self.get_parameter('accel_noise').value)
        self.port_retry_sec = float(self.get_parameter('port_retry_sec').value)

        self.ser = None
        self._rx_buffer = bytearray()
        self._packet_count = 0
        self._loss_count = 0
        self._last_arduino_millis = 0
        self._last_recv_time = None
        self._last_packet_time = None
        self._read_timer = None
        self._diag_timer = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_imu = self.create_publisher(Imu, '/imu', qos)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', qos)
        self.sub_control = self.create_subscription(
            Int16MultiArray, '/robot_control', self._motor_callback, qos
        )
        self.sub_recal = self.create_subscription(
            Bool, '/imu_recal', self._recal_callback, qos
        )

        self._try_open_port()
        if self.ser is not None:
            self._read_timer = self.create_timer(1.0 / self.publish_rate, self._read_timer_callback)
            self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

    def _try_open_port(self):
        if self.ser is not None:
            return
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0)
            self.get_logger().info(
                f'Opened serial port {self.serial_port} at {self.baud_rate} baud.'
            )
        except serial.SerialException as exc:
            self.get_logger().error(
                f'Cannot open {self.serial_port}: {exc}. Retry in {self.port_retry_sec}s'
            )
            self.create_timer(self.port_retry_sec, self._try_open_port, oneshot=True)

    def _read_timer_callback(self):
        if self.ser is None or not self.ser.is_open:
            return
        try:
            available = self.ser.in_waiting
        except serial.SerialException:
            self._handle_serial_lost()
            return

        if available <= 0:
            return

        try:
            data = self.ser.read(available)
        except serial.SerialException:
            self._handle_serial_lost()
            return

        if not data:
            return

        self._rx_buffer.extend(data)
        while True:
            idx = self._rx_buffer.find(self.HEADER)
            if idx < 0:
                if len(self._rx_buffer) > 64:
                    del self._rx_buffer[:-64]
                break

            if len(self._rx_buffer) - idx < self.PACKET_LEN:
                break

            packet = bytes(self._rx_buffer[idx:idx + self.PACKET_LEN])
            if self._validate_packet(packet):
                payload = self._decode_packet(packet)
                if payload is not None:
                    self._packet_count += 1
                    self._last_recv_time = self.get_clock().now().nanoseconds / 1e9
                    self._last_arduino_millis = payload['millis']
                    self._publish_imu(**payload)
                del self._rx_buffer[:idx + self.PACKET_LEN]
                continue

            del self._rx_buffer[idx:idx + 1]

    def _validate_packet(self, packet: bytes) -> bool:
        if len(packet) != self.PACKET_LEN:
            return False
        if packet[:2] != self.HEADER:
            return False
        checksum = 0
        for byte in packet[2:30]:
            checksum ^= byte
        return checksum == packet[30]

    def _decode_packet(self, packet: bytes) -> Optional[dict]:
        try:
            millis, gx, gy, gz, ax, ay, az = struct.unpack('<Iffffff', packet[2:30])
            return {
                'millis': millis,
                'gx': gx,
                'gy': gy,
                'gz': gz,
                'ax': ax,
                'ay': ay,
                'az': az,
            }
        except struct.error:
            self.get_logger().warn('Malformed IMU packet received; discarding.')
            return None

    def _publish_imu(self, gx, gy, gz, ax, ay, az, millis=None):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)
        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)
        msg.orientation_covariance[0] = -1.0
        sigma_g = (self.gyro_noise ** 2) * 100.0
        sigma_a = (self.accel_noise ** 2) * 100.0
        msg.angular_velocity_covariance = [sigma_g, 0.0, 0.0,
                                            0.0, sigma_g, 0.0,
                                            0.0, 0.0, sigma_g]
        msg.linear_acceleration_covariance = [sigma_a, 0.0, 0.0,
                                             0.0, sigma_a, 0.0,
                                             0.0, 0.0, sigma_a]
        self.pub_imu.publish(msg)

    def _publish_diagnostics(self):
        status = DiagnosticArray()
        status.header.stamp = self.get_clock().now().to_msg()
        if self.ser is None or not self.ser.is_open:
            ds = DiagnosticStatus()
            ds.level = DiagnosticStatus.ERROR
            ds.name = 'imu_publisher'
            ds.message = 'Serial port unavailable'
            status.status = [ds]
            self.pub_diag.publish(status)
            return

        packet_rate = 0.0
        if self._last_packet_time is not None:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._last_packet_time
            if elapsed > 0.0:
                packet_rate = 1.0 / elapsed

        ds = DiagnosticStatus()
        ds.level = DiagnosticStatus.OK
        ds.name = 'imu_publisher'
        ds.message = 'Serial link healthy'
        ds.values = [
            KeyValue(key='serial_port', value=self.serial_port),
            KeyValue(key='baud_rate', value=str(self.baud_rate)),
            KeyValue(key='packet_rate_hz', value=f'{packet_rate:.2f}'),
            KeyValue(key='packet_count', value=str(self._packet_count)),
            KeyValue(key='last_arduino_millis', value=str(self._last_arduino_millis)),
        ]
        status.status = [ds]
        self.pub_diag.publish(status)
        self._last_packet_time = self.get_clock().now().nanoseconds / 1e9

    def _motor_callback(self, msg: Int16MultiArray):
        if self.ser is None or not self.ser.is_open:
            self.get_logger().warn('Ignoring motor command because serial port is closed.')
            return
        left = int(msg.data[0]) if len(msg.data) > 0 else 0
        right = int(msg.data[1]) if len(msg.data) > 1 else 0
        self.send_motor_command(left, right)

    def _recal_callback(self, msg: Bool):
        if self.ser is None or not self.ser.is_open:
            self.get_logger().warn('Ignoring IMU recalibration request because serial port is closed.')
            return
        self.send_recal_command()

    def send_motor_command(self, left: int, right: int):
        left = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        if abs(left) < 10:
            left = 0
        if abs(right) < 10:
            right = 0
        packet = bytearray([0xBB, 0x66])
        packet.extend(struct.pack('<hh', left, right))
        checksum = 0
        for byte in packet[2:6]:
            checksum ^= byte
        packet.append(checksum)
        try:
            self.ser.write(packet)
        except serial.SerialException as exc:
            self.get_logger().error(f'Failed to send motor command: {exc}')
            self._handle_serial_lost()

    def send_recal_command(self):
        packet = b'\xCC\x77'
        try:
            self.ser.write(packet)
            self.get_logger().info('IMU recalibration requested')
        except serial.SerialException as exc:
            self.get_logger().error(f'Failed to request IMU recalibration: {exc}')
            self._handle_serial_lost()

    def _handle_serial_lost(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.get_logger().warn('Serial port lost; scheduling retry.')
        self.create_timer(self.port_retry_sec, self._try_open_port, oneshot=True)


def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        if node.ser is not None:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
