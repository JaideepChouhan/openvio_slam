#!/usr/bin/env python3
"""
test_protocol.py — Verify Arduino protocol without ROS.
Run this BEFORE launching any ROS nodes.
Connects directly to Arduino, reads raw bytes, validates packets.

Usage:
    python3 test_protocol.py [/dev/ttyUSB0]

Expected output when working correctly:
    ✅ Port opened: /dev/ttyUSB0
    ✅ Packet #1: ts=1234 gx=0.001 gy=-0.000 gz=0.002 | ax=0.04 ay=0.01 az=9.81 | mx=23.4 my=-5.1 mz=41.2
    ✅ Packet #2: ts=1244 ...
    ...
    ✅ Rate: 99.8 Hz over 100 packets
"""

import serial
import struct
import sys
import time
from functools import reduce
from operator import xor

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = 115200
PKT_LEN = 43

# Colours
G = '\033[0;32m'; Y = '\033[1;33m'; R = '\033[0;31m'; N = '\033[0m'

print(f"Opening {PORT} @ {BAUD} baud…")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0)
except serial.SerialException as e:
    print(f"{R}❌ Cannot open {PORT}: {e}{N}")
    print("   Fix: sudo usermod -aG dialout $USER && newgrp dialout")
    sys.exit(1)

time.sleep(2.0)   # let Arduino reset

print(f"{G}✅ Port opened: {PORT}{N}")
print(f"   Waiting for packets (PKT_LEN={PKT_LEN}, checksum bytes 2-41)…\n")

buf = bytearray()
pkt_count = 0
bad_count  = 0
t_start    = None

def try_parse(data):
    """Attempt to parse 43-byte packet. Returns dict or None."""
    if len(data) < PKT_LEN:
        return None
    if data[0] != 0xAA or data[1] != 0x55:
        return None
    # Checksum: XOR of bytes 2..41
    expected = reduce(xor, data[2:42], 0)
    if expected != data[42]:
        return None
    ts, gx, gy, gz, ax, ay, az, mx, my, mz = \
        struct.unpack_from('<Ifffffffff', data, 2)
    return {'ts': ts, 'gx': gx, 'gy': gy, 'gz': gz,
            'ax': ax, 'ay': ay, 'az': az, 'mx': mx, 'my': my, 'mz': mz}

try:
    while pkt_count < 110:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))

        while len(buf) >= PKT_LEN:
            try:
                i = buf.index(0xAA)
            except ValueError:
                buf.clear(); break
            if i > 0:
                del buf[:i]
            if len(buf) < PKT_LEN:
                break
            if buf[1] != 0x55:
                del buf[:1]; continue

            pkt = bytes(buf[:PKT_LEN])
            result = try_parse(pkt)
            if result is None:
                bad_count += 1
                del buf[:1]
                continue

            del buf[:PKT_LEN]
            pkt_count += 1

            if t_start is None:
                t_start = time.time()

            # Print first 10 and every 10th after
            if pkt_count <= 10 or pkt_count % 10 == 0:
                r = result
                # Sanity check: |accel| should be near 9.81 when stationary
                a_mag = (r['ax']**2 + r['ay']**2 + r['az']**2)**0.5
                g_mag = (r['gx']**2 + r['gy']**2 + r['gz']**2)**0.5
                accel_ok = abs(a_mag - 9.81) < 1.0
                color = G if accel_ok else Y
                print(
                    f"{color}✅ Pkt #{pkt_count:3d}{N}  "
                    f"ts={r['ts']:6d}ms  "
                    f"G({r['gx']:+.3f},{r['gy']:+.3f},{r['gz']:+.3f})rad/s  "
                    f"A({r['ax']:+.2f},{r['ay']:+.2f},{r['az']:+.2f})m/s²  "
                    f"M({r['mx']:+.1f},{r['my']:+.1f},{r['mz']:+.1f})µT"
                    + (f"  {Y}⚠ |a|={a_mag:.2f} (expected ~9.81){N}" if not accel_ok else ""))

        time.sleep(0.001)

except KeyboardInterrupt:
    pass
finally:
    ser.close()

elapsed = time.time() - t_start if t_start else 1
rate = pkt_count / elapsed

print(f"\n{'─'*70}")
print(f"Packets received: {pkt_count}")
print(f"Bad checksum:     {bad_count}")
print(f"Duration:         {elapsed:.1f}s")

if rate >= 90:
    print(f"{G}✅ Rate: {rate:.1f} Hz  (target: 100 Hz){N}")
else:
    print(f"{Y}⚠️  Rate: {rate:.1f} Hz  (target: 100 Hz — check Arduino firmware){N}")

if bad_count == 0:
    print(f"{G}✅ Checksum: 0 errors{N}")
else:
    print(f"{R}❌ Checksum: {bad_count} errors — check baud rate or serial noise{N}")

if pkt_count > 0:
    print(f"\n{G}✅ Arduino protocol is CORRECT (43-byte 9-DOF packets){N}")
    print(f"   Ready to launch: ros2 launch openvio_slam openvio_slam.launch.py")
else:
    print(f"\n{R}❌ No packets received. Upload robot_controller.ino to Arduino first.{N}")
