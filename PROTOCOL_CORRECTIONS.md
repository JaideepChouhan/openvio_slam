This document is the definitive record of every difference between the
Parts 1-3 specification and the actual protocol used by your Arduino.
Pass this alongside any multi-agent prompt to prevent re-generating
the wrong protocol.

# Protocol Corrections: Parts 1-3 vs Actual Arduino (visualizer.py ground truth)

How we determined the actual protocol

Your `visualizer.py` is the authoritative ground truth. It was tested against
your actual Arduino and confirmed working. Any specification that contradicts
`visualizer.py` is wrong by definition.

---

Correction 1 — Packet size: 31 bytes → 43 bytes

**Parts 1-3 said:**
```
IMU packet: 31 bytes
[AA][55][ts:4][gx:4][gy:4][gz:4][ax:4][ay:4][az:4][chk:1]
```

**Actual (from visualizer.py `parse_packet`):**
```
IMU packet: 43 bytes
[AA][55][ts:4][gx:4][gy:4][gz:4][ax:4][ay:4][az:4][mx:4][my:4][mz:4][chk:1]
```

**Why:** Your Arduino has an MPU-9250 which includes the AK8963 magnetometer.
The firmware sends all 9 axes (gyro XYZ + accel XYZ + mag XYZ) = 9 × 4 = 36
data bytes, plus 2 header + 4 timestamp + 1 checksum = **43 total**.

**Impact:** Every component that parses or builds IMU packets must use 43 bytes.

---

Correction 2 — Checksum byte range: bytes 2-29 → bytes 2-41

**Parts 1-3 said:**
```python
checksum = XOR of bytes 2 through 29  (covers 28 bytes)
```

**Actual (from visualizer.py `parse_packet`):**
```python
for i in range(2, 42):   # range(2,42) = indices 2,3,...,41 = 40 bytes
    chk ^= data[i]
```
Checksum = XOR of **bytes 2 through 41** (covers all payload bytes including mag).
Checksum is stored at **byte 42**.

**Impact:** Any code that computes or validates checksum with the wrong range
will reject every valid packet. This is a silent failure — packets are dropped,
IMU rate appears as 0 Hz, but no error is printed.

---

Correction 3 — Motor command format: binary → ASCII text

**Parts 1-3 said:**
```
Motor command: 7 bytes binary
[0xBB][0x66][left:int16 LE][right:int16 LE][checksum:XOR]
```

**Actual (from visualizer.py `send_motor`):**
```python
cmd = f"M {left} {right}\n"
ser.write(cmd.encode())
# Example: "M 200 -150\n"
```

Plain ASCII text. No binary. No header bytes. No checksum. Just a string.

**Impact:** Sending binary motor commands to your Arduino will be silently
ignored (the Arduino text parser will not recognise `0xBB` as the start of
a command). Motors will not move.

---

Correction 4 — Calibration command: binary → ASCII text

**Parts 1-3 said:**
```
Calibration: 2 bytes binary [0xCC][0x77]
```

**Actual (from visualizer.py main loop):**
```python
ser.write(b"CAL\n")
```

Plain ASCII text `"CAL\n"`. No binary header.

**Impact:** Sending `0xCC 0x77` to your Arduino triggers nothing. Calibration
button in the web UI would have no effect.

---

Correction 5 — Sensor count: 6-DOF → 9-DOF

**Parts 1-3 said:** IMU has gyroscope + accelerometer only (6-DOF).

**Actual:** IMU has gyroscope + accelerometer + magnetometer (9-DOF, MPU-9250
with embedded AK8963). All 9 axes are published and displayed.

**New ROS topics added:**
- `/imu/mag`  `sensor_msgs/MagneticField`  (magnetometer in Tesla)

**New UI elements added:**
- Tilt-compensated magnetic compass widget (replaces odometry-yaw compass)
- Magnetometer XYZ chart (third chart, below gyro and accel)
- Magnetic heading readout (`id="mag-heading"`)
- Magnetic XYZ values readout (`id="mag-readout"`)

**New bag recording topic:**
- `/imu/mag` added to `ros2 bag record` command in web_controller_node.py

---

Correction 6 — Struct unpack format

**Parts 1-3 said:**
```python
struct.unpack_from('<Iffffff', data, 2)   # 1 uint32 + 6 floats = 28 bytes
```

**Actual (matching 43-byte packet):**
```python
struct.unpack_from('<Ifffffffff', data, 2)  # 1 uint32 + 9 floats = 40 bytes
```
Nine `f` characters, not six.

---

Summary Table

| Property | Parts 1-3 (WRONG) | Actual (from visualizer.py) |
|---|---|---|
| Packet size | 31 bytes | **43 bytes** |
| Checksum range | bytes 2–29 | **bytes 2–41** |
| Checksum byte index | 30 | **42** |
| Motor command format | Binary (7 bytes) | **ASCII "M L R\n"** |
| Calibration command | Binary `0xCC 0x77` | **ASCII "CAL\n"** |
| Sensor axes | 6-DOF (gyro+accel) | **9-DOF (gyro+accel+mag)** |
| Struct format | `<Iffffff` | **`<Ifffffffff`** |
| ROS mag topic | Not present | **/imu/mag MagneticField** |

---

Files Affected and What Changed

| File | Change |
|---|---|
| `imu_publisher_node.py` | PKT_LEN=43, checksum range 2-41, 9-float unpack, publishes `/imu/mag`, ASCII motor/recal |
| `robot_controller.ino` | 43-byte send_packet, ASCII command parser, AK8963 magnetometer init+read |
| `web_controller_node.py` | Subscribes to `/imu/mag`, forwards mag data to browser |
| `controller.js` | `handleMag()` with tilt-compensated heading, mag chart, compass uses real mag |
| `index.html` | `compass-canvas`, `mag-chart`, `mag-readout`, `mag-heading` elements |
| `style.css` | `.compass-wrap`, `.minimap-wrap`, `.compass-map-row`, `.badge-purple` |
| `openvio_slam.launch.py` | No change to SLAM pipeline; imu_node params unchanged |

---

Files NOT Changed

These files are correct and do not need modification:

- `openvio_slam/ros/vio_node.py` — subscribes to `/imu`, unchanged
- `openvio_slam/estimator/msckf.py` — unchanged
- `openvio_slam/utils/so3.py` — unchanged
- `openvio_slam/utils/quaternion.py` — unchanged
- All files in `openvio_slam/frontend/` — unchanged
- `config/slam.yaml` — unchanged
- `config/extrinsic.yaml` — unchanged

