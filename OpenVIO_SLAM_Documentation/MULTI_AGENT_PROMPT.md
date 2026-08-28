# OpenVIO-SLAM: Arduino IMU Integration + Industrial Web Controller
## Complete Implementation Prompt for Multi-Agent LLM Systems

---

> **HOW TO USE THIS PROMPT**
> This document is a self-contained technical specification. Pass it in full
> to any multi-agent LLM (AutoGen, CrewAI, LangGraph, Claude, GPT-4, Gemini).
> Each agent should be assigned one numbered section. The agents must read
> every section before generating code because cross-section dependencies
> are real and violations cause runtime failures.
>
> **Do not summarize or truncate this prompt. Pass it verbatim.**

---

## 0. Existing System Context

The target system is a tracked ground robot running **OpenVIO-SLAM** on a
**Raspberry Pi 5** with **Ubuntu 24.04** and **ROS 2 Jazzy**.

### Current hardware
| Component | Details |
|---|---|
| Compute | Raspberry Pi 5, 8 GB RAM |
| Camera | Android phone mounted in **landscape orientation** on robot |
| Current IMU | Phone IMU via WebSocket (being replaced) |
| Motor controller | Cytron MDD20A |
| Motors | Two 45 RPM high-torque DC motors (left track, right track) |
| New IMU | MPU-9250 or MPU-6500 on **Arduino UNO** via I2C |
| Communication | Arduino ↔ Pi via USB serial at 115200 baud |

### Current ROS 2 node topology
```
Phone → rosbridge → slam_bridge_node → /image, /camera_info, /imu (OLD)
                                                ↓
                                       openvio_slam_node
                                                ↓
                                     /odom, /openvio/map_points, /tf
```

### Target topology after this implementation
```
Phone → rosbridge → slam_bridge_node → /image, /camera_info
                                                ↓
Arduino UNO → USB Serial → imu_publisher_node → /imu
                    ↑                             ↓
          motor_cmd (serial)           openvio_slam_node
                    ↑                             ↓
         web_controller_node ← /odom, /openvio/map_points
                    ↑
         Browser (laptop) → keyboard/joystick
```

### Key constraint
`openvio_slam_node` subscribes to `/imu` (topic name is fixed — do not
rename it). `slam_bridge_node` must stop publishing `/imu` (or be
conditionally disabled). Everything else in `openvio_slam_node` is
unchanged — no modifications to MSCKF, triangulation, or any estimator code.

### File locations (all paths relative to `~/slam_ws/src/`)
```
openvio_slam/
├── config/
│   ├── slam.yaml
│   └── extrinsic.yaml          ← update p_IC and R_IC for Arduino IMU position
├── launch/
│   └── openvio_slam.launch.py  ← add new nodes here
├── openvio_slam/
│   └── ros/
│       └── vio_node.py         ← DO NOT MODIFY
├── imu_publisher/              ← NEW package (or new module in openvio_slam)
│   └── imu_publisher_node.py
└── web_controller/             ← NEW package
    ├── web_controller_node.py
    └── static/
        ├── index.html
        ├── controller.js
        └── style.css
```

---

## 1. Arduino Firmware — `robot_controller.ino`

### 1.1 Complete responsibilities
1. Read MPU-9250/MPU-6500 via I2C at **200 Hz** using a hardware timer interrupt
2. Perform **gyro bias calibration** at startup (1000 samples, robot must be stationary)
3. Convert raw values to SI units (rad/s, m/s²)
4. Apply **axis remapping** so output matches ROS convention: X=forward, Y=left, Z=up
5. Send calibrated IMU packets over USB serial at **100 Hz**
6. Receive motor command packets from Pi and drive Cytron MDD20A
7. Implement **watchdog**: stop motors if no command received for 500 ms
8. Implement **encoder ticks** if encoders are available (optional, use #ifdef guard)
9. Respond to **recalibration command** from Pi

### 1.2 Pin assignments — Cytron MDD20A
```
LEFT motor:
  PWM pin  → 5   (Timer0, avoid conflicts with I2C)
  DIR pin  → 4

RIGHT motor:
  PWM pin  → 6
  DIR pin  → 7

MPU I2C:
  SDA → A4
  SCL → A5
  AD0 → GND (address 0x68)
  INT → 2   (optional, for data-ready interrupt)
```

### 1.3 IMU data packet — binary, 31 bytes
```
Byte  0:    0xAA  (header byte 1)
Byte  1:    0x55  (header byte 2)
Bytes 2–5:  uint32_t  millis()  (little-endian)
Bytes 6–9:  float     gyro_x   (rad/s)
Bytes 10–13: float    gyro_y   (rad/s)
Bytes 14–17: float    gyro_z   (rad/s)
Bytes 18–21: float    accel_x  (m/s²)
Bytes 22–25: float    accel_y  (m/s²)
Bytes 26–29: float    accel_z  (m/s²)
Byte  30:   uint8_t  checksum  (XOR of bytes 2–29)
```
Checksum is XOR of every byte from index 2 to 29 inclusive.
Send this packet at exactly **100 Hz** (every 10 ms) using `millis()` scheduling.

### 1.4 Motor command packet — received from Pi, 7 bytes
```
Byte 0:    0xBB  (header byte 1)
Byte 1:    0x66  (header byte 2)
Bytes 2–3: int16_t  left_speed   (-255 to 255, little-endian)
Bytes 4–5: int16_t  right_speed  (-255 to 255, little-endian)
Byte 6:    uint8_t  checksum     (XOR of bytes 2–5)
```
Positive = forward, negative = reverse. Values outside ±255 are clamped.
Deadband: ±10 (treat |speed| < 10 as 0).

### 1.5 Recalibration command packet — received from Pi, 2 bytes
```
Byte 0: 0xCC
Byte 1: 0x77
```
On receipt: re-run gyro bias calibration routine (1000 samples, ~5 seconds).
During calibration: stop motors and send a status byte `0xAA 0x55 0xFF 0xFF`
(4-byte status packet, distinct from IMU packet).

### 1.6 Axis remapping requirement
The MPU-9250/6500 mounted flat on the robot with the chip label facing up and
the **X-axis of the chip pointing toward the front of the robot**:
- No remapping needed: chip X = robot forward, chip Y = robot left, chip Z = up
- If mounted differently, apply the rotation in firmware before sending.
- **Document the assumed mounting orientation in a comment at the top of the file.**
- Gyro sign convention: positive Z = counterclockwise rotation when viewed from above.

### 1.7 MPU-9250/6500 register configuration
Use direct I2C register writes (no library dependency except Wire.h):
```
PWR_MGMT_1    (0x6B) = 0x01   // Clock source: PLL with gyro
GYRO_CONFIG   (0x1B) = 0x18   // ±2000 dps  → scale = 16.4 LSB/(°/s)
ACCEL_CONFIG  (0x1C) = 0x10   // ±8g        → scale = 4096 LSB/g
ACCEL_CONFIG2 (0x1D) = 0x04   // DLPF 20 Hz for accel
CONFIG        (0x1A) = 0x04   // DLPF 20 Hz for gyro
SMPLRT_DIV    (0x19) = 0x04   // Sample rate = 1000/(1+4) = 200 Hz
INT_PIN_CFG   (0x37) = 0x02   // INT pin active high, push-pull
INT_ENABLE    (0x38) = 0x01   // Data ready interrupt
```
Conversion constants:
```cpp
#define GYRO_SCALE    (1.0f / 16.4f * M_PI / 180.0f)   // LSB → rad/s
#define ACCEL_SCALE   (1.0f / 4096.0f * 9.80665f)       // LSB → m/s²
```

### 1.8 Gyro bias calibration algorithm
```cpp
void calibrate_gyro() {
    const int N = 1000;
    long gx_sum = 0, gy_sum = 0, gz_sum = 0;
    for (int i = 0; i < N; i++) {
        int16_t gx, gy, gz;
        read_raw_gyro(&gx, &gy, &gz);
        gx_sum += gx; gy_sum += gy; gz_sum += gz;
        delay(2);   // 500 Hz during calibration
    }
    bias_gx = (float)gx_sum / N;
    bias_gy = (float)gy_sum / N;
    bias_gz = (float)gz_sum / N;
}
```
Store bias in EEPROM (addresses 0–11, three floats) so calibration persists
across power cycles. Load from EEPROM on startup; recalibrate only if
commanded or if EEPROM checksum invalid.

### 1.9 Required code structure
```cpp
// robot_controller.ino
// --- Includes ---
// --- Constants and pin definitions ---
// --- Global state ---
// --- setup() ---
//   init Serial, Wire, MPU, motors
//   load bias from EEPROM or calibrate
// --- loop() ---
//   check millis() for 10ms IMU send interval
//   check millis() for motor watchdog (500ms)
//   parse incoming serial bytes (state machine, NOT blocking)
// --- Functions ---
//   void read_imu(float* gx, float* gy, float* gz, float* ax, float* ay, float* az)
//   void send_imu_packet(float gx, float gy, float gz, float ax, float ay, float az)
//   void set_motor(int left, int right)
//   void parse_serial_byte(uint8_t b)
//   void calibrate_gyro()
//   void save_bias_eeprom()
//   void load_bias_eeprom()
//   uint8_t compute_checksum(uint8_t* data, int len)
```

### 1.10 Serial parser — state machine (non-blocking)
```
States: IDLE → HDR1_RECV → HDR2_RECV → DATA (6 bytes) → CHECKSUM_CHECK
For motor command: separate state machine triggered by 0xBB header
For recal command: separate state machine triggered by 0xCC header
```
**Never use `Serial.readBytes()` with a blocking timeout in the main loop.**
Use a single `Serial.available()` check and a state machine.

---

## 2. ROS 2 IMU Publisher Node — `imu_publisher_node.py`

### 2.1 File location
`~/slam_ws/src/openvio_slam/openvio_slam/imu/imu_publisher_node.py`
Also install as executable `imu_publisher_node` in `CMakeLists.txt`.

### 2.2 Complete class structure
```python
class IMUPublisherNode(Node):
    def __init__(self):
        # ROS parameters (see 2.5)
        # Serial port open with timeout=0 (non-blocking)
        # Publishers: /imu (sensor_msgs/Imu), /diagnostics
        # Timer: _read_timer at publish_rate Hz
        # Internal state: ring buffer for jitter smoothing,
        #                 packet counter, loss counter,
        #                 last_recv_time

    def _read_timer_callback(self):
        # Read all available bytes from serial (non-blocking)
        # Feed to packet parser
        # If complete packet: validate checksum, convert, publish
        # If no packet for >100ms: log warning (throttled 1Hz)

    def _parse_packet(self, data: bytes) -> Optional[dict]:
        # Validate header (0xAA, 0x55)
        # Validate length (31 bytes)
        # Validate checksum (XOR bytes 2–29)
        # Unpack: millis (uint32), gx,gy,gz,ax,ay,az (float32)
        # Return dict or None

    def _publish_imu(self, gx, gy, gz, ax, ay, az):
        # Build sensor_msgs/Imu
        # header.frame_id = self.frame_id
        # header.stamp = self.get_clock().now().to_msg()
        # angular_velocity.x/y/z
        # linear_acceleration.x/y/z
        # orientation_covariance[0] = -1.0 (unknown)
        # angular_velocity_covariance = diagonal [σ_g², σ_g², σ_g²]
        # linear_acceleration_covariance = diagonal [σ_a², σ_a², σ_a²]

    def _publish_diagnostics(self):
        # DiagnosticArray with:
        # - serial_port status
        # - packet rate (Hz)
        # - packet loss %
        # - last IMU timestamp (Arduino millis)

    def send_motor_command(self, left: int, right: int):
        # Build motor command packet (0xBB 0x66 ...)
        # Write to serial

    def send_recal_command(self):
        # Write 0xCC 0x77 to serial
        # Log: "IMU recalibration requested"
```

### 2.3 Covariance values (from slam.yaml)
```python
GYRO_NOISE_DENSITY  = 0.005    # rad/s/√Hz  (from slam.yaml imu.gyro_noise)
ACCEL_NOISE_DENSITY = 0.085    # m/s²/√Hz   (from slam.yaml imu.accel_noise)
# Convert to variance for the message (assumes 100 Hz sampling):
# σ² = noise_density² × sample_rate
sigma_g_sq  = GYRO_NOISE_DENSITY**2  * 100   # per element of 3×3 diagonal
sigma_a_sq  = ACCEL_NOISE_DENSITY**2 * 100
```

### 2.4 Packet byte buffer and synchronization
Use a `bytearray` ring buffer. Scan for header `[0xAA, 0x55]`. Once found,
wait for 29 more bytes. Parse. If checksum fails: discard 1 byte and re-scan.
Never discard a full 31-byte window on a single failure.

### 2.5 ROS parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `serial_port` | string | `/dev/ttyUSB0` | Serial device |
| `baud_rate` | int | `115200` | Baud rate |
| `publish_rate` | double | `100.0` | Hz |
| `frame_id` | string | `imu_link` | IMU frame |
| `gyro_noise` | double | `0.005` | rad/s/√Hz |
| `accel_noise` | double | `0.085` | m/s²/√Hz |
| `port_retry_sec` | double | `3.0` | Retry interval if port fails |

### 2.6 Serial port error handling
```python
try:
    self.ser = serial.Serial(port, baud, timeout=0)
except serial.SerialException as e:
    self.get_logger().error(f"Cannot open {port}: {e}. Retry in {retry}s")
    self.create_timer(retry, self._try_open_port)
    return
```
If serial is lost mid-run: close, log error, schedule retry. Do not crash.

### 2.7 Motor command forwarding
The `web_controller_node` publishes `/robot_control` as `std_msgs/Int16MultiArray`
with `data = [left_speed, right_speed]`. The `imu_publisher_node` subscribes
to `/robot_control` and forwards commands over serial to the Arduino.
This way `imu_publisher_node` is the single owner of the serial port.

```python
self.sub_control = self.create_subscription(
    Int16MultiArray, '/robot_control',
    self._motor_callback, qos_profile_sensor_data)
```

### 2.8 Required imports
```python
import serial
import struct
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16MultiArray
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
```

---

## 3. Web Controller Node — `web_controller_node.py`

### 3.1 File location
`~/slam_ws/src/openvio_slam/web_controller/web_controller_node.py`

### 3.2 Architecture
The node runs two concurrent components:
1. **ROS 2 subscriber/publisher** — standard rclpy node
2. **HTTP server** (Python `http.server`, port 8080) — serves static files
3. **WebSocket server** (Python `websockets`, port 8765) — bridges ROS → browser

All three run in the same process using `asyncio` + `rclpy`'s external executor.

```python
# Threading model:
# Thread 1: rclpy.spin(node)       — all ROS callbacks
# Thread 2: asyncio event loop     — HTTP server + WebSocket server
# Shared: thread-safe queue for ROS→browser messages
# Shared: threading.Lock for motor command state
```

### 3.3 ROS subscriptions (all data forwarded to browser via WebSocket)
| Topic | Type | Forward as |
|---|---|---|
| `/image` | `sensor_msgs/Image` | JPEG bytes (base64) at max 10 fps |
| `/imu` | `sensor_msgs/Imu` | JSON `{gx,gy,gz,ax,ay,az,stamp}` |
| `/odom` | `nav_msgs/Odometry` | JSON `{x,y,z,qw,qx,qy,qz,vx,vy}` |
| `/openvio/map_points` | `sensor_msgs/PointCloud2` | JSON `{n_points}` (count only) |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | JSON forwarded |

### 3.4 ROS publications (from browser commands)
| Topic | Type | Trigger |
|---|---|---|
| `/robot_control` | `std_msgs/Int16MultiArray` | Keyboard/joystick input |
| `/imu_recal` | `std_msgs/Bool` | Recalibrate button |
| `/slam_reset` | `std_msgs/Bool` | Reset SLAM button |

### 3.5 WebSocket message protocol (browser ↔ node)
All messages are JSON strings.

**Node → Browser:**
```json
{"type": "image",     "data": "<base64-jpeg>", "stamp": 1234567890.123}
{"type": "imu",       "gx": 0.01, "gy": 0.00, "gz": 0.002,
                      "ax": 0.12, "ay": -9.79, "az": 0.05, "stamp": 1234}
{"type": "odom",      "x": 1.2, "y": -0.3, "z": 0.0,
                      "vx": 0.05, "vy": 0.0, "yaw": 0.34}
{"type": "mapstats",  "n_points": 1234, "stamp": 1234}
{"type": "diag",      "status": [...]}
{"type": "sys",       "cpu": 45.2, "mem_mb": 1230, "ros_hz_imu": 99.8,
                      "ros_hz_img": 9.7, "stamp": 1234}
```

**Browser → Node:**
```json
{"type": "motor",     "left": 128, "right": 128}
{"type": "recal"}
{"type": "reset_slam"}
{"type": "stop"}
{"type": "set_speed", "value": 0.75}
```

### 3.6 System telemetry
The node collects system stats every 1 second:
```python
import psutil
cpu    = psutil.cpu_percent(interval=None)
mem_mb = psutil.virtual_memory().used // (1024*1024)
```
Publish these as `{"type": "sys", ...}` to browser.

---

## 4. Web Controller UI — `static/index.html` + `controller.js` + `style.css`

### 4.1 Layout specification
```
┌────────────────────────────────────────────────────────────────────────┐
│  TOP BAR: [OpenVIO SLAM] [●CONNECTED / ○DISCONNECTED] [IP] [ROS TIME] │
│           [CPU: 45%] [MEM: 1.2GB] [IMU: 99Hz] [CAM: 10Hz] [WARN:0]  │
├──────────────────────────────────────┬─────────────────────────────────┤
│                                      │  TELEMETRY PANEL                │
│  CAMERA FEED                         │  ┌─────────────────────────┐   │
│  (canvas, 60% width)                 │  │ IMU CHARTS (Chart.js)   │   │
│                                      │  │  • Gyro XYZ (line, 5s)  │   │
│  [overlay: crosshair, FPS, latency]  │  │  • Accel XYZ (line, 5s) │   │
│  [overlay: feature count if avail]   │  └─────────────────────────┘   │
│  [overlay: ZUPT indicator]           │  ┌─────────────────────────┐   │
│                                      │  │ MINI MAP (canvas)       │   │
│                                      │  │  • Robot path from /odom│   │
│                                      │  │  • Robot pose indicator │   │
│                                      │  └─────────────────────────┘   │
│                                      │  ┌─────────────────────────┐   │
│                                      │  │ ODOMETRY                │   │
│                                      │  │  X: +1.23m  Y: -0.45m  │   │
│                                      │  │  Yaw: +34.2°            │   │
│                                      │  │  Vx: +0.12 m/s          │   │
│                                      │  │  MapPts: 1234           │   │
│                                      │  └─────────────────────────┘   │
├──────────────────────────────────────┴─────────────────────────────────┤
│  CONTROL BAR                                                           │
│  [SPEED: ████░░░░ 60%]  [W] [A] [S] [D]  [JOYSTICK]  [●REC] [STOP]  │
├────────────────────────────────────────────────────────────────────────┤
│  DEBUG CONSOLE (collapsible, last 200 lines)                           │
│  [RECAL IMU] [RESET SLAM] [EXPORT PATH] [SCREENSHOT] [SHUTDOWN]       │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Visual design requirements

**Color palette:**
```css
:root {
  --bg-primary:     #0a0e1a;   /* deep navy black */
  --bg-secondary:   #0f1629;   /* panel background */
  --bg-tertiary:    #151e35;   /* card background */
  --accent-cyan:    #00d4ff;   /* primary accent */
  --accent-green:   #00ff88;   /* success / active */
  --accent-orange:  #ff6b00;   /* warning */
  --accent-red:     #ff2244;   /* error / stop */
  --accent-purple:  #8b5cf6;   /* secondary accent */
  --text-primary:   #e2e8f0;
  --text-secondary: #64748b;
  --text-dim:       #334155;
  --border:         #1e2d4d;
  --border-glow:    rgba(0, 212, 255, 0.3);
  --grid-line:      rgba(0, 212, 255, 0.05);
}
```

**Typography:**
- Primary font: `'JetBrains Mono'` (Google Fonts) for all data displays
- UI font: `'Inter'` for labels and buttons
- All imported via `@import` from Google Fonts CDN

**Effects:**
- Neon glow on active elements: `box-shadow: 0 0 12px var(--accent-cyan)`
- Subtle scanline overlay on camera feed: CSS `repeating-linear-gradient`
- Panel borders with `border: 1px solid var(--border)` and glow on hover
- Animated connection pulse (blinking green dot when connected)
- Smooth transitions: `transition: all 0.15s ease`
- Camera feed: `filter: contrast(1.05) saturate(1.1)` for crisp look

### 4.3 Camera feed implementation
```javascript
// WebSocket receives: {"type": "image", "data": "<base64>", "stamp": N}
// Decode and draw on <canvas> using ImageBitmap for performance

const canvas = document.getElementById('cam-feed');
const ctx = canvas.getContext('2d');

function updateFrame(base64data) {
    const img = new Image();
    img.onload = () => {
        canvas.width  = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        drawOverlays(ctx);   // crosshair, FPS, latency
    };
    img.src = 'data:image/jpeg;base64,' + base64data;
}

function drawOverlays(ctx) {
    // Crosshair at center
    // FPS counter (top-right, green)
    // Latency ms (top-right, below FPS)
    // ZUPT indicator: "ZUPT" badge (bottom-left, cyan when active)
    // Resolution (bottom-right, dim)
}
```

### 4.4 IMU charts (Chart.js)
```javascript
// Two charts: gyro (XYZ) and accel (XYZ)
// Rolling 5-second window at 100 Hz → 500 samples max
// Line chart, no fill, thin lines (1.5px)
// Colors: X=cyan, Y=green, Z=orange
// Y-axis auto-scale with fixed padding
// X-axis: "time ago" in seconds (0 to -5s)
// Update at 20 Hz max (throttle IMU data for chart)
// Chart background: transparent, grid: var(--grid-line)

const gyroChart  = new Chart(gyroCtx,  chartConfig('Gyroscope (rad/s)'));
const accelChart = new Chart(accelCtx, chartConfig('Accelerometer (m/s²)'));

function chartConfig(title) {
    return {
        type: 'line',
        data: { datasets: [...] },
        options: {
            animation: false,
            responsive: true,
            // ... full config
        }
    };
}
```

### 4.5 Mini-map (canvas)
```javascript
// 250×250 canvas
// Coordinate frame: center = robot start (0,0)
// Scale: auto-fit based on trajectory extent (min 2m × 2m visible)
// Draw: trajectory as polyline (cyan, thin), robot as arrow (white, filled)
// Grid: 1m × 1m lines (dim)
// Reset on SLAM reset event

class MiniMap {
    constructor(canvas) { ... }
    addPose(x, y, yaw) { ... }   // from /odom
    render() { ... }              // called on each odom update
    reset() { ... }
}
```

### 4.6 Keyboard control
```javascript
// Key state map (handles key repeat correctly)
const keyState = {};

document.addEventListener('keydown', e => {
    if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',
         'w','a','s','d','W','A','S','D',' '].includes(e.key)) {
        e.preventDefault();
    }
    keyState[e.key] = true;
    updateMotorFromKeys();
});
document.addEventListener('keyup', e => {
    keyState[e.key] = false;
    updateMotorFromKeys();
});

function updateMotorFromKeys() {
    const spd  = Math.round(speedSlider.value * 255 / 100);
    const fwd  = (keyState['ArrowUp']    || keyState['w'] || keyState['W']);
    const bwd  = (keyState['ArrowDown']  || keyState['s'] || keyState['S']);
    const lft  = (keyState['ArrowLeft']  || keyState['a'] || keyState['A']);
    const rgt  = (keyState['ArrowRight'] || keyState['d'] || keyState['D']);

    let L = 0, R = 0;
    if (fwd)  { L =  spd;   R =  spd;   }
    if (bwd)  { L = -spd;   R = -spd;   }
    if (lft)  { L -= spd/2; R += spd/2; }  // differential turn
    if (rgt)  { L += spd/2; R -= spd/2; }
    if (keyState[' '])  { L = 0; R = 0; }  // emergency stop

    L = Math.max(-255, Math.min(255, Math.round(L)));
    R = Math.max(-255, Math.min(255, Math.round(R)));
    sendMotor(L, R);
}
```

### 4.7 On-screen joystick
Use `nipplejs` (CDN: `https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js`).
```javascript
const joystick = nipplejs.create({
    zone:  document.getElementById('joystick-zone'),
    mode:  'static',
    position: { left: '50%', top: '50%' },
    color: 'rgba(0, 212, 255, 0.5)',
    size:  120,
});
joystick.on('move', (evt, data) => {
    const angle  = data.angle.radian;
    const force  = Math.min(data.force, 1.0);
    const spd    = Math.round(speedSlider.value * 255 / 100 * force);
    const L = Math.round(spd * (Math.cos(angle) + Math.sin(angle)));
    const R = Math.round(spd * (Math.cos(angle) - Math.sin(angle)));
    sendMotor(Math.max(-255,Math.min(255,L)),
              Math.max(-255,Math.min(255,R)));
});
joystick.on('end', () => sendMotor(0, 0));
```

### 4.8 Speed slider
```html
<input type="range" id="speed-slider" min="0" max="100" value="60"
       style="accent-color: var(--accent-cyan)">
```
Display as styled progress bar with neon fill. Current value shown as `XX%`.
All motor commands are scaled by this value before sending.

### 4.9 Debug console
```javascript
// Max 200 lines in a <pre> or <div> with overflow-y: auto
// Auto-scroll to bottom (with override if user scrolled up manually)
// Color coding:
//   [INFO]  → var(--text-secondary)
//   [WARN]  → var(--accent-orange)
//   [ERROR] → var(--accent-red)
//   [SYS]   → var(--accent-purple)
//   [IMU]   → var(--accent-cyan)   (throttled: show every 50th message)
// Collapsible: clicking the header toggles height
// Filter: text input to search/filter log lines

function log(level, msg) {
    const line = document.createElement('div');
    line.className = `log-line log-${level.toLowerCase()}`;
    line.textContent = `${timestamp()} [${level}] ${msg}`;
    consoleEl.appendChild(line);
    if (consoleEl.children.length > 200)
        consoleEl.removeChild(consoleEl.firstChild);
    if (autoScroll) consoleEl.scrollTop = consoleEl.scrollHeight;
}
```

### 4.10 Advanced control buttons

| Button | Action | Confirmation required |
|---|---|---|
| RECAL IMU | Send `{"type":"recal"}` to WebSocket | No |
| RESET SLAM | Send `{"type":"reset_slam"}` to WebSocket | Yes (modal) |
| EXPORT PATH | Download `/odom` trajectory as CSV | No |
| SCREENSHOT | `canvas.toBlob()` → download PNG | No |
| ●REC / ■STOP | Toggle rosbag recording | No |
| SHUTDOWN | `{"type":"shutdown"}` | Yes (modal) |

Rosbag recording: the node runs `subprocess.Popen(['ros2', 'bag', 'record', ...])`.

### 4.11 Connection state machine
```javascript
const WS_URL = `ws://${location.hostname}:8765`;
let ws, reconnectTimer;

function connect() {
    ws = new WebSocket(WS_URL);
    ws.onopen    = () => { setStatus('connected');    log('SYS','WebSocket connected'); };
    ws.onmessage = (e)  => handleMessage(JSON.parse(e.data));
    ws.onerror   = ()   => setStatus('error');
    ws.onclose   = ()   => { setStatus('disconnected');
                              reconnectTimer = setTimeout(connect, 2000); };
}

// Status indicator: blinking green dot when connected, red when not
function setStatus(s) { /* update DOM */ }
```

### 4.12 Keyboard shortcuts (global)
| Key | Action |
|---|---|
| `Space` | Emergency stop (motors = 0) |
| `R` | Reset SLAM (with confirmation) |
| `C` | Recalibrate IMU |
| `F` | Toggle fullscreen on camera feed |
| `D` | Toggle debug console |
| `S` (hold Ctrl) | Screenshot |
| `E` | Export path |
| `+/-` | Increase/decrease speed by 10% |

---

## 5. Updated Launch File — `openvio_slam.launch.py`

### 5.1 New node additions
```python
# Add to existing launch file:

imu_node = Node(
    package='openvio_slam',
    executable='imu_publisher_node',
    name='imu_publisher',
    output='screen',
    parameters=[{
        'serial_port':   '/dev/ttyUSB0',
        'baud_rate':     115200,
        'publish_rate':  100.0,
        'frame_id':      'imu_link',
        'gyro_noise':    0.005,
        'accel_noise':   0.085,
    }],
)

web_node = Node(
    package='openvio_slam',
    executable='web_controller_node',
    name='web_controller',
    output='screen',
    parameters=[{
        'http_port': 8080,
        'ws_port':   8765,
    }],
)
```

### 5.2 Remove IMU from slam_bridge_node
Add a parameter to `slam_bridge_node`:
```python
bridge_node = Node(
    ...
    parameters=[{
        'publish_imu': False,   # ← NEW: disable phone IMU publishing
    }],
)
```
The slam_bridge_node must read this parameter and skip IMU publishing
if `publish_imu == False`.

### 5.3 Launch timing (staggered)
```python
return LaunchDescription([
    rviz_arg,
    rosbridge_node,                                    # t=0
    TimerAction(period=2.0,  actions=[bridge_node]),   # t=2  (camera only)
    TimerAction(period=3.0,  actions=[imu_node]),      # t=3  (Arduino IMU)
    TimerAction(period=5.0,  actions=[slam_node]),     # t=5  (SLAM after both streams ready)
    TimerAction(period=7.0,  actions=[web_node]),      # t=7  (UI last)
    TimerAction(period=9.0,  actions=[rviz_node]),     # t=9
])
```

---

## 6. Extrinsic Calibration Update — `config/extrinsic.yaml`

### 6.1 Frame definitions
```
imu_link   → frame of MPU-9250/6500 on Arduino UNO (body frame)
camera     → frame of phone camera lens (landscape, front-facing)
base_link  → robot chassis center
```

### 6.2 Measurement procedure
With the Arduino IMU mounted on the robot and the phone camera mounted
in landscape on the robot:

1. Measure the physical offset from camera lens to IMU sensor (ruler):
   - In robot body frame (X=forward, Y=left, Z=up)
   - Example: camera 5cm ahead of IMU, 10cm to the right, 3cm below IMU
   - `p_IC` = position of camera in IMU frame (metres)

2. Measure rotation: if both camera and IMU have the same axis orientation
   (very common if the phone is mounted flat and Arduino is flat),
   `R_IC = I₃` (identity).

3. Default placeholder values (replace with measured values):
```yaml
extrinsic:
  R_IC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  p_IC: [0.0, 0.0, 0.0]   # ← MEASURE AND REPLACE
  comment: "IMU=Arduino MPU-9250. p_IC = camera position in IMU frame (metres).
            Measure with ruler. Positive X = forward, Y = left, Z = up."
```

### 6.3 Static TF publisher (add to launch file)
```python
static_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='imu_to_camera_tf',
    # args: x y z yaw pitch roll parent_frame child_frame
    # Replace 0 0 0 0 0 0 with your measured values
    arguments=['0', '0', '0', '0', '0', '0', 'imu_link', 'base_link'],
    output='screen',
)
```

---

## 7. `slam.yaml` Updates

### 7.1 Changes required
Add under `imu:` section:
```yaml
imu:
  gyro_noise:       0.005      # MPU-9250 spec (rad/s/√Hz) — verify with Allan variance
  accel_noise:      0.085      # MPU-9250 spec (m/s²/√Hz)
  gyro_bias_walk:   0.0001
  accel_bias_walk:  0.221
  gravity:          9.81
  frame_id:         "imu_link" # ← NEW: must match imu_publisher_node frame_id
  rate_hz:          100        # ← NEW: expected IMU publish rate
```

Add new top-level section:
```yaml
web_controller:
  http_port:    8080
  ws_port:      8765
  max_cam_fps:  10
  record_path:  "~/.ros/bags/"
```

---

## 8. `CMakeLists.txt` Updates

### 8.1 New executables to install
```cmake
install(PROGRAMS
  openvio_slam/imu/imu_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME imu_publisher_node
)

install(PROGRAMS
  web_controller/web_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME web_controller_node
)

install(DIRECTORY
  web_controller/static
  DESTINATION share/${PROJECT_NAME}/web_controller/
)
```

### 8.2 New Python dependency
```cmake
# In requirements or package.xml:
# python3-serial  (pyserial)
# python3-psutil
# python3-websockets
```

### 8.3 package.xml additions
```xml
<exec_depend>python3-serial</exec_depend>
<exec_depend>python3-psutil</exec_depend>
<exec_depend>python3-websockets</exec_depend>
<exec_depend>diagnostic_msgs</exec_depend>
```

---

## 9. Agent Task Assignment

> Multi-agent systems should assign these tasks as follows.
> Each task is independent given the specifications above.
> Agents must not modify sections outside their assigned scope.

### Agent 1 — Embedded Firmware
**Task:** Write `robot_controller.ino` per Section 1.
**Output:** Single `.ino` file, compiles with Arduino IDE 2.x, no external libraries except `Wire.h` and `EEPROM.h`.
**Validation:** Must compile with `arduino-cli compile --fqbn arduino:avr:uno`.

### Agent 2 — ROS 2 IMU Publisher
**Task:** Write `imu_publisher_node.py` per Section 2.
**Output:** Single `.py` file. Must run as a ROS 2 node. Must import from `openvio_slam` package namespace.
**Validation:** `ros2 run openvio_slam imu_publisher_node --ros-args -p serial_port:=/dev/null` must start without crash (serial fail is expected, should log and retry).

### Agent 3 — ROS 2 Web Controller Node
**Task:** Write `web_controller_node.py` per Section 3.
**Output:** Single `.py` file using `asyncio` + `rclpy`.
**Validation:** Must start with `ros2 run openvio_slam web_controller_node` and serve HTTP on port 8080.

### Agent 4 — Web UI (HTML/JS/CSS)
**Task:** Write `index.html`, `controller.js`, `style.css` per Section 4.
**Output:** Three files. Must use only CDN-hosted dependencies (no npm build step).
**Required CDNs:**
```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.10.1/nipplejs.min.js"></script>
```
**Validation:** Must open in Chrome/Firefox without errors. WebSocket connection must attempt to connect to `ws://localhost:8765`.

### Agent 5 — Launch File + Config Updates
**Task:** Update `openvio_slam.launch.py`, `slam.yaml`, `extrinsic.yaml`, `CMakeLists.txt`, `package.xml` per Sections 5–8.
**Output:** Modified versions of these 5 files.
**Validation:** `colcon build --packages-select openvio_slam` must succeed.

### Agent 6 — Integration Test Script
**Task:** Write `tools/test_integration.sh` that:
1. Checks Arduino is connected: `ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM*`
2. Checks Python packages: `python3 -c "import serial, psutil, websockets"`
3. Verifies ROS environment: `source /opt/ros/jazzy/setup.bash`
4. Runs `ros2 topic hz /imu` for 5 seconds and checks output ≥ 90 Hz
5. Runs `ros2 topic hz /image` for 5 seconds and checks ≥ 8 Hz
6. Opens `http://localhost:8080` and checks HTTP 200
7. Prints PASS/FAIL summary for each check

---

## 10. Cross-Agent Dependencies and Interface Contracts

> These are the precise interfaces that all agents must agree on.
> Any deviation breaks the system.

### 10.1 Serial packet format (Agent 1 ↔ Agent 2)
- IMU packet: exactly 31 bytes, header `[0xAA, 0x55]`, checksum = XOR of bytes 2–29
- Motor command: exactly 7 bytes, header `[0xBB, 0x66]`
- Recal command: exactly 2 bytes `[0xCC, 0x77]`
- **Byte order: little-endian for all multi-byte values**

### 10.2 ROS topic contracts (Agent 2/3 ↔ Agent 4/5)
| Topic | Publisher | Subscriber | QoS |
|---|---|---|---|
| `/imu` | Agent 2 | `openvio_slam_node`, Agent 3 | RELIABLE, KEEP_LAST 10 |
| `/robot_control` | Agent 3 | Agent 2 | RELIABLE, KEEP_LAST 1 |
| `/imu_recal` | Agent 3 | Agent 2 | RELIABLE, KEEP_LAST 1 |
| `/slam_reset` | Agent 3 | `openvio_slam_node` | RELIABLE, KEEP_LAST 1 |

### 10.3 WebSocket message contract (Agent 3 ↔ Agent 4)
All messages JSON. All numeric values are Python/JS native numbers (not strings).
Timestamps are Unix epoch seconds as float.
The `"type"` field is always present and always a string.

### 10.4 Files that must NOT be modified
- `openvio_slam/ros/vio_node.py` (any version)
- `openvio_slam/estimator/msckf.py`
- `openvio_slam/utils/so3.py`
- `openvio_slam/utils/quaternion.py`
- Any file in `openvio_slam/frontend/`

---

## 11. Testing and Validation Checklist

> Every agent must confirm their output satisfies each relevant item before submitting.

### Arduino firmware
- [ ] Compiles with `arduino-cli` without warnings on `avr:uno`
- [ ] Sends IMU packet every 10 ms (±1 ms) as measured by `screen /dev/ttyUSB0 115200`
- [ ] Checksum verified: Python `reduce(xor, packet[2:30]) == packet[30]`
- [ ] Motor stops within 500 ms of serial disconnect
- [ ] Recalibration triggered by `0xCC 0x77` packet
- [ ] Gyro bias saved to EEPROM and loaded on restart

### IMU publisher node
- [ ] `ros2 topic hz /imu` shows ≥ 95 Hz when Arduino connected
- [ ] `ros2 topic echo /imu --once` shows sensible values (|accel| ≈ 9.81 when static)
- [ ] Node does not crash if serial port missing; retries automatically
- [ ] Motor commands forwarded correctly: `/robot_control [128, -128]` → serial packet

### Web controller node
- [ ] HTTP 200 on `http://localhost:8080`
- [ ] WebSocket accepts connection on port 8765
- [ ] `/image` frames arrive in browser (base64 JPEG in `{"type":"image"}` message)
- [ ] Motor commands from browser arrive on `/robot_control` topic

### Web UI
- [ ] Loads without console errors in Chrome 120+
- [ ] Camera feed updates at ≥ 8 FPS when phone connected
- [ ] Arrow keys produce `/robot_control` messages (verify with `ros2 topic echo`)
- [ ] Joystick produces motor commands
- [ ] IMU charts update in real time
- [ ] Mini-map grows as robot moves
- [ ] Debug console shows ROS messages
- [ ] Emergency stop (Space) sends `[0, 0]` immediately

### Integration
- [ ] `ros2 topic hz /imu` shows ≥ 95 Hz
- [ ] `ros2 topic hz /image` shows ≥ 8 Hz
- [ ] `openvio_slam_node` shows `MapPts > 0` after 30 seconds of movement
- [ ] No `Frame error` messages in `openvio_slam_node` terminal

---

## 12. Environment Setup Commands

> Include these at the top of any setup script for reproducibility.

```bash
# System dependencies
sudo apt install python3-serial python3-psutil ros-jazzy-diagnostic-msgs -y

# Python packages
pip3 install pyserial psutil websockets --break-system-packages

# Arduino CLI (if not already installed)
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
arduino-cli core install arduino:avr
arduino-cli lib install "Wire"    # built-in, but explicit
# No external IMU library required (direct register access)

# Verify serial port after Arduino connected
ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
sudo usermod -aG dialout $USER && newgrp dialout   # permission for serial port

# Build workspace
cd ~/slam_ws
colcon build --packages-select openvio_slam
source install/setup.bash

# Correct launch command (NO use_sim_time for live streaming)
ros2 launch openvio_slam openvio_slam.launch.py
```

---

## 13. Known Pitfalls and Mitigations

> Every agent must read this section. These are real runtime failures that will
> occur if any of these points are missed.

| Pitfall | Symptom | Mitigation |
|---|---|---|
| `use_sim_time:=true` with no bag | `slam_bridge_node: Frames: 0` | Never use `use_sim_time` for live phone streaming |
| `_pose_cache` not declared in `__init__` | `MapPts: 0` forever, silent AttributeError | Declare `self._pose_cache = {}` in `__init__` |
| numpy array in bare Python `if` | `Frame error: truth value ambiguous` | Wrap with `float()` or use `.any()/.all()` |
| Wrong IMU topic name | MSCKF never propagates | Subscribe to `/imu` (matches slam_bridge_node AND imu_publisher_node) |
| Serial port not in `dialout` group | `Permission denied: /dev/ttyUSB0` | `sudo usermod -aG dialout $USER` |
| Arduino timestamp not synced to ROS | Time-sync errors in MSCKF | Stamp IMU messages with `self.get_clock().now()`, ignore Arduino `millis()` |
| Forced init with garbage visual pose | SLAM position at [-10, 6, 4]m | Clamp visual pose: if `\|p_vis\| > 5m`, reset to origin before forced init |
| RViz PointCloud2 History Policy: Keep All | Cloud never appears after late connect | Use `History Policy: Keep Last` with `depth: 5` |
| Serial blocking read in Arduino loop | IMU packets delayed, motor response sluggish | State machine parser, never use `Serial.readBytes()` with timeout |
| Web UI WebSocket to wrong host | `ERR_CONNECTION_REFUSED` in browser | Use `location.hostname` dynamically, not hardcoded `localhost` |
| `ros2 bag record` blocks node thread | Web controller freezes | Use `subprocess.Popen()` (non-blocking) |

---

*End of specification. Total: 13 sections, 6 agent tasks, complete interface contracts.*
*This document is self-contained. No external references are required to implement it.*
