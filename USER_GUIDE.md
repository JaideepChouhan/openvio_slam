# OpenVIO-SLAM — Testing Platform User Guide
### Robot Setup · Operation · Tuning · Troubleshooting
**Version 2.0 — 9-DOF Arduino Protocol**

---

## Who this guide is for

This guide is for anyone operating the tracked ground robot that serves as the **physical testing platform for OpenVIO-SLAM**. The robot is not the final product — it is the tool used to generate real-world motion data for the SLAM estimator. Understanding this distinction matters: when something goes wrong, the question is always *"is the SLAM system failing, or is the test platform failing to feed it good data?"* This guide helps you answer that question systematically.

Prior knowledge assumed: you can use a terminal, you have ROS 2 Jazzy installed and sourced, and you have already built the `openvio_slam` package with `colcon build`.

---

## Section 1 — Hardware setup

### 1.1 Component checklist

Before powering anything on, verify every item:

```
☐ Raspberry Pi 5 (8 GB) — Ubuntu 24.04, ROS 2 Jazzy
☐ Arduino UNO — robot_controller.ino firmware uploaded
☐ MPU-9250 or MPU-6500 breakout board
    ☐ VCC → Arduino 3.3V  (NOT 5V — will damage the chip)
    ☐ GND → Arduino GND
    ☐ SDA → Arduino A4
    ☐ SCL → Arduino A5
    ☐ AD0 → Arduino GND   (sets I2C address to 0x68)
☐ Cytron MDD20A motor driver
    ☐ VM+/VM- → motor battery (7–30V, separate from Pi power)
    ☐ VCC/GND  → Arduino 5V / GND
    ☐ AN1/AIN1 → Arduino pins 6, 7  (right motor: PWM, DIR)
    ☐ AN2/AIN2 → Arduino pins 5, 4  (left  motor: PWM, DIR)
    ☐ AOUT1 → left  motor terminals
    ☐ AOUT2 → right motor terminals
☐ USB cable: Arduino → Raspberry Pi (appears as /dev/ttyUSB0 or /dev/ttyACM0)
☐ Android phone
    ☐ Mounted in LANDSCAPE orientation on robot
    ☐ Camera lens facing FORWARD (same direction as robot travel)
    ☐ Phone and Pi on the same WiFi network
☐ Power
    ☐ Pi: 27W USB-C supply (official Pi 5 PSU recommended)
    ☐ Motors: separate 12V battery — NOT from Pi or Arduino
    ☐ 100µF cap across Arduino 5V/GND (prevents motor switching brownout)
    ☐ 1000µF cap across motor battery terminals at Cytron
```

### 1.2 Wiring overview

```
Raspberry Pi 5                   Arduino UNO
┌──────────────┐  USB-B cable   ┌──────────────────────────────┐
│ /dev/ttyUSB0 │◄──────────────►│ Serial0 (D0/D1)              │
└──────────────┘                │                              │
                                │  A4 (SDA) ─────┐            │
                                │  A5 (SCL) ─────┤  MPU-9250  │
                                │  3.3V ─────────┤  breakout  │
                                │  GND  ─────────┘            │
                                │                              │
                                │  D5~ ──► Cytron AN2 (L-PWM) │
                                │  D4  ──► Cytron AIN2 (L-DIR) │
                                │  D6~ ──► Cytron AN1 (R-PWM) │
                                │  D7  ──► Cytron AIN1 (R-DIR) │
                                └──────────────────────────────┘

Cytron MDD20A
┌──────────────────────────────────────────────────┐
│  VM+ / VM- ──────────────────────► Motor battery │
│  AOUT1+/AOUT1- ──────────────────► Left  motor  │
│  AOUT2+/AOUT2- ──────────────────► Right motor  │
└──────────────────────────────────────────────────┘
```

### 1.3 IMU mounting orientation

The firmware assumes the MPU-9250 is mounted **flat on the robot chassis with the chip X-axis pointing forward** (toward the phone mount). With this orientation:

- Chip X = robot forward (+X)
- Chip Y = robot left (+Y)
- Chip Z = robot up (+Z)

This matches ROS convention directly — no axis remapping is needed. If your board is mounted differently, edit `convert_imu()` in `robot_controller.ino` before uploading.

**Verify orientation at runtime:**
```bash
ros2 topic echo /imu --once | grep -A 3 linear_acceleration
# When robot is flat and stationary:
# linear_acceleration.x ≈  0.0   (no forward acceleration)
# linear_acceleration.y ≈  0.0   (no lateral acceleration)
# linear_acceleration.z ≈ +9.81  (gravity = +Z when chip faces up)
```

If `az ≈ -9.81`, the Z-axis is inverted — flip it in firmware. If `ax ≈ ±9.81`, the chip is mounted sideways.

---

## Section 2 — Software setup

### 2.1 One-time installation

```bash
# System packages
sudo apt update
sudo apt install -y \
    python3-serial python3-psutil \
    ros-jazzy-diagnostic-msgs \
    ros-jazzy-tf2-ros ros-jazzy-cv-bridge \
    ros-jazzy-rosbridge-server

# Python packages
pip3 install pyserial psutil websockets --break-system-packages

# Serial port permission (log out and back in after this)
sudo usermod -aG dialout $USER
```

### 2.2 Arduino udev rule (gives stable port name)

```bash
# For official Arduino UNO (Atmega16U2 USB chip):
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", SYMLINK+="ttyArduino", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/99-arduino.rules

# For UNO clones with CH340 USB chip (very common):
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="ttyArduinoClone", MODE="0666"' \
    | sudo tee -a /etc/udev/rules.d/99-arduino.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
# After reconnecting Arduino: ls /dev/ttyArduino*
```

### 2.3 Upload Arduino firmware

1. Open **Arduino IDE 2.x**
2. File → Open → `~/slam_ws/src/openvio_slam/arduino/robot_controller/robot_controller.ino`
3. Tools → Board → Arduino UNO
4. Tools → Port → `/dev/ttyUSB0` (or `/dev/ttyACM0`)
5. Click **Upload** (→)
6. Wait for *"Done uploading"*

The firmware will immediately start sending 43-byte IMU packets and blinking the built-in LED on startup to signal it is running.

### 2.4 Build the ROS package

```bash
cd ~/slam_ws
colcon build --packages-select openvio_slam \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash

# Add to .bashrc so you don't have to source manually:
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source ~/slam_ws/install/setup.bash" >> ~/.bashrc
```

---

## Section 3 — Pre-flight verification

Run these checks **in order** before every test session. Each check isolates one layer of the system. Do not skip to the launch step — a failure anywhere in this sequence will produce silent wrong behaviour in the SLAM estimator.

### Check 1 — Arduino is sending data (no ROS, no phone)

```bash
python3 ~/slam_ws/src/openvio_slam/tools/test_protocol.py /dev/ttyUSB0
```

**Good output:**
```
✅ Port opened: /dev/ttyUSB0
✅ Pkt  #1: ts=  1234ms  G(+0.001,-0.000,+0.002)rad/s  A(+0.04,+0.01,+9.81)m/s²  M(+23.4,-5.1,+41.2)µT
✅ Pkt  #2: ts=  1244ms  G(+0.001,-0.001,+0.002)rad/s  A(+0.04,+0.01,+9.81)m/s²  M(+23.3,-5.2,+41.1)µT
...
✅ Rate: 99.8 Hz  (target: 100 Hz)
✅ Checksum: 0 errors
✅ Arduino protocol is CORRECT (43-byte 9-DOF packets)
```

**If this fails:** see Troubleshooting Section 6.

### Check 2 — Math tests pass

```bash
cd ~/slam_ws/src/openvio_slam
python3 -m pytest tests/test_math.py -v --tb=short
```

All tests must be green. A wrong sign in `hat()` or `quat_multiply()` produces trajectory drift that looks like a tuning problem but is a code bug. These tests catch it.

### Check 3 — Phone camera is reachable

On the phone:
1. Open **Chrome**
2. Navigate to `https://<pi-ip>:9090` (accept the self-signed cert warning)
3. Grant camera and motion sensor permissions
4. Set FPS → **10**, JPEG quality → **60**
5. Tap **Connect**

Then on the Pi:
```bash
ros2 topic hz /image        # expect: average rate: 9.x Hz
ros2 topic hz /camera_info  # expect: at least 1 message
```

### Check 4 — Full pre-launch checklist

```
☐ test_protocol.py shows ✅ at ~100 Hz
☐ test_math.py — all green
☐ ros2 topic hz /image ≥ 8 Hz
☐ ros2 topic hz /camera_info — at least 1 message received
☐ Robot is on flat floor, not on a ramp or angled surface
☐ Motors are powered (motor battery connected)
☐ Phone mount is secure — no play or vibration
☐ Lighting is adequate (≥ 100 lux, avoid direct sunlight on camera)
☐ Launch will NOT use use_sim_time:=true
```

---

## Section 4 — Launch and operation

### 4.1 Launch the system

```bash
# Always source first if you haven't in this terminal session
source ~/slam_ws/install/setup.bash

# Launch — do NOT add use_sim_time:=true for live robot testing
ros2 launch openvio_slam openvio_slam.launch.py
```

Nodes start in a staggered sequence (2-second gaps) to ensure dependencies are ready:

| Time | Node | What it does |
|:---:|:---|:---|
| t = 0 s | `rosbridge_websocket` | Opens port 9090 for phone connection |
| t = 2 s | `slam_bridge_node` | Subscribes to phone stream, publishes `/image` `/camera_info` |
| t = 3 s | `imu_publisher_node` | Opens Arduino serial, publishes `/imu` `/imu/mag` |
| t = 4 s | `static_tf_publisher` | Broadcasts `imu_link → base_link` transform |
| t = 5 s | `openvio_slam_node` | SLAM estimator starts (waits for camera + IMU) |
| t = 7 s | `web_controller_node` | HTTP :8080 and WebSocket :8765 go live |
| t = 9 s | `rviz2` | RViz opens (if `rviz:=true`, the default) |

### 4.2 Initialization procedure

The SLAM estimator cannot start without a valid initial state. Follow this sequence every time:

```
1. Launch the system (Section 4.1)

2. Wait for:
   [openvio_slam] Hold robot stationary 2 seconds for gyro bias init...

3. Place the robot on a flat surface and DO NOT TOUCH IT for 3 seconds.
   The gyroscope bias is estimated during this window. If the robot moves,
   the bias estimate will be wrong and trajectory will drift immediately.

4. Connect the phone (if not already connected).

5. Watch for:
   [openvio_slam] ✅ VIO initialized! Scale=1.000, |v|=0.008 m/s

   OR after 50 frames without convergence:
   [openvio_slam] ⚠️  VIO force-initialized at origin.

6. After initialization, move the robot SLOWLY (< 0.3 m/s) for the first
   10 seconds. This lets the MSCKF build up its first map points and
   establishes a reliable sliding window.

7. Watch MapPts in the terminal stats line. It should reach > 20 within
   30 seconds of slow movement:
   Frames:120 | Avg:24ms | Feats:264 | MapPts:67 | ZUPT:no
```

### 4.3 Browser controller

Open `http://<pi-ip>:8080` on any device on the same network.

**Layout overview:**

```
┌────────────────────────────────────────────────────────────────────────┐
│ TOP BAR: [OpenVIOSLAM 9-DOF] [●CONNECTED] [CPU] [MEM] [IMU Hz] [CAM Hz]│
├──────────────────────────────────────┬─────────────────────────────────┤
│                                      │  ODOMETRY                       │
│  CAMERA FEED                         │  X: +1.23m  Y: -0.45m          │
│  (with crosshair + latency overlay)  │  YAW: +34.2°  Vx: +0.12 m/s   │
│  [ZUPT badge when stationary]        ├─────────────────────────────────┤
│                                      │  COMPASS + TRAJECTORY MAP       │
│                                      │  [tilt-compensated mag compass] │
│                                      │  [odometry path minimap]        │
│                                      ├─────────────────────────────────┤
│                                      │  GYROSCOPE chart (Gx Gy Gz)     │
│                                      │  ACCELEROMETER chart (Ax Ay Az)  │
│                                      │  MAGNETOMETER chart (Mx My Mz)   │
├──────────────────────────────────────┴─────────────────────────────────┤
│  DRIVE: [speed slider] [▲ W] [◄ A] [■ stop] [D ►] [▼ S] [joystick]   │
│  ACTIONS: [RECAL IMU] [RESET SLAM] [EXPORT PATH] [SCREENSHOT] [●REC]  │
├────────────────────────────────────────────────────────────────────────┤
│  DEBUG CONSOLE (collapsible — filter by keyword)                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Keyboard shortcuts

| Key | Action |
|:---:|:---|
| `W` / `↑` | Forward |
| `S` / `↓` | Backward |
| `A` / `←` | Turn left |
| `D` / `→` | Turn right |
| `Space` | **Emergency stop** (send `[0,0]` immediately) |
| `+` / `-` | Increase / decrease speed by 10% |
| `C` | Recalibrate IMU (sends `"CAL\n"` to Arduino) |
| `R` | Reset SLAM map (with confirmation modal) |
| `E` | Export trajectory as CSV |
| `F` | Toggle camera fullscreen |
| `D` | Toggle debug console |
| `Ctrl+S` | Screenshot |

### 4.5 Drive recommendations for good SLAM data

The quality of the SLAM map depends entirely on how the robot moves. These guidelines produce the best results:

**Do:**
- Move slowly (0.1–0.3 m/s). Fast motion causes motion blur on the phone camera and degrades KLT tracking.
- Pause at corners for 1–2 seconds. ZUPT fires during pauses and eliminates accumulated velocity drift.
- Make deliberate loops — drive out, turn, come back roughly the same path. Loop closure corrects all accumulated drift.
- Keep the camera pointed at textured surfaces. Blank white walls have no features for ORB to detect.
- Maintain consistent lighting. Sudden brightness changes (doorways, windows) cause tracking loss.

**Avoid:**
- Spinning in place rapidly. Pure rotation makes scale unobservable and can cause initialization failure.
- Driving over bumps or cables at speed. Vibration spikes corrupt the IMU pre-integration.
- Pointing the camera at the floor or ceiling. All features must be in the camera frustum.
- Moving immediately after power-on. Wait for initialization confirmation.

### 4.6 Saving and reloading the map

```bash
# Save current map:
ros2 service call /save_map std_srvs/Trigger

# Map is saved to: ~/.openvio_slam/maps/map_<timestamp>.npz

# On next run, load a saved map for re-localization:
ros2 service call /load_map std_srvs/SetString \
    "data: '/home/user/.openvio_slam/maps/map_20250901_143022.npz'"
```

After loading, the robot will attempt to re-localize by matching its current view against stored keyframes. This takes 5–15 seconds. Watch the terminal for:
```
[openvio_slam] Relocalization succeeded: KF47 (sim=0.731, inliers=34)
```

### 4.7 Recording a test session

The browser **●REC** button records all relevant topics to a ROS bag:

Topics recorded: `/image`, `/imu`, `/imu/mag`, `/odom`, `/openvio/map_points`

Bags are saved to `~/.ros/bags/slam_<timestamp>/`.

To record manually:
```bash
ros2 bag record -o ~/test_session_01 \
    /image /imu /imu/mag /odom /openvio/map_points /tf
```

---

## Section 5 — IMU calibration

### 5.1 Why calibration matters

The MSCKF covariance propagation is parameterised by four noise values in `config/slam.yaml`. If these values are wrong:
- Too small: filter over-trusts the IMU, trajectory oscillates
- Too large: filter under-trusts the IMU, visual updates dominate, trajectory is jerky

The default values in `slam.yaml` are conservative estimates for MPU-9250. Running the Allan variance calibration on your specific chip takes 10 minutes and can cut trajectory error by 30–50%.

### 5.2 Gyro bias calibration (every session)

The Arduino firmware calibrates gyro bias at startup automatically (1000 samples, ~2 seconds). The bias is saved to EEPROM and reloaded on subsequent power-cycles.

To force a recalibration mid-session (e.g., after the robot warms up):
- Press **C** in the browser controller, or
- `ros2 topic pub /imu_recal std_msgs/Bool "data: true" --once`

Keep the robot completely stationary for 5 seconds during calibration. The terminal will show:
```
[imu_publisher] Sending calibration command: CAL\n
[imu_publisher] ⚠️  Keep robot completely stationary during calibration!
```

### 5.3 Allan variance calibration (once per chip)

This estimates the four noise parameters for `slam.yaml` from a long stationary recording.

```bash
# Step 1: Record 10 minutes of stationary IMU data
ros2 bag record -o ~/.ros/imu_calibration/static_$(date +%Y%m%d) /imu

# Step 2: Extract to CSV (timestamp,gx,gy,gz,ax,ay,az)
# Use your preferred bag-to-csv tool, or the extract script:
python3 ~/slam_ws/src/openvio_slam/tools/extract_imu_csv.py \
    ~/.ros/imu_calibration/static_<date> imu_static.csv

# Step 3: Run Allan variance analysis
python3 ~/slam_ws/src/openvio_slam/tools/imu_noise_calibration.py \
    --data imu_static.csv --rate 100
```

Copy the printed values into `config/slam.yaml`:
```yaml
imu:
  gyro_noise:       <printed value>   # rad/s/√Hz
  accel_noise:      <printed value>   # m/s²/√Hz
  gyro_bias_walk:   <printed value>   # rad/s²/√Hz
  accel_bias_walk:  <printed value>   # m/s³/√Hz
```

### 5.4 IMU-camera extrinsic calibration

The `T_IC` transform in `config/extrinsic.yaml` tells the estimator where the camera is relative to the IMU. A 5-degree error here causes systematic trajectory curvature that no amount of noise tuning can fix.

**Measure with a ruler (sufficient for most tests):**

```
p_IC = [x_forward, y_left, z_up]
     = physical distance from MPU-9250 chip center
       to camera lens center, in the robot body frame

Example: camera is 11 cm ahead, 1.3 cm to the right, 14.6 cm above the IMU:
  p_IC: [0.110, -0.013, 0.146]
```

Update `config/extrinsic.yaml` with your measured values. For higher accuracy, run the hand-eye calibration tool:
```bash
python3 ~/slam_ws/src/openvio_slam/tools/calibrate_imu_camera.py \
    --data calib_data.csv \
    --fx 518.5 --fy 519.4 --cx 320.2 --cy 243.0 \
    --output config/extrinsic.yaml
```

---

## Section 6 — Troubleshooting

### Arduino / Serial

**`Permission denied: /dev/ttyUSB0`**
```bash
sudo usermod -aG dialout $USER
newgrp dialout
# If still failing: ls -la /dev/ttyUSB0 — should show crw-rw---- 1 root dialout
```

**`No Arduino found on /dev/ttyUSB*`**
```bash
lsusb | grep -i "arduino\|2341\|1a86"   # check USB detection
dmesg | tail -20 | grep tty              # check kernel port assignment
# Unplug and replug, wait 3 seconds, retry
```

**`test_protocol.py` shows 0 Hz or no packets**
```bash
# Confirm firmware is running — should see data:
screen /dev/ttyUSB0 115200
# You should see binary garbage (IMU packets). Ctrl-A then K to exit.
# If you see nothing: firmware not uploaded, or wrong baud rate.
# If you see ASCII text: old debug firmware is on the Arduino.
```

**Checksum errors reported by `test_protocol.py`**
```bash
# Cause: baud rate mismatch, or noise on long I2C wires
# Fix 1: verify firmware and node both use 115200
# Fix 2: shorten I2C wires to < 20 cm
# Fix 3: add 4.7kΩ pull-ups on SDA and SCL if not on breakout board
```

**`ros2 topic hz /imu` shows 0 Hz after launch**
```bash
ros2 node list | grep imu              # check node is running
ros2 node info /imu_publisher          # check subscriptions/publications
# If node is running but /imu is empty:
ros2 topic echo /imu --once            # should show an Imu message
# Check serial_port parameter matches actual Arduino port:
ros2 param get /imu_publisher serial_port
```

---

### SLAM estimator

**`MapPts: 0` forever (most common issue)**

This has three distinct causes — check them in order:

```bash
# Cause 1: _pose_cache missing (silent AttributeError in old vio_node.py)
# Fix: ensure you are using vio_node.py v5 (check file header)
grep "FIXED v5\|Protocol-Corrected" \
    ~/slam_ws/src/openvio_slam/openvio_slam/ros/vio_node.py

# Cause 2: robot not moving (parallax < 2° → all triangulations rejected)
# Fix: move robot forward at least 0.5 m after initialization

# Cause 3: pose_cache is empty (no camera states registered)
# Add temporarily to vio_node.py to diagnose:
# self.get_logger().info(f"cache={len(self._pose_cache)}", throttle_duration_sec=2.0)
```

**`Frame error: The truth value of an array is ambiguous`**
```bash
# numpy array used in bare Python if/bool context
# Most common location: chi-squared test in msckf.py
# Fix: wrap scalar results with float():
#   WRONG:  if gamma > threshold:
#   CORRECT: if float(gamma) > threshold:
# Enable full traceback to find the exact line:
# In _image_worker except handler, use traceback.format_exc()
```

**Robot position immediately jumps to [±10, ±10]m**
```bash
# Cause: force-init used garbage visual pose from init phase
# This usually happens when use_sim_time:=true froze the bridge clock
# vio_node.py v5 already clamps this — if it still happens:
grep "use_sim_time" ~/slam_ws/src/openvio_slam/launch/openvio_slam.launch.py
# The launch file must NOT pass use_sim_time:=true
```

**SLAM drifts rapidly despite Arduino IMU**
```bash
# Step 1: Verify IMU alignment (robot flat and stationary):
ros2 topic echo /imu --once | grep -A 3 linear_acceleration
# Expected: az ≈ +9.81, ax ≈ 0, ay ≈ 0

# Step 2: Check extrinsic is not identity when it should not be:
cat ~/slam_ws/src/openvio_slam/config/extrinsic.yaml

# Step 3: Run Allan variance calibration (Section 5.3)
# Wrong noise params are the most common cause of drift
```

**`slam_bridge_node: Frames: 0 | FPS: 0.0` throughout**
```bash
# Cause: use_sim_time:=true with no bag playing — clock is frozen
# Fix: kill all nodes, relaunch WITHOUT use_sim_time
pkill -f ros2
sleep 2
ros2 launch openvio_slam openvio_slam.launch.py
# Never add use_sim_time:=true for live robot testing
```

---

### Web controller

**Browser shows blank page**
```bash
# Check node is running and port is bound:
ros2 node list | grep web_controller
ss -tlnp | grep 8080
# If port in use: sudo fuser -k 8080/tcp
```

**Camera feed not appearing in browser**
```bash
ros2 topic hz /image                            # must be > 0
ros2 topic echo /image --once --field encoding  # should be bgr8
python3 -c "import cv2; print(cv2.__version__)" # must not error
```

**Keyboard keys not working**
The browser requires document focus. Click the camera feed area first, then use keyboard controls.

**Motors not responding despite commands**
```bash
# Check /robot_control is publishing:
ros2 topic echo /robot_control

# Check imu_publisher_node is subscribed:
ros2 node info /imu_publisher | grep Subscriptions

# Manual test (sends forward command):
ros2 topic pub /robot_control std_msgs/Int16MultiArray \
    "data: [100, 100]" --once

# Check motor watchdog: if no command in 500ms, Arduino stops motors
# The browser must be sending commands at ≥ 2 Hz to keep motors running
```

**IMU charts not updating**
```bash
ros2 topic hz /imu       # must be ~100 Hz
# If /imu is publishing but charts are empty:
# Check browser WebSocket connection (top bar status dot must be green)
# Reload the page and reconnect
```

---

## Section 7 — Tuning guide

Work through these in order. Each step assumes the previous ones are complete.

### Step 1 — IMU noise parameters

Wrong noise params are the most common source of trajectory problems. Do this before adjusting anything else.

```bash
# Record 10 min stationary:
ros2 bag record -o static /imu
# Extract + calibrate (Section 5.3)
# Copy values to config/slam.yaml imu: section
```

Signs of wrong params:
- `gyro_noise` too small → trajectory oscillates left/right unrealistically
- `gyro_noise` too large → trajectory lags behind actual motion
- `accel_bias_walk` too small → velocity drift during long straight segments

### Step 2 — Extrinsic calibration T_IC

Verify `p_IC` in `config/extrinsic.yaml` matches physical measurement. A wrong `p_IC` causes trajectory to curve even on a straight path.

Quick sanity check: drive the robot in a 2 m straight line forward. The estimated X position should increase by approximately 2 m. If it curves left or right, `p_IC` needs adjustment.

### Step 3 — ZUPT threshold

```yaml
# config/slam.yaml
constraints:
  zupt_gyro_var_threshold: 0.001   # default
```

Watch the browser controller — the **ZUPT** badge should appear only when the robot is genuinely stationary.

- If ZUPT fires while moving slowly: increase threshold to 0.005
- If ZUPT never fires when stationary: decrease threshold to 0.0005

### Step 4 — Feature count and window size

Reduce these only if the Pi cannot keep up (frame time > 80 ms):

```yaml
# config/slam.yaml  — Pi 5 defaults
frontend:
  n_features:  300   # reduce to 150 for Pi 4
estimator:
  window_size: 20    # reduce to 12 for Pi 4
```

Watch the stats line: `Avg:` should stay below 80 ms on Pi 5 with the defaults.

### Step 5 — Loop closure sensitivity

If the robot returns to a mapped area but no loop closure fires:
```yaml
loop_closure:
  bow_similarity_threshold: 0.5    # default 0.6 — lower to detect more loops
  min_loop_inliers: 15             # default 20  — lower for smaller rooms
```

If false loop closures fire (trajectory jumps unexpectedly):
```yaml
loop_closure:
  bow_similarity_threshold: 0.7    # raise threshold
  min_loop_inliers: 30             # require more geometric evidence
```

---

## Section 8 — Evaluating SLAM accuracy

### 8.1 Quick visual check

In RViz or the browser minimap, drive the robot in a complete loop and return to the start. Trajectory error is the distance between the start pose and the pose when the robot physically returns to the same spot.

Target: < 5 cm after loop closure in a 5 m × 5 m room.

### 8.2 Quantitative evaluation with ground truth

If you have a ground truth trajectory (from a motion capture system or tape-measure waypoints):

```bash
# Export trajectory from bag:
python3 ~/slam_ws/src/openvio_slam/tools/evaluate_trajectory.py \
    --estimated trajectory_estimated.txt \
    --groundtruth trajectory_groundtruth.txt

# Output:
# ATE RMSE:    4.2 cm  ← primary metric
# RTE RMSE:    0.8 cm/frame
# Rating: EXCELLENT (< 5 cm)
```

Trajectory files use TUM format: `timestamp tx ty tz qx qy qz qw`

### 8.3 Interpretation

| ATE RMSE | Rating | Likely cause if worse |
|:---:|:---:|:---|
| < 5 cm | Excellent | — |
| 5–20 cm | Good | IMU noise params not tuned |
| 20–50 cm | Acceptable | Wrong extrinsic, or no loop closure |
| > 50 cm | Poor | Check IMU alignment, `use_sim_time` bug, MapPts = 0 |

---

## Section 9 — Diagnostic commands cheat sheet

```bash
# ── System health ──────────────────────────────────────────────────────
ros2 topic list
ros2 topic hz /imu /imu/mag /image /odom /openvio/map_points
ros2 node list
ros2 node info /openvio_slam
ros2 node info /imu_publisher

# ── IMU sanity (robot flat, stationary) ───────────────────────────────
ros2 topic echo /imu --once | grep -A 3 linear_acceleration
# Expected: ax≈0 ay≈0 az≈+9.81

ros2 topic echo /imu/mag --once | grep -A 3 magnetic_field
# Expected: non-zero, |mag|≈30–60 µT (Earth's field)

# ── Map point check ────────────────────────────────────────────────────
ros2 topic echo /openvio/map_points --once --field width
# Expected: > 0 after 30 seconds of movement

# ── Arduino serial direct test ─────────────────────────────────────────
python3 ~/slam_ws/src/openvio_slam/tools/test_protocol.py /dev/ttyUSB0

# ── Web controller check ───────────────────────────────────────────────
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080
# Expected: 200

# ── CPU / memory while running ─────────────────────────────────────────
top -p $(pgrep -d',' -f 'openvio_slam|imu_publisher|web_controller|slam_bridge')

# ── Full clean restart ─────────────────────────────────────────────────
pkill -f ros2 && sleep 2
ros2 launch openvio_slam openvio_slam.launch.py
```

---

## Appendix A — Configuration reference

All parameters live in `config/slam.yaml`. Key values with their effects:

```yaml
frontend:
  n_features:         300     # features per frame — reduce for speed
  grid_rows:          4       # spatial bucketing rows
  grid_cols:          5       # spatial bucketing columns
  min_features:       150     # extract new features below this count
  min_track_length:   2       # minimum observations before MSCKF update

estimator:
  window_size:        20      # sliding window camera states (reduce for Pi 4)
  pixel_noise:        1.0     # feature measurement noise (pixels)

imu:
  gyro_noise:         0.005   # rad/s/√Hz  — from Allan variance
  accel_noise:        0.085   # m/s²/√Hz   — from Allan variance
  gyro_bias_walk:     0.0001  # rad/s²/√Hz
  accel_bias_walk:    0.001   # m/s³/√Hz
  gravity:            9.81    # m/s² — adjust for your latitude

constraints:
  zupt_gyro_var_threshold: 0.001   # stationary detection sensitivity
  zupt_noise:              0.001   # velocity pseudo-measurement noise
  planar_enabled:          true
  planar_vz_noise:         0.05    # vertical velocity constraint

loop_closure:
  bow_similarity_threshold: 0.6    # 0.5=more loops, 0.8=fewer false positives
  min_loop_inliers:         20     # geometric verification minimum
  min_keyframe_spacing:     20     # minimum keyframes between candidates
  pose_graph_max_iter:      50

map:
  max_map_points:     8000    # cull oldest beyond this limit
```

---

## Appendix B — File locations after deploy

```
~/slam_ws/src/openvio_slam/
├── openvio_slam/imu/imu_publisher_node.py   ← Arduino serial reader
├── web_controller/web_controller_node.py     ← HTTP + WebSocket server
├── web_controller/static/index.html          ← Browser dashboard
├── web_controller/static/controller.js       ← Drive control + compass
├── web_controller/static/style.css           ← Dark industrial theme
├── arduino/robot_controller/
│   └── robot_controller.ino                  ← 9-DOF Arduino firmware
├── launch/openvio_slam.launch.py             ← System launch
├── config/slam.yaml                          ← All tunable parameters
├── config/extrinsic.yaml                     ← T_IC (measure and fill in)
├── tools/test_protocol.py                    ← Run this first
├── tools/deploy.sh                           ← One-shot file installer
└── PROTOCOL_CORRECTIONS.md                   ← Protocol diff (Parts 1-3 vs actual)
```

---

*For questions about the estimator algorithms, read the papers in the References section of README.md. For questions about the robot hardware, read WIRING_GUIDE.md. For protocol differences between the spec and actual Arduino, read PROTOCOL_CORRECTIONS.md.*
