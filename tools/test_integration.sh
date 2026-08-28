#!/usr/bin/env bash
set -u

PASS=0
FAIL=0

log_pass() {
  echo "PASS: $1"
  PASS=$((PASS + 1))
}

log_fail() {
  echo "FAIL: $1"
  FAIL=$((FAIL + 1))
}

check_serial() {
  if ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM* 2>/dev/null; then
    log_pass "Arduino serial device detected"
  else
    log_fail "No Arduino serial device detected"
  fi
}

check_python() {
  python3 - <<'PY'
import serial, psutil, websockets
print('deps-ok')
PY
  if [ $? -eq 0 ]; then
    log_pass "Python dependencies available"
  else
    log_fail "Missing Python dependencies"
  fi
}

check_ros() {
  if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    log_pass "ROS 2 Jazzy environment available"
  else
    log_fail "ROS 2 Jazzy not installed"
  fi
}

check_topic_hz() {
  source /opt/ros/jazzy/setup.bash 2>/dev/null || true
  if command -v ros2 >/dev/null 2>&1; then
    timeout 5s ros2 topic hz /imu 2>/dev/null | tail -n 1 | grep -E "average rate|" >/dev/null && log_pass "IMU topic heartbeat present" || log_fail "IMU topic heartbeat missing"
    timeout 5s ros2 topic hz /image 2>/dev/null | tail -n 1 | grep -E "average rate|" >/dev/null && log_pass "Image topic heartbeat present" || log_fail "Image topic heartbeat missing"
  else
    log_fail "ros2 command not available"
  fi
}

check_http() {
  if command -v curl >/dev/null 2>&1; then
    curl -I --max-time 5 http://localhost:8080 >/tmp/openvio_http_check.txt 2>/dev/null
    if [ $? -eq 0 ]; then
      log_pass "Web dashboard HTTP endpoint responding"
    else
      log_fail "Web dashboard HTTP endpoint unavailable"
    fi
  else
    log_fail "curl not available"
  fi
}

echo "Running OpenVIO-SLAM integration checks"
check_serial
check_python
check_ros
check_topic_hz
check_http

echo "Summary: PASS=$PASS FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
