const socket = new WebSocket(`ws://${location.hostname}:8765`);
const state = {
  speed: 0.6,
  imuValues: { gx: [], gy: [], gz: [], ax: [], ay: [], az: [] },
  currentImageStamp: 0,
  lastMessageTs: performance.now(),
  yaw: 0,
};

const logEl = document.getElementById('console-log');
const speedFill = document.getElementById('speed-fill');
const fpsBadge = document.getElementById('fps-badge');
const latencyBadge = document.getElementById('latency-badge');
const statusDot = document.getElementById('status-dot');
const connectionState = document.getElementById('connection-state');

function logLine(text) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${text}\n` + logEl.textContent.slice(0, 2000);
}

function updateSpeedMeter() {
  speedFill.style.width = `${Math.round(state.speed * 100)}%`;
}

socket.addEventListener('open', () => {
  statusDot.classList.add('connected');
  statusDot.classList.remove('disconnected');
  connectionState.textContent = 'CONNECTED';
  logLine('WebSocket connected.');
});

socket.addEventListener('close', () => {
  statusDot.classList.add('disconnected');
  statusDot.classList.remove('connected');
  connectionState.textContent = 'DISCONNECTED';
  logLine('WebSocket closed. Retrying...');
});

socket.addEventListener('error', () => {
  logLine('WebSocket error.');
});

socket.addEventListener('message', (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'image') {
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById('cam-feed');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.9)';
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 0);
      ctx.lineTo(canvas.width / 2, canvas.height);
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      const now = performance.now();
      const latency = Math.max(0, now - state.lastMessageTs);
      latencyBadge.textContent = `LAT ${Math.round(latency)} ms`;
      state.lastMessageTs = now;
    };
    img.src = 'data:image/jpeg;base64,' + msg.data;
    const now = performance.now();
    fpsBadge.textContent = `FPS ${Math.round(1000 / Math.max(1, now - state.currentImageStamp || 1))}`;
    state.currentImageStamp = now;
    return;
  }

  if (msg.type === 'imu') {
    const list = [msg.gx, msg.gy, msg.gz];
    state.imuValues.gx.push(msg.gx);
    state.imuValues.gy.push(msg.gy);
    state.imuValues.gz.push(msg.gz);
    state.imuValues.ax.push(msg.ax);
    state.imuValues.ay.push(msg.ay);
    state.imuValues.az.push(msg.az);
    for (const key of Object.keys(state.imuValues)) {
      if (state.imuValues[key].length > 120) {
        state.imuValues[key].shift();
      }
    }
    return;
  }

  if (msg.type === 'odom') {
    document.getElementById('odom-x').textContent = `${msg.x.toFixed(2)}m`;
    document.getElementById('odom-y').textContent = `${msg.y.toFixed(2)}m`;
    document.getElementById('odom-yaw').textContent = `${(msg.yaw * 180 / Math.PI).toFixed(1)}°`;
    document.getElementById('odom-vx').textContent = `${msg.vx.toFixed(2)}`;
    state.yaw = msg.yaw;
    return;
  }

  if (msg.type === 'mapstats') {
    document.getElementById('map-points').textContent = `${msg.n_points}`;
    return;
  }

  if (msg.type === 'sys') {
    document.getElementById('cpu-value').textContent = `CPU: ${msg.cpu.toFixed(0)}%`;
    document.getElementById('mem-value').textContent = `MEM: ${(msg.mem_mb / 1024).toFixed(1)}GB`;
    document.getElementById('imu-hz').textContent = `IMU: ${msg.ros_hz_imu.toFixed(1)}Hz`;
    document.getElementById('cam-hz').textContent = `CAM: ${msg.ros_hz_img.toFixed(1)}Hz`;
    return;
  }

  if (msg.type === 'diag') {
    logLine(`diag: ${JSON.stringify(msg.status).slice(0, 180)}`);
  }
});

function sendMotor(left, right) {
  socket.send(JSON.stringify({ type: 'motor', left, right }));
}

function handleKeyState(key, pressed) {
  const map = {
    w: [1, 1],
    s: [-1, -1],
    a: [-1, 1],
    d: [1, -1],
  };
  if (!pressed || !(key in map)) return;
  const scale = 255 * state.speed;
  const [l, r] = map[key].map(v => Math.round(v * scale));
  sendMotor(l, r);
}

window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  if (key === ' ') {
    event.preventDefault();
    sendMotor(0, 0);
    return;
  }
  if (key === 'r') {
    socket.send(JSON.stringify({ type: 'reset_slam' }));
    logLine('Reset SLAM requested.');
    return;
  }
  if (key === 'c') {
    socket.send(JSON.stringify({ type: 'recal' }));
    logLine('Recalibration requested.');
    return;
  }
  if (key === 'e') {
    socket.send(JSON.stringify({ type: 'stop' }));
    logLine('Emergency stop sent.');
    return;
  }
  handleKeyState(key, true);
});

window.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (key in { w:1, a:1, s:1, d:1 }) {
    sendMotor(0, 0);
  }
});

document.getElementById('recal-btn').addEventListener('click', () => socket.send(JSON.stringify({ type: 'recal' })));
document.getElementById('stop-btn').addEventListener('click', () => sendMotor(0, 0));
document.getElementById('reset-btn').addEventListener('click', () => socket.send(JSON.stringify({ type: 'reset_slam' })));

const gyroChart = new Chart(document.getElementById('gyroChart'), {
  type: 'line',
  data: { labels: Array(60).fill(''), datasets: [
    { label: 'gx', borderColor: '#00d4ff', data: Array(60).fill(0), tension: 0.2 },
    { label: 'gy', borderColor: '#00ff88', data: Array(60).fill(0), tension: 0.2 },
    { label: 'gz', borderColor: '#ff6b00', data: Array(60).fill(0), tension: 0.2 },
  ] },
  options: { animation: false, responsive: true, maintainAspectRatio: false, scales: { y: { min: -2, max: 2 } } }
});

const accelChart = new Chart(document.getElementById('accelChart'), {
  type: 'line',
  data: { labels: Array(60).fill(''), datasets: [
    { label: 'ax', borderColor: '#00d4ff', data: Array(60).fill(0), tension: 0.2 },
    { label: 'ay', borderColor: '#00ff88', data: Array(60).fill(0), tension: 0.2 },
    { label: 'az', borderColor: '#ff6b00', data: Array(60).fill(0), tension: 0.2 },
  ] },
  options: { animation: false, responsive: true, maintainAspectRatio: false, scales: { y: { min: -12, max: 12 } } }
});

setInterval(() => {
  const gx = state.imuValues.gx.slice(-60);
  const gy = state.imuValues.gy.slice(-60);
  const gz = state.imuValues.gz.slice(-60);
  const ax = state.imuValues.ax.slice(-60);
  const ay = state.imuValues.ay.slice(-60);
  const az = state.imuValues.az.slice(-60);

  gyroChart.data.datasets[0].data = gx.length ? gx : Array(60).fill(0);
  gyroChart.data.datasets[1].data = gy.length ? gy : Array(60).fill(0);
  gyroChart.data.datasets[2].data = gz.length ? gz : Array(60).fill(0);
  accelChart.data.datasets[0].data = ax.length ? ax : Array(60).fill(0);
  accelChart.data.datasets[1].data = ay.length ? ay : Array(60).fill(0);
  accelChart.data.datasets[2].data = az.length ? az : Array(60).fill(0);
  gyroChart.update();
  accelChart.update();
}, 200);

updateSpeedMeter();
