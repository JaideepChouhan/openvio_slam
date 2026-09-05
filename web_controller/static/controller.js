/**
 * controller.js  —  OpenVIO-SLAM Industrial Web Controller  (Protocol-Corrected v2)
 * ===================================================================================
 *
 * CHANGES FROM PARTS 1-3:
 *   • Compass widget uses REAL magnetometer data from {"type":"mag"} messages
 *   • Tilt-compensated heading computed from mag + latest roll/pitch
 *   • Mag XYZ values shown in telemetry panel
 *   • Accel chart replaced with Accel+Mag dual-tab chart
 *   • drawCompass() now uses mag heading, not odometry yaw
 *   • All other logic unchanged from Part 2 reference implementation
 */

'use strict';

// ── WebSocket ─────────────────────────────────────────────────────────────────
const WS_URL = `ws://${location.hostname}:8765`;
let ws, reconnectTimer;
let wsConnected = false;

function connect() {
    clearTimeout(reconnectTimer);
    ws = new WebSocket(WS_URL);
    ws.onopen    = onOpen;
    ws.onmessage = onMessage;
    ws.onerror   = ()   => setStatus('error');
    ws.onclose   = ()   => {
        wsConnected = false;
        setStatus('disconnected');
        log('SYS', 'Disconnected. Reconnecting in 2s…');
        reconnectTimer = setTimeout(connect, 2000);
    };
}

function onOpen() {
    wsConnected = true;
    setStatus('connected');
    log('SYS', `Connected to ${WS_URL}`);
}

function onMessage(e) {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    switch (msg.type) {
        case 'image':      handleImage(msg);      break;
        case 'imu':        handleIMU(msg);        break;
        case 'mag':        handleMag(msg);        break;
        case 'odom':       handleOdom(msg);       break;
        case 'mapstats':   handleMapStats(msg);   break;
        case 'diag':       handleDiag(msg);       break;
        case 'sys':        handleSys(msg);        break;
        case 'path_export':handlePathExport(msg); break;
    }
}

function wsSend(obj) {
    if (ws && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify(obj));
}

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(s) {
    const dot  = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    dot.className = 'status-dot ' + s;
    text.textContent = s.toUpperCase();
}

// ── Camera Feed ───────────────────────────────────────────────────────────────
const camCanvas = document.getElementById('cam-feed');
const camCtx    = camCanvas.getContext('2d');
let camFPS = 0, camFPSCount = 0, camFPSTimer = Date.now();

function handleImage(msg) {
    const img  = new Image();
    img.onload = () => {
        camCanvas.width  = img.width;
        camCanvas.height = img.height;
        camCtx.drawImage(img, 0, 0);
        drawCamOverlays(msg);
        trackCamFPS();
    };
    img.src = 'data:image/jpeg;base64,' + msg.data;
    document.getElementById('res-badge').textContent = `${msg.w}×${msg.h}`;
}

function drawCamOverlays(msg) {
    const latency = ((Date.now() / 1000) - msg.stamp) * 1000;
    document.getElementById('latency-display').textContent =
        `${Math.round(latency)}ms`;
}

function trackCamFPS() {
    camFPSCount++;
    const now = Date.now();
    if (now - camFPSTimer >= 1000) {
        camFPS = camFPSCount; camFPSCount = 0; camFPSTimer = now;
        document.getElementById('fps-badge').textContent = `${camFPS}fps`;
        document.getElementById('m-cam').textContent    = `${camFPS}Hz`;
    }
}

// ── IMU State (shared between IMU and Mag handlers) ───────────────────────────
// Used for tilt-compensated compass heading
let latestRoll  = 0;   // radians
let latestPitch = 0;   // radians
let imuHz = 0, imuCount = 0, lastImuTime = Date.now();
let imuThrottle = 0;

function handleIMU(msg) {
    // Track rate
    imuCount++;
    const now = Date.now();
    if (now - lastImuTime >= 1000) {
        imuHz = imuCount; imuCount = 0; lastImuTime = now;
        document.getElementById('m-imu').textContent = `${imuHz}Hz`;
    }

    // Compute roll and pitch from accelerometer for tilt compensation
    const ax = msg.ax, ay = msg.ay, az = msg.az;
    latestRoll  = Math.atan2(ay, az);
    latestPitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az));

    // ZUPT indicator: |a| ≈ 9.81 AND |ω| < 0.05
    const aMag = Math.sqrt(ax*ax + ay*ay + az*az);
    const gMag = Math.sqrt(msg.gx**2 + msg.gy**2 + msg.gz**2);
    const isZupt = Math.abs(aMag - 9.81) < 0.3 && gMag < 0.05;
    document.getElementById('zupt-badge').classList.toggle('hidden', !isZupt);

    // Charts (throttled to ~20Hz)
    imuThrottle++;
    if (imuThrottle % 5 !== 0) return;

    pushChart(gyroChart,  [msg.gx, msg.gy, msg.gz]);
    pushChart(accelChart, [ax, ay, az]);
}

// ── Magnetometer Handler ──────────────────────────────────────────────────────
let latestMagHeading = 0;   // degrees, tilt-compensated

function handleMag(msg) {
    // Tilt-compensated compass heading
    // Corrects for roll and pitch using latest IMU accelerometer data
    const mx = msg.mx, my = msg.my, mz = msg.mz;
    const r = latestRoll, p = latestPitch;

    // Tilt-compensated X and Y magnetic components
    const mx_comp =  mx * Math.cos(p)
                   + my * Math.sin(r) * Math.sin(p)
                   + mz * Math.cos(r) * Math.sin(p);
    const my_comp =  my * Math.cos(r)
                   - mz * Math.sin(r);

    // Heading: 0° = North, clockwise positive
    let heading = Math.degrees
        ? Math.degrees(Math.atan2(my_comp, mx_comp))
        : Math.atan2(my_comp, mx_comp) * 180 / Math.PI;
    heading = ((heading % 360) + 360) % 360;
    latestMagHeading = heading;

    // Update compass widget with real magnetic heading
    drawCompass(heading);

    // Update mag telemetry readouts
    const el = document.getElementById('mag-readout');
    if (el) {
        el.textContent =
            `Mx:${mx.toFixed(1)} My:${my.toFixed(1)} Mz:${mz.toFixed(1)} µT`;
    }
    const hdg_el = document.getElementById('mag-heading');
    if (hdg_el) {
        hdg_el.textContent = `${heading.toFixed(1)}°`;
    }

    // Push to mag chart if exists
    if (typeof magChart !== 'undefined') {
        pushChart(magChart, [mx, my, mz]);
    }
}

// ── Compass Widget (canvas-based, uses real mag heading) ──────────────────────
const compassCanvas = document.getElementById('compass-canvas');
const compassCtx    = compassCanvas
    ? compassCanvas.getContext('2d') : null;

function drawCompass(headingDeg) {
    if (!compassCtx) return;
    const canvas = compassCanvas;
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const r  = Math.min(W, H) / 2 - 8;

    compassCtx.clearRect(0, 0, W, H);

    // Outer ring
    compassCtx.strokeStyle = 'rgba(0,212,255,0.4)';
    compassCtx.lineWidth   = 2;
    compassCtx.beginPath();
    compassCtx.arc(cx, cy, r, 0, Math.PI * 2);
    compassCtx.stroke();

    // Cardinal labels
    const cardinals = [
        [0,   'N', '#ff4455'],
        [90,  'E', '#aaaaaa'],
        [180, 'S', '#aaaaaa'],
        [270, 'W', '#aaaaaa'],
    ];
    compassCtx.font = 'bold 11px JetBrains Mono, monospace';
    cardinals.forEach(([deg, label, color]) => {
        const rad = (deg - 90) * Math.PI / 180;
        const lx  = cx + (r - 14) * Math.cos(rad);
        const ly  = cy + (r - 14) * Math.sin(rad);
        compassCtx.fillStyle = color;
        compassCtx.textAlign = 'center';
        compassCtx.textBaseline = 'middle';
        compassCtx.fillText(label, lx, ly);
    });

    // Tick marks every 30°
    compassCtx.strokeStyle = 'rgba(0,212,255,0.25)';
    compassCtx.lineWidth   = 1;
    for (let deg = 0; deg < 360; deg += 30) {
        const rad  = (deg - 90) * Math.PI / 180;
        const r1   = r - 4, r2 = r - 10;
        compassCtx.beginPath();
        compassCtx.moveTo(cx + r1 * Math.cos(rad), cy + r1 * Math.sin(rad));
        compassCtx.lineTo(cx + r2 * Math.cos(rad), cy + r2 * Math.sin(rad));
        compassCtx.stroke();
    }

    // Needle — points to magnetic North
    const needleRad = (headingDeg - 90) * Math.PI / 180;
    const needleLen = r - 20;

    // North half (red)
    compassCtx.beginPath();
    compassCtx.moveTo(cx, cy);
    compassCtx.lineTo(
        cx + needleLen * Math.cos(needleRad),
        cy + needleLen * Math.sin(needleRad));
    compassCtx.strokeStyle = '#ff2244';
    compassCtx.lineWidth   = 3;
    compassCtx.lineCap     = 'round';
    compassCtx.stroke();

    // South half (cyan)
    compassCtx.beginPath();
    compassCtx.moveTo(cx, cy);
    compassCtx.lineTo(
        cx - (needleLen / 2) * Math.cos(needleRad),
        cy - (needleLen / 2) * Math.sin(needleRad));
    compassCtx.strokeStyle = '#00d4ff';
    compassCtx.lineWidth   = 3;
    compassCtx.stroke();

    // Center pivot
    compassCtx.beginPath();
    compassCtx.arc(cx, cy, 4, 0, Math.PI * 2);
    compassCtx.fillStyle = '#ffffff';
    compassCtx.fill();

    // Heading text below needle
    compassCtx.font      = '12px JetBrains Mono, monospace';
    compassCtx.fillStyle = '#00d4ff';
    compassCtx.textAlign = 'center';
    compassCtx.textBaseline = 'top';
    compassCtx.fillText(`${headingDeg.toFixed(1)}°`, cx, cy + r - 18);
}

// Initialize compass with 0° heading
document.addEventListener('DOMContentLoaded', () => {
    drawCompass(0);
});

// ── IMU Charts ────────────────────────────────────────────────────────────────
const CHART_WINDOW = 300;

const makeDatasets = (labels, colors) => labels.map((label, i) => ({
    label,
    data:         Array(CHART_WINDOW).fill(0),
    borderColor:  colors[i],
    borderWidth:  1.5,
    pointRadius:  0,
    tension:      0,
    fill:         false,
}));

const CHART_OPTS = () => ({
    type: 'line',
    data: {
        labels:   Array(CHART_WINDOW).fill(''),
        datasets: [],
    },
    options: {
        animation:   false,
        responsive:  true,
        maintainAspectRatio: false,
        interaction: { intersect: false },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color:     '#64748b',
                    font:      { family: 'JetBrains Mono', size: 10 },
                    boxWidth:  10,
                    padding:   8,
                },
            },
        },
        scales: {
            x: { display: false },
            y: {
                ticks: {
                    color: '#64748b',
                    font:  { family: 'JetBrains Mono', size: 9 },
                    maxTicksLimit: 4,
                },
                grid: { color: 'rgba(0,212,255,0.04)' },
            },
        },
    },
});

function makeChart(canvasId, labels, colors) {
    const el = document.getElementById(canvasId);
    if (!el) return null;
    const opts = CHART_OPTS();
    opts.data.datasets = makeDatasets(labels, colors);
    return new Chart(el, opts);
}

const gyroChart  = makeChart('gyro-chart',
    ['Gx', 'Gy', 'Gz'], ['#00d4ff', '#00ff88', '#ff6b00']);
const accelChart = makeChart('accel-chart',
    ['Ax', 'Ay', 'Az'], ['#00d4ff', '#00ff88', '#ff6b00']);
const magChart   = makeChart('mag-chart',
    ['Mx', 'My', 'Mz'], ['#8b5cf6', '#f472b6', '#fbbf24']);

function pushChart(chart, vals) {
    if (!chart) return;
    chart.data.datasets.forEach((ds, i) => {
        ds.data.push(vals[i]);
        ds.data.shift();
    });
    chart.update('none');
}

// ── Mini Map ──────────────────────────────────────────────────────────────────
const mapCanvas = document.getElementById('mini-map');
const mapCtx    = mapCanvas ? mapCanvas.getContext('2d') : null;
let   mapPath   = [];
let   mapRobot  = { x: 0, y: 0, yaw: 0 };

function handleOdom(msg) {
    mapPath.push({ x: msg.x, y: msg.y });
    if (mapPath.length > 5000) mapPath.shift();
    mapRobot = { x: msg.x, y: msg.y, yaw: msg.yaw * Math.PI / 180 };
    renderMap();

    document.getElementById('odom-x').textContent   = fmtSigned(msg.x, 3);
    document.getElementById('odom-y').textContent   = fmtSigned(msg.y, 3);
    document.getElementById('odom-yaw').textContent = fmtSigned(msg.yaw, 1);
    document.getElementById('odom-vx').textContent  = fmtSigned(msg.vx, 3);
}

function renderMap() {
    if (!mapCtx || mapPath.length < 2) return;
    const W = mapCanvas.width, H = mapCanvas.height;
    mapCtx.clearRect(0, 0, W, H);

    const xs = mapPath.map(p => p.x);
    const ys = mapPath.map(p => p.y);
    const minX = Math.min(...xs) - 1, maxX = Math.max(...xs) + 1;
    const minY = Math.min(...ys) - 1, maxY = Math.max(...ys) + 1;
    const scale = Math.min(W / (maxX - minX), H / (maxY - minY), 80);

    const toC = (x, y) => ({
        cx: (x - minX) * scale + (W - (maxX - minX) * scale) / 2,
        cy: H - ((y - minY) * scale + (H - (maxY - minY) * scale) / 2),
    });

    // Grid lines (1m spacing)
    mapCtx.strokeStyle = 'rgba(0,212,255,0.06)';
    mapCtx.lineWidth   = 1;
    for (let gx = Math.floor(minX); gx <= Math.ceil(maxX); gx++) {
        const { cx } = toC(gx, 0);
        mapCtx.beginPath(); mapCtx.moveTo(cx, 0); mapCtx.lineTo(cx, H); mapCtx.stroke();
    }
    for (let gy = Math.floor(minY); gy <= Math.ceil(maxY); gy++) {
        const { cy } = toC(0, gy);
        mapCtx.beginPath(); mapCtx.moveTo(0, cy); mapCtx.lineTo(W, cy); mapCtx.stroke();
    }

    // Trajectory
    mapCtx.beginPath();
    mapCtx.strokeStyle = 'rgba(0,212,255,0.7)';
    mapCtx.lineWidth   = 1.5;
    mapPath.forEach((p, i) => {
        const { cx, cy } = toC(p.x, p.y);
        i === 0 ? mapCtx.moveTo(cx, cy) : mapCtx.lineTo(cx, cy);
    });
    mapCtx.stroke();

    // Robot arrow
    const { cx, cy } = toC(mapRobot.x, mapRobot.y);
    mapCtx.save();
    mapCtx.translate(cx, cy);
    mapCtx.rotate(-mapRobot.yaw);
    mapCtx.fillStyle = '#ffffff';
    mapCtx.beginPath();
    mapCtx.moveTo(0, -8); mapCtx.lineTo(5, 6);
    mapCtx.lineTo(0, 3);  mapCtx.lineTo(-5, 6);
    mapCtx.closePath();
    mapCtx.fill();
    mapCtx.restore();
}

function handleMapStats(msg) {
    document.getElementById('m-pts').textContent = msg.n_points;
}

function handleSys(msg) {
    document.getElementById('m-cpu').textContent = `${Math.round(msg.cpu)}%`;
    document.getElementById('m-mem').textContent =
        `${(msg.mem_mb / 1024).toFixed(1)}G`;
    const recBtn = document.getElementById('btn-rec');
    if (msg.recording) {
        recBtn.classList.add('recording');
        recBtn.textContent = '■ STOP';
    } else {
        recBtn.classList.remove('recording');
        recBtn.textContent = '● REC';
    }
}

function handleDiag(msg) {
    msg.status.forEach(s => {
        const lvl = s.level === 0 ? 'INFO' : s.level === 1 ? 'WARN' : 'ERROR';
        log(lvl, `[DIAG] ${s.name}: ${s.message}`);
    });
}

// ── Keyboard control ──────────────────────────────────────────────────────────
const keyState = {};

document.addEventListener('keydown', e => {
    if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',
         'w','a','s','d','W','A','S','D',' '].includes(e.key))
        e.preventDefault();
    keyState[e.key] = true;
    applyKeyActions(e.key, e.ctrlKey);
    updateMotorFromKeys();
    updateDpadVisual();
});
document.addEventListener('keyup', e => {
    keyState[e.key] = false;
    updateMotorFromKeys();
    updateDpadVisual();
});

function applyKeyActions(key, ctrl) {
    if (key === 'f' || key === 'F') toggleFullscreen();
    if (key === 'd' || key === 'D') toggleConsole();
    if (key === 'c' || key === 'C') { wsSend({type:'recal'}); log('SYS','Recal IMU'); }
    if (key === 'e' || key === 'E') wsSend({type:'export_path'});
    if ((key === 's' || key === 'S') && ctrl) doScreenshot();
    if (key === '+' || key === '=') {
        speedSlider.value = Math.min(100, +speedSlider.value + 10);
        updateSpeedUI();
    }
    if (key === '-') {
        speedSlider.value = Math.max(0, +speedSlider.value - 10);
        updateSpeedUI();
    }
    if (key === 'r' || key === 'R') {
        showModal('Reset SLAM',
            'This will clear the current map. Continue?',
            () => { wsSend({type:'reset_slam'}); mapPath = []; renderMap(); });
    }
}

function updateMotorFromKeys() {
    const spd  = Math.round(speedSlider.value * 255 / 100);
    const fwd  = keyState['ArrowUp']    || keyState['w'] || keyState['W'];
    const bwd  = keyState['ArrowDown']  || keyState['s'] || keyState['S'];
    const lft  = keyState['ArrowLeft']  || keyState['a'] || keyState['A'];
    const rgt  = keyState['ArrowRight'] || keyState['d'] || keyState['D'];
    const stop = keyState[' '];

    let L = 0, R = 0;
    if (fwd)  { L += spd;        R += spd;        }
    if (bwd)  { L -= spd;        R -= spd;        }
    if (lft)  { L -= spd * 0.6; R += spd * 0.6; }
    if (rgt)  { L += spd * 0.6; R -= spd * 0.6; }
    if (stop) { L = 0; R = 0; }

    L = Math.max(-255, Math.min(255, Math.round(L)));
    R = Math.max(-255, Math.min(255, Math.round(R)));
    sendMotor(L, R);
}

function updateDpadVisual() {
    document.getElementById('btn-fwd').classList.toggle('active',
        !!(keyState['ArrowUp']    || keyState['w'] || keyState['W']));
    document.getElementById('btn-bwd').classList.toggle('active',
        !!(keyState['ArrowDown']  || keyState['s'] || keyState['S']));
    document.getElementById('btn-lft').classList.toggle('active',
        !!(keyState['ArrowLeft']  || keyState['a'] || keyState['A']));
    document.getElementById('btn-rgt').classList.toggle('active',
        !!(keyState['ArrowRight'] || keyState['d'] || keyState['D']));
}

// D-pad buttons
['btn-fwd','btn-bwd','btn-lft','btn-rgt','btn-stp'].forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    const keyMap = {
        'btn-fwd': 'ArrowUp', 'btn-bwd': 'ArrowDown',
        'btn-lft': 'ArrowLeft', 'btn-rgt': 'ArrowRight',
    };
    const press   = () => {
        if (id === 'btn-stp') { sendMotor(0, 0); return; }
        keyState[keyMap[id]] = true; updateMotorFromKeys();
    };
    const release = () => {
        if (keyMap[id]) delete keyState[keyMap[id]];
        updateMotorFromKeys();
    };
    btn.addEventListener('mousedown',  press);
    btn.addEventListener('mouseup',    release);
    btn.addEventListener('mouseleave', release);
    btn.addEventListener('touchstart', e => { e.preventDefault(); press(); });
    btn.addEventListener('touchend',   e => { e.preventDefault(); release(); });
});

// ── Joystick ──────────────────────────────────────────────────────────────────
const joystickZone = document.getElementById('joystick-zone');
if (joystickZone && typeof nipplejs !== 'undefined') {
    const joystick = nipplejs.create({
        zone:     joystickZone,
        mode:     'static',
        position: { left: '50%', top: '50%' },
        color:    'rgba(0,212,255,0.5)',
        size:     90,
    });
    joystick.on('move', (evt, data) => {
        const angle = data.angle.radian;
        const force = Math.min(data.force, 1.0);
        const spd   = Math.round(speedSlider.value * 255 / 100 * force);
        const L = Math.round(spd * (Math.sin(angle) + Math.cos(angle)));
        const R = Math.round(spd * (Math.sin(angle) - Math.cos(angle)));
        sendMotor(
            Math.max(-255, Math.min(255, L)),
            Math.max(-255, Math.min(255, R)));
    });
    joystick.on('end', () => sendMotor(0, 0));
}

// ── Motor rate limiting (20 Hz max) ──────────────────────────────────────────
let lastMotorSend = 0;
function sendMotor(L, R) {
    const now = Date.now();
    if (now - lastMotorSend >= 50) {
        wsSend({type:'motor', left:L, right:R});
        lastMotorSend = now;
    }
}

// ── Speed Slider ──────────────────────────────────────────────────────────────
const speedSlider = document.getElementById('speed-slider');
function updateSpeedUI() {
    const v = speedSlider.value;
    document.getElementById('speed-val').textContent  = `${v}%`;
    document.getElementById('speed-fill').style.width = `${v}%`;
}
speedSlider.addEventListener('input', () => { updateSpeedUI(); updateMotorFromKeys(); });
updateSpeedUI();

// ── Action Buttons ────────────────────────────────────────────────────────────
let isRecording = false;

document.getElementById('btn-recal')?.addEventListener('click', () => {
    wsSend({type:'recal'});
    log('SYS','IMU recalibration requested — keep robot stationary!');
});
document.getElementById('btn-reset')?.addEventListener('click', () =>
    showModal('Reset SLAM',
        'This clears the map and restarts odometry. Continue?',
        () => { wsSend({type:'reset_slam'}); mapPath=[]; renderMap(); }));
document.getElementById('btn-export')?.addEventListener('click', () =>
    wsSend({type:'export_path'}));
document.getElementById('btn-shot')?.addEventListener('click', doScreenshot);
document.getElementById('btn-rec')?.addEventListener('click', () => {
    isRecording = !isRecording;
    wsSend({type: isRecording ? 'start_record' : 'stop_record'});
    log('SYS', isRecording ? 'Recording started' : 'Recording stopped');
});
document.getElementById('btn-shutdown')?.addEventListener('click', () =>
    showModal('Shutdown',
        '⚠️  Power off the Raspberry Pi?',
        () => wsSend({type:'shutdown'})));
document.getElementById('map-reset-btn')?.addEventListener('click', () => {
    mapPath = []; renderMap(); log('SYS', 'Mini-map reset');
});
document.getElementById('fullscreen-btn')?.addEventListener('click', toggleFullscreen);

// ── Screenshot ────────────────────────────────────────────────────────────────
function doScreenshot() {
    camCanvas.toBlob(blob => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `slam_${Date.now()}.png`;
        a.click();
        log('SYS', 'Screenshot saved');
    }, 'image/png');
}

function handlePathExport(msg) {
    const blob = new Blob([msg.csv], {type:'text/csv'});
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = `path_${Date.now()}.csv`;
    a.click();
    log('SYS', `Path exported (${msg.csv.split('\n').length-1} points)`);
}

// ── Fullscreen ────────────────────────────────────────────────────────────────
function toggleFullscreen() {
    const el = document.querySelector('.cam-panel');
    if (!document.fullscreenElement) el?.requestFullscreen?.();
    else document.exitFullscreen?.();
}

// ── Debug Console ─────────────────────────────────────────────────────────────
const consoleBody = document.getElementById('console-body');
const consoleWrap = document.getElementById('console-wrap');
let   autoScroll  = true;
let   warnCount   = 0;
let   consoleOpen = true;

function log(level, msg) {
    if (level === 'WARN' || level === 'ERROR') {
        warnCount++;
        const el = document.getElementById('warn-count');
        if (el) el.textContent = `${warnCount} ${level === 'ERROR' ? 'err' : 'warn'}`;
    }
    const ts   = new Date().toISOString().substr(11, 12);
    const line = document.createElement('div');
    line.className   = `log-line log-${level.toLowerCase()}`;
    line.textContent = `${ts} [${level}] ${msg}`;
    line.dataset.raw = `${level} ${msg}`;

    const f = document.getElementById('console-filter')?.value?.toLowerCase() || '';
    if (f && !line.dataset.raw.toLowerCase().includes(f))
        line.style.display = 'none';

    if (consoleBody) {
        consoleBody.appendChild(line);
        while (consoleBody.children.length > 200)
            consoleBody.removeChild(consoleBody.firstChild);
        if (autoScroll) consoleBody.scrollTop = consoleBody.scrollHeight;
    }
}

consoleBody?.addEventListener('scroll', () => {
    const el = consoleBody;
    autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
});

document.getElementById('console-toggle')?.addEventListener('click', toggleConsole);
function toggleConsole() {
    consoleOpen = !consoleOpen;
    consoleWrap?.classList.toggle('collapsed', !consoleOpen);
}

document.getElementById('console-clear')?.addEventListener('click', e => {
    e.stopPropagation();
    if (consoleBody) consoleBody.innerHTML = '';
    warnCount = 0;
    const el = document.getElementById('warn-count');
    if (el) el.textContent = '0 warn';
});

document.getElementById('console-filter')?.addEventListener('input', function () {
    const f = this.value.toLowerCase();
    [...(consoleBody?.children || [])].forEach(line => {
        line.style.display =
            (!f || line.dataset.raw.toLowerCase().includes(f)) ? '' : 'none';
    });
});

// ── Modal ─────────────────────────────────────────────────────────────────────
let modalCb = null;
function showModal(title, msg, onConfirm) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-msg').textContent   = msg;
    document.getElementById('modal')?.classList.remove('hidden');
    modalCb = onConfirm;
}
document.getElementById('modal-confirm')?.addEventListener('click', () => {
    document.getElementById('modal')?.classList.add('hidden');
    if (modalCb) { modalCb(); modalCb = null; }
});
document.getElementById('modal-cancel')?.addEventListener('click', () => {
    document.getElementById('modal')?.classList.add('hidden');
    modalCb = null;
});

// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(() => {
    const el = document.getElementById('ros-time');
    if (el) el.textContent = new Date().toISOString().substr(11, 8);
}, 1000);

// ── Utility ───────────────────────────────────────────────────────────────────
function fmtSigned(v, dp) {
    const s = v.toFixed(dp);
    return v >= 0 ? '+' + s : s;
}

// ── Init ──────────────────────────────────────────────────────────────────────
log('SYS', `OpenVIO-SLAM Controller v2 (9-DOF + tilt-compensated compass)`);
log('SYS', `Connecting to ${WS_URL}…`);
log('SYS', `Keys: WASD/arrows=drive  Space=stop  C=recalIMU  R=resetSLAM  F=fullscreen`);
connect();
