/**
 * robot_controller.ino  —  OpenVIO-SLAM Robot Controller  (Protocol-Corrected v2)
 * ==================================================================================
 *
 * PROTOCOL CORRECTION:
 * Parts 1-3 of the spec described a 31-byte binary protocol.
 * Your actual firmware (confirmed by visualizer.py) uses:
 *
 *   SENDING  → Pi: 43-byte binary packet (9-DOF: gyro + accel + mag)
 *   RECEIVING← Pi: ASCII text commands ("M <L> <R>\n" and "CAL\n")
 *
 * This file is the COMPLETE corrected firmware. It matches your visualizer.py
 * exactly so they work together without any changes on either side.
 *
 * ── PACKET FORMAT (SENDING, 43 bytes) ─────────────────────────────────────────
 *
 *   Byte  0:     0xAA  (header 1)
 *   Byte  1:     0x55  (header 2)
 *   Bytes 2–5:   uint32_t  millis()         (little-endian)
 *   Bytes 6–9:   float     gyro_x  (rad/s)  (little-endian)
 *   Bytes 10–13: float     gyro_y  (rad/s)
 *   Bytes 14–17: float     gyro_z  (rad/s)
 *   Bytes 18–21: float     accel_x (m/s²)
 *   Bytes 22–25: float     accel_y (m/s²)
 *   Bytes 26–29: float     accel_z (m/s²)
 *   Bytes 30–33: float     mag_x   (µT)
 *   Bytes 34–37: float     mag_y   (µT)
 *   Bytes 38–41: float     mag_z   (µT)
 *   Byte  42:    uint8_t   checksum (XOR of bytes 2..41)
 *
 * ── COMMAND FORMAT (RECEIVING FROM Pi) ────────────────────────────────────────
 *
 *   Motor:        "M <left> <right>\n"   e.g. "M 200 -150\n"
 *   Calibrate:    "CAL\n"
 *   Values: integers -255..255
 *
 * ── HARDWARE ──────────────────────────────────────────────────────────────────
 *
 *   Arduino UNO
 *   MPU-9250 (I2C: SDA=A4, SCL=A5, AD0=GND → addr 0x68)
 *   AK8963 magnetometer (inside MPU-9250, accessed via I2C bypass)
 *   Cytron MDD20A:  LEFT (PWM=5, DIR=4)  RIGHT (PWM=6, DIR=7)
 *   USB Serial to Raspberry Pi 5 @ 115200 baud
 *
 * ── AXIS MOUNTING ASSUMPTION ──────────────────────────────────────────────────
 *
 *   MPU-9250 mounted flat on robot chassis.
 *   Chip X-axis pointing FORWARD (toward phone camera mount).
 *   ROS convention: X=forward, Y=left, Z=up.
 *   With this mounting: chip X = robot X (forward). No remapping needed.
 *   If mounted differently: adjust remap() function.
 */

#include <Wire.h>
#include <EEPROM.h>
#include <math.h>

// ─── Motor pins (Cytron MDD20A) ───────────────────────────────────────────────
#define PIN_LEFT_PWM   5
#define PIN_LEFT_DIR   4
#define PIN_RIGHT_PWM  6
#define PIN_RIGHT_DIR  7

// ─── MPU-9250 I2C addresses ───────────────────────────────────────────────────
#define MPU_ADDR       0x68   // AD0 = GND
#define AK8963_ADDR    0x0C   // Magnetometer inside MPU-9250

// ─── MPU-9250 register map ────────────────────────────────────────────────────
#define REG_SMPLRT_DIV  0x19
#define REG_CONFIG      0x1A
#define REG_GYRO_CFG    0x1B
#define REG_ACCEL_CFG   0x1C
#define REG_ACCEL_CFG2  0x1D
#define REG_INT_PIN     0x37
#define REG_INT_EN      0x38
#define REG_ACCEL_XOUT  0x3B   // 6 bytes: accel XYZ
#define REG_TEMP_OUT    0x41   // 2 bytes
#define REG_GYRO_XOUT   0x43   // 6 bytes: gyro XYZ
#define REG_USER_CTRL   0x6A
#define REG_PWR_MGMT_1  0x6B
#define REG_I2C_MST_CTRL 0x24
#define REG_INT_BYPASS  0x37

// ─── AK8963 magnetometer registers ───────────────────────────────────────────
#define AK_WIA         0x00   // Who Am I → should be 0x48
#define AK_HXL         0x03   // X-axis Low byte (start of 7-byte burst read)
#define AK_ST2         0x09   // Status 2 (must be read to unlock next measurement)
#define AK_CNTL        0x0A   // Control
#define AK_CNTL2       0x0B   // Control 2
#define AK_ASAX        0x10   // X sensitivity adjustment

// ─── Conversion constants ─────────────────────────────────────────────────────
// Gyro ±2000 dps → 16.4 LSB/(°/s) → rad/s
#define GYRO_SCALE    (1.0f / 16.4f * (float)M_PI / 180.0f)
// Accel ±8g → 4096 LSB/g → m/s²
#define ACCEL_SCALE   (1.0f / 4096.0f * 9.80665f)
// Magnetometer: 14-bit output → 0.6 µT/LSB (with ASAX correction applied)
#define MAG_SCALE     0.6f

// ─── EEPROM layout (gyro bias persistence) ───────────────────────────────────
// Bytes 0–11:  float bias_gx, bias_gy, bias_gz
// Bytes 12–15: uint32 magic marker
#define EEPROM_MAGIC_VAL  0xDEADBEEF
#define EEPROM_BIAS_BASE  0
#define EEPROM_MAGIC_ADDR 12

// ─── Timing ───────────────────────────────────────────────────────────────────
#define IMU_INTERVAL_MS    10     // 100 Hz
#define WATCHDOG_MS        500    // stop motors if no command

// ─── Global state ─────────────────────────────────────────────────────────────
float bias_gx = 0, bias_gy = 0, bias_gz = 0;  // gyro bias (raw LSB units)
float mag_asa_x = 1.0f, mag_asa_y = 1.0f, mag_asa_z = 1.0f;  // sensitivity adj

unsigned long last_imu_ms    = 0;
unsigned long last_motor_ms  = 0;
int  motor_left   = 0;
int  motor_right  = 0;
bool calibrating  = false;

// Raw sensor readings
int16_t raw_ax, raw_ay, raw_az;
int16_t raw_gx, raw_gy, raw_gz;
int16_t raw_mx, raw_my, raw_mz;
bool    mag_ready = false;

// ASCII command parser (non-blocking)
char   cmd_buf[32];
int    cmd_len = 0;

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);

    // Motors
    pinMode(PIN_LEFT_PWM,  OUTPUT);
    pinMode(PIN_LEFT_DIR,  OUTPUT);
    pinMode(PIN_RIGHT_PWM, OUTPUT);
    pinMode(PIN_RIGHT_DIR, OUTPUT);
    set_motor(0, 0);

    Wire.begin();
    Wire.setClock(400000);

    // Init MPU-9250 (and embedded AK8963)
    if (!init_mpu9250()) {
        // Fatal: blink error code
        while (true) {
            digitalWrite(LED_BUILTIN, HIGH); delay(200);
            digitalWrite(LED_BUILTIN, LOW);  delay(200);
        }
    }

    // Load or compute gyro bias
    if (!load_bias_eeprom()) {
        calibrate_gyro();
    }

    last_imu_ms   = millis();
    last_motor_ms = millis();
    digitalWrite(LED_BUILTIN, LOW);
}

// =============================================================================
// LOOP
// =============================================================================
void loop() {
    unsigned long now = millis();

    // ── 100 Hz IMU send ───────────────────────────────────────────────────────
    if (!calibrating && (now - last_imu_ms) >= IMU_INTERVAL_MS) {
        last_imu_ms = now;
        read_mpu();
        read_ak8963();

        float gx, gy, gz, ax, ay, az, mx, my, mz;
        convert_imu(&gx, &gy, &gz, &ax, &ay, &az);
        convert_mag(&mx, &my, &mz);
        send_packet(now, gx, gy, gz, ax, ay, az, mx, my, mz);
    }

    // ── Motor watchdog ────────────────────────────────────────────────────────
    if ((now - last_motor_ms) >= WATCHDOG_MS) {
        if (motor_left != 0 || motor_right != 0) {
            motor_left  = 0;
            motor_right = 0;
            set_motor(0, 0);
        }
    }

    // ── Parse ASCII commands from Pi (non-blocking) ───────────────────────────
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\n') {
            cmd_buf[cmd_len] = '\0';
            process_command(cmd_buf);
            cmd_len = 0;
        } else if (cmd_len < 31) {
            cmd_buf[cmd_len++] = c;
        }
    }
}

// =============================================================================
// COMMAND PARSER (ASCII text from Pi)
// =============================================================================
void process_command(const char* cmd) {
    // Motor command: "M <left> <right>"
    if (cmd[0] == 'M' && cmd[1] == ' ') {
        int L, R;
        if (sscanf(cmd + 2, "%d %d", &L, &R) == 2) {
            L = constrain(L, -255, 255);
            R = constrain(R, -255, 255);
            motor_left  = L;
            motor_right = R;
            last_motor_ms = millis();
            set_motor(L, R);
        }
        return;
    }

    // Calibration command: "CAL"
    if (strncmp(cmd, "CAL", 3) == 0) {
        calibrate_gyro();
        return;
    }
}

// =============================================================================
// MPU-9250 INIT
// =============================================================================
bool init_mpu9250() {
    // Check WHO_AM_I
    uint8_t who = mpu_read(REG_PWR_MGMT_1 - 1 + 0x75 - 0x6B);
    // WHO_AM_I is at 0x75; use direct read
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x75);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
    who = Wire.available() ? Wire.read() : 0;

    if (who != 0x71 && who != 0x70 && who != 0x68 && who != 0x73) {
        // Unknown WHO_AM_I — warn but continue
        // (some MPU-6500 clones return different values)
    }

    // Wake up, PLL clock from gyro
    mpu_write(REG_PWR_MGMT_1, 0x01);
    delay(100);

    // Gyro: ±2000 dps
    mpu_write(REG_GYRO_CFG, 0x18);
    // Accel: ±8g
    mpu_write(REG_ACCEL_CFG, 0x10);
    // Accel DLPF 20 Hz
    mpu_write(REG_ACCEL_CFG2, 0x04);
    // Gyro DLPF 20 Hz
    mpu_write(REG_CONFIG, 0x04);
    // Sample rate: 1000/(1+4) = 200 Hz internal
    mpu_write(REG_SMPLRT_DIV, 0x04);
    // Enable I2C bypass so we can talk to AK8963 directly
    mpu_write(0x37, 0x02);   // INT_PIN_CFG: BYPASS_EN = 1

    delay(50);

    // Init AK8963 (magnetometer)
    return init_ak8963();
}

bool init_ak8963() {
    // Check WHO_AM_I
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(AK_WIA);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)AK8963_ADDR, (uint8_t)1);
    if (!Wire.available()) return false;
    uint8_t wia = Wire.read();
    if (wia != 0x48) {
        // AK8963 not found (some MPU-6500 boards don't have magnetometer)
        // Continue without mag — values will be 0.0
        return true;
    }

    // Reset
    ak_write(AK_CNTL2, 0x01);
    delay(10);

    // Read sensitivity adjustment values (Fuse ROM mode)
    ak_write(AK_CNTL, 0x0F);   // Fuse ROM access mode
    delay(10);
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(AK_ASAX);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)AK8963_ADDR, (uint8_t)3);
    if (Wire.available() >= 3) {
        uint8_t asa_x = Wire.read();
        uint8_t asa_y = Wire.read();
        uint8_t asa_z = Wire.read();
        // Correction formula from AK8963 datasheet
        mag_asa_x = ((float)(asa_x - 128) / 256.0f) + 1.0f;
        mag_asa_y = ((float)(asa_y - 128) / 256.0f) + 1.0f;
        mag_asa_z = ((float)(asa_z - 128) / 256.0f) + 1.0f;
    }

    // Power down, then continuous measurement mode 2 (100 Hz, 16-bit)
    ak_write(AK_CNTL, 0x00);
    delay(10);
    ak_write(AK_CNTL, 0x16);   // 16-bit, continuous mode 2 (100 Hz)
    delay(10);
    return true;
}

// =============================================================================
// SENSOR READS
// =============================================================================
void read_mpu() {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(REG_ACCEL_XOUT);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);

    raw_ax = (int16_t)((Wire.read() << 8) | Wire.read());
    raw_ay = (int16_t)((Wire.read() << 8) | Wire.read());
    raw_az = (int16_t)((Wire.read() << 8) | Wire.read());
    Wire.read(); Wire.read();   // skip temperature
    raw_gx = (int16_t)((Wire.read() << 8) | Wire.read());
    raw_gy = (int16_t)((Wire.read() << 8) | Wire.read());
    raw_gz = (int16_t)((Wire.read() << 8) | Wire.read());
}

void read_ak8963() {
    // Check data-ready bit in ST1 register (0x02)
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(0x02);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)AK8963_ADDR, (uint8_t)1);
    if (!Wire.available()) return;
    uint8_t st1 = Wire.read();
    if (!(st1 & 0x01)) return;   // not ready

    // Read 7 bytes: HXL, HXH, HYL, HYH, HZL, HZH, ST2
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(AK_HXL);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)AK8963_ADDR, (uint8_t)7);
    if (Wire.available() < 7) return;

    uint8_t buf[7];
    for (int i = 0; i < 7; i++) buf[i] = Wire.read();

    uint8_t st2 = buf[6];
    if (st2 & 0x08) return;   // magnetic overflow

    raw_mx = (int16_t)((buf[1] << 8) | buf[0]);
    raw_my = (int16_t)((buf[3] << 8) | buf[2]);
    raw_mz = (int16_t)((buf[5] << 8) | buf[4]);
    mag_ready = true;
}

// =============================================================================
// UNIT CONVERSION
// =============================================================================
void convert_imu(float* gx, float* gy, float* gz,
                  float* ax, float* ay, float* az) {
    // Remove gyro bias (in raw LSB units), then scale to rad/s
    *gx = (raw_gx - bias_gx) * GYRO_SCALE;
    *gy = (raw_gy - bias_gy) * GYRO_SCALE;
    *gz = (raw_gz - bias_gz) * GYRO_SCALE;

    // Scale accel to m/s² (no bias removal — MSCKF handles accel bias)
    *ax = raw_ax * ACCEL_SCALE;
    *ay = raw_ay * ACCEL_SCALE;
    *az = raw_az * ACCEL_SCALE;

    // ── Axis remapping ────────────────────────────────────────────────────────
    // Default: no remapping (chip X = robot forward, chip Y = left, chip Z = up)
    // If your mounting is different, apply rotation here.
    // Example for 90° rotation around Z (chip X = robot left):
    //   float tmp = *gx; *gx = -*gy; *gy = tmp;
    //   tmp = *ax;       *ax = -*ay; *ay = tmp;
}

void convert_mag(float* mx, float* my, float* mz) {
    if (!mag_ready) {
        *mx = 0.0f; *my = 0.0f; *mz = 0.0f;
        return;
    }
    // Apply sensitivity correction and scale to µT
    // AK8963 is Little-endian, X/Y are swapped relative to MPU axes
    *mx = raw_mx * mag_asa_x * MAG_SCALE;
    *my = raw_my * mag_asa_y * MAG_SCALE;
    *mz = raw_mz * mag_asa_z * MAG_SCALE;
    mag_ready = false;
}

// =============================================================================
// SEND 43-BYTE PACKET
// Matches visualizer.py parse_packet() exactly
// =============================================================================
void send_packet(uint32_t ts_ms,
                  float gx, float gy, float gz,
                  float ax, float ay, float az,
                  float mx, float my, float mz) {
    uint8_t pkt[43];
    pkt[0] = 0xAA;
    pkt[1] = 0x55;

    memcpy(&pkt[2],  &ts_ms, 4);
    memcpy(&pkt[6],  &gx, 4);
    memcpy(&pkt[10], &gy, 4);
    memcpy(&pkt[14], &gz, 4);
    memcpy(&pkt[18], &ax, 4);
    memcpy(&pkt[22], &ay, 4);
    memcpy(&pkt[26], &az, 4);
    memcpy(&pkt[30], &mx, 4);
    memcpy(&pkt[34], &my, 4);
    memcpy(&pkt[38], &mz, 4);

    // Checksum: XOR of bytes 2..41 (matches visualizer.py range(2,42))
    uint8_t chk = 0;
    for (int i = 2; i <= 41; i++) chk ^= pkt[i];
    pkt[42] = chk;

    Serial.write(pkt, 43);
}

// =============================================================================
// MOTOR CONTROL
// =============================================================================
void set_motor(int left, int right) {
    // Deadband ±10
    if (abs(left)  < 10) left  = 0;
    if (abs(right) < 10) right = 0;

    left  = constrain(left,  -255, 255);
    right = constrain(right, -255, 255);

    // LEFT motor
    digitalWrite(PIN_LEFT_DIR, left >= 0 ? HIGH : LOW);
    analogWrite(PIN_LEFT_PWM, abs(left));

    // RIGHT motor
    // ⚠️  If the right motor spins backward: invert the HIGH/LOW below
    digitalWrite(PIN_RIGHT_DIR, right >= 0 ? HIGH : LOW);
    analogWrite(PIN_RIGHT_PWM, abs(right));
}

// =============================================================================
// GYRO BIAS CALIBRATION
// =============================================================================
void calibrate_gyro() {
    calibrating = true;
    set_motor(0, 0);

    const int N = 1000;
    long gx_sum = 0, gy_sum = 0, gz_sum = 0;

    for (int i = 0; i < N; i++) {
        Wire.beginTransmission(MPU_ADDR);
        Wire.write(REG_GYRO_XOUT);
        Wire.endTransmission(false);
        Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6);
        int16_t gx_raw = (int16_t)((Wire.read() << 8) | Wire.read());
        int16_t gy_raw = (int16_t)((Wire.read() << 8) | Wire.read());
        int16_t gz_raw = (int16_t)((Wire.read() << 8) | Wire.read());
        gx_sum += gx_raw;
        gy_sum += gy_raw;
        gz_sum += gz_raw;
        delay(2);
    }

    bias_gx = (float)gx_sum / N;
    bias_gy = (float)gy_sum / N;
    bias_gz = (float)gz_sum / N;

    save_bias_eeprom();
    calibrating = false;
}

// =============================================================================
// EEPROM
// =============================================================================
void save_bias_eeprom() {
    EEPROM.put(EEPROM_BIAS_BASE + 0, bias_gx);
    EEPROM.put(EEPROM_BIAS_BASE + 4, bias_gy);
    EEPROM.put(EEPROM_BIAS_BASE + 8, bias_gz);
    uint32_t magic = EEPROM_MAGIC_VAL;
    EEPROM.put(EEPROM_MAGIC_ADDR, magic);
}

bool load_bias_eeprom() {
    uint32_t magic;
    EEPROM.get(EEPROM_MAGIC_ADDR, magic);
    if (magic != EEPROM_MAGIC_VAL) return false;
    EEPROM.get(EEPROM_BIAS_BASE + 0, bias_gx);
    EEPROM.get(EEPROM_BIAS_BASE + 4, bias_gy);
    EEPROM.get(EEPROM_BIAS_BASE + 8, bias_gz);
    return true;
}

// =============================================================================
// I2C HELPERS
// =============================================================================
void mpu_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

uint8_t mpu_read(uint8_t reg) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0;
}

void ak_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}
