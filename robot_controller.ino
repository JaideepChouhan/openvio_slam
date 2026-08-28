// Arduino UNO + MPU-9250/MPU-6500 robot controller
// Assumed mounting: chip X points forward, chip Y points left, chip Z points up.
// Positive Z gyro is CCW when viewed from above.

#include <Wire.h>
#include <EEPROM.h>

#define LEFT_PWM_PIN 5
#define LEFT_DIR_PIN 4
#define RIGHT_PWM_PIN 6
#define RIGHT_DIR_PIN 7

#define MPU_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define GYRO_CONFIG 0x1B
#define ACCEL_CONFIG 0x1C
#define ACCEL_CONFIG2 0x1D
#define CONFIG 0x1A
#define SMPLRT_DIV 0x19
#define INT_PIN_CFG 0x37
#define INT_ENABLE 0x38

#define GYRO_SCALE (1.0f / 16.4f * PI / 180.0f)
#define ACCEL_SCALE (1.0f / 4096.0f * 9.80665f)

struct BiasData {
  float gx;
  float gy;
  float gz;
};

volatile bool imuReady = false;
volatile uint32_t lastMicros = 0;
uint32_t lastSendMs = 0;
uint32_t lastMotorCmdMs = 0;
bool recalibrationActive = false;
BiasData imuBias = {0.0f, 0.0f, 0.0f};

void setup() {
  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(LEFT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);
  analogWrite(LEFT_PWM_PIN, 0);
  analogWrite(RIGHT_PWM_PIN, 0);
  digitalWrite(LEFT_DIR_PIN, LOW);
  digitalWrite(RIGHT_DIR_PIN, LOW);

  Serial.begin(115200);
  Wire.begin();

  configureMpu();

  if (!loadBiasFromEeprom()) {
    calibrateGyro();
    saveBiasToEeprom();
  }

  lastSendMs = millis();
  lastMotorCmdMs = millis();
}

void loop() {
  if (imuReady) {
    imuReady = false;
    float gx, gy, gz, ax, ay, az;
    readImu(&gx, &gy, &gz, &ax, &ay, &az);
    if (!recalibrationActive && (millis() - lastSendMs >= 10)) {
      lastSendMs = millis();
      sendImuPacket(gx, gy, gz, ax, ay, az);
    }
  }

  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    parseSerialByte(b);
  }

  if ((millis() - lastMotorCmdMs) > 500) {
    setMotor(0, 0);
  }
}

void configureMpu() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x01);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(GYRO_CONFIG);
  Wire.write(0x18);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_CONFIG);
  Wire.write(0x10);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_CONFIG2);
  Wire.write(0x04);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(CONFIG);
  Wire.write(0x04);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(SMPLRT_DIV);
  Wire.write(0x04);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(INT_PIN_CFG);
  Wire.write(0x02);
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(INT_ENABLE);
  Wire.write(0x01);
  Wire.endTransmission();
}

void readImu(float* gx, float* gy, float* gz, float* ax, float* ay, float* az) {
  int16_t rawGx, rawGy, rawGz;
  int16_t rawAx, rawAy, rawAz;

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x43);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  rawGx = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  rawGy = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  rawGz = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  rawAx = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  rawAy = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  rawAz = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));

  *gx = (float)(rawGx - imuBias.gx) * GYRO_SCALE;
  *gy = (float)(rawGy - imuBias.gy) * GYRO_SCALE;
  *gz = (float)(rawGz - imuBias.gz) * GYRO_SCALE;

  *ax = (float)rawAx * ACCEL_SCALE;
  *ay = (float)rawAy * ACCEL_SCALE;
  *az = (float)rawAz * ACCEL_SCALE;
}

void calibrateGyro() {
  const int N = 1000;
  long gxSum = 0;
  long gySum = 0;
  long gzSum = 0;

  for (int i = 0; i < N; ++i) {
    int16_t gx, gy, gz;
    readRawGyro(&gx, &gy, &gz);
    gxSum += gx;
    gySum += gy;
    gzSum += gz;
    delay(2);
  }

  imuBias.gx = (float)gxSum / N;
  imuBias.gy = (float)gySum / N;
  imuBias.gz = (float)gzSum / N;
}

void readRawGyro(int16_t* gx, int16_t* gy, int16_t* gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x43);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);
  *gx = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  *gy = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
  *gz = (int16_t)((uint16_t)Wire.read() | ((uint16_t)Wire.read() << 8));
}

void sendImuPacket(float gx, float gy, float gz, float ax, float ay, float az) {
  uint8_t buffer[31];
  memset(buffer, 0, sizeof(buffer));
  buffer[0] = 0xAA;
  buffer[1] = 0x55;

  uint32_t millisNow = millis();
  memcpy(&buffer[2], &millisNow, 4);
  memcpy(&buffer[6], &gx, 4);
  memcpy(&buffer[10], &gy, 4);
  memcpy(&buffer[14], &gz, 4);
  memcpy(&buffer[18], &ax, 4);
  memcpy(&buffer[22], &ay, 4);
  memcpy(&buffer[26], &az, 4);

  uint8_t checksum = 0;
  for (int i = 2; i < 30; ++i) {
    checksum ^= buffer[i];
  }
  buffer[30] = checksum;
  Serial.write(buffer, sizeof(buffer));
}

void setMotor(int left, int right) {
  left = constrain(left, -255, 255);
  right = constrain(right, -255, 255);
  if (abs(left) < 10) left = 0;
  if (abs(right) < 10) right = 0;

  if (left >= 0) {
    analogWrite(LEFT_PWM_PIN, left);
    digitalWrite(LEFT_DIR_PIN, LOW);
  } else {
    analogWrite(LEFT_PWM_PIN, abs(left));
    digitalWrite(LEFT_DIR_PIN, HIGH);
  }

  if (right >= 0) {
    analogWrite(RIGHT_PWM_PIN, right);
    digitalWrite(RIGHT_DIR_PIN, LOW);
  } else {
    analogWrite(RIGHT_PWM_PIN, abs(right));
    digitalWrite(RIGHT_DIR_PIN, HIGH);
  }

  lastMotorCmdMs = millis();
}

void parseSerialByte(uint8_t b) {
  static uint8_t state = 0;
  static uint8_t motorBytes[7];
  static uint8_t motorIndex = 0;
  static uint8_t recalState = 0;

  if (b == 0xCC && recalState == 0) {
    recalState = 1;
    return;
  }
  if (recalState == 1 && b == 0x77) {
    recalibrationActive = true;
    setMotor(0, 0);
    Serial.write((uint8_t)0xAA);
    Serial.write((uint8_t)0x55);
    Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)0xFF);
    calibrateGyro();
    saveBiasToEeprom();
    recalibrationActive = false;
    recalState = 0;
    return;
  }
  recalState = 0;

  if (b == 0xBB) {
    state = 1;
    motorIndex = 0;
    return;
  }

  if (state == 1 && b == 0x66) {
    state = 2;
    return;
  }

  if (state == 2 && motorIndex < 6) {
    motorBytes[motorIndex++] = b;
    if (motorIndex == 6) {
      uint8_t checksum = 0;
      for (int i = 0; i < 4; ++i) {
        checksum ^= motorBytes[i];
      }
      if (checksum == motorBytes[4]) {
        int16_t left = (int16_t)((uint16_t)motorBytes[0] | ((uint16_t)motorBytes[1] << 8));
        int16_t right = (int16_t)((uint16_t)motorBytes[2] | ((uint16_t)motorBytes[3] << 8));
        setMotor((int)left, (int)right);
      }
      state = 0;
    }
  } else {
    state = 0;
  }
}

bool loadBiasFromEeprom() {
  uint8_t data[12];
  for (int i = 0; i < 12; ++i) {
    data[i] = EEPROM.read(i);
  }

  float savedGx = 0.0f, savedGy = 0.0f, savedGz = 0.0f;
  memcpy(&savedGx, &data[0], 4);
  memcpy(&savedGy, &data[4], 4);
  memcpy(&savedGz, &data[8], 4);

  imuBias.gx = savedGx;
  imuBias.gy = savedGy;
  imuBias.gz = savedGz;
  return true;
}

void saveBiasToEeprom() {
  uint8_t data[12];
  memcpy(&data[0], &imuBias.gx, 4);
  memcpy(&data[4], &imuBias.gy, 4);
  memcpy(&data[8], &imuBias.gz, 4);
  for (int i = 0; i < 12; ++i) {
    EEPROM.update(i, data[i]);
  }
}
