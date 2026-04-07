#include <Arduino.h>
#include <math.h>
#include <string.h>
#include <Wire.h>
#include <TFT_eSPI.h>
#include "LIS3DHTR.h"
#include <machine_learning_inferencing.h>

TFT_eSPI tft = TFT_eSPI();
LIS3DHTR<TwoWire> lis;

// =====================================================
// 基本配置
// =====================================================
#define SCREEN_W 320
#define SCREEN_H 240

#define SAMPLE_RATE_HZ 100
#define SAMPLE_INTERVAL_MS (1000 / SAMPLE_RATE_HZ)

#ifndef WIO_LIGHT
#define WIO_LIGHT A5
#endif

enum UIMode {
  MODE_HOME = 0,
  MODE_SOUND,
  MODE_LIGHT,
  MODE_STATUS,
  MODE_COUNT
};

UIMode currentMode = MODE_HOME;
UIMode lastRenderedMode = MODE_COUNT;

// =====================================================
// STATUS 页面状态枚举
// =====================================================
enum DeviceStatus {
  STATUS_FOCUS = 0,
  STATUS_REST,
  STATUS_SPEECH,
  STATUS_NOISE
};

DeviceStatus currentStatus = STATUS_FOCUS;
DeviceStatus lastStatus = STATUS_FOCUS;
float currentStatusConfidence = 0.0f;

// =====================================================
// 颜色
// =====================================================
#define COL_BG         TFT_WHITE
#define COL_PANEL      0xE71C
#define COL_PANEL_ALT  0xF79E
#define COL_TEXT       TFT_BLACK
#define COL_SUBTEXT    TFT_DARKGREY
#define COL_ACCENT     0x051D
#define COL_CYAN2      0x867F
#define COL_GREEN2     0x3FE0
#define COL_ORANGE2    0xFD20
#define COL_PINK2      0xF81F
#define COL_RED2       TFT_RED
#define COL_LINE       0xC618
#define COL_FACE       0xFD20
#define COL_FACE_EDGE  TFT_ORANGE
#define COL_FACE_BG    0xFFF9
#define COL_PANEL2     COL_LINE

#define COL_STATUS_FOCUS   COL_GREEN2
#define COL_STATUS_REST    COL_CYAN2
#define COL_STATUS_SPEECH  COL_ORANGE2
#define COL_STATUS_NOISE   COL_RED2

// =====================================================
// 手势触发参数
// =====================================================
const float SHAKE_TRIGGER_THRESHOLD = 0.20f;
const float LR_TRIGGER_THRESHOLD    = 0.18f;
const float UD_TRIGGER_THRESHOLD    = 0.55f;
const float LR_FIX_MIN_SCORE        = 0.07f;
const float AXIS_DOMINANCE_RATIO    = 1.25f;

unsigned long lastModeSwitchMs = 0;
const unsigned long MODE_SWITCH_COOLDOWN = 1200;

unsigned long lastShakeTriggerMs = 0;
const unsigned long SHAKE_TRIGGER_COOLDOWN = 1200;

// =====================================================
// 状态变量
// =====================================================
String lastGesture = "idle";
float lastGestureScore = 0.0f;

bool dizzyActive = false;
unsigned long dizzyUntilMs = 0;

int mockTempC = 12;

// 亮度阈值：小于此值视为偏暗
const int LIGHT_REST_THRESHOLD = 400;

static const int LIGHT_HISTORY = 80;
int lightHistory[LIGHT_HISTORY];
int lightIndex = 0;
String lightContext = "indoor";

// =====================================================
// 真麦克风声音数据
// =====================================================
int soundBars[8] = {18, 20, 22, 25, 23, 21, 19, 18};
float soundDb = 0.0f;
float soundDbSmooth = 0.0f;
int micDcEstimate = 2048;
const int MIC_SAMPLE_COUNT = 128;
const int MIC_BAR_COUNT = 8;

// STATUS 历史
int statusHistory[24] = {
  35, 36, 38, 40, 42, 44, 41, 39,
  36, 34, 32, 30, 33, 37, 43, 47,
  52, 58, 61, 56, 49, 45, 40, 38
};

int gestureHistory[24] = {
  20, 24, 28, 25, 30, 35, 32, 38,
  42, 40, 45, 50, 46, 52, 55, 53,
  49, 57, 60, 58, 62, 59, 64, 61
};

// 最近一次采样的 IMU 值，仅用于手势方向判定
float lastAx = 0.0f;
float lastAy = 0.0f;
float lastAz = 0.0f;

// =====================================================
// 局部刷新控制
// =====================================================
bool headerDirty = true;
bool footerDirty = true;
bool homeStaticDirty = true;
bool homeCornerDirty = true;
bool homeFaceDirty = true;
bool soundStaticDirty = true;
bool soundBarsDirty = true;
bool lightStaticDirty = true;
bool lightGraphDirty = true;
bool statusStaticDirty = true;
bool statusDynamicDirty = true;

unsigned long lastUiMs = 0;
const unsigned long UI_INTERVAL_MS = 120;

unsigned long lastFaceAnimMs = 0;
const unsigned long FACE_ANIM_INTERVAL_MS = 90;

// =====================================================
// Edge Impulse 输入缓存
// =====================================================
float imuBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
size_t imuBufferIx = 0;
bool bufferReady = false;

// =====================================================
// Edge Impulse callback
// =====================================================
static int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, imuBuffer + offset, length * sizeof(float));
  return 0;
}

// =====================================================
// 工具函数
// =====================================================
const char* statusName(DeviceStatus s) {
  switch (s) {
    case STATUS_FOCUS:  return "FOCUS";
    case STATUS_REST:   return "REST";
    case STATUS_SPEECH: return "SPEECH";
    case STATUS_NOISE:  return "NOISE";
    default:            return "?";
  }
}

uint16_t statusColor(DeviceStatus s) {
  switch (s) {
    case STATUS_FOCUS:  return COL_STATUS_FOCUS;
    case STATUS_REST:   return COL_STATUS_REST;
    case STATUS_SPEECH: return COL_STATUS_SPEECH;
    case STATUS_NOISE:  return COL_STATUS_NOISE;
    default:            return COL_ACCENT;
  }
}

const char* modeName(UIMode m) {
  switch (m) {
    case MODE_HOME:   return "HOME";
    case MODE_SOUND:  return "SOUND";
    case MODE_LIGHT:  return "LIGHT";
    case MODE_STATUS: return "STATUS";
    default:          return "?";
  }
}

uint16_t accentColorForMode(UIMode m) {
  switch (m) {
    case MODE_HOME:   return COL_ACCENT;
    case MODE_SOUND:  return COL_PINK2;
    case MODE_LIGHT:  return COL_CYAN2;
    case MODE_STATUS: return statusColor(currentStatus);
    default:          return COL_ACCENT;
  }
}

int statusScore(DeviceStatus s) {
  switch (s) {
    case STATUS_REST:   return 20;
    case STATUS_FOCUS:  return 40;
    case STATUS_SPEECH: return 70;
    case STATUS_NOISE:  return 95;
    default:            return 50;
  }
}

String predictLabel(const ei_impulse_result_t &result, float &scoreOut) {
  size_t bestIx = 0;
  float bestVal = result.classification[0].value;
  for (size_t i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > bestVal) {
      bestVal = result.classification[i].value;
      bestIx = i;
    }
  }
  scoreOut = bestVal;
  return String(result.classification[bestIx].label);
}

float getLabelScore(const ei_impulse_result_t &result, const char *target) {
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (strcmp(result.classification[i].label, target) == 0) {
      return result.classification[i].value;
    }
  }
  return 0.0f;
}

void updateLightContext(int rawLight) {
  if (rawLight < 250) lightContext = "dark";
  else if (rawLight < 1200) lightContext = "indoor";
  else lightContext = "outdoor";
}

void pushLightSample(int rawLight) {
  lightHistory[lightIndex] = rawLight;
  lightIndex = (lightIndex + 1) % LIGHT_HISTORY;
}

void triggerDizzy() {
  dizzyActive = true;
  dizzyUntilMs = millis() + 1500;
  lastShakeTriggerMs = millis();
  if (currentMode == MODE_HOME) homeFaceDirty = true;
}

void markAllDirtyForMode(UIMode m) {
  headerDirty = true;
  footerDirty = true;

  if (m == MODE_HOME) {
    homeStaticDirty = true;
    homeCornerDirty = true;
    homeFaceDirty = true;
  } else if (m == MODE_SOUND) {
    soundStaticDirty = true;
    soundBarsDirty = true;
  } else if (m == MODE_LIGHT) {
    lightStaticDirty = true;
    lightGraphDirty = true;
  } else if (m == MODE_STATUS) {
    statusStaticDirty = true;
    statusDynamicDirty = true;
  }
}

void switchToNextMode() {
  currentMode = (UIMode)((currentMode + 1) % MODE_COUNT);
  lastModeSwitchMs = millis();
  lastRenderedMode = MODE_COUNT;
  markAllDirtyForMode(currentMode);
  Serial.println(">>> MODE NEXT");
}

void switchToPrevMode() {
  currentMode = (UIMode)((currentMode - 1 + MODE_COUNT) % MODE_COUNT);
  lastModeSwitchMs = millis();
  lastRenderedMode = MODE_COUNT;
  markAllDirtyForMode(currentMode);
  Serial.println(">>> MODE PREV");
}

void fillRoundPanel(int x, int y, int w, int h, int r, uint16_t fill, uint16_t border) {
  tft.fillRoundRect(x, y, w, h, r, fill);
  tft.drawRoundRect(x, y, w, h, r, border);
}

void drawMiniCard(int x, int y, int w, int h, const char* title, uint16_t edgeColor) {
  tft.fillRoundRect(x, y, w, h, 10, COL_PANEL);
  tft.drawRoundRect(x, y, w, h, 10, edgeColor);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawString(title, x + 6, y + 5, 1);
}

void drawMiniLineChart(int x, int y, int w, int h, int *data, int len, int minV, int maxV, uint16_t color) {
  int gx = x + 6;
  int gy = y + 18;
  int gw = w - 12;
  int gh = h - 24;

  for (int i = 0; i < 3; i++) {
    int yy = gy + (i * gh) / 2;
    tft.drawFastHLine(gx, yy, gw, COL_LINE);
  }

  for (int i = 0; i < len - 1; i++) {
    int v0 = constrain(data[i], minV, maxV);
    int v1 = constrain(data[i + 1], minV, maxV);

    int x0 = gx + (i * gw) / (len - 1);
    int x1 = gx + ((i + 1) * gw) / (len - 1);

    int y0 = gy + gh - map(v0, minV, maxV, 0, gh);
    int y1 = gy + gh - map(v1, minV, maxV, 0, gh);

    tft.drawLine(x0, y0, x1, y1, color);
    tft.drawLine(x0, y0 + 1, x1, y1 + 1, color);
  }
}

void drawStatusFace(int cx, int cy, int r, DeviceStatus s, uint16_t faceColor, uint16_t bgColor) {
  tft.fillCircle(cx, cy, r, faceColor);
  tft.drawCircle(cx, cy, r, TFT_BLACK);
  tft.drawCircle(cx, cy, r + 1, TFT_BLACK);

  if (s == STATUS_FOCUS) {
    tft.fillCircle(cx - 12, cy - 8, 4, TFT_BLACK);
    tft.fillCircle(cx + 12, cy - 8, 4, TFT_BLACK);
    tft.drawFastHLine(cx - 10, cy + 12, 20, TFT_BLACK);
    tft.drawFastHLine(cx - 8, cy + 13, 16, TFT_BLACK);
  }
  else if (s == STATUS_REST) {
    tft.drawFastHLine(cx - 16, cy - 8, 10, TFT_BLACK);
    tft.drawFastHLine(cx + 6, cy - 8, 10, TFT_BLACK);
    tft.drawFastHLine(cx - 8, cy + 12, 16, TFT_BLACK);
    tft.setTextColor(TFT_BLACK, bgColor);
    tft.drawString("z", cx + 18, cy - 22, 2);
    tft.drawString("z", cx + 28, cy - 32, 1);
  }
  else if (s == STATUS_SPEECH) {
    tft.fillCircle(cx - 12, cy - 8, 4, TFT_BLACK);
    tft.fillCircle(cx + 12, cy - 8, 4, TFT_BLACK);
    tft.drawCircle(cx, cy + 12, 6, TFT_BLACK);
    tft.drawCircle(cx, cy + 12, 5, TFT_BLACK);
  }
  else if (s == STATUS_NOISE) {
    tft.drawLine(cx - 16, cy - 12, cx - 8, cy - 4, TFT_BLACK);
    tft.drawLine(cx - 16, cy - 4, cx - 8, cy - 12, TFT_BLACK);
    tft.drawLine(cx + 8, cy - 12, cx + 16, cy - 4, TFT_BLACK);
    tft.drawLine(cx + 8, cy - 4, cx + 16, cy - 12, TFT_BLACK);
    tft.drawFastHLine(cx - 12, cy + 14, 24, TFT_BLACK);
    tft.drawFastHLine(cx - 10, cy + 15, 20, TFT_BLACK);
  }
}

// =====================================================
// 状态融合逻辑：只看声音 + 亮度
// 规则：
// < 50        -> REST / FOCUS（由亮度决定）
// 50 ~ 75     -> SPEECH
// > 75        -> NOISE
// =====================================================
void updateStatusFromSensors() {
  int latestLight = lightHistory[(lightIndex - 1 + LIGHT_HISTORY) % LIGHT_HISTORY];

  lastStatus = currentStatus;

  if (soundDb > 75.0f) {
    currentStatus = STATUS_NOISE;
    currentStatusConfidence = min(1.0f, 0.75f + (soundDb - 75.0f) / 20.0f);
  }
  else if (soundDb >= 50.0f) {
    currentStatus = STATUS_SPEECH;
    currentStatusConfidence = 0.82f;
  }
  else {
    if (latestLight < LIGHT_REST_THRESHOLD) {
      currentStatus = STATUS_REST;
      currentStatusConfidence = 0.86f;
    } else {
      currentStatus = STATUS_FOCUS;
      currentStatusConfidence = 0.84f;
    }
  }

  if (currentStatus != lastStatus) {
    headerDirty = true;
    footerDirty = true;

    if (currentMode == MODE_STATUS) {
      statusStaticDirty = true;
      statusDynamicDirty = true;
    }

    if (currentMode == MODE_HOME) {
      homeCornerDirty = true;
    }
  }
}

// =====================================================
// 真麦克风采样
// =====================================================
void updateRealSoundBars() {
#ifdef WIO_MIC
  const int samplesPerBar = MIC_SAMPLE_COUNT / MIC_BAR_COUNT;

  long totalAbs = 0;

  long dcSum = 0;
  for (int i = 0; i < 16; i++) {
    dcSum += analogRead(WIO_MIC);
  }
  int dcNow = dcSum / 16;
  micDcEstimate = (micDcEstimate * 7 + dcNow) / 8;

  for (int b = 0; b < MIC_BAR_COUNT; b++) {
    long segAbs = 0;

    for (int i = 0; i < samplesPerBar; i++) {
      int v = analogRead(WIO_MIC);
      int diff = abs(v - micDcEstimate);
      segAbs += diff;
      totalAbs += diff;
    }

    int avgSeg = segAbs / samplesPerBar;
    int h = map(avgSeg, 2, 180, 14, 82);
    h = constrain(h, 14, 82);

    soundBars[b] = (soundBars[b] * 2 + h) / 3;
  }

  float avgAbs = (float)totalAbs / MIC_SAMPLE_COUNT;
  float db = 20.0f * log10f((avgAbs + 1.0f) / 8.0f) + 42.0f;

  if (db < 0.0f) db = 0.0f;
  if (db > 99.9f) db = 99.9f;

  soundDbSmooth = soundDbSmooth * 0.75f + db * 0.25f;
  soundDb = soundDbSmooth;
#else
  for (int i = 0; i < 8; i++) {
    soundBars[i] = 18;
  }
  soundDb = 0.0f;
#endif
}

// =====================================================
// IMU 采样
// =====================================================
void sampleIMUToBuffer() {
  static unsigned long lastSampleMs = 0;
  unsigned long now = millis();

  if (lastSampleMs == 0) {
    lastSampleMs = now;
    return;
  }

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;
  lastSampleMs += SAMPLE_INTERVAL_MS;

  float ax = lis.getAccelerationX();
  float ay = lis.getAccelerationY();
  float az = lis.getAccelerationZ();

  lastAx = ax;
  lastAy = ay;
  lastAz = az;

  if (imuBufferIx + 3 <= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    imuBuffer[imuBufferIx++] = ax;
    imuBuffer[imuBufferIx++] = ay;
    imuBuffer[imuBufferIx++] = az;
  }

  if (imuBufferIx >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    imuBufferIx = 0;
    bufferReady = true;
  }
}

// =====================================================
// 头部 / 底部
// =====================================================
void drawHeader() {
  uint16_t accent = accentColorForMode(currentMode);
  tft.fillRoundRect(10, 8, 300, 28, 10, COL_PANEL_ALT);
  tft.drawRoundRect(10, 8, 300, 28, 10, COL_PANEL2);

  unsigned long sec = millis() / 1000;
  int mm = (sec / 60) % 60;
  int ss = sec % 60;
  char timeBuf[16];
  snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d", mm, ss);

  tft.setTextColor(TFT_DARKGREY, COL_PANEL_ALT);
  tft.drawString(timeBuf, 20, 14, 2);

  tft.setTextColor(accent, COL_PANEL_ALT);
  tft.drawCentreString(modeName(currentMode), 160, 14, 2);

  tft.setTextColor(TFT_DARKGREY, COL_PANEL_ALT);
  tft.drawRightString(String(mockTempC) + "C", 295, 14, 2);

  headerDirty = false;
}

void drawFooter() {
  tft.fillRoundRect(10, 206, 300, 24, 10, COL_PANEL_ALT);
  tft.drawRoundRect(10, 206, 300, 24, 10, COL_PANEL2);

  tft.setTextColor(TFT_DARKGREY, COL_PANEL_ALT);

  if (currentMode == MODE_STATUS) {
    tft.drawString(String(soundDb, 1) + " dB", 20, 212, 2);
    tft.setTextColor(statusColor(currentStatus), COL_PANEL_ALT);
    tft.drawRightString(String(statusName(currentStatus)) + " " + String((int)(currentStatusConfidence * 100)) + "%", 295, 212, 2);
  } else {
    tft.drawString("Light " + lightContext, 20, 212, 2);
    tft.setTextColor(accentColorForMode(currentMode), COL_PANEL_ALT);
    tft.drawRightString(lastGesture + " " + String(lastGestureScore, 2), 295, 212, 2);
  }

  footerDirty = false;
}

// =====================================================
// HOME 页面
// =====================================================
void drawHomeStatic() {
  if (lastRenderedMode != MODE_HOME) {
    tft.fillScreen(COL_BG);
    lastRenderedMode = MODE_HOME;
  }

  drawMiniCard(16, 46, 92, 52, "Light", COL_CYAN2);
  drawMiniCard(212, 46, 92, 52, "Score", COL_PINK2);
  drawMiniCard(16, 144, 92, 52, "Trend", COL_CYAN2);
  drawMiniCard(212, 144, 92, 52, "Status", statusColor(currentStatus));

  fillRoundPanel(112, 52, 96, 144, 18, COL_FACE_BG, COL_ACCENT);

  tft.setTextColor(TFT_DARKGREY, COL_FACE_BG);
  tft.drawCentreString("Gesture", 160, 60, 2);

  homeStaticDirty = false;
}

void drawHomeCornerWidgets() {
  tft.fillRect(22, 58, 80, 34, COL_PANEL);
  int latestLight = lightHistory[(lightIndex - 1 + LIGHT_HISTORY) % LIGHT_HISTORY];
  tft.setTextColor(COL_CYAN2, COL_PANEL);
  tft.drawCentreString(String(latestLight), 62, 62, 4);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawCentreString(lightContext, 62, 86, 1);

  tft.fillRect(218, 58, 80, 34, COL_PANEL);
  int scorePercent = (int)(lastGestureScore * 100.0f);
  tft.setTextColor(COL_PINK2, COL_PANEL);
  tft.drawCentreString(String(scorePercent), 258, 62, 4);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawCentreString(lastGesture.c_str(), 258, 86, 1);

  tft.fillRect(18, 150, 88, 40, COL_PANEL);
  drawMiniLineChart(16, 144, 92, 52, gestureHistory, 24, 0, 100, COL_CYAN2);

  uint16_t sc = statusColor(currentStatus);
  tft.fillRoundRect(212, 144, 92, 52, 10, COL_PANEL);
  tft.drawRoundRect(212, 144, 92, 52, 10, sc);
  tft.setTextColor(sc, COL_PANEL);
  tft.drawCentreString(statusName(currentStatus), 258, 156, 2);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawCentreString(String((int)(currentStatusConfidence * 100)) + "%", 258, 175, 2);

  homeCornerDirty = false;
}

void drawHomeFaceOnly() {
  tft.fillRect(120, 82, 80, 82, COL_FACE_BG);

  int cx = 160;
  int cy = 116;

  if (dizzyActive && millis() < dizzyUntilMs) {
    for (int i = 0; i < 2; i++) {
      int lx = cx - 34 - i * 5;
      int rx = cx + 34 + i * 5;
      tft.drawFastVLine(lx, cy - 22, 44, COL_LINE);
      tft.drawFastVLine(rx, cy - 22, 44, COL_LINE);
    }
  }

  tft.fillCircle(cx, cy, 28, COL_FACE);
  tft.drawCircle(cx, cy, 28, COL_FACE_EDGE);
  tft.drawCircle(cx, cy, 29, COL_FACE_EDGE);

  if (dizzyActive && millis() < dizzyUntilMs) {
    for (int r = 2; r <= 7; r += 2) {
      tft.drawCircle(cx - 10, cy - 7, r, TFT_BLACK);
      tft.drawCircle(cx + 10, cy - 7, r, TFT_BLACK);
    }
    tft.drawFastHLine(cx - 14, cy + 12, 28, COL_RED2);
    tft.drawFastHLine(cx - 14, cy + 13, 28, COL_RED2);
    tft.drawFastHLine(cx - 14, cy + 14, 28, COL_RED2);
  } else {
    tft.fillCircle(cx - 10, cy - 7, 4, TFT_BLACK);
    tft.fillCircle(cx + 10, cy - 7, 4, TFT_BLACK);
    tft.drawFastHLine(cx - 12, cy + 12, 24, TFT_BLACK);
    tft.drawFastHLine(cx - 12, cy + 13, 24, TFT_BLACK);
  }

  tft.fillRect(124, 148, 72, 34, COL_FACE_BG);
  tft.setTextColor(COL_TEXT, COL_FACE_BG);
  tft.drawCentreString(lastGesture, 160, 150, 2);
  tft.setTextColor(COL_ACCENT, COL_FACE_BG);
  tft.drawCentreString(String(lastGestureScore, 2), 160, 170, 2);

  homeFaceDirty = false;
}

// =====================================================
// SOUND 页面
// =====================================================
void drawSoundStatic() {
  if (lastRenderedMode != MODE_SOUND) {
    tft.fillScreen(COL_BG);
    lastRenderedMode = MODE_SOUND;
  }

  fillRoundPanel(16, 46, 288, 150, 18, COL_PANEL, COL_PINK2);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawString("Sound Dashboard", 28, 56, 2);

  tft.fillRoundRect(232, 54, 58, 24, 8, COL_FACE_BG);
  tft.drawRoundRect(232, 54, 58, 24, 8, COL_PINK2);
  tft.setTextColor(COL_PINK2, COL_FACE_BG);
  tft.drawCentreString("dB", 261, 61, 2);

  soundStaticDirty = false;
}

void drawSoundBarsOnly() {
  tft.fillRect(28, 82, 260, 102, COL_PANEL);

  for (int i = 0; i < 4; i++) {
    int yy = 94 + i * 24;
    tft.drawFastHLine(32, yy, 248, COL_LINE);
  }

  tft.fillRect(220, 82, 70, 26, COL_PANEL);
  tft.setTextColor(COL_PINK2, COL_PANEL);
  tft.drawRightString(String(soundDb, 1), 276, 86, 4);

  int baseY = 178;
  int startX = 40;
  int barW = 20;
  int gap = 10;

  for (int i = 0; i < 8; i++) {
    int h = soundBars[i];
    int x = startX + i * (barW + gap);
    int y = baseY - h;

    tft.fillRoundRect(x, y, barW, h, 6, COL_PINK2);
    tft.drawRoundRect(x, y, barW, h, 6, COL_PINK2);

    if (h > 10) {
      tft.drawFastVLine(x + 4, y + 4, h - 8, TFT_WHITE);
    }
  }

  soundBarsDirty = false;
}

// =====================================================
// LIGHT 页面
// =====================================================
void drawLightStatic() {
  if (lastRenderedMode != MODE_LIGHT) {
    tft.fillScreen(COL_BG);
    lastRenderedMode = MODE_LIGHT;
  }

  fillRoundPanel(16, 46, 288, 150, 18, COL_PANEL, COL_CYAN2);
  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawString("Ambient Light", 28, 56, 2);

  lightStaticDirty = false;
}

void drawLightDynamicOnly() {
  tft.fillRect(24, 76, 260, 110, COL_PANEL);

  int latestLight = lightHistory[(lightIndex - 1 + LIGHT_HISTORY) % LIGHT_HISTORY];
  tft.setTextColor(COL_CYAN2, COL_PANEL);
  tft.drawString(String(latestLight), 28, 78, 6);

  tft.setTextColor(COL_TEXT, COL_PANEL);
  tft.drawString(lightContext, 210, 88, 2);

  int gx = 28, gy = 126, gw = 248, gh = 52;

  for (int i = 0; i < 4; i++) {
    int y = gy + (i * gh) / 3;
    tft.drawFastHLine(gx, y, gw, COL_LINE);
  }

  for (int i = 0; i < LIGHT_HISTORY - 1; i++) {
    int idx0 = (lightIndex + i) % LIGHT_HISTORY;
    int idx1 = (lightIndex + i + 1) % LIGHT_HISTORY;

    int y0 = gy + gh - map(lightHistory[idx0], 0, 2000, 0, gh);
    int y1 = gy + gh - map(lightHistory[idx1], 0, 2000, 0, gh);

    int x0 = gx + (i * gw) / (LIGHT_HISTORY - 1);
    int x1 = gx + ((i + 1) * gw) / (LIGHT_HISTORY - 1);

    tft.drawLine(x0, y0, x1, y1, COL_CYAN2);
    tft.drawLine(x0, y0 + 1, x1, y1 + 1, COL_CYAN2);
  }

  lightGraphDirty = false;
}

// =====================================================
// STATUS 页面
// =====================================================
void drawStatusStatic() {
  if (lastRenderedMode != MODE_STATUS) {
    tft.fillScreen(COL_BG);
    lastRenderedMode = MODE_STATUS;
  }

  uint16_t sc = statusColor(currentStatus);

  fillRoundPanel(16, 46, 136, 150, 18, COL_PANEL, sc);
  fillRoundPanel(168, 46, 136, 150, 18, COL_FACE_BG, sc);

  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawString("Status", 28, 56, 2);

  tft.setTextColor(COL_SUBTEXT, COL_FACE_BG);
  tft.drawString("Mood", 180, 56, 2);

  statusStaticDirty = false;
}

void drawStatusDynamicOnly() {
  uint16_t sc = statusColor(currentStatus);

  tft.fillRect(22, 76, 124, 108, COL_PANEL);

  tft.setTextColor(sc, COL_PANEL);
  tft.drawCentreString(statusName(currentStatus), 84, 82, 4);

  tft.setTextColor(COL_TEXT, COL_PANEL);
  if (currentStatus == STATUS_FOCUS) {
    tft.drawCentreString("focused", 84, 118, 2);
  } else if (currentStatus == STATUS_REST) {
    tft.drawCentreString("resting", 84, 118, 2);
  } else if (currentStatus == STATUS_SPEECH) {
    tft.drawCentreString("speaking", 84, 118, 2);
  } else {
    tft.drawCentreString("noisy", 84, 118, 2);
  }

  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawCentreString(String(soundDb, 1) + " dB", 84, 145, 2);
  tft.drawCentreString(lightContext, 84, 164, 2);

  tft.fillRoundRect(176, 76, 120, 104, 12, COL_FACE_BG);
  tft.drawRoundRect(176, 76, 120, 104, 12, sc);
  drawStatusFace(236, 124, 30, currentStatus, sc, COL_FACE_BG);

  drawMiniLineChart(16, 144, 136, 52, statusHistory, 24, 0, 100, sc);

  statusDynamicDirty = false;
}

// =====================================================
// 页面渲染入口
// =====================================================
void renderCurrentModeIfNeeded() {
  if (currentMode == MODE_HOME) {
    if (homeStaticDirty) drawHomeStatic();
    if (homeCornerDirty) drawHomeCornerWidgets();
    if (homeFaceDirty) drawHomeFaceOnly();
  }
  else if (currentMode == MODE_SOUND) {
    if (soundStaticDirty) drawSoundStatic();
    if (soundBarsDirty) drawSoundBarsOnly();
  }
  else if (currentMode == MODE_LIGHT) {
    if (lightStaticDirty) drawLightStatic();
    if (lightGraphDirty) drawLightDynamicOnly();
  }
  else if (currentMode == MODE_STATUS) {
    if (statusStaticDirty) drawStatusStatic();
    if (statusDynamicDirty) drawStatusDynamicOnly();
  }
}

// =====================================================
// setup / loop
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("boot");

  pinMode(LCD_BACKLIGHT, OUTPUT);
  digitalWrite(LCD_BACKLIGHT, HIGH);

  tft.begin();
  tft.setRotation(3);
  tft.fillScreen(COL_BG);
  tft.setTextColor(COL_TEXT, COL_BG);
  tft.drawString("Booting...", 20, 20, 2);
  Serial.println("tft ok");

  randomSeed(analogRead(A0));

  analogReadResolution(12);
#ifdef WIO_MIC
  pinMode(WIO_MIC, INPUT);
#endif

  for (int i = 0; i < LIGHT_HISTORY; i++) {
    lightHistory[i] = 0;
  }

  lis.begin(Wire1);
  lis.setOutputDataRate(LIS3DHTR_DATARATE_100HZ);
  lis.setFullScaleRange(LIS3DHTR_RANGE_2G);
  Serial.println("imu ok");

  markAllDirtyForMode(currentMode);
  drawHeader();
  drawFooter();
  renderCurrentModeIfNeeded();
  Serial.println("setup done");
}

void loop() {
  sampleIMUToBuffer();

  static unsigned long lastLightMs = 0;
  if (millis() - lastLightMs >= 120) {
    lastLightMs = millis();

    int rawLight = analogRead(WIO_LIGHT);
    pushLightSample(rawLight);

    String oldContext = lightContext;
    updateLightContext(rawLight);

    for (int i = 0; i < 23; i++) gestureHistory[i] = gestureHistory[i + 1];
    gestureHistory[23] = (int)(lastGestureScore * 100.0f);

    if (currentMode == MODE_HOME) homeCornerDirty = true;
    if (currentMode == MODE_LIGHT) lightGraphDirty = true;

    if (lightContext != oldContext) {
      footerDirty = true;
      updateStatusFromSensors();

      if (currentMode == MODE_STATUS) {
        statusStaticDirty = true;
        statusDynamicDirty = true;
      }
    }
  }

  static unsigned long lastSoundAnimMs = 0;
  if (millis() - lastSoundAnimMs >= 80) {
    lastSoundAnimMs = millis();

    updateRealSoundBars();
    updateStatusFromSensors();

    if (currentMode == MODE_SOUND) {
      soundBarsDirty = true;
    }
    if (currentMode == MODE_STATUS) {
      statusDynamicDirty = true;
    }
    if (currentMode == MODE_HOME) {
      homeCornerDirty = true;
    }

    footerDirty = true;

    Serial.print("Mic dB=");
    Serial.print(soundDb, 1);
    Serial.print(" status=");
    Serial.println(statusName(currentStatus));
  }

  static unsigned long lastStatusHistMs = 0;
  if (millis() - lastStatusHistMs >= 700) {
    lastStatusHistMs = millis();

    for (int i = 0; i < 23; i++) statusHistory[i] = statusHistory[i + 1];
    statusHistory[23] = statusScore(currentStatus);

    if (currentMode == MODE_STATUS) {
      statusDynamicDirty = true;
    }
  }

  // =================================================
  // 手势模型
  // =================================================
  if (bufferReady) {
    bufferReady = false;

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &raw_feature_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

    if (res == EI_IMPULSE_OK) {
      float score = 0.0f;
      String label = predictLabel(result, score);

      float shakeScore = getLabelScore(result, "shake");
      float lrScore    = getLabelScore(result, "left-right");
      float udScore    = getLabelScore(result, "up-down");

      lastGesture = label;
      lastGestureScore = score;
      footerDirty = true;

      if (currentMode == MODE_HOME) {
        homeCornerDirty = true;
        homeFaceDirty = true;
      }

      float absAx = fabs(lastAx);
      float absAy = fabs(lastAy);
      bool axisSuggestLR = absAx > absAy * AXIS_DOMINANCE_RATIO;
      bool axisSuggestUD = absAy > absAx * AXIS_DOMINANCE_RATIO;

      Serial.print("Pred: ");
      Serial.print(label);
      Serial.print(" score=");
      Serial.println(score, 3);

      Serial.print("shake=");
      Serial.print(shakeScore, 3);
      Serial.print(" lr=");
      Serial.print(lrScore, 3);
      Serial.print(" ud=");
      Serial.println(udScore, 3);

      bool canSwitch = millis() - lastModeSwitchMs >= MODE_SWITCH_COOLDOWN;
      bool canShake  = millis() - lastShakeTriggerMs >= SHAKE_TRIGGER_COOLDOWN;

      if (canShake && shakeScore > SHAKE_TRIGGER_THRESHOLD) {
        triggerDizzy();
        Serial.println(">>> SHAKE TRIGGERED");
      }
      else if (canSwitch && lrScore > LR_TRIGGER_THRESHOLD) {
        switchToNextMode();
        Serial.println(">>> LEFT-RIGHT TRIGGERED (model direct)");
      }
      else if (canSwitch && udScore > UD_TRIGGER_THRESHOLD) {
        if (axisSuggestLR && lrScore > LR_FIX_MIN_SCORE) {
          switchToNextMode();
          Serial.println(">>> LEFT-RIGHT TRIGGERED (fix from up-down)");
        }
        else {
          switchToPrevMode();
          Serial.println(">>> UP-DOWN TRIGGERED");
        }
      }
    } else {
      Serial.print("run_classifier error: ");
      Serial.println((int)res);
    }
  }

  if (dizzyActive && millis() > dizzyUntilMs) {
    dizzyActive = false;
    if (currentMode == MODE_HOME) homeFaceDirty = true;
  }

  if (currentMode == MODE_HOME &&
      homeFaceDirty &&
      millis() - lastFaceAnimMs >= FACE_ANIM_INTERVAL_MS) {
    lastFaceAnimMs = millis();
    drawHomeFaceOnly();
  }

  if (millis() - lastUiMs >= UI_INTERVAL_MS) {
    lastUiMs = millis();

    if (headerDirty) drawHeader();
    if (footerDirty) drawFooter();
    renderCurrentModeIfNeeded();
  }
}
