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
  MODE_ACTIVITY,
  MODE_COUNT
};

UIMode currentMode = MODE_HOME;
UIMode lastRenderedMode = MODE_COUNT;

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

// =====================================================
// 模型与触发参数
// =====================================================
const float SHAKE_TRIGGER_THRESHOLD = 0.20f;
const float LR_TRIGGER_THRESHOLD    = 0.35f;
const float UD_TRIGGER_THRESHOLD    = 0.35f;
const float GESTURE_DOMINANCE_GAP   = 0.10f;

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

uint32_t stepCount = 0;
unsigned long lastStepMs = 0;
const unsigned long STEP_DEBOUNCE_MS = 350;

int mockTempC = 12;
const char* mockWeather = "Cloudy";

static const int LIGHT_HISTORY = 80;
int lightHistory[LIGHT_HISTORY];
int lightIndex = 0;
String lightContext = "indoor";

int soundBars[8] = {26, 35, 40, 48, 60, 45, 34, 24};

int stepHistory[24] = {
  2, 3, 5, 4, 6, 8, 7, 9,
  10, 12, 11, 13, 12, 15, 16, 14,
  13, 17, 18, 16, 15, 19, 20, 18
};

int gestureHistory[24] = {
  20, 24, 28, 25, 30, 35, 32, 38,
  42, 40, 45, 50, 46, 52, 55, 53,
  49, 57, 60, 58, 62, 59, 64, 61
};

// =====================================================
// shake 后处理增强
// =====================================================
float shakeAccum = 0.0f;

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
bool activityStaticDirty = true;
bool activityDynamicDirty = true;

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
// 声音条更新
// =====================================================
void updateMockSoundBars() {
  for (int i = 0; i < 8; i++) {
    int delta = random(-10, 11);
    soundBars[i] += delta;
    if (soundBars[i] < 16) soundBars[i] = 16;
    if (soundBars[i] > 78) soundBars[i] = 78;
  }
}

// =====================================================
// 标签辅助判断
// 按常见命名都兼容一下，避免标签名不一致
// =====================================================
bool isShakeLabel(const String &label) {
  return label == "shake";
}

bool isLeftRightLabel(const String &label) {
  return label == "left-right" ||
         label == "leftright" ||
         label == "left_right" ||
         label == "leftRight";
}

bool isUpDownLabel(const String &label) {
  return label == "up-down" ||
         label == "updown" ||
         label == "up_down" ||
         label == "upDown";
}

// =====================================================
// 工具函数
// =====================================================
const char* modeName(UIMode m) {
  switch (m) {
    case MODE_HOME: return "HOME";
    case MODE_SOUND: return "SOUND";
    case MODE_LIGHT: return "LIGHT";
    case MODE_ACTIVITY: return "STEP";
    default: return "?";
  }
}

uint16_t accentColorForMode(UIMode m) {
  switch (m) {
    case MODE_HOME: return COL_ACCENT;
    case MODE_SOUND: return COL_PINK2;
    case MODE_LIGHT: return COL_CYAN2;
    case MODE_ACTIVITY: return COL_GREEN2;
    default: return COL_ACCENT;
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

void maybeCountStep(float ax, float ay, float az) {
  float mag = sqrt(ax * ax + ay * ay + az * az);
  if (mag > 14.0f && millis() - lastStepMs > STEP_DEBOUNCE_MS) {
    stepCount++;
    lastStepMs = millis();
    footerDirty = true;
    if (currentMode == MODE_HOME) homeCornerDirty = true;
    if (currentMode == MODE_ACTIVITY) activityDynamicDirty = true;
  }
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
  } else if (m == MODE_ACTIVITY) {
    activityStaticDirty = true;
    activityDynamicDirty = true;
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

void drawGauge(int cx, int cy, int r, float value01, uint16_t arcColor, const char* label, uint16_t bg) {
  value01 = constrain(value01, 0.0f, 1.0f);

  for (int a = -140; a <= 140; a += 4) {
    float rad = a * 0.0174533f;
    int x0 = cx + cos(rad) * (r - 8);
    int y0 = cy + sin(rad) * (r - 8);
    int x1 = cx + cos(rad) * r;
    int y1 = cy + sin(rad) * r;
    tft.drawLine(x0, y0, x1, y1, COL_LINE);
  }

  int endAngle = -140 + (int)(280.0f * value01);
  for (int a = -140; a <= endAngle; a += 4) {
    float rad = a * 0.0174533f;
    int x0 = cx + cos(rad) * (r - 8);
    int y0 = cy + sin(rad) * (r - 8);
    int x1 = cx + cos(rad) * r;
    int y1 = cy + sin(rad) * r;
    tft.drawLine(x0, y0, x1, y1, arcColor);
    tft.drawLine(x0, y0 - 1, x1, y1 - 1, arcColor);
  }

  float needleA = (-140 + 280.0f * value01) * 0.0174533f;
  int nx = cx + cos(needleA) * (r - 14);
  int ny = cy + sin(needleA) * (r - 14);
  tft.drawLine(cx, cy, nx, ny, arcColor);
  tft.drawLine(cx + 1, cy, nx + 1, ny, arcColor);
  tft.fillCircle(cx, cy, 3, arcColor);

  tft.setTextColor(COL_TEXT, bg);
  tft.drawCentreString(label, cx, cy + 8, 2);
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

  maybeCountStep(ax, ay, az);

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
  tft.drawString("Steps " + String(stepCount), 20, 212, 2);

  tft.setTextColor(accentColorForMode(currentMode), COL_PANEL_ALT);
  tft.drawRightString(lastGesture + " " + String(lastGestureScore, 2), 295, 212, 2);

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
  drawMiniCard(212, 144, 92, 52, "Motion", COL_GREEN2);

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

  tft.fillRoundRect(212, 144, 92, 52, 10, COL_PANEL);
  tft.drawRoundRect(212, 144, 92, 52, 10, COL_GREEN2);
  drawGauge(258, 173, 18, min(stepCount % 100, 100U) / 100.0f, COL_GREEN2, "step", COL_PANEL);

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
  tft.setTextColor(COL_PINK2, COL_PANEL);
  tft.drawString("AUDIO", 220, 56, 2);
  tft.setTextColor(COL_TEXT, COL_PANEL);
  tft.drawString("Energy", 28, 86, 4);

  soundStaticDirty = false;
}

void drawSoundBarsOnly() {
  int baseY = 178;
  int startX = 36;
  int barW = 22;
  int gap = 10;

  tft.fillRect(30, 96, 250, 88, COL_PANEL);

  for (int i = 0; i < 8; i++) {
    int h = soundBars[i];
    int x = startX + i * (barW + gap);
    int y = baseY - h;

    tft.fillRoundRect(x, y, barW, h, 6, COL_PINK2);
    tft.drawRoundRect(x, y, barW, h, 6, COL_PINK2);
    tft.drawFastVLine(x + 4, y + 4, max(0, h - 8), TFT_WHITE);
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
// STEP 页面
// =====================================================
void drawActivityStatic() {
  if (lastRenderedMode != MODE_ACTIVITY) {
    tft.fillScreen(COL_BG);
    lastRenderedMode = MODE_ACTIVITY;
  }

  fillRoundPanel(16, 46, 136, 150, 18, COL_PANEL, COL_GREEN2);
  fillRoundPanel(168, 46, 136, 150, 18, COL_PANEL, COL_GREEN2);

  tft.setTextColor(COL_SUBTEXT, COL_PANEL);
  tft.drawString("Steps", 28, 56, 2);
  tft.drawString("Gauge", 180, 56, 2);

  activityStaticDirty = false;
}

void drawActivityDynamicOnly() {
  tft.fillRect(22, 76, 124, 108, COL_PANEL);

  tft.setTextColor(COL_GREEN2, COL_PANEL);
  tft.drawCentreString(String(stepCount), 84, 92, 6);
  drawMiniLineChart(24, 132, 120, 52, stepHistory, 24, 0, 24, COL_GREEN2);

  tft.fillRoundRect(176, 76, 120, 104, 12, COL_FACE_BG);
  tft.drawRoundRect(176, 76, 120, 104, 12, COL_GREEN2);

  float ratio = min(stepCount % 100, 100U) / 100.0f;
  drawGauge(236, 124, 34, ratio, COL_GREEN2, "walk", COL_FACE_BG);

  tft.setTextColor(COL_GREEN2, COL_FACE_BG);
  tft.drawCentreString(String((int)(ratio * 100)), 236, 156, 2);

  activityDynamicDirty = false;
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
  else if (currentMode == MODE_ACTIVITY) {
    if (activityStaticDirty) drawActivityStatic();
    if (activityDynamicDirty) drawActivityDynamicOnly();
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
    if (lightContext != oldContext) footerDirty = true;
  }

  static unsigned long lastSoundAnimMs = 0;
  if (millis() - lastSoundAnimMs >= 220) {
    lastSoundAnimMs = millis();
    updateMockSoundBars();
    if (currentMode == MODE_SOUND) soundBarsDirty = true;
  }

  static unsigned long lastStepHistMs = 0;
  if (millis() - lastStepHistMs >= 700) {
    lastStepHistMs = millis();

    for (int i = 0; i < 23; i++) stepHistory[i] = stepHistory[i + 1];
    stepHistory[23] = min((int)(stepCount % 24), 24);

    if (currentMode == MODE_HOME) homeCornerDirty = true;
    if (currentMode == MODE_ACTIVITY) activityDynamicDirty = true;
  }

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

      if (shakeScore > 0.18f) {
        shakeAccum += shakeScore;
        if (shakeAccum > 1.2f) shakeAccum = 1.2f;
      } else {
        shakeAccum *= 0.65f;
      }

      lastGesture = label;
      lastGestureScore = score;
      footerDirty = true;

      if (currentMode == MODE_HOME) {
        homeCornerDirty = true;
        homeFaceDirty = true;
      }

      Serial.print("Pred: ");
      Serial.print(label);
      Serial.print(" score=");
      Serial.println(score, 3);

      Serial.print("shake=");
      Serial.print(shakeScore, 3);
      Serial.print(" lr=");
      Serial.print(lrScore, 3);
      Serial.print(" ud=");
      Serial.print(udScore, 3);
      Serial.print(" accum=");
      Serial.println(shakeAccum, 3);

      for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        Serial.print(result.classification[i].label);
        Serial.print(": ");
        Serial.println(result.classification[i].value, 3);
      }
      Serial.println("-----");

      bool canSwitch = millis() - lastModeSwitchMs >= MODE_SWITCH_COOLDOWN;
      bool canShake  = millis() - lastShakeTriggerMs >= SHAKE_TRIGGER_COOLDOWN;

      // 1) shake 优先，且当 shake 证据明显时不允许翻页
      bool shakeLikely = (shakeAccum > 0.55f || isShakeLabel(label));

      if (canShake &&
          shakeAccum > 0.75f &&
          shakeScore > SHAKE_TRIGGER_THRESHOLD &&
          shakeScore > lrScore - 0.03f &&
          shakeScore > udScore - 0.03f) {
        triggerDizzy();
        shakeAccum = 0.0f;
        Serial.println(">>> SHAKE TRIGGERED");
      }
      // 2) 只有 label 明确就是 left-right，才翻到下一页
      else if (canSwitch &&
               !shakeLikely &&
               isLeftRightLabel(label) &&
               score > LR_TRIGGER_THRESHOLD) {
        switchToNextMode();
      }
      // 3) 只有 label 明确就是 up-down，才返回上一页
      else if (canSwitch &&
               !shakeLikely &&
               isUpDownLabel(label) &&
               score > UD_TRIGGER_THRESHOLD) {
        switchToPrevMode();
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
