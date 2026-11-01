#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <math.h>
#include <stdint.h>

// Enable SMTP and debugging features (must be before ReadyMail.h)
#define ENABLE_SMTP
#define ENABLE_DEBUG
#define READYMAIL_DEBUG_PORT Serial
#include <ReadyMail.h>

#include "hrv_model.h"   // Kaggle model

// ---------------- TEST SWITCH ----------------
// 1 = only test WiFi + email (skips everything else)
// 0 = normal operation (measure, or type 3 for manual test mode)
#define TEST_EMAIL_ONLY 0

// ---------------- WIFI ----------------
#define WIFI_SSID     "FUJIN"
#define WIFI_PASSWORD "123123123"

// ---------------- SMTP ----------------
#define SMTP_SERVER     "smtp.gmail.com"
#define SMTP_PORT       465
#define SENDER_EMAIL    "kuk17550@gmail.com"
#define SENDER_PASSWORD "vald lqfc ford yzyq"
#define RECIPIENT_EMAIL "ananttindori@gmail.com"

WiFiClientSecure ssl_client;
SMTPClient smtp(ssl_client);

auto smtpStatusCallback = [](SMTPStatus status) {
  Serial.println(status.text);
};

// ---------------- HRV SETTINGS ----------------
#define PULSE_PIN 34
#define SENSOR_POWER_PIN 27
#define WIN_BEATS 64                 // MUST match the training window (unverified)
const int SAMPLE_RATE = 250;
const unsigned long SAMPLE_US = 1000000UL / SAMPLE_RATE;
const unsigned long MEASURE_MS = 120000UL;   // 2 minutes
const int TARGET_BEATS = 200;                // or stop at 200 clean beats
const int MAX_RR_N = 600;

float rr[MAX_RR_N]; int rrN = 0;
float filt = 0, base = 0, env = 0;
float a0 = 0, a1 = 0, a2 = 0;
int64_t t1 = 0, t2 = 0, lastPeakUs = 0;

int userAge = 0, smoking = 0, activity = 0;   // context only, NOT model inputs

const char *names[] = {"caution", "danger", "healthy"};

// ---------- model ----------
int hrv_predict(const float *x, float *p) {
  p[0] = p[1] = p[2] = 0;
  for (int t = 0; t < HRV_TREES; t++) {
    int k = hrv_root[t];
    while (hrv_feat[k] >= 0)
      k = (x[hrv_feat[k]] <= hrv_thr[k]) ? hrv_left[k] : hrv_right[k];
    for (int c = 0; c < 3; c++) p[c] += hrv_val[k][c];
  }
  int b = 0; for (int c = 1; c < 3; c++) if (p[c] > p[b]) b = c;
  return b;                          // 0=caution 1=danger 2=healthy
}

// ---------- features: MEAN_RR,RMSSD,SDRR,MEDIAN_RR,SDSD,pNN50,SD1,SD2 ----------
void features(const float *x, float *f) {
  const int n = WIN_BEATS;
  float mean = 0; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
  float v = 0; for (int i = 0; i < n; i++) v += (x[i]-mean)*(x[i]-mean);
  float sdrr = sqrtf(v / (n-1));
  float dm = 0, sq = 0; int over = 0;
  for (int i = 0; i < n-1; i++) {
    float d = x[i+1]-x[i]; dm += d; sq += d*d;
    if (fabsf(d) > 50.0f) over++;
  }
  dm /= (n-1);
  float dv = 0;
  for (int i = 0; i < n-1; i++) { float d = x[i+1]-x[i]; dv += (d-dm)*(d-dm); }
  float sdsd = sqrtf(dv / (n-2));
  float rmssd = sqrtf(sq / (n-1));
  float s[WIN_BEATS]; for (int i = 0; i < n; i++) s[i] = x[i];
  for (int i = 1; i < n; i++) { float k = s[i]; int j = i-1;
    while (j >= 0 && s[j] > k) { s[j+1] = s[j]; j--; } s[j+1] = k; }
  float med = (n % 2) ? s[n/2] : 0.5f*(s[n/2-1] + s[n/2]);
  float sd2sq = 2*sdrr*sdrr - 0.5f*sdsd*sdsd;
  f[0] = mean; f[1] = rmssd; f[2] = sdrr; f[3] = med; f[4] = sdsd;
  f[5] = 100.0f * over / (n-1);                  // percent, as in training
  f[6] = sqrtf(0.5f) * sdsd;
  f[7] = sqrtf(sd2sq > 0 ? sd2sq : 0);
}

// ---------- beat detection ----------
float median5() {
  float s[5]; for (int i = 0; i < 5; i++) s[i] = rr[rrN-5+i];
  for (int i = 1; i < 5; i++) { float k = s[i]; int j = i-1;
    while (j >= 0 && s[j] > k) { s[j+1] = s[j]; j--; } s[j+1] = k; }
  return s[2];
}

void processSample(int64_t tUs) {
  int raw = analogRead(PULSE_PIN);
  filt += 0.3f * (raw - filt);
  base += 0.005f * (filt - base);
  float ac = filt - base;
  env += 0.01f * (fabsf(ac) - env);
  a0 = a1; a1 = a2; a2 = ac; t1 = t2; t2 = tUs;

  if (a1 > a0 && a1 >= a2 && a1 > 0.5f * env && env > 5.0f) {
    float den = a0 - 2*a1 + a2;
    float delta = (den != 0) ? 0.5f * (a0 - a2) / den : 0;
    int64_t peakUs = t1 + (int64_t)(delta * SAMPLE_US);
    if (lastPeakUs == 0 || peakUs - lastPeakUs > 300000) {
      if (lastPeakUs != 0) {
        float r = (peakUs - lastPeakUs) / 1000.0f;
        bool ok = (r >= 300 && r <= 2000);
        if (ok && rrN >= 5) { float m = median5(); ok = fabsf(r - m) < 0.25f * m; }
        if (ok && rrN < MAX_RR_N) rr[rrN++] = r;
      }
      lastPeakUs = peakUs;
    }
  }
}

// ---------- serial input ----------
String readLine(const char *msg) {
  Serial.println(msg);
  while (Serial.available()) Serial.read();
  while (true) {
    if (Serial.available()) {
      String s = Serial.readStringUntil('\n'); s.trim();
      if (s.length() > 0) return s;
    }
    delay(10);
  }
}

float readFloat(const char *msg) {
  return readLine(msg).toFloat();
}

void getUserInfo() {
  userAge = readLine("Age in years:").toInt();
  while (userAge < 1 || userAge > 120) userAge = readLine("Invalid. Age:").toInt();
  smoking = readLine("Smoking? 0=No 1=Yes:").toInt();
  while (smoking != 0 && smoking != 1) smoking = readLine("Enter 0 or 1:").toInt();
  activity = readLine("Activity: 0=Rest 1=Walking 2=Exercise:").toInt();
  while (activity < 0 || activity > 2) activity = readLine("Enter 0, 1 or 2:").toInt();
}

// ---------- WiFi ----------
bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  unsigned long st = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - st < 20000) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi connected! IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }
  Serial.println("Wi-Fi connection failed!");
  return false;
}

// ---------- email ----------
// isTest = true sends a clearly labelled TEST email (connectivity check only)
bool sendAlertEmail(int age, float meanRR, int dangerVotes, int windows, bool isTest) {
  if (!connectWiFi()) { Serial.println("Email not sent (no Wi-Fi)."); return false; }

  Serial.println("\nStarting SMTP configuration...");
  ssl_client.stop();          // close any previous connection (allows several sends per run)
  ssl_client.setInsecure();   // testing only; verify the root CA in production

  Serial.println("Connecting to SMTP server...");
  smtp.connect(SMTP_SERVER, SMTP_PORT, smtpStatusCallback);
  if (!smtp.isConnected()) {
    Serial.println("SMTP connection failed!");
    return false;
  }
  Serial.println("SMTP server connected!");

  smtp.authenticate(SENDER_EMAIL, SENDER_PASSWORD, readymail_auth_password);
  if (!smtp.isAuthenticated()) {
    Serial.println("SMTP authentication failed!");
    return false;
  }
  Serial.println("SMTP authentication successful!");

  // Sync clock for the email Date header
  configTime(0, 0, "pool.ntp.org");
  Serial.println("Synchronizing time...");
  unsigned long st = millis();
  while (time(nullptr) < 100000 && millis() - st < 15000) delay(200);
  if (time(nullptr) < 100000) {
    Serial.println("Time synchronization failed!");
    return false;
  }

  SMTPMessage message;
  message.headers.add(rfc822_from, "HRV Monitor <" + String(SENDER_EMAIL) + ">");
  message.headers.add(rfc822_to, "Receiver <" + String(RECIPIENT_EMAIL) + ">");
  message.headers.add(rfc822_subject,
      isTest ? "TEST: HRV monitor email check" : "ALERT: HRV monitor flagged DANGER");

  String body;
  if (isTest) {
    body = "This is a TEST email from the HRV monitor. No real alert.\r\n\r\n";
  } else {
    body = "The HRV monitor flagged a possible danger state.\r\n\r\n";
  }
  body += "Age: " + String(age) + "\r\n";
  body += "Danger windows: " + String(dangerVotes) + " of " + String(windows) + "\r\n";
  body += "Mean RR: " + String(meanRR, 0) + " ms\r\n\r\n";
  body += "This is a rough screening tendency, not a medical diagnosis. "
          "Please check on the person and seek medical help if they are unwell.";
  message.text.body(body);
  message.timestamp = time(nullptr);

  Serial.println("Sending email...");
  smtp.send(message);   // final result is printed by the status callback
  return true;
}

// ---------- manual test mode ----------
// Type the 8 HRV features by hand, run the model, email if it says "danger".
void manualTest() {
  const char *labels[8] = {
    "MEAN_RR (ms):", "RMSSD (ms):", "SDRR (ms):", "MEDIAN_RR (ms):",
    "SDSD (ms):", "pNN50 (%):", "SD1 (ms):", "SD2 (ms):"
  };

  Serial.println("\n=== MANUAL TEST MODE ===");
  Serial.println("Enter the 8 HRV values. The sensor is not used.");

  while (true) {
    float f[8], p[3];
    for (int i = 0; i < 8; i++) f[i] = readFloat(labels[i]);

    int c = hrv_predict(f, p);
    Serial.println("\n--- Model result ---");
    Serial.printf("Scores: caution=%.2f danger=%.2f healthy=%.2f\n",
                  p[0] / HRV_TREES, p[1] / HRV_TREES, p[2] / HRV_TREES);
    Serial.printf("CLASSIFIED AS: %s\n", names[c]);

    if (c == 1) {
      Serial.println("DANGER detected -> sending alert email...");
      sendAlertEmail(userAge, f[0], 1, 1, false);
    } else {
      Serial.println("Not danger, no alert email sent.");
    }

    String a = readLine("\nType a=enter new values, e=send TEST email anyway, q=quit:");
    a.toLowerCase();
    if (a == "e") {
      sendAlertEmail(userAge, f[0], 0, 1, true);
      a = readLine("Type a=enter new values, q=quit:");
      a.toLowerCase();
    }
    if (a == "q") { Serial.println("Manual test finished."); return; }
  }
}

// ---------- main ----------
void setup() {
  Serial.begin(115200);

#if TEST_EMAIL_ONLY
  Serial.println("\n=== EMAIL TEST MODE ===");
  sendAlertEmail(25, 800, 0, 1, true);
  return;
#endif

  analogReadResolution(12);
  analogSetPinAttenuation(PULSE_PIN, ADC_11db);
  pinMode(PULSE_PIN, INPUT);
  pinMode(SENSOR_POWER_PIN, OUTPUT);
  digitalWrite(SENSOR_POWER_PIN, HIGH);     // power sensor first
  delay(2000);

  Serial.println("\n=== ESP32 HRV STRESS MONITOR ===");
  getUserInfo();

  String go = readLine("Place finger on sensor, keep still, press Enter to measure (or type 3 for manual test):");
  if (go == "3") {
    manualTest();
    return;
  }

  filt = base = analogRead(PULSE_PIN); env = 0;
  unsigned long st = millis();
  while (millis() - st < 4000) {            // warm-up, no detection
    int raw = analogRead(PULSE_PIN);
    filt += 0.3f * (raw - filt); base += 0.02f * (filt - base);
    env += 0.02f * (fabsf(filt - base) - env);
    delay(4);
  }

  Serial.println("Measuring for up to 2 minutes or 200 beats...");
  unsigned long startMs = millis();
  int64_t next = micros();
  while (millis() - startMs < MEASURE_MS && rrN < TARGET_BEATS) {
    if ((int64_t)micros() - next >= 0) { processSample(next); next += SAMPLE_US; }
  }
  Serial.printf("\nAccepted beats: %d\n", rrN);

  int windows = rrN / WIN_BEATS;
  if (windows < 1) { Serial.println("Not enough clean beats. Check finger/signal."); return; }

  int votes[3] = {0, 0, 0};
  int goodWindows = 0;                      // windows that look like a real heartbeat
  for (int w = 0; w < windows; w++) {
    float f[8], p[3];
    features(&rr[w * WIN_BEATS], f);
    if (f[0] >= 400 && f[0] <= 1500 && f[2] > 5.0f) goodWindows++;
    Serial.printf("win %d: meanRR=%.0f RMSSD=%.1f SDRR=%.1f pNN50=%.1f",
                  w, f[0], f[1], f[2], f[5]);
    if (activity == 0) {
      int c = hrv_predict(f, p); votes[c]++;
      Serial.printf(" -> %s (%.2f)\n", names[c], p[c] / HRV_TREES);
    } else Serial.println(" -> skipped (not at rest)");
  }

  Serial.printf("\nAge=%d Smoking=%d Activity=%d (context only)\n", userAge, smoking, activity);
  if (activity != 0) {
    Serial.println("No classification: exercise/walking changes HRV and would look like stress. Re-measure at rest.");
  } else {
    int best = 0; for (int c = 1; c < 3; c++) if (votes[c] > votes[best]) best = c;
    Serial.printf("VOTES caution=%d danger=%d healthy=%d\n", votes[0], votes[1], votes[2]);
    Serial.printf("Good-quality windows: %d of %d\n", goodWindows, windows);
    Serial.printf("OVERALL (rough tendency, not a diagnosis): %s\n", names[best]);

    if (goodWindows * 2 < windows) {
      Serial.println("Signal quality too low, no alert sent.");
    } else if (votes[1] * 2 >= windows) {
      float avgRR = 0;
      for (int i = 0; i < windows * WIN_BEATS; i++) avgRR += rr[i];
      avgRR /= (windows * WIN_BEATS);
      sendAlertEmail(userAge, avgRR, votes[1], windows, false);
    } else {
      Serial.println("Not in danger, no email sent.");
    }
  }
}

void loop() {}
