"""HRV stress dashboard for the ESP32 sketch.

Run:  py -m pip install streamlit pyserial pandas
      py -m streamlit run hrv_dashboard.py

Close the Arduino Serial Monitor first: only one program can use the COM port.
"""
import re
import time
import random

import pandas as pd
import serial
import serial.tools.list_ports
import streamlit as st

st.set_page_config(page_title="HRV stress monitor", page_icon="🫀", layout="centered")

MEASURE_S = 120  # must match MEASURE_MS in the sketch
TRAIN_MEANS = {"MEAN_RR": 847, "RMSSD": 15, "SDRR": 109, "pNN50": 0.87}
COLORS = {"healthy": "#2f7d4f", "caution": "#b7791f", "danger": "#b83b3b"}

RE_WIN = re.compile(r"win (\d+): meanRR=(-?\d+) RMSSD=([\d.]+) SDRR=([\d.]+) pNN50=([\d.]+)(.*)")
RE_CLS = re.compile(r"-> (\w+) \(([\d.]+)\)")
RE_VOTES = re.compile(r"VOTES caution=(\d+) danger=(\d+) healthy=(\d+)")
RE_BEATS = re.compile(r"Accepted beats: (\d+)")
RE_OVERALL = re.compile(r"OVERALL.*: (\w+)\s*$")


# ---------- line sources ----------
def serial_lines(port, answers):
    """Yield lines from the ESP32 and answer its prompts. Yields '' on read timeout."""
    ser = serial.Serial(port, 115200, timeout=1)
    idx = 0
    try:
        while True:
            raw = ser.readline()
            if not raw:
                yield ""
                continue
            line = raw.decode(errors="ignore").strip()
            if idx < len(answers) and answers[idx][0] in line:
                time.sleep(0.5)  # the sketch flushes input right after printing a prompt
                ser.write((answers[idx][1] + "\n").encode())
                idx += 1
            yield line
    finally:
        ser.close()


def demo_lines(activity):
    """Fake device output so the UI can be tested without hardware."""
    rnd = random.Random(7)
    yield "=== ESP32 HRV STRESS MONITOR ==="
    time.sleep(0.5)
    yield "Measuring for 5 minutes..."
    time.sleep(1.5)
    yield "Accepted beats: 340"
    names = ["healthy", "caution", "healthy", "danger", "healthy"]
    votes = {"caution": 0, "danger": 0, "healthy": 0}
    for i, n in enumerate(names):
        f = (rnd.randint(780, 900), rnd.uniform(11, 22), rnd.uniform(60, 140), rnd.uniform(0.2, 2))
        tail = f" -> {n} ({rnd.uniform(.4, .7):.2f})" if activity == 0 else " -> skipped (not at rest)"
        votes[n] += 1
        yield f"win {i}: meanRR={f[0]} RMSSD={f[1]:.1f} SDRR={f[2]:.1f} pNN50={f[3]:.1f}{tail}"
        time.sleep(0.4)
    if activity != 0:
        yield "No classification: exercise/walking changes HRV and would look like stress. Re-measure at rest."
    else:
        yield f"VOTES caution={votes['caution']} danger={votes['danger']} healthy={votes['healthy']}"
        yield "OVERALL (rough tendency, not a diagnosis): healthy"


# ---------- page ----------
st.title("HRV stress monitor")
st.caption("Pulse sensor on ESP32, random forest on the device. A rough tendency, not a diagnosis.")

ports = [p.device for p in serial.tools.list_ports.comports()]
port = st.selectbox("Serial port", ["Demo (no device)"] + ports)

c1, c2, c3 = st.columns(3)
age = c1.number_input("Age", 1, 120, 25)
smoking = c2.radio("Smoking", ["No", "Yes"], horizontal=True)
activity_label = c3.selectbox("Activity", ["Rest", "Walking", "Exercise"])
activity = ["Rest", "Walking", "Exercise"].index(activity_label)

start = st.button("Start 5-minute measurement", type="primary")

if start:
    answers = [
        ("Age in years", str(int(age))),
        ("Smoking?", "1" if smoking == "Yes" else "0"),
        ("Activity:", str(activity)),
        ("Place finger", "go"),  # the sketch ignores an empty Enter, so send a word
    ]
    src = demo_lines(activity) if port.startswith("Demo") else serial_lines(port, answers)

    wins, votes, overall, beats, note = [], None, None, None, None
    t_start, t_meas = time.time(), None
    status = st.empty()
    bar = st.progress(0.0)
    log = st.empty()
    lines = []
    status.info("Waiting for the ESP32. If nothing happens, press the EN or RESET button on the board.")

    try:
        for line in src:
            now = time.time()
            if line:
                lines.append(line)
                log.code("\n".join(lines[-8:]))
                m = RE_WIN.search(line)
                if m:
                    c = RE_CLS.search(m.group(6))
                    wins.append({
                        "window": int(m[1]), "MEAN_RR": float(m[2]), "RMSSD": float(m[3]),
                        "SDRR": float(m[4]), "pNN50": float(m[5]),
                        "result": c[1] if c else "skipped",
                        "confidence": float(c[2]) if c else None,
                    })
                if RE_BEATS.search(line):
                    beats = int(RE_BEATS.search(line)[1])
                if RE_VOTES.search(line):
                    votes = [int(x) for x in RE_VOTES.search(line).groups()]
                if RE_OVERALL.search(line):
                    overall = RE_OVERALL.search(line)[1]
                if "Measuring for" in line:
                    t_meas = now
                    status.info("Measuring. Keep the finger still on the sensor.")
                if "No classification" in line:
                    note = "Activity was not Rest, so no classification was made. Measure again at rest."
                if "Not enough clean beats" in line:
                    note = "Not enough clean beats. Check finger placement and signal quality."
            if t_meas:
                bar.progress(min((now - t_meas) / MEASURE_S, 1.0))
            if overall or note:
                break
            if now - t_start > MEASURE_S + 180:
                note = "Timed out waiting for the device."
                break
        status.success("Measurement finished.") if (overall and not note) else status.warning(note or "No result.")
    except serial.SerialException as e:
        status.error(f"Serial error: {e}. Close the Arduino Serial Monitor and try again.")
    finally:
        src.close()

    st.session_state["result"] = dict(wins=wins, votes=votes, overall=overall, beats=beats, note=note)

# ---------- results ----------
res = st.session_state.get("result")
if res:
    st.divider()
    if res["overall"]:
        color = COLORS.get(res["overall"], "#444")
        st.markdown(
            f"<div style='padding:1.2rem 1.4rem;border-left:8px solid {color};"
            f"background:{color}18;border-radius:6px'>"
            f"<div style='font-size:0.9rem;opacity:.75'>Overall tendency</div>"
            f"<div style='font-size:2.2rem;font-weight:700;color:{color}'>{res['overall'].capitalize()}</div></div>",
            unsafe_allow_html=True,
        )
    elif res["note"]:
        st.warning(res["note"])

    if res["beats"] is not None:
        st.metric("Clean beats accepted", res["beats"])

    if res["votes"]:
        st.subheader("Window votes")
        st.bar_chart(pd.DataFrame({"windows": res["votes"]}, index=["caution", "danger", "healthy"]))

    if res["wins"]:
        df = pd.DataFrame(res["wins"])
        st.subheader("Windows")
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("Sanity check against training data")
        cmp = pd.DataFrame({
            "your mean": [df[k].mean() for k in TRAIN_MEANS],
            "training mean": list(TRAIN_MEANS.values()),
        }, index=list(TRAIN_MEANS))
        st.dataframe(cmp.round(2), use_container_width=True)
        if df["RMSSD"].mean() > 3 * TRAIN_MEANS["RMSSD"] or not (500 < df["MEAN_RR"].mean() < 1300):
            st.warning("Your values are far from the training data. The sensor signal may be noisy, so the "
                       "result is not reliable. Check finger pressure and stay still.")

        st.download_button("Download windows as CSV", df.to_csv(index=False), "hrv_windows.csv", "text/csv")

    st.caption("Estimated on a cluster-based split of window-level data without subject IDs: expect lower "
               "accuracy on new people. Age, smoking and activity are not model inputs.")
