import streamlit as st
import pyaudio
import numpy as np
import tensorflow_hub as hub
import math
import pandas as pd
import time
from scipy.signal import find_peaks
import base64
import os

st.set_page_config(page_title="Live OR Monitor", layout="wide")

# =============================
# CONFIG (ORIGINAL DEVPOST MATH)
# =============================
RATE = 16000
CHUNK = 16000  # 1 second
FORMAT = pyaudio.paInt16
CHANNELS = 1

ALARM_MULTIPLIER = 2.5
CRITICAL_THRESHOLD = 60.0       # Lowered to 60 for easier demo triggering
LOG_COOLDOWN_SEC = 3.0          
ALERT_COOLDOWN_SEC = 10.0       

SPIKE_HEIGHT_PCT = 35.0
SPIKE_DISTANCE_SEC = 1.0

# --- ORIGINAL WEIGHTS ---
WEIGHTS = {
    "volume": 25,
    "alarm": 30,
    "speech": 20,
    "volatility": 15,
    "spikes": 10,
}

# =============================
# 1) AI Setup (YAMNet + classes)
# =============================
@st.cache_resource
def load_yamnet_and_classes():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    lines = open(class_map_path, "r").read().splitlines()
    classes = [ln.split(",")[2] for ln in lines[1:]]
    return model, classes

yamnet_model, yamnet_classes = load_yamnet_and_classes()

speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]

alarm_keywords = ["alarm", "beep", "buzzer", "siren", "bell", "ring", "pager", "chime", "tone"]
alarm_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in alarm_keywords)]

# =============================
# Helpers
# =============================
def get_decibels(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 0 else 0.0

def alarm_spikes_per_minute_from_history(history: pd.DataFrame, height_pct: float, distance_sec: float) -> float:
    if history is None or len(history) < 3:
        return 0.0
    alarm_series = history["Alarm Prob"].to_numpy(dtype=float)
    times = history["Time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    distance = max(1, int(distance_sec / max(dt, 1e-6)))
    peaks, _ = find_peaks(alarm_series, height=height_pct, distance=distance)
    duration_min = (times[-1] - times[0]) / 60.0 if times[-1] > times[0] else 0.0
    if duration_min <= 0:
        return 0.0
    return float(len(peaks) / duration_min)

# =============================
# 2) UI
# =============================
st.title("🎙️ AcoustiCare: Live OR Sentinel")
st.markdown("Real-time acoustic analysis using **YAMNet** + RMS Decibel Tracking.")

with st.sidebar:
    st.header("Controls")
    run_live = st.checkbox("🔴 ACTIVATE LIVE MIC", value=False)
    reset_log = st.button("🧹 Reset event log")

# =============================
# Session state
# =============================
if "events" not in st.session_state:
    st.session_state["events"] = []

if "start_epoch" not in st.session_state:
    st.session_state["start_epoch"] = None

if "last_alert_epoch" not in st.session_state:
    st.session_state["last_alert_epoch"] = 0.0

if "last_log_epoch" not in st.session_state:
    st.session_state["last_log_epoch"] = 0.0

if reset_log:
    st.session_state["events"] = []
    st.session_state["last_alert_epoch"] = 0.0
    st.session_state["last_log_epoch"] = 0.0

status_text = st.empty()
metric_row = st.empty()
sri_alert = st.empty()
chart_placeholder = st.empty()
audio_placeholder = st.empty() 

st.markdown("---")
st.subheader("Critical Risk Events Log")
log_placeholder = st.empty()
download_placeholder = st.empty()

# =============================
# 3) Main loop
# =============================
if run_live:
    if st.session_state["start_epoch"] is None:
        st.session_state["start_epoch"] = time.time()
        st.session_state["last_alert_epoch"] = 0.0
        st.session_state["last_log_epoch"] = 0.0

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    status_text.success("Listening… original algorithms loaded.")
    history = pd.DataFrame(columns=["Time", "Volume (dB)", "Speech Prob", "Alarm Prob", "Volatility", "Alarm Spikes/min", "SRI"])

    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            if len(audio_data) == 0:
                continue

            now_epoch = time.time()
            current_time = round(now_epoch - st.session_state["start_epoch"], 1)

            # --- ORIGINAL MATH SCALING ---
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
            db = get_decibels(rms)
            db_norm = min(db / 100.0, 1.0) 

            waveform = audio_data.astype(np.float32) / 32768.0
            scores, _, _ = yamnet_model(waveform)
            max_scores = np.max(scores.numpy(), axis=0)

            speech_prob = (max_scores[speech_idx].max() * 100.0) if len(speech_idx) else 0.0
            alarm_prob_raw = (np.clip(max_scores[alarm_idx].sum(), 0, 1) * 100.0) if len(alarm_idx) else 0.0
            alarm_prob = min(alarm_prob_raw * ALARM_MULTIPLIER, 100.0)

            if len(history) > 2:
                loud_volatility = float(np.std(history["Volume (dB)"])) / 100.0
            else:
                loud_volatility = 0.0

            instruction_confidence = max(0.0, 1.0 - ((speech_prob / 100.0) * loud_volatility)) * 100.0

            new_row = pd.DataFrame({
                "Time": [current_time],
                "Volume (dB)": [db],
                "Speech Prob": [speech_prob],
                "Alarm Prob": [alarm_prob],
                "Volatility": [loud_volatility],
                "Alarm Spikes/min": [np.nan],
                "SRI": [np.nan],
            })
            history = pd.concat([history, new_row], ignore_index=True)

            if len(history) > 30:
                history = history.iloc[-30:].reset_index(drop=True)

            alarm_spikes_pm = alarm_spikes_per_minute_from_history(history, float(SPIKE_HEIGHT_PCT), float(SPIKE_DISTANCE_SEC))
            alarm_spikes_norm = min(alarm_spikes_pm / 20.0, 1.0) 

            vol_contrib = db_norm * WEIGHTS["volume"]
            alarm_contrib = (alarm_prob / 100.0) * WEIGHTS["alarm"]
            speech_contrib = (speech_prob / 100.0) * WEIGHTS["speech"]
            volatility_contrib = loud_volatility * WEIGHTS["volatility"]
            spike_contrib = alarm_spikes_norm * WEIGHTS["spikes"]

            sri_score = float(vol_contrib + alarm_contrib + speech_contrib + volatility_contrib + spike_contrib)

            components_map = {
                "High Volume Levels": vol_contrib,
                "Active Equipment Alarms": alarm_contrib,
                "High Speech Density": speech_contrib,
                "Acoustic Volatility (Chaos)": volatility_contrib,
                "Alarm Spike Frequency": spike_contrib,
            }
            top_stressor = max(components_map, key=components_map.get)

            history.loc[history.index[-1], "Alarm Spikes/min"] = alarm_spikes_pm
            history.loc[history.index[-1], "SRI"] = sri_score

            if sri_score >= CRITICAL_THRESHOLD:
                if (now_epoch - st.session_state["last_log_epoch"]) >= LOG_COOLDOWN_SEC:
                    st.session_state["events"].append({
                        "Time (s)": current_time,
                        "SRI_Score": round(sri_score, 1),
                        "Primary_Stressor": top_stressor,
                    })
                    st.session_state["last_log_epoch"] = now_epoch

                if (now_epoch - st.session_state["last_alert_epoch"]) >= ALERT_COOLDOWN_SEC:
                    if os.path.exists("alert.mp3"):
                        b64 = base64.b64encode(open("alert.mp3", "rb").read()).decode()
                        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                        audio_placeholder.empty() 
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                        st.session_state["last_alert_epoch"] = now_epoch

            if now_epoch - st.session_state["last_alert_epoch"] > 3.0:
                audio_placeholder.empty()

            with metric_row.container():
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
                c1.metric("Live Volume", f"{db:.1f} dB")
                c2.metric("Speech", f"{speech_prob:.1f}%")
                c3.metric("Alarms", f"{alarm_prob:.1f}%")
                c4.metric("Volatility", f"{loud_volatility:.2f}")
                c5.metric("Spikes/min", f"{alarm_spikes_pm:.1f}")
                c6.metric("Instr. Clarity", f"{instruction_confidence:.1f}%")
                c7.metric("SRI", f"{sri_score:.1f}/100")
                c8.metric("Top stressor", top_stressor)

            with sri_alert.container():
                if sri_score < 40:
                    st.success(f"🟢 **LOW RISK: {sri_score:.1f} / 100**")
                elif sri_score < CRITICAL_THRESHOLD:
                    st.warning(f"🟡 **ELEVATED RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")
                else:
                    st.error(f"🔴 **CRITICAL RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")

            chart_placeholder.line_chart(
                history.set_index("Time")[["Volume (dB)", "Speech Prob", "Alarm Prob", "SRI"]]
            )

            events_df = pd.DataFrame(st.session_state["events"])
            log_placeholder.dataframe(events_df if not events_df.empty else events_df, use_container_width=True)

    except Exception as e:
        status_text.error(f"Audio Stream Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        status_text.warning("Microphone disconnected safely.")

else:
    st.session_state["start_epoch"] = None 
    status_text.info("Check the box above to start the live feed.")
    events_df = pd.DataFrame(st.session_state["events"])
    log_placeholder.dataframe(events_df if not events_df.empty else events_df, use_container_width=True)

    if not events_df.empty:
        csv = events_df.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="📥 Download Audit Log (CSV)",
            data=csv,
            file_name="acousticare_audit_log.csv",
            mime="text/csv",
        )