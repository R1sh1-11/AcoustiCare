import streamlit as st
import pyaudio
import numpy as np
import tensorflow_hub as hub
import math
import pandas as pd
import time
from scipy.signal import find_peaks

st.set_page_config(page_title="Live OR Monitor", layout="wide")

# -----------------------------
# 1) AI Setup (YAMNet + classes)
# -----------------------------
@st.cache_resource
def load_yamnet_and_classes():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    lines = open(class_map_path, "r").read().splitlines()
    # CSV: index,mid,display_name
    classes = [ln.split(",")[2] for ln in lines[1:]]
    return model, classes

yamnet_model, yamnet_classes = load_yamnet_and_classes()

speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]

alarm_keywords = ["alarm", "beep", "buzzer", "siren", "bell", "ring", "pager", "chime", "tone"]
alarm_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in alarm_keywords)]

# -----------------------------
# 2) Audio Setup
# -----------------------------
RATE = 16000
CHUNK = 16000  # 1 second
FORMAT = pyaudio.paInt16
CHANNELS = 1

def get_decibels(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 0 else 0.0

def alarm_spikes_per_minute_from_history(history: pd.DataFrame,
                                        height_pct: float = 35.0,
                                        distance_sec: float = 1.0) -> float:
    """
    Count peaks in Alarm Prob (%) over the current history window and return spikes/min.
    height_pct: threshold in percent (0..100)
    distance_sec: min time between spikes
    """
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

# -----------------------------
# 3) UI
# -----------------------------
st.title("🎙️ AcoustiCare: Live OR Sentinel")
st.markdown("Real-time acoustic analysis using **YAMNet** + RMS Decibel Tracking.")

with st.sidebar:
    st.header("Live settings")
    run_live = st.checkbox("🔴 ACTIVATE LIVE MIC")

    st.subheader("Alarm sensitivity")
    alarm_multiplier = st.slider("Alarm boost (for phone speakers)", 1.0, 4.0, 2.5, 0.1)

    st.subheader("Event logging")
    critical_threshold = st.slider("Critical SRI threshold", 50, 95, 75, 1)
    log_cooldown_sec = st.slider("Log cooldown (sec)", 1.0, 10.0, 3.0, 0.5)

    st.subheader("Alarm spikes detection")
    spike_height_pct = st.slider("Alarm peak threshold (%)", 5, 95, 35, 1)
    spike_distance_sec = st.slider("Min seconds between spikes", 0.5, 5.0, 1.0, 0.5)

    st.subheader("SRI weights (sum ≈ 100)")
    w_volume = st.slider("Volume weight", 0, 60, 25, 1)
    w_alarm  = st.slider("Alarm burden weight", 0, 60, 30, 1)
    w_speech = st.slider("Speech weight", 0, 60, 20, 1)
    w_volat  = st.slider("Volatility weight", 0, 60, 15, 1)
    w_spike  = st.slider("Alarm spikes weight", 0, 60, 10, 1)

# session event log
if "events" not in st.session_state:
    st.session_state["events"] = []

status_text = st.empty()
metric_row = st.empty()
sri_alert = st.empty()
chart_placeholder = st.empty()

st.markdown("---")
st.subheader("Critical Risk Events Log")
log_placeholder = st.empty()
download_placeholder = st.empty()

# -----------------------------
# 4) Run loop
# -----------------------------
if run_live:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    status_text.success("Listening… talk / clap / play alarm audio.")
    history = pd.DataFrame(columns=["Time", "Volume (dB)", "Speech Prob", "Alarm Prob", "Volatility", "Alarm Spikes/min", "SRI"])
    start_time = time.time()

    try:
        while run_live:
            try: # <-- THE BULLETPROOF AUDIO WRAPPER
                raw_data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                if len(audio_data) == 0:
                    continue

                # volume
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
                db = get_decibels(rms)

                # YAMNet
                waveform = audio_data.astype(np.float32) / 32768.0
                scores, _, _ = yamnet_model(waveform)
                max_scores = np.max(scores.numpy(), axis=0)

                # speech/alarm probs (percent)
                speech_prob = (max_scores[speech_idx].max() * 100) if len(speech_idx) else 0.0
                alarm_prob_raw = (np.clip(max_scores[alarm_idx].sum(), 0, 1) * 100) if len(alarm_idx) else 0.0
                alarm_prob = min(alarm_prob_raw * alarm_multiplier, 100.0)

                # volatility from last ~30 seconds of dB history
                if len(history) > 2:
                    loud_volatility = float(np.std(history["Volume (dB)"])) / 100.0  # normalize
                else:
                    loud_volatility = 0.0

                instruction_confidence = max(0.0, 1.0 - ((speech_prob / 100.0) * loud_volatility)) * 100

                # update time + history first (so spike calc sees newest point)
                current_time = round(time.time() - start_time, 1)

                # temporary row (spike rate computed after concat)
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

                # keep last 30 seconds (approx 30 samples)
                if len(history) > 30:
                    history = history.iloc[-30:].reset_index(drop=True)

                # alarm spikes/min
                alarm_spikes_pm = alarm_spikes_per_minute_from_history(
                    history,
                    height_pct=float(spike_height_pct),
                    distance_sec=float(spike_distance_sec),
                )
                alarm_spikes_norm = min(alarm_spikes_pm / 20.0, 1.0)  # 20 spikes/min ~ max

                # SRI contributions (0..100)
                db_norm = min(db / 100.0, 1.0)
                vol_contrib = db_norm * w_volume
                alarm_contrib = (alarm_prob / 100.0) * w_alarm
                speech_contrib = (speech_prob / 100.0) * w_speech
                volatility_contrib = loud_volatility * w_volat
                spike_contrib = alarm_spikes_norm * w_spike

                sri_score = vol_contrib + alarm_contrib + speech_contrib + volatility_contrib + spike_contrib

                components = {
                    "High Volume Levels": vol_contrib,
                    "Active Equipment Alarms": alarm_contrib,
                    "High Speech Density": speech_contrib,
                    "Acoustic Volatility (Chaos)": volatility_contrib,
                    "Alarm Spike Frequency": spike_contrib
                }
                top_stressor = max(components, key=components.get)

                # write back computed columns for last row
                history.loc[history.index[-1], "Alarm Spikes/min"] = alarm_spikes_pm
                history.loc[history.index[-1], "SRI"] = sri_score

                # log critical events with cooldown
                if sri_score >= critical_threshold:
                    if len(st.session_state["events"]) == 0 or (current_time - st.session_state["events"][-1]["time_sec"] > log_cooldown_sec):
                        st.session_state["events"].append({
                            "time_sec": current_time,
                            "SRI_Score": round(float(sri_score), 1),
                            "Primary_Stressor": top_stressor
                        })

                # UI
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
                    elif sri_score < critical_threshold:
                        st.warning(f"🟡 **ELEVATED RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")
                    else:
                        st.error(f"🔴 **CRITICAL RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")

                chart_placeholder.line_chart(
                    history.set_index("Time")[["Volume (dB)", "Speech Prob", "Alarm Prob", "SRI"]]
                )

                events_df = pd.DataFrame(st.session_state["events"])
                log_placeholder.dataframe(events_df, use_container_width=True)

                # THE DOWNLOAD BUTTON HAS BEEN SAFELY REMOVED FROM THIS LOOP!

            except Exception as inner_e:
                continue # Ignore dropped audio frames and keep going!

    except Exception as e:
        status_text.error(f"Audio Stream Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        status_text.warning("Microphone disconnected safely.")

# --- THE SAFE OUTSIDE BLOCK ---
else:
    status_text.info("Check the box above to start the live feed.")
    events_df = pd.DataFrame(st.session_state["events"])
    log_placeholder.dataframe(events_df, use_container_width=True)
    
    # Download button is safely outside the active loop!
    if not events_df.empty:
        csv = events_df.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="📥 Download Audit Log (CSV)",
            data=csv,
            file_name="acousticare_audit_log.csv",
            mime="text/csv",
        )