import streamlit as st
import pyaudio
import numpy as np
import tensorflow_hub as hub
import math
import pandas as pd
import time

st.set_page_config(page_title="Live OR Monitor", layout="wide")

# --- 1. AI Setup (Dynamic Mapping) ---
@st.cache_resource
def load_yamnet_and_classes():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    lines = open(class_map_path, "r").read().splitlines()
    classes = [ln.split(",")[2] for ln in lines[1:]]
    return model, classes

yamnet_model, yamnet_classes = load_yamnet_and_classes()

# Dynamically find indices instead of hardcoding
speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]

alarm_keywords = ["alarm", "beep", "buzzer", "siren", "bell", "ring", "pager", "chime", "tone"]
alarm_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in alarm_keywords)]

# --- 2. Audio Setup ---
RATE = 16000
CHUNK = 16000 # 1 second of audio
FORMAT = pyaudio.paInt16
CHANNELS = 1

def get_decibels(rms):
    return 20 * math.log10(rms) if rms > 0 else 0.0

# --- 3. UI Layout ---
st.title("🎙️ AcoustiCare: Live OR Sentinel")
st.markdown("Real-time acoustic analysis using YAMNet AI and RMS Decibel Tracking.")

run_live = st.checkbox("🔴 ACTIVATE LIVE MIC")

# Session state for our Event Log
if "events" not in st.session_state:
    st.session_state["events"] = []

# Placeholders for our live updating UI
status_text = st.empty()
metric_row = st.empty()
sri_alert = st.empty()
chart_placeholder = st.empty()

st.markdown("---")
st.subheader("Critical Risk Events Log")
log_placeholder = st.empty()
download_placeholder = st.empty()

if run_live:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    status_text.success("Listening... Make some noise or play an alarm!")
    
    history = pd.DataFrame(columns=["Time", "Volume (dB)", "Speech Prob", "Alarm Prob", "Volatility", "SRI"])
    start_time = time.time()

    try:
        while run_live:
            # Read audio chunk
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # Math & AI Processing
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
            db = get_decibels(rms)
            
            waveform = audio_data.astype(np.float32) / 32768.0
            scores, _, _ = yamnet_model(waveform)
            max_scores = np.max(scores.numpy(), axis=0)
            
            # Dynamic AI Probabilities
            speech_prob = (max_scores[speech_idx].max() * 100) if len(speech_idx) else 0.0
            alarm_prob  = (np.clip(max_scores[alarm_idx].sum(), 0, 1) * 100 * 2.5) if len(alarm_idx) else 0.0 # 2.5x multiplier for phone speakers
            
            # --- NEW METRICS: Volatility & Instruction Confidence ---
            if len(history) > 1:
                loud_volatility = np.std(history["Volume (dB)"]) / 100.0
            else:
                loud_volatility = 0.0
            
            instruction_confidence = max(0.0, 1.0 - ((speech_prob / 100.0) * loud_volatility)) * 100

            # --- THE NEW SRI FORMULA ---
            db_norm = min(db / 100.0, 1.0) 
            vol_contrib = db_norm * 30
            alarm_contrib = (alarm_prob / 100) * 30
            speech_contrib = (speech_prob / 100) * 20
            volatility_contrib = loud_volatility * 20
            
            sri_score = vol_contrib + alarm_contrib + speech_contrib + volatility_contrib

            components = {
                "High Volume Levels": vol_contrib,
                "Active Equipment Alarms": alarm_contrib,
                "High Speech Density": speech_contrib,
                "Acoustic Volatility (Chaos)": volatility_contrib
            }
            top_stressor = max(components, key=components.get)

            # Update Data History (ONE TIME)
            current_time = round(time.time() - start_time, 1)
            new_row = pd.DataFrame({
                "Time": [current_time], 
                "Volume (dB)": [db], 
                "Speech Prob": [speech_prob], 
                "Alarm Prob": [alarm_prob],
                "Volatility": [loud_volatility],
                "SRI": [sri_score]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            if len(history) > 30:
                history = history.iloc[-30:]

            # Log Critical Events (Judge Candy)
            # We use a simple time-gate so we don't spam the log 10 times for the same 10-second alarm
            if sri_score >= 75:
                if len(st.session_state["events"]) == 0 or (current_time - st.session_state["events"][-1]["time_sec"] > 3.0):
                    st.session_state["events"].append({
                        "time_sec": current_time,
                        "SRI_Score": round(sri_score, 1),
                        "Primary_Stressor": top_stressor
                    })

            # --- LIVE UI UPDATES (ONE TIME) ---
            with metric_row.container():
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Live Volume", f"{db:.1f} dB")
                c2.metric("Speech", f"{speech_prob:.1f}%")
                c3.metric("Alarms", f"{alarm_prob:.1f}%")
                c4.metric("Volatility", f"{loud_volatility:.2f}")
                c5.metric("Instr. Clarity", f"{instruction_confidence:.1f}%")
                c6.metric("SRI Score", f"{sri_score:.1f}/100")
            
            with sri_alert.container():
                if sri_score < 40:
                    st.success(f"🟢 **LOW RISK: {sri_score:.1f} / 100** (OR Environment is Stable)")
                elif sri_score < 75:
                    st.warning(f"🟡 **ELEVATED RISK: {sri_score:.1f} / 100** (Monitor Cognitive Load)\n\n**Primary Stressor:** {top_stressor}")
                else:
                    st.error(f"🔴 **CRITICAL RISK: {sri_score:.1f} / 100** (High Noise & Alarm Fatigue!)\n\n**Primary Stressor:** {top_stressor}")
            
            # Draw the live graph
            chart_placeholder.line_chart(history.set_index("Time")[["Volume (dB)", "Speech Prob", "Alarm Prob", "SRI"]])

            # Draw the Event Log
            events_df = pd.DataFrame(st.session_state["events"])
            log_placeholder.dataframe(events_df, use_container_width=True)
            
            # Draw Download Button
            if not events_df.empty:
                csv = events_df.to_csv(index=False).encode('utf-8')
                download_placeholder.download_button(
                    label="📥 Download Audit Log (CSV)",
                    data=csv,
                    file_name='acousticare_audit_log.csv',
                    mime='text/csv',
                )

    except Exception as e:
        status_text.error(f"Audio Stream Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        status_text.warning("Microphone disconnected.")
else:
    status_text.info("Check the box above to start the live feed.")
    
    # Still show the log even if mic is off!
    events_df = pd.DataFrame(st.session_state["events"])
    log_placeholder.dataframe(events_df, use_container_width=True)