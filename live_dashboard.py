import streamlit as st
import pyaudio
import numpy as np
import tensorflow_hub as hub
import math
import pandas as pd
import time

st.set_page_config(page_title="Live OR Monitor", layout="wide")

# --- 1. AI Setup ---
@st.cache_resource
def load_yamnet():
    return hub.load('https://tfhub.dev/google/yamnet/1')

yamnet_model = load_yamnet()

SPEECH_IDX = 0
ALARM_INDICES = [382, 387, 388, 389, 393, 403, 404, 415, 418]

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

# Placeholders for our live updating UI
status_text = st.empty()
metric_row = st.empty()
sri_alert = st.empty()
chart_placeholder = st.empty()

if run_live:
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    status_text.success("Listening... Make some noise or play an alarm!")
    
    # Store history for the live graph
    history = pd.DataFrame(columns=["Time", "Volume (dB)", "Speech Prob", "Alarm Prob"])
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
            
            speech_prob = max_scores[SPEECH_IDX] * 100
            alarm_prob = min(sum([max_scores[idx] for idx in ALARM_INDICES]) * 100, 100.0)
            
            # --- NEW METRICS: Volatility & Instruction Confidence ---
            # Calculate rolling volatility from the last 30 seconds of history
            if len(history) > 1:
                loud_volatility = np.std(history["Volume (dB)"]) / 100.0 # Normalized
            else:
                loud_volatility = 0.0
            
            # The Proxy Metric: If it's loud/volatile AND people are talking, instructions are likely lost.
            instruction_confidence = max(0.0, 1.0 - ((speech_prob / 100.0) * loud_volatility)) * 100

            # --- THE NEW SRI FORMULA ---
            # 30% Volume | 30% Alarms | 20% Speech Density | 20% Loudness Volatility
            db_norm = min(db / 100.0, 1.0) 
            sri_score = ((db_norm * 0.3) + ((alarm_prob / 100) * 0.3) + ((speech_prob / 100) * 0.2) + (loud_volatility * 0.2)) * 100

            # Update Data History
            current_time = round(time.time() - start_time, 1)
            new_row = pd.DataFrame({
                "Time": [current_time], 
                "Volume (dB)": [db], 
                "Speech Prob": [speech_prob], 
                "Alarm Prob": [alarm_prob],
                "Volatility": [loud_volatility] # Added to history
            })
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Keep only the last 30 seconds so the graph doesn't lag
            if len(history) > 30:
                history = history.iloc[-30:]

            # --- LIVE UI UPDATES ---
            with metric_row.container():
                # Expanded to 5 columns to show off the new math!
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Live Volume", f"{db:.1f} dB")
                c2.metric("Speech Prob", f"{speech_prob:.1f}%")
                c3.metric("Alarm Prob", f"{alarm_prob:.1f}%")
                c4.metric("Volatility", f"{loud_volatility:.2f}")
                c5.metric("Instruction Clarity", f"{instruction_confidence:.1f}%")
            # Update Data History
            current_time = round(time.time() - start_time, 1)
            new_row = pd.DataFrame({"Time": [current_time], "Volume (dB)": [db], "Speech Prob": [speech_prob], "Alarm Prob": [alarm_prob]})
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Keep only the last 30 seconds so the graph doesn't lag
            if len(history) > 30:
                history = history.iloc[-30:]

            # --- LIVE UI UPDATES ---
            with metric_row.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Live Volume", f"{db:.1f} dB")
                c2.metric("Speech Probability", f"{speech_prob:.1f}%")
                c3.metric("Alarm Probability", f"{alarm_prob:.1f}%")
            
            with sri_alert.container():
                if sri_score < 40:
                    st.success(f"🟢 LOW RISK: {sri_score:.1f} / 100 (OR Environment is Stable)")
                elif sri_score < 75:
                    st.warning(f"🟡 ELEVATED RISK: {sri_score:.1f} / 100 (Monitor Cognitive Load)")
                else:
                    st.error(f"🔴 CRITICAL RISK: {sri_score:.1f} / 100 (High Noise & Alarm Fatigue!)")
            
            # Draw the live graph
            chart_placeholder.line_chart(history.set_index("Time"))

    except Exception as e:
        status_text.error(f"Audio Stream Error: {e}")
    finally:
        # Always clean up the mic when the checkbox is unchecked
        stream.stop_stream()
        stream.close()
        p.terminate()
        status_text.warning("Microphone disconnected.")
else:
    status_text.info("Check the box above to start the live feed.")