import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import tensorflow_hub as hub
from scipy.signal import find_peaks

st.set_page_config(page_title="Static OR Monitor", layout="wide")
TARGET_SR = 16000

# --- HARDCODED DEMO SETTINGS (Synced with Live) ---
alarm_multiplier = 2.5
critical_threshold = 60.0    # Lowered to match live dashboard
spike_height_pct = 35.0 
spike_distance_sec = 1.0

# --- ORIGINAL WEIGHTS ---
w_volume = 25
w_alarm  = 30
w_speech = 20
w_volat  = 15
w_spike  = 10

# -----------------------------
# Load YAMNet + classes
# -----------------------------
@st.cache_resource
def load_yamnet_and_classes():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    lines = open(class_map_path, "r").read().splitlines()
    classes = [ln.split(",")[2] for ln in lines[1:]]
    return model, classes

yamnet, yamnet_classes = load_yamnet_and_classes()

speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]

alarm_keywords = ["alarm", "beep", "buzzer", "siren", "bell", "ring", "pager", "chime", "tone"]
alarm_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in alarm_keywords)]

# -----------------------------
# Helpers
# -----------------------------
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = y.astype(np.float32)
    return y, sr

def chunk_audio(y, sr, chunk_sec=1.0, hop_sec=1.0):
    chunk = int(chunk_sec * sr)
    hop = int(hop_sec * sr)
    chunks, times = [], []
    for start in range(0, len(y) - chunk + 1, hop):
        end = start + chunk
        chunks.append(y[start:end])
        times.append(start / sr)
    if len(chunks) == 0:
        return np.zeros((0, chunk), dtype=np.float32), np.array([])
    return np.stack(chunks).astype(np.float32), np.array(times)

def yamnet_probs_for_chunks(chunks):
    norm_chunks = chunks / (np.max(np.abs(chunks)) + 1e-9)
    out = []
    for c in norm_chunks:
        scores, _, _ = yamnet(c)
        out.append(scores.numpy().mean(axis=0))
    return np.stack(out) if len(out) else np.zeros((0, len(yamnet_classes)))

def alarm_spikes_per_minute(alarm_prob: np.ndarray, times: np.ndarray,
                            height: float, distance_sec: float) -> float:
    if len(alarm_prob) < 3 or len(times) < 3:
        return 0.0
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    distance = max(1, int(distance_sec / max(dt, 1e-6)))
    peaks, _ = find_peaks(alarm_prob, height=height, distance=distance)
    duration_min = (times[-1] - times[0]) / 60.0 if times[-1] > times[0] else 0.0
    if duration_min <= 0:
        return 0.0
    return float(len(peaks) / duration_min)

# -----------------------------
# UI
# -----------------------------
st.title("📂 AcoustiCare: Post-Op Audit Dashboard")
st.markdown("Upload surgical audio logs to analyze the **Surgical Risk Index (SRI)**.")

with st.sidebar:
    st.header("Upload")
    audio_file = st.file_uploader("Upload audio (wav/mp3/m4a/ogg)", type=["wav", "mp3", "m4a", "ogg"])
    st.info("Demo mode engaged. Acoustic sensitivity locked to original project logic.")

if not audio_file:
    st.info("Upload an audio clip to begin the audit. Tip: use a clip with clear alarms and overlapping speech.")
    st.stop()

st.audio(audio_file)

# -----------------------------
# Processing
# -----------------------------
y, sr = load_audio(audio_file)
duration_sec = len(y) / sr
chunks, times = chunk_audio(y, sr, chunk_sec=1.0, hop_sec=1.0)

if len(chunks) == 0:
    st.error("Audio is too short for the selected chunk settings.")
    st.stop()

with st.spinner("Running AI inference & Acoustic Analysis…"):
    probs = yamnet_probs_for_chunks(chunks)

# --- MATH SYNC WITH LIVE DASHBOARD ---

# 1. Volume & dB Math (Simulating PyAudio int16 amplitude to match live mic)
chunks_int16 = chunks * 32768.0
rms_array = np.sqrt(np.mean(chunks_int16**2, axis=1) + 1e-12)
db_array = np.clip(20 * np.log10(rms_array), 0, None)

# 2. Probability Math (Raw, no memory buffer)
speech_prob_array = probs[:, speech_idx].max(axis=1) * 100.0 if len(speech_idx) else np.zeros(len(times))
raw_alarm_prob = probs[:, alarm_idx].sum(axis=1) * 100.0 if len(alarm_idx) else np.zeros(len(times))
alarm_prob_array = np.clip(raw_alarm_prob * alarm_multiplier, 0, 100.0)

# 3. File-Level Aggregation for SRI
avg_db = float(np.mean(db_array))
speech_fraction = float(np.mean(speech_prob_array)) / 100.0
peak_alarm = float(np.max(alarm_prob_array)) / 100.0

# 4. Volatility (Original / 100.0 scaling)
loud_volatility = float(np.std(db_array)) / 100.0

instruction_confidence = max(0.0, 1.0 - (speech_fraction * loud_volatility)) * 100.0

# 5. Alarm Spikes (Original / 20.0 scaling)
alarm_spikes_pm = alarm_spikes_per_minute(
    alarm_prob_array, times, height=spike_height_pct, distance_sec=spike_distance_sec
)
alarm_spikes_norm = min(alarm_spikes_pm / 20.0, 1.0)

# -----------------------------
# SRI Calculation
# -----------------------------
# 6. Volume Sensitivity (Original / 100.0 scaling)
db_norm = min(avg_db / 100.0, 1.0)

vol_contrib = db_norm * w_volume
alarm_contrib = peak_alarm * w_alarm
speech_contrib = speech_fraction * w_speech
volatility_contrib = loud_volatility * w_volat
spike_contrib = alarm_spikes_norm * w_spike

sri_score = float(vol_contrib + alarm_contrib + speech_contrib + volatility_contrib + spike_contrib)

components = {
    "High Volume Levels": vol_contrib,
    "Active Equipment Alarms": alarm_contrib,
    "High Speech Density": speech_contrib,
    "Acoustic Volatility (Chaos)": volatility_contrib,
    "Alarm Spike Frequency": spike_contrib,
}
top_stressor = max(components, key=components.get)

st.markdown("---")
st.subheader("Global Surgical Risk Index (SRI)")

if sri_score < 40:
    st.success(f"🟢 **LOW RISK: {sri_score:.1f} / 100** (Environment is stable)")
elif sri_score < critical_threshold:
    st.warning(f"🟡 **ELEVATED RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")
else:
    st.error(f"🔴 **CRITICAL RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")
    # NO AUDIO ALERT IN THIS FILE

st.markdown("---")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Duration", f"{duration_sec:.1f}s")
c2.metric("Speech", f"{speech_fraction*100:.1f}%")
c3.metric("Peak Alarm", f"{peak_alarm*100:.1f}%")
c4.metric("Avg Loudness", f"{avg_db:.1f} dB")
c5.metric("Volatility", f"{float(np.std(db_array)):.1f} dB (Dev)")
c6.metric("Alarm Spikes/min", f"{alarm_spikes_pm:.1f}")
c7.metric("Instr. Clarity", f"{instruction_confidence:.1f}%")

# timeline plot
timeline = pd.DataFrame({
    "time_sec": times,
    "Volume (dB)": db_array,
    "Speech Prob (%)": speech_prob_array,
    "Alarm Prob (%)": alarm_prob_array,
})

st.subheader("Post-Op Timeline Analysis")
fig = plt.figure()
plt.plot(timeline["time_sec"], timeline["Volume (dB)"], label="Loudness (dB)")
plt.plot(timeline["time_sec"], timeline["Speech Prob (%)"], label="Speech prob (%)")
plt.plot(timeline["time_sec"], timeline["Alarm Prob (%)"], label="Alarm prob (%)")
plt.ylim(0, 105)
plt.xlabel("Time (sec)")
plt.ylabel("Metrics (0-100)")
plt.legend()
st.pyplot(fig)

with st.expander("Timeline data"):
    st.dataframe(timeline, use_container_width=True)