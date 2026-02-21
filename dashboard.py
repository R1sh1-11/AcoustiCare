import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import tensorflow_hub as hub
from scipy.signal import find_peaks

st.set_page_config(page_title="Static OR Monitor", layout="wide")
TARGET_SR = 16000

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
    y = y / (np.max(np.abs(y)) + 1e-9)
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

def loudness_rms(chunks):
    return np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)

def yamnet_probs_for_chunks(chunks):
    out = []
    for c in chunks:
        scores, _, _ = yamnet(c)
        out.append(scores.numpy().mean(axis=0))
    return np.stack(out) if len(out) else np.zeros((0, len(yamnet_classes)))

def alarm_spikes_per_minute(alarm_prob: np.ndarray, times: np.ndarray,
                            height: float = 0.35, distance_sec: float = 1.0) -> float:
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

    st.header("Alarm spike detection")
    spike_height = st.slider("Alarm peak threshold (0..1)", 0.05, 0.95, 0.35, 0.05)
    spike_distance = st.slider("Min seconds between spikes", 0.5, 5.0, 1.0, 0.5)

    st.header("SRI weights (sum ≈ 100)")
    w_volume = st.slider("Avg loudness weight", 0, 60, 25, 1)
    w_alarm  = st.slider("Peak alarm weight", 0, 60, 30, 1)
    w_speech = st.slider("Speech density weight", 0, 60, 20, 1)
    w_volat  = st.slider("Volatility weight", 0, 60, 15, 1)
    w_spike  = st.slider("Alarm spikes weight", 0, 60, 10, 1)

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

with st.spinner("Running AI inference…"):
    probs = yamnet_probs_for_chunks(chunks)

# loudness (normalized)
rms = loudness_rms(chunks)
rms95 = np.percentile(rms, 95) + 1e-9
loud_norm = np.clip(rms / rms95, 0, 1)

speech_prob = probs[:, speech_idx].max(axis=1) if len(speech_idx) else np.zeros(len(times))
alarm_prob = np.clip(probs[:, alarm_idx].sum(axis=1), 0, 1) if len(alarm_idx) else np.zeros(len(times))

speech_fraction = float(np.mean(speech_prob))
peak_alarm = float(np.max(alarm_prob))
avg_loudness = float(np.mean(loud_norm))
loud_volatility = float(np.std(loud_norm))

instruction_confidence = max(0.0, 1.0 - (speech_fraction * loud_volatility)) * 100

# alarm spikes/min
alarm_spikes_pm = alarm_spikes_per_minute(
    alarm_prob, times, height=float(spike_height), distance_sec=float(spike_distance)
)
alarm_spikes_norm = min(alarm_spikes_pm / 20.0, 1.0)

# -----------------------------
# SRI
# -----------------------------
vol_contrib = avg_loudness * w_volume
alarm_contrib = peak_alarm * w_alarm
speech_contrib = speech_fraction * w_speech
volatility_contrib = min(loud_volatility * 2.0, 1.0) * w_volat
spike_contrib = alarm_spikes_norm * w_spike

sri_score = vol_contrib + alarm_contrib + speech_contrib + volatility_contrib + spike_contrib

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
elif sri_score < 75:
    st.warning(f"🟡 **ELEVATED RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")
else:
    st.error(f"🔴 **CRITICAL RISK: {sri_score:.1f} / 100**\n\n**Primary Stressor:** {top_stressor}")

st.markdown("---")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Duration", f"{duration_sec:.1f}s")
c2.metric("Speech", f"{speech_fraction:.2f}")
c3.metric("Peak Alarm", f"{peak_alarm:.2f}")
c4.metric("Avg Loudness", f"{avg_loudness:.2f}")
c5.metric("Volatility", f"{loud_volatility:.2f}")
c6.metric("Alarm Spikes/min", f"{alarm_spikes_pm:.1f}")
c7.metric("Instr. Clarity", f"{instruction_confidence:.1f}%")

# timeline plot
timeline = pd.DataFrame({
    "time_sec": times,
    "loud_norm": loud_norm,
    "speech_prob": speech_prob,
    "alarm_prob": alarm_prob,
})

st.subheader("Post-Op Timeline Analysis")
fig = plt.figure()
plt.plot(timeline["time_sec"], timeline["loud_norm"], label="Loudness (norm)")
plt.plot(timeline["time_sec"], timeline["speech_prob"], label="Speech prob")
plt.plot(timeline["time_sec"], timeline["alarm_prob"], label="Alarm prob")
plt.ylim(0, 1.05)
plt.xlabel("Time (sec)")
plt.ylabel("Intensity (0..1)")
plt.legend()
st.pyplot(fig)

with st.expander("Timeline data"):
    st.dataframe(timeline, use_container_width=True)