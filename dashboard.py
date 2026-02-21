import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

import tensorflow as tf
import tensorflow_hub as hub


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Decibel + YAMNet Monitor", layout="wide")
TARGET_SR = 16000  # YAMNet expects 16kHz mono


# -----------------------------
# Load YAMNet once
# -----------------------------
@st.cache_resource
def load_yamnet_and_classes():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")

    # CSV: index,mid,display_name
    lines = open(class_map_path, "r").read().splitlines()
    classes = [ln.split(",")[2] for ln in lines[1:]]  # skip header
    return model, classes

yamnet, yamnet_classes = load_yamnet_and_classes()


# -----------------------------
# Helpers
# -----------------------------
def load_audio(file) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(file, sr=None, mono=True)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = y.astype(np.float32)
    # Normalize to avoid huge differences between files
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def chunk_audio(y: np.ndarray, sr: int, chunk_sec=1.0, hop_sec=1.0):
    chunk = int(chunk_sec * sr)
    hop = int(hop_sec * sr)
    chunks = []
    times = []
    for start in range(0, len(y) - chunk + 1, hop):
        end = start + chunk
        chunks.append(y[start:end])
        times.append(start / sr)
    if len(chunks) == 0:
        return np.zeros((0, chunk), dtype=np.float32), np.array([])
    return np.stack(chunks).astype(np.float32), np.array(times)

def loudness_rms(chunks: np.ndarray) -> np.ndarray:
    # RMS per chunk (proxy for loudness)
    return np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)

def yamnet_probs_for_chunks(chunks: np.ndarray) -> np.ndarray:
    """
    For each chunk, YAMNet outputs frame-level class probabilities.
    We average frames within chunk to get a chunk-level probability vector.
    """
    out = []
    for c in chunks:
        scores, embeddings, spectrogram = yamnet(c)
        out.append(scores.numpy().mean(axis=0))
    return np.stack(out) if len(out) else np.zeros((0, len(yamnet_classes)))


# -----------------------------
# UI
# -----------------------------
st.title("Decibel Monitor + YAMNet (Speech/Alarm)")

st.sidebar.header("Input")
audio_file = st.sidebar.file_uploader("Upload audio (wav/mp3/m4a/ogg)", type=["wav", "mp3", "m4a", "ogg"])

st.sidebar.header("Chunk settings")
chunk_sec = st.sidebar.slider("Chunk size (sec)", 0.25, 2.0, 1.0, 0.25)
hop_sec   = st.sidebar.slider("Hop size (sec)",   0.25, 2.0, 1.0, 0.25)

st.sidebar.header("Alarm label picker")
candidate_alarm_labels = [
    c for c in yamnet_classes
    if any(k in c.lower() for k in ["alarm", "beep", "bleep", "buzzer", "siren", "bell", "ring", "ringtone", "pager", "chime", "tone"])
]
default_alarm = candidate_alarm_labels[:10] if len(candidate_alarm_labels) >= 10 else candidate_alarm_labels

selected_alarm_labels = st.sidebar.multiselect(
    "Select which YAMNet classes count as 'alarm/equipment'",
    options=candidate_alarm_labels,
    default=default_alarm
)

selected_alarm_idx = [i for i, c in enumerate(yamnet_classes) if c in selected_alarm_labels]
st.sidebar.caption(f"Selected alarm classes: {len(selected_alarm_idx)}")

st.sidebar.header("Speech detection")
speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]
st.sidebar.caption(f"Speech-related classes found: {len(speech_idx)}")


if not audio_file:
    st.info("Upload an audio clip to analyze. Tip: for alarm testing, use a simple repeated beep (monitor-like).")
    st.stop()

# Play audio
st.audio(audio_file)


# -----------------------------
# Processing
# -----------------------------
y, sr = load_audio(audio_file)
duration_sec = len(y) / sr

chunks, times = chunk_audio(y, sr, chunk_sec=chunk_sec, hop_sec=hop_sec)

if len(chunks) == 0:
    st.error("Audio is too short for the selected chunk/hop settings. Reduce chunk size.")
    st.stop()

with st.spinner("Running YAMNet on audio chunks…"):
    probs = yamnet_probs_for_chunks(chunks)

# Loudness
rms = loudness_rms(chunks)
rms95 = np.percentile(rms, 95) + 1e-9
loud_norm = np.clip(rms / rms95, 0, 1)  # 0..1 scaled loudness proxy

# Speech probability (use max across speech-related classes)
if len(speech_idx) > 0:
    speech_prob = probs[:, speech_idx].max(axis=1)
else:
    speech_prob = np.zeros(len(times), dtype=np.float32)

# Alarm probability (sum across selected alarm classes, clipped)
if len(selected_alarm_idx) > 0:
    alarm_prob = np.clip(probs[:, selected_alarm_idx].sum(axis=1), 0, 1)
else:
    alarm_prob = np.zeros(len(times), dtype=np.float32)

speech_fraction = float(np.mean(speech_prob))
alarm_fraction = float(np.mean(alarm_prob))

# Debug: what labels are most present overall?
mean_probs = probs.mean(axis=0)
topk_overall = np.argsort(mean_probs)[::-1][:12]
top_overall_labels = [(yamnet_classes[i], float(mean_probs[i])) for i in topk_overall]

# Debug: what labels appear at the loudest chunk?
peak_chunk = int(np.argmax(loud_norm))
topk_peak = np.argsort(probs[peak_chunk])[::-1][:12]
top_peak_labels = [(yamnet_classes[i], float(probs[peak_chunk][i])) for i in topk_peak]


# -----------------------------
# Display metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration (sec)", f"{duration_sec:.1f}")
c2.metric("Avg loudness (0–1)", f"{float(np.mean(loud_norm)):.2f}")
c3.metric("Speech activity (avg)", f"{speech_fraction:.2f}")
c4.metric("Alarm activity (avg)", f"{alarm_fraction:.2f}")

# Timeline dataframe
timeline = pd.DataFrame({
    "time_sec": times,
    "loud_norm": loud_norm,
    "speech_prob": speech_prob,
    "alarm_prob": alarm_prob,
})

st.subheader("Timeline (normalized)")
fig = plt.figure()
plt.plot(timeline["time_sec"], timeline["loud_norm"], label="loudness (norm)")
plt.plot(timeline["time_sec"], timeline["speech_prob"], label="speech prob")
plt.plot(timeline["time_sec"], timeline["alarm_prob"], label="alarm prob")
plt.ylim(0, 1.05)
plt.xlabel("time (sec)")
plt.ylabel("level (0..1)")
plt.legend()
st.pyplot(fig)

st.subheader("Timeline data")
st.dataframe(timeline, use_container_width=True)

# Debug panels (super useful for fixing “alarm=0%”)
with st.expander("DEBUG: Top detected classes (overall)"):
    st.write(top_overall_labels)

with st.expander("DEBUG: Top detected classes (at loudest moment)"):
    st.write(f"Peak chunk starts at ~{times[peak_chunk]:.2f}s")
    st.write(top_peak_labels)
