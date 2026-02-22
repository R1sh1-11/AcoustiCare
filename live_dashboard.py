import streamlit as st
import pyaudio
import numpy as np
import tensorflow_hub as hub
import math
import pandas as pd
import time
from scipy.signal import find_peaks
import streamlit.components.v1 as components

st.set_page_config(page_title="Live OR Monitor", layout="wide")

# =============================
# CONFIG (NO SLIDERS)
# =============================
RATE = 16000
CHUNK = 16000  # 1 second
FORMAT = pyaudio.paInt16
CHANNELS = 1

ALARM_MULTIPLIER = 2.5

CRITICAL_THRESHOLD = 75.0
LOG_COOLDOWN_SEC = 3.0          # log repeat cooldown
ALERT_COOLDOWN_SEC = 10.0       # audio repeat cooldown

SPIKE_HEIGHT_PCT = 35.0
SPIKE_DISTANCE_SEC = 1.0

# Weights (will be normalized to sum to 100 automatically)
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
    classes = [ln.split(",")[2] for ln in lines[1:]]  # index,mid,display_name
    return model, classes

yamnet_model, yamnet_classes = load_yamnet_and_classes()

speech_keywords = ["speech", "conversation", "narration", "shout", "screaming"]
speech_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in speech_keywords)]

alarm_keywords = ["alarm", "beep", "buzzer", "siren", "bell", "ring", "pager", "chime", "tone"]
alarm_idx = [i for i, c in enumerate(yamnet_classes) if any(k in c.lower() for k in alarm_keywords)]

# =============================
# Helpers
# =============================
def compute_dbfs(audio_int16: np.ndarray) -> float:
    """
    True digital loudness measure: dBFS (0 dBFS = max possible digital level).
    Range: (-inf..0]. Typical room audio might sit around -50 to -15 dBFS.
    """
    if audio_int16.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio_int16.astype(np.float64)))))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)

def norm_dbfs(dbfs: float, floor_dbfs: float = -60.0) -> float:
    """
    Map dbfs in [floor_dbfs..0] to [0..1]. Anything below floor -> 0.
    """
    x = (dbfs - floor_dbfs) / (0.0 - floor_dbfs)
    return float(np.clip(x, 0.0, 1.0))

def normalize_weights(w: dict) -> dict:
    s = float(sum(max(0.0, float(v)) for v in w.values()))
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: (float(v) / s) * 100.0 for k, v in w.items()}

def alarm_spikes_per_minute_from_history(history: pd.DataFrame,
                                        height_pct: float = 35.0,
                                        distance_sec: float = 1.0) -> float:
    if history is None or len(history) < 3:
        return 0.0

    alarm_series = history["Alarm Prob"].to_numpy(dtype=float)
    times = history["t_rel"].to_numpy(dtype=float)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    distance = max(1, int(distance_sec / max(dt, 1e-6)))

    peaks, _ = find_peaks(alarm_series, height=height_pct, distance=distance)

    duration_min = (times[-1] - times[0]) / 60.0 if times[-1] > times[0] else 0.0
    if duration_min <= 0:
        return 0.0

    return float(len(peaks) / duration_min)

def play_alert_autoplay(mp3_path: str, key: str):
    """
    Streamlit's st.audio autoplay can be inconsistent; this HTML audio tag
    tends to re-trigger more reliably when the HTML changes (unique key).
    """
    # Note: browser autoplay policies still apply; user may need to click once initially.
    html = f"""
    <audio autoplay>
      <source src="{mp3_path}?v={time.time()}" type="audio/mpeg">
    </audio>
    """
    components.html(html, height=0, width=0)

# =============================
# 2) UI
# =============================
st.title("🎙️ AcoustiCare: Live OR Sentinel")
st.markdown("Real-time acoustic analysis using **YAMNet** + dBFS + spike detection.")

with st.sidebar:
    st.header("Controls")
    run_live = st.checkbox("🔴 ACTIVATE LIVE MIC", value=False)

    reset_log = st.button("🧹 Reset event log")
    st.caption("Make sure `alert.mp3` is in the same folder as this app.")

# =============================
# Session state
# =============================
if "events" not in st.session_state:
    st.session_state["events"] = []  # each: {"t_epoch":..., "t_rel":..., "SRI":..., "Primary_Stressor":...}

if "start_epoch" not in st.session_state:
    st.session_state["start_epoch"] = None

if "last_alert_epoch" not in st.session_state:
    st.session_state["last_alert_epoch"] = 0.0

if "last_log_epoch" not in st.session_state:
    st.session_state["last_log_epoch"] = 0.0

if "alert_count" not in st.session_state:
    st.session_state["alert_count"] = 0

if reset_log:
    st.session_state["events"] = []
    st.session_state["last_alert_epoch"] = 0.0
    st.session_state["last_log_epoch"] = 0.0

status_text = st.empty()
metric_row = st.empty()
sri_alert = st.empty()
chart_placeholder = st.empty()

st.markdown("---")
st.subheader("Critical Risk Events Log")
log_placeholder = st.empty()
download_placeholder = st.empty()

# =============================
# 3) Main loop
# =============================
W = normalize_weights(WEIGHTS)

if run_live:
    # Reset timers on (re)start so cooldown works across runs
    if st.session_state["start_epoch"] is None:
        st.session_state["start_epoch"] = time.time()
        st.session_state["last_alert_epoch"] = 0.0
        st.session_state["last_log_epoch"] = 0.0

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    status_text.success("Listening… talk / clap / play alarm audio.")
    history = pd.DataFrame(columns=["t_rel", "dBFS", "Speech Prob", "Alarm Prob", "Volatility", "Alarm Spikes/min", "SRI"])

    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            if len(audio_data) == 0:
                continue

            now_epoch = time.time()
            t_rel = round(now_epoch - st.session_state["start_epoch"], 1)

            # Volume: dBFS + normalize
            dbfs = compute_dbfs(audio_data)
            db_norm = norm_dbfs(dbfs, floor_dbfs=-60.0)  # 0..1

            # YAMNet
            waveform = audio_data.astype(np.float32) / 32768.0
            scores, _, _ = yamnet_model(waveform)
            max_scores = np.max(scores.numpy(), axis=0)

            speech_prob = (max_scores[speech_idx].max() * 100.0) if len(speech_idx) else 0.0
            alarm_prob_raw = (np.clip(max_scores[alarm_idx].sum(), 0, 1) * 100.0) if len(alarm_idx) else 0.0
            alarm_prob = float(min(alarm_prob_raw * ALARM_MULTIPLIER, 100.0))

            # Volatility: std dev of dBFS over last ~30 seconds, scaled to 0..1-ish
            if len(history) >= 3:
                vol_raw = float(np.std(history["dBFS"].to_numpy(dtype=float)))
                # typical dBFS std might be ~0..10; cap at 15
                loud_volatility = float(np.clip(vol_raw / 15.0, 0.0, 1.0))
            else:
                loud_volatility = 0.0

            instruction_confidence = max(0.0, 1.0 - ((speech_prob / 100.0) * loud_volatility)) * 100.0

            # Append row first
            new_row = pd.DataFrame({
                "t_rel": [t_rel],
                "dBFS": [dbfs],
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

            # Spikes/min
            alarm_spikes_pm = alarm_spikes_per_minute_from_history(
                history,
                height_pct=float(SPIKE_HEIGHT_PCT),
                distance_sec=float(SPIKE_DISTANCE_SEC),
            )
            alarm_spikes_norm = min(alarm_spikes_pm / 20.0, 1.0)  # 20 spikes/min ~ max

            # SRI components (0..100, weights normalized)
            vol_contrib = db_norm * W["volume"]
            alarm_contrib = (alarm_prob / 100.0) * W["alarm"]
            speech_contrib = (speech_prob / 100.0) * W["speech"]
            volatility_contrib = loud_volatility * W["volatility"]
            spike_contrib = alarm_spikes_norm * W["spikes"]

            sri_score = float(vol_contrib + alarm_contrib + speech_contrib + volatility_contrib + spike_contrib)

            components_map = {
                "High Volume Levels": vol_contrib,
                "Active Equipment Alarms": alarm_contrib,
                "High Speech Density": speech_contrib,
                "Acoustic Volatility (Chaos)": volatility_contrib,
                "Alarm Spike Frequency": spike_contrib,
            }
            top_stressor = max(components_map, key=components_map.get)

            # Write back computed columns
            history.loc[history.index[-1], "Alarm Spikes/min"] = alarm_spikes_pm
            history.loc[history.index[-1], "SRI"] = sri_score

            # Critical handling (LOG + ALERT) using epoch time cooldowns
            if sri_score >= CRITICAL_THRESHOLD:
                if (now_epoch - st.session_state["last_log_epoch"]) >= LOG_COOLDOWN_SEC:
                    st.session_state["events"].append({
                        "t_epoch": now_epoch,
                        "t_rel": t_rel,
                        "SRI_Score": round(sri_score, 1),
                        "Primary_Stressor": top_stressor,
                    })
                    st.session_state["last_log_epoch"] = now_epoch

                if (now_epoch - st.session_state["last_alert_epoch"]) >= ALERT_COOLDOWN_SEC:
                    st.session_state["alert_count"] += 1
                    play_alert_autoplay("alert.mp3", key=f"alert_{st.session_state['alert_count']}")
                    st.session_state["last_alert_epoch"] = now_epoch

            # UI
            with metric_row.container():
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
                c1.metric("Live Volume", f"{dbfs:.1f} dBFS")
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
                history.set_index("t_rel")[["dBFS", "Speech Prob", "Alarm Prob", "SRI"]]
            )

            events_df = pd.DataFrame(st.session_state["events"])
            log_placeholder.dataframe(events_df[["t_rel", "SRI_Score", "Primary_Stressor"]] if not events_df.empty else events_df,
                                     use_container_width=True)

    except Exception as e:
        status_text.error(f"Audio Stream Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        status_text.warning("Microphone disconnected safely.")

else:
    # Not running live
    st.session_state["start_epoch"] = None  # important: allow a clean restart
    status_text.info("Check the box above to start the live feed.")
    events_df = pd.DataFrame(st.session_state["events"])
    log_placeholder.dataframe(events_df[["t_rel", "SRI_Score", "Primary_Stressor"]] if not events_df.empty else events_df,
                             use_container_width=True)

    if not events_df.empty:
        csv = events_df[["t_rel", "SRI_Score", "Primary_Stressor"]].to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="📥 Download Audit Log (CSV)",
            data=csv,
            file_name="acousticare_audit_log.csv",
            mime="text/csv",
        )
