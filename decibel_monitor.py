import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import math

print("🧠 Loading YAMNet AI Model... (This will take a few seconds)")
# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# YAMNet Class Indices we care about for the MVP
SPEECH_IDX = 0
# 388 is 'Alarm', 383 is 'Beep', 393 is 'Siren'. We can check a few!
# Expanded list: Beeps, Alarms, Sirens, Smoke Detectors, and Ringtones
ALARM_INDICES = [382, 387, 388, 389, 393, 403, 404, 415, 418]

# --- Audio Configuration ---
RATE = 16000             # YAMNet requires 16kHz!
CHUNK = 16000            # Process 1 second of audio at a time for the AI
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1             # Mono audio

def get_decibels(rms):
    if rms > 0:
        return 20 * math.log10(rms)
    return 0.0

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎙️ AcoustiCare Phase 2: Live AI Listening... (Press Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            # 1. Capture 1 second of audio
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # 2. Calculate raw loudness (Decibels)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
            db = get_decibels(rms)
            
            # 3. Format audio for YAMNet
            # YAMNet expects float32 values between -1.0 and 1.0
            waveform = audio_data.astype(np.float32) / 32768.0
            
            # 4. Run AI Prediction
            scores, embeddings, spectrogram = yamnet_model(waveform)
            
            # scores is a matrix of probabilities. Let's get the max probability for this 1-second chunk
            # YAMNet outputs multiple frames per second, so we take the mean across the chunk
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            # 5. Extract our MVP metrics
            speech_prob = mean_scores[SPEECH_IDX] * 100
            alarm_prob = sum([mean_scores[idx] for idx in ALARM_INDICES]) * 100
            
            # 6. Display the metrics
            print(f"🔊 Volume: {db:5.1f} dB | 🗣️ Speech Prob: {speech_prob:5.1f}% | 🚨 Alarm Prob: {alarm_prob:5.1f}%")

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()