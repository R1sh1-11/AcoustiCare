import pyaudio
import numpy as np
import math
import time

# --- Configuration ---
CHUNK = 1024             # Number of audio frames per buffer
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1             # Mono audio
RATE = 44100             # 44.1kHz sampling rate

def get_decibels(rms):
    """Converts RMS amplitude to decibels."""
    if rms > 0:
        return 20 * math.log10(rms)
    return 0.0

def main():
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎙️ AcoustiCare Phase 1: Listening... (Make some noise! Press Ctrl+C to stop)")

    try:
        while True:
            # Read a chunk of raw data from the microphone
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert raw bytes into a numpy array of integers
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # Calculate the Root Mean Square (RMS)
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))
            
            # Convert RMS to Decibels
            db = get_decibels(rms)
            
            # Print a visual bar and the dB value
            bar = "|" * int(db / 2)  
            print(f"Volume: {db:5.1f} dB  {bar}")
            
            # Sleep slightly to make the terminal output readable
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        # Clean up to avoid locking the microphone
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()