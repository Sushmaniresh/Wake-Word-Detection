### For small Pharses ###
# import sounddevice as sd
# import numpy as np
# import librosa
# import time
# import datetime
# import simpleaudio as sa
# from tensorflow.keras.models import load_model

# # ✅ Model and audio parameters
# MODEL_PATH = "model/wakeword_cnn_stage1.h5"
# SAMPLE_RATE = 16000      # must match training
# DURATION = 1.2           # seconds per frame
# OVERLAP = 0.5            # seconds overlap between windows
# THRESHOLD = 0.7          # wake-word detection confidence

# # ✅ Load trained model
# print(f"[INFO] Loading model from {MODEL_PATH} ...")
# model = load_model(MODEL_PATH)
# print("[INFO] Model loaded successfully!")

# # ✅ Optional: short beep sound when wake word is detected
# def play_beep():
#     frequency = 880  # Hz
#     duration_ms = 200
#     t = np.linspace(0, duration_ms / 1000, int(SAMPLE_RATE * duration_ms / 1000), False)
#     tone = np.sin(frequency * 2 * np.pi * t)
#     audio = np.int16(tone * 32767)
#     sa.play_buffer(audio, 1, 2, SAMPLE_RATE)

# def extract_mfcc_from_array(audio_array, sr=16000, n_mfcc=40):
#     """Extract normalized MFCCs from a numpy audio array."""
#     try:
#         if len(audio_array) < sr:
#             audio_array = np.pad(audio_array, (0, sr - len(audio_array)))
#         elif len(audio_array) > sr:
#             audio_array = audio_array[:sr]
#         mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
#         mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
#         return mfcc
#     except Exception as e:
#         print(f"[WARNING] MFCC extraction failed: {e}")
#         return None

# def listen_continuously():
#     """Continuous detection with overlapping windows."""
#     print("\nListening for wake word... (Ctrl+C to stop)\n")

#     buffer_size = int(SAMPLE_RATE * DURATION)
#     step_size = int(SAMPLE_RATE * (DURATION - OVERLAP))
#     audio_buffer = np.zeros(buffer_size, dtype='float32')

#     try:
#         with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='float32') as stream:
#             while True:
#                 # Shift buffer left by step_size
#                 audio_buffer[:-step_size] = audio_buffer[step_size:]
#                 # Read new audio chunk
#                 new_chunk, _ = stream.read(step_size)
#                 audio_buffer[-step_size:] = np.squeeze(new_chunk)

#                 mfcc = extract_mfcc_from_array(audio_buffer, sr=SAMPLE_RATE)
#                 if mfcc is None:
#                     continue

#                 mfcc = mfcc[np.newaxis, ..., np.newaxis]
#                 pred = model.predict(mfcc, verbose=0)[0][0]

#                 if pred > THRESHOLD:
#                     timestamp = datetime.datetime.now().strftime("%H:%M:%S")
#                     print(f"[{timestamp}] Wake word detected! (Confidence: {pred:.2f})")
#                     play_beep()
#                     time.sleep(1.0)  # cool-down
#                 else:
#                     print(f"Listening... ({pred:.2f})", end="\r")

#     except KeyboardInterrupt:
#         print("\nDetection stopped by user.")
#     except Exception as e:
#         print(f"[ERROR] {e}")

# if __name__ == "__main__":
#     listen_continuously()

### For longer Pharses ###
import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model/wakeword_cnn_stage1.h5"
SAMPLE_RATE = 16000      # Hz
DURATION = 2.0            # seconds (longer than before, for "cat bed")
THRESHOLD = 0.8           # detection confidence threshold
MFCC_N = 40               # number of MFCC features

# -----------------------------
# Load Model
# -----------------------------
print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!\n")

# -----------------------------
# Helper Functions
# -----------------------------
def extract_mfcc(audio, sr):
    """Extract MFCC features compatible with training data."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_N)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

def listen_and_detect():
    print("🎧 Listening for wake phrase 'cat bed' ...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Record audio
            recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = np.squeeze(recording)

            # Normalize
            audio = audio / np.max(np.abs(audio) + 1e-8)

            # Extract MFCC features
            mfcc = extract_mfcc(audio, SAMPLE_RATE)
            mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

            # Predict
            preds = model.predict(mfcc, verbose=0)
            confidence = float(preds[0][0])

            if confidence >= THRESHOLD:
                print(f"Wake phrase 'cat bed' detected! (Confidence: {confidence:.2f})")
                time.sleep(2.0)  # Small delay before listening again
            else:
                print(f"Listening... (Confidence: {confidence:.2f})")

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    listen_and_detect()
