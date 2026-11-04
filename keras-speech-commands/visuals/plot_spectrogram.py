import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Path to a wake word file (example)
file_path = "data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"

y, sr = librosa.load(file_path, sr=16000)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10,4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-Spectrogram of Wake Word 'Yes'")
plt.tight_layout()
plt.show()
