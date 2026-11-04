# import librosa, matplotlib.pyplot as plt
# import numpy as np, librosa

# def extract_mfcc(file_path, sr=16000, n_mfcc=40):
#     y, _ = librosa.load(file_path, sr=sr)
#     if len(y) < sr: y = np.pad(y, (0, sr-len(y)))
#     elif len(y) > sr: y = y[:sr]
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     return (mfcc - np.mean(mfcc)) / np.std(mfcc)

# file = "data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"
# y, sr = librosa.load(file, sr=16000)
# plt.plot(y)
# plt.title(f"Waveform – {len(y)/sr:.2f}s, sr={sr}")
# plt.show()

import librosa
import numpy as np

def extract_mfcc(file_path, sr=16000, n_mfcc=40):
    """Extracts MFCC features from an audio file."""
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[WARNING] Could not read file {file_path}: {e}")
        return None

    # Ensure consistent length (1 second = 16000 samples)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    elif len(y) > sr:
        y = y[:sr]
    
    # if len(y) < 2 * sr:
    #     y = np.pad(y, (0, 2*sr - len(y)))
    # elif len(y) > 2 * sr:
    #     y = y[:2*sr]


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc
