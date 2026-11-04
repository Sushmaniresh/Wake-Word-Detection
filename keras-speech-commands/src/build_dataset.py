import os
import glob
import numpy as np
from feature_extraction import extract_mfcc

# Correct dataset path (matches your actual folder)
DATASET_PATH = "data/speech_commands_v0.02"

# Folder where processed features will be saved
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Define your wake word
# WAKE_WORD = "custom_cat_bed"
WAKE_WORD = "yes"

# Get list of non-wake classes
nonwake_classes = [
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
    and d not in [WAKE_WORD, "_background_noise_"]
]

X, y = [], []

print(f"[INFO] Using dataset from: {DATASET_PATH}")
print(f"[INFO] Wake word: '{WAKE_WORD}'")
print(f"[INFO] Found {len(nonwake_classes)} non-wake classes")

# Load wake word samples
wake_path = os.path.join(DATASET_PATH, WAKE_WORD)
for file in glob.glob(f"{wake_path}/*.wav"):
    X.append(extract_mfcc(file))
    y.append(1)

# Load a subset of non-wake samples (3x wake samples)
for cls in nonwake_classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    files = glob.glob(f"{cls_path}/*.wav")
    for file in files[:len(X) * 3]:
        X.append(extract_mfcc(file))
        y.append(0)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN
X = X[..., np.newaxis]

# Save processed arrays
np.save(os.path.join(PROCESSED_PATH, "features.npy"), X)
np.save(os.path.join(PROCESSED_PATH, "labels.npy"), y)

print("\nDataset built successfully!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
