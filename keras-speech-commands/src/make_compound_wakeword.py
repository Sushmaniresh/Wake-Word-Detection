import os
import random
import numpy as np
import soundfile as sf

BASE_PATH = "data/speech_commands_v0.02"
OUT_PATH = os.path.join(BASE_PATH, "custom_cat_bed")

os.makedirs(OUT_PATH, exist_ok=True)

cat_files = [os.path.join(BASE_PATH, "cat", f) for f in os.listdir(os.path.join(BASE_PATH, "cat")) if f.endswith(".wav")]
bed_files = [os.path.join(BASE_PATH, "bed", f) for f in os.listdir(os.path.join(BASE_PATH, "bed")) if f.endswith(".wav")]

# Generate ~2000 combined samples (you can adjust)
N = 2000
print(f"[INFO] Generating {N} 'cat bed' combined samples...")

for i in range(N):
    f1 = random.choice(cat_files)
    f2 = random.choice(bed_files)

    y1, sr1 = sf.read(f1)
    y2, sr2 = sf.read(f2)

    # Ensure same sampling rate
    assert sr1 == sr2, "Sampling rates don't match!"

    # Optional short pause between words (0.1s)
    pause = np.zeros(int(0.1 * sr1))

    # Concatenate cat + pause + bed
    combined = np.concatenate([y1, pause, y2])

    # Normalize to avoid clipping
    combined = combined / np.max(np.abs(combined))

    out_file = os.path.join(OUT_PATH, f"cat_bed_{i:04d}.wav")
    sf.write(out_file, combined, sr1)

print(f"Done! Created {len(os.listdir(OUT_PATH))} combined samples in {OUT_PATH}")
