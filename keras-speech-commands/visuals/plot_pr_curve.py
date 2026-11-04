import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve, average_precision_score

X = np.load("data/processed/features.npy")
y = np.load("data/processed/labels.npy")
model = load_model("model/wakeword_cnn_stage1.h5")

y_scores = model.predict(X).flatten()
precision, recall, _ = precision_recall_curve(y, y_scores)
avg_precision = average_precision_score(y, y_scores)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='b', lw=2, label=f"AP = {avg_precision:.2f}")
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()
