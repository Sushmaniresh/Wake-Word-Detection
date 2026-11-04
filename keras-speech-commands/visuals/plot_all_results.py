# Combine ROC + PR curves (optional for paper figure)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

X = np.load("data/processed/features.npy")
y = np.load("data/processed/labels.npy")
model = load_model("model/wakeword_cnn_stage1.h5")

y_scores = model.predict(X).flatten()
fpr, tpr, _ = roc_curve(y, y_scores)
precision, recall, _ = precision_recall_curve(y, y_scores)
roc_auc = auc(fpr, tpr)
avg_precision = average_precision_score(y, y_scores)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='darkorange', label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, color='blue', label=f"AP = {avg_precision:.2f}")
plt.title("Precision–Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()

plt.tight_layout()
plt.show()
