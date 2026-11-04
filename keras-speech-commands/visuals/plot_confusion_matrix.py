import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

# Load data and model
X = np.load("data/processed/features.npy")
y = np.load("data/processed/labels.npy")
model = load_model("model/wakeword_cnn_stage1.h5")

preds = (model.predict(X) > 0.5).astype(int).flatten()

cm = confusion_matrix(y, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Wake", "Wake"],
            yticklabels=["Non-Wake", "Wake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Wake vs Non-Wake")
plt.show()
