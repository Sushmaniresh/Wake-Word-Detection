import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def evaluate_model(model_path="model/wakeword_cnn_stage1.h5", data_path="data/processed"):
    """
    Evaluates the trained wake word detection model.
    """
    print("[INFO] Loading data and model...")
    X = np.load(f"{data_path}/features.npy")
    y = np.load(f"{data_path}/labels.npy")
    model = load_model(model_path)

    print("[INFO] Running predictions...")
    preds = (model.predict(X) > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["Non-Wake", "Wake"]))

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Wake", "Wake"],
                yticklabels=["Non-Wake", "Wake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
