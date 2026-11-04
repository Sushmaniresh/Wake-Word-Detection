import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_cnn import build_cnn
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
import json
import os

def train_model(data_path="data/processed", model_path="model/wakeword_cnn_stage1.h5"):
    """
    Trains the CNN model using preprocessed MFCC features.
    """
    print("[INFO] Loading preprocessed data...")
    X = np.load(f"{data_path}/features.npy")
    y = np.load(f"{data_path}/labels.npy")

    # Normalize data
    X = (X - np.mean(X)) / np.std(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build model
    model = build_cnn(X_train.shape[1:])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

    # Add callbacks for stability
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    print("[INFO] Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15, batch_size=32,
        callbacks=callbacks, verbose=1
    )

    print(f"\nTraining complete! Model saved to: {model_path}")

    os.makedirs("model", exist_ok=True)
    with open("model/training_history.json", "w") as f:
        json.dump(history.history, f)

    print("[INFO] Training history saved to model/training_history.json")

    return model, history

if __name__ == "__main__":
    train_model()
