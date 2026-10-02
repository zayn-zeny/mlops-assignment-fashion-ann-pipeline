"""
train.py — Stage 3: Build and train Sequential ANN, save model and history.
Architecture: Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax)
Hyperparameters are read from params.yaml.
"""

import os
import csv
import yaml
import numpy as np
import tensorflow as tf


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def build_model(dense_units, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="fashion_ann")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=== Stage: train ===")

    # Load hyperparameters from params.yaml
    params = load_params()
    dense_units  = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs       = params["epochs"]
    batch_size   = params["batch_size"]

    print(f"  dense_units={dense_units}, dropout={dropout_rate}, "
          f"lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

    # Load processed data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val   = np.load("data/processed/x_val.npy")
    y_val   = np.load("data/processed/y_val.npy")

    print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}")

    # Build ANN
    model = build_model(dense_units, dropout_rate, learning_rate)
    model.summary()

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    print("  Saved: models/model.h5")

    # Save training history as CSV
    history_dict = history.history
    fieldnames = list(history_dict.keys())
    with open("models/history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history_dict[fieldnames[0]])):
            writer.writerow({k: history_dict[k][i] for k in fieldnames})
    print("  Saved: models/history.csv")
    print("=== train complete ===")


if __name__ == "__main__":
    main()
