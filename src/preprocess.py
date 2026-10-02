"""
preprocess.py — Stage 2: Normalize pixels, split validation set, save to data/processed/
Hyperparameters are read from params.yaml.
"""

import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess"]


def main():
    print("=== Stage: preprocess ===")

    # Load hyperparameters from params.yaml
    params = load_params()
    test_size = params["test_size"]
    seed = params["seed"]
    print(f"  test_size={test_size}, seed={seed}")

    # Load raw arrays
    x_train_raw = np.load("data/raw/x_train.npy")
    y_train_raw = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    print(f"  Loaded raw: x_train={x_train_raw.shape}, x_test={x_test.shape}")

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train_raw = x_train_raw.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split training data into train/validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_raw, y_train_raw,
        test_size=test_size,
        random_state=seed,
        stratify=y_train_raw
    )

    print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")

    # Save processed arrays
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("  Saved all processed arrays to data/processed/")
    print("=== preprocess complete ===")


if __name__ == "__main__":
    main()
