"""
prepare.py — Stage 1: Load Fashion-MNIST and save raw arrays to data/raw/
"""

import os
import numpy as np
import tensorflow as tf


def main():
    print("=== Stage: prepare ===")
    print("Loading Fashion-MNIST via tf.keras.datasets...")

    # Load dataset directly from Keras (no manual download needed)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"  x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  x_test shape:  {x_test.shape}")
    print(f"  y_test shape:  {y_test.shape}")

    # Ensure output directory exists
    os.makedirs("data/raw", exist_ok=True)

    # Save as .npy files
    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("  Saved: data/raw/x_train.npy")
    print("  Saved: data/raw/y_train.npy")
    print("  Saved: data/raw/x_test.npy")
    print("  Saved: data/raw/y_test.npy")
    print("=== prepare complete ===")


if __name__ == "__main__":
    main()
