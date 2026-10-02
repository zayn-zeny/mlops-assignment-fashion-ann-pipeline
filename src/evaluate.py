"""
evaluate.py — Stage 4: Load trained model, evaluate on test set,
generate confusion matrix image, write metrics.json to project root.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def main():
    print("=== Stage: evaluate ===")

    # Load processed test data
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    print(f"  x_test: {x_test.shape}")

    # Load trained model
    model = tf.keras.models.load_model("models/model.h5")
    print("  Model loaded successfully")

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Predictions for confusion matrix
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate confusion matrix image
    os.makedirs("report_assets", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("report_assets/confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: report_assets/confusion_matrix.png")

    # Write metrics.json at project root
    metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json -> {metrics}")
    print("=== evaluate complete ===")


if __name__ == "__main__":
    main()
