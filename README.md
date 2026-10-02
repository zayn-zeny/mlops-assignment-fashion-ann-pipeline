# Fashion-ANN Pipeline

End-to-End ML Versioning with Git, DVC & Google Drive

## Assignment 3 — MLOps

This project implements a fully-connected Artificial Neural Network (ANN) trained on the Fashion-MNIST dataset, managed with Git for code versioning and DVC for data/model versioning, backed by Google Drive as remote storage.

## Dataset

**Fashion-MNIST**: 70,000 grayscale 28×28 images across 10 clothing categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Loaded directly via `tf.keras.datasets.fashion_mnist.load_data()`.

## Project Structure

```
fashion-ann-pipeline/
├── src/
│   ├── prepare.py       # Load Fashion-MNIST → data/raw/
│   ├── preprocess.py    # Normalize + split → data/processed/
│   ├── train.py         # Train ANN → models/
│   └── evaluate.py      # Evaluate → metrics.json
├── data/
│   ├── raw/             # DVC-tracked raw arrays
│   └── processed/       # DVC-tracked processed arrays
├── models/              # DVC-tracked trained model
├── params.yaml          # Hyperparameters
├── dvc.yaml             # DVC pipeline definition
├── dvc.lock             # DVC pipeline lock file
├── metrics.json         # Model evaluation metrics
└── report_assets/       # Screenshots for PDF report
```

## Pipeline

Run the full pipeline with:

```bash
dvc repro
```

Or run each stage independently:

```bash
python src/prepare.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Hyperparameters

All hyperparameters are defined in `params.yaml`. Edit that file and re-run `dvc repro` to reproduce with new settings.

## DVC Remote

Artifacts (raw data, processed data, models) are versioned with DVC and stored on Google Drive.

```bash
dvc pull   # Download artifacts
dvc push   # Upload artifacts
```

## Author

zayn-zeny
