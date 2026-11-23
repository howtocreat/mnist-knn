from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
FULL_DATASET = DATA_DIR / "mnist.npz"
QUICK_DATASET = DATA_DIR / "mnist-quick.npz"
FEATURE_COUNT = 28 * 28


def load_split(dataset, split):
    with np.load(dataset, allow_pickle=False) as arrays:
        images = arrays[f"{split}_images"]
        labels = arrays[f"{split}_labels"]
    features = images.reshape(-1, FEATURE_COUNT).astype(np.float32)
    features /= 255
    return features, labels
