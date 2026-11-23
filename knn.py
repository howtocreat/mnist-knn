from collections import Counter

import numpy as np


class CustomKNNClassifier:
    """KNN with Euclidean or Manhattan distance and majority voting."""

    def __init__(self, k=3, metric="euclidean"):
        if k < 1:
            raise ValueError("k must be at least 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("Choose euclidean or manhattan distance.")
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training images.")
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X_test, progress_interval=0):
        predictions = []
        for count, sample in enumerate(X_test, start=1):
            distances = self._compute_distances(sample)
            neighbors = np.argsort(distances)[:self.k]
            votes = Counter(self.y_train[neighbors])
            predictions.append(votes.most_common(1)[0][0])
            if progress_interval and count % progress_interval == 0:
                print(f"  [Custom KNN] Predicted {count}/{len(X_test)} samples...")
        return np.asarray(predictions)

    def _compute_distances(self, sample):
        difference = self.X_train - sample
        if self.metric == "manhattan":
            np.abs(difference, out=difference)
            return np.sum(difference, axis=1)
        np.square(difference, out=difference)
        return np.sqrt(np.sum(difference, axis=1))
