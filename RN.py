import argparse
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from dataset import FULL_DATASET, QUICK_DATASET, load_split
from knn import CustomKNNClassifier


def plot_samples(images, labels):
    import matplotlib.pyplot as plt

    indices = np.random.default_rng().choice(len(images), size=5, replace=False)
    figure, axes = plt.subplots(1, len(indices), figsize=(10, 2))
    for axis, index in zip(axes, indices):
        axis.imshow(images[index].reshape(28, 28), cmap="gray")
        axis.set_title(f"Label: {labels[index]}")
        axis.axis("off")
    figure.suptitle("Training samples")
    plt.show()


def plot_confusion_matrix(labels, predictions, k):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(labels, predictions, cmap="Blues")
    plt.title(f"Custom KNN confusion matrix (k={k})")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare custom KNN with scikit-learn on MNIST.")
    parser.add_argument("--quick", action="store_true", help="Use the bundled coursework subset")
    parser.add_argument("--k", type=int, nargs="+", help="Override the list of neighbor counts")
    parser.add_argument("--plot", action="store_true", help="Show samples and custom KNN confusion matrices")
    args = parser.parse_args()

    dataset = QUICK_DATASET if args.quick else FULL_DATASET
    X_train, y_train = load_split(dataset, "train")
    X_test, y_test = load_split(dataset, "test")
    default_k = [1, 3, 5, len(X_train)] if args.quick else [1, 3, 5, 7, 9, 30000, 59999]
    k_values = args.k or default_k
    if not all(1 <= k <= len(X_train) for k in k_values):
        parser.error(f"k must be between 1 and {len(X_train)} for this dataset.")
    if args.plot:
        plot_samples(X_train, y_train)

    print(f"Training images: {len(X_train)}. Test images: {len(X_test)}.")
    print(f"{'k':<7} {'Classifier':<18} {'Accuracy':>10} {'Prediction time':>17}")
    for k in k_values:
        custom = CustomKNNClassifier(k=k).fit(X_train, y_train)
        baseline = KNeighborsClassifier(n_neighbors=k, metric="euclidean").fit(X_train, y_train)

        for name, predict in (
            ("Custom KNN", lambda samples: custom.predict(samples, progress_interval=1000)),
            ("scikit-learn KNN", baseline.predict),
        ):
            start = perf_counter()
            predictions = predict(X_test)
            elapsed = perf_counter() - start
            accuracy = accuracy_score(y_test, predictions)
            print(f"{k:<7} {name:<18} {accuracy:>10.2%} {elapsed:>15.2f} s")
            if args.plot and name == "Custom KNN":
                plot_confusion_matrix(y_test, predictions, k)


if __name__ == "__main__":
    main()
