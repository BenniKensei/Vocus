"""demo.py - Deterministic toy example for the Vocus portfolio README.

This script demonstrates the shape of the feature pipeline without requiring a
camera or MediaPipe runtime. It synthesizes dummy landmark vectors, fits a
simple nearest-centroid classifier, and reports a reproducible accuracy score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyDataset:
    """Container for the synthetic demo dataset.

    Attributes:
        features: Feature matrix with shape ``(num_samples, feature_dim)``.
        labels: Integer labels with shape ``(num_samples,)``.
    """

    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class NearestCentroidModel:
    """Minimal classifier used for the deterministic demo.

    Attributes:
        centroids: Class centroid matrix with shape ``(num_classes, feature_dim)``.
    """

    centroids: np.ndarray


def build_toy_dataset(
    seed: int = 7,
    samples_per_class: int = 24,
    feature_dim: int = 20,
) -> ToyDataset:
    """Create a seeded synthetic dataset that mimics landmark vectors.

    Args:
        seed: Random seed used to make the portfolio demo reproducible.
        samples_per_class: Number of dummy samples generated for each class.
        feature_dim: Number of scalar features per sample. The production
            gesture model also consumes 20 normalized landmark-distance values.

    Returns:
        A ``ToyDataset`` instance containing synthetic features and labels.

    Raises:
        ValueError: If ``samples_per_class`` or ``feature_dim`` is not positive.
    """

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    rng = np.random.default_rng(seed)
    # The seed is fixed so the README demo always prints the same result.
    centers = np.array([
        np.linspace(0.15, 0.45, feature_dim),
        np.linspace(0.55, 0.75, feature_dim),
        np.linspace(0.85, 1.05, feature_dim),
    ])

    features = []
    labels = []
    for label, center in enumerate(centers):
        # Expected shape: (samples_per_class, feature_dim)
        class_block = rng.normal(loc=center, scale=0.03, size=(samples_per_class, feature_dim))
        features.append(class_block)
        labels.append(np.full(samples_per_class, label, dtype=np.int64))

    stacked_features = np.vstack(features)
    stacked_labels = np.concatenate(labels)
    return ToyDataset(features=stacked_features, labels=stacked_labels)


def fit_nearest_centroid(dataset: ToyDataset) -> NearestCentroidModel:
    """Compute one centroid per class from the synthetic training data.

    Args:
        dataset: Synthetic features and labels returned by ``build_toy_dataset``.

    Returns:
        A ``NearestCentroidModel`` with one centroid per label.

    Raises:
        ValueError: If the dataset is empty or labels are missing.
    """

    if dataset.features.size == 0:
        raise ValueError("dataset.features must not be empty")
    if dataset.labels.size == 0:
        raise ValueError("dataset.labels must not be empty")

    class_ids = np.unique(dataset.labels)
    centroids = []
    for class_id in class_ids:
        # Expected shape: (num_samples_for_class, feature_dim)
        class_vectors = dataset.features[dataset.labels == class_id]
        centroids.append(class_vectors.mean(axis=0))

    return NearestCentroidModel(centroids=np.vstack(centroids))


def predict_nearest_centroid(model: NearestCentroidModel, features: np.ndarray) -> np.ndarray:
    """Predict labels by selecting the closest class centroid.

    Args:
        model: Fitted centroid model returned by ``fit_nearest_centroid``.
        features: Input feature matrix with shape ``(batch_size, feature_dim)``.

    Returns:
        Integer class predictions with shape ``(batch_size,)``.

    Raises:
        ValueError: If the input feature matrix has the wrong dimensionality.
    """

    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if features.shape[1] != model.centroids.shape[1]:
        raise ValueError("feature_dim does not match the fitted model")

    # Expected shape: (batch_size, num_classes, feature_dim)
    deltas = features[:, None, :] - model.centroids[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return np.argmin(distances, axis=1)


def main() -> None:
    """Run the deterministic toy example and print a reproducible score.

    Args:
        None.

    Returns:
        None.

    Raises:
        ValueError: Propagated if the synthetic dataset or prediction path is invalid.
    """

    dataset = build_toy_dataset()
    model = fit_nearest_centroid(dataset)
    predictions = predict_nearest_centroid(model, dataset.features)
    accuracy = float(np.mean(predictions == dataset.labels))

    print("Vocus toy demo")
    print(f"Samples: {dataset.features.shape[0]}")
    print(f"Feature dimension: {dataset.features.shape[1]}")
    print(f"Training accuracy on synthetic data: {accuracy:.3f}")
    print(f"Example prediction for the first sample: {predictions[0]}")


if __name__ == "__main__":
    main()
