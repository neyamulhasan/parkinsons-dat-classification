"""Baseline training pipeline for binary classification on DaT scans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from parkinsons_dat_classification.data.dataset import DaTDataset, DaTSample, split_samples
from parkinsons_dat_classification.evaluation.error_analysis import PredictionObservation
from parkinsons_dat_classification.training.features import extract_volume_features, stack_feature_rows
from parkinsons_dat_classification.training.metrics import (
    BinaryClassificationMetrics,
    compute_binary_classification_metrics,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineTrainingResult:
    """Artifacts and metrics produced by the baseline training run."""

    model: "BaselineBinaryClassifier"
    train_metrics: BinaryClassificationMetrics
    validation_metrics: BinaryClassificationMetrics
    validation_predictions: tuple[PredictionObservation, ...]
    train_sample_count: int
    validation_sample_count: int
    feature_count: int


class BaselineBinaryClassifier:
    """Thin wrapper around logistic regression for the baseline slice."""

    def __init__(self, *, max_iter: int = 1000, class_weight: str | dict[int, float] | None = "balanced", random_state: int = 42) -> None:
        self._model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BaselineBinaryClassifier":
        self._model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = self._model.predict_proba(features)
        return probabilities[:, 1]


def _build_feature_matrix(
    dataset: DaTDataset,
    *,
    histogram_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = []
    labels = []

    for index in range(len(dataset)):
        example = dataset[index]
        feature_rows.append(extract_volume_features(example.image, histogram_bins=histogram_bins))
        labels.append(example.label)

    return stack_feature_rows(feature_rows), np.asarray(labels, dtype=np.int32)


def train_baseline_classifier(
    samples: Iterable[DaTSample],
    *,
    image_loader,
    validation_fraction: float,
    histogram_bins: int = 16,
    max_iter: int = 1000,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
) -> BaselineTrainingResult:
    """Train a reproducible baseline classifier on DaT scans.

    The function keeps the training slice compatible with the manifest-backed
    dataset and does not require any hardcoded dataset assumptions.
    """

    train_samples, validation_samples = split_samples(
        samples,
        validation_fraction=validation_fraction,
        seed=random_state,
    )

    train_dataset = DaTDataset(train_samples, image_loader=image_loader)
    validation_dataset = DaTDataset(validation_samples, image_loader=image_loader)

    train_features, train_labels = _build_feature_matrix(
        train_dataset,
        histogram_bins=histogram_bins,
    )
    validation_features, validation_labels = _build_feature_matrix(
        validation_dataset,
        histogram_bins=histogram_bins,
    )

    model = BaselineBinaryClassifier(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    ).fit(train_features, train_labels)

    train_predictions = model.predict(train_features)
    validation_predictions = model.predict(validation_features)
    train_scores = model.predict_proba(train_features)
    validation_scores = model.predict_proba(validation_features)

    validation_records = tuple(
        PredictionObservation(
            uid=example.uid,
            y_true=int(example.label),
            y_pred=int(predicted_label),
            y_score=float(score),
        )
        for example, predicted_label, score in zip(
            validation_dataset.samples,
            validation_predictions,
            validation_scores,
            strict=True,
        )
    )

    train_metrics = compute_binary_classification_metrics(
        train_labels,
        train_predictions,
        train_scores,
    )
    validation_metrics = compute_binary_classification_metrics(
        validation_labels,
        validation_predictions,
        validation_scores,
    )

    LOGGER.info(
        "Baseline trained on %s samples with %s validation samples",
        len(train_samples),
        len(validation_samples),
    )

    return BaselineTrainingResult(
        model=model,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        validation_predictions=validation_records,
        train_sample_count=len(train_samples),
        validation_sample_count=len(validation_samples),
        feature_count=train_features.shape[1],
    )
