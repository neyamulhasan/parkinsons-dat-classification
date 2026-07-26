"""Validation error analysis and calibration reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import brier_score_loss

from parkinsons_dat_classification.training.metrics import (
    BinaryClassificationMetrics,
    compute_binary_classification_metrics,
)


@dataclass(frozen=True, slots=True)
class PredictionObservation:
    """A single prediction observation used for error analysis."""

    uid: str
    y_true: int
    y_pred: int
    y_score: float


@dataclass(frozen=True, slots=True)
class CalibrationBinSummary:
    """Observed calibration statistics for a probability bin."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float
    observed_positive_rate: float


@dataclass(frozen=True, slots=True)
class ErrorAnalysisReport:
    """Structured summary of validation performance and failure modes."""

    sample_count: int
    metrics: BinaryClassificationMetrics
    brier_score: float
    false_positive_uids: tuple[str, ...]
    false_negative_uids: tuple[str, ...]
    most_confident_false_positives: tuple[str, ...]
    most_confident_false_negatives: tuple[str, ...]
    calibration_bins: tuple[CalibrationBinSummary, ...]


def _validation_arrays(observations: Iterable[PredictionObservation]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    uids: list[str] = []

    for observation in observations:
        uids.append(observation.uid)
        y_true.append(int(observation.y_true))
        y_pred.append(int(observation.y_pred))
        y_score.append(float(observation.y_score))

    if not uids:
        raise ValueError("At least one validation prediction is required")

    return (
        np.asarray(y_true, dtype=np.int32),
        np.asarray(y_pred, dtype=np.int32),
        np.asarray(y_score, dtype=np.float32),
        uids,
    )


def _calibration_bins(y_true: np.ndarray, y_score: np.ndarray, bin_count: int) -> tuple[CalibrationBinSummary, ...]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bins: list[CalibrationBinSummary] = []

    for index in range(bin_count):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)

        if not np.any(mask):
            bins.append(
                CalibrationBinSummary(
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    mean_predicted_probability=0.0,
                    observed_positive_rate=0.0,
                )
            )
            continue

        bin_scores = y_score[mask]
        bin_true = y_true[mask]
        bins.append(
            CalibrationBinSummary(
                lower_bound=lower,
                upper_bound=upper,
                count=int(mask.sum()),
                mean_predicted_probability=float(np.mean(bin_scores)),
                observed_positive_rate=float(np.mean(bin_true)),
            )
        )

    return tuple(bins)


def analyze_validation_predictions(
    observations: Iterable[PredictionObservation],
    *,
    calibration_bins: int = 10,
) -> ErrorAnalysisReport:
    """Create a structured report describing validation failure modes."""

    y_true, y_pred, y_score, uids = _validation_arrays(observations)
    metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)
    brier = float(brier_score_loss(y_true, y_score))

    false_positive_uids = tuple(uid for uid, truth, pred in zip(uids, y_true, y_pred, strict=True) if truth == 0 and pred == 1)
    false_negative_uids = tuple(uid for uid, truth, pred in zip(uids, y_true, y_pred, strict=True) if truth == 1 and pred == 0)

    positive_errors = sorted(
        (uid for uid, truth, pred, score in zip(uids, y_true, y_pred, y_score, strict=True) if truth == 0 and pred == 1),
        key=lambda uid: next(
            score for u, _, _, score in zip(uids, y_true, y_pred, y_score, strict=True) if u == uid
        ),
        reverse=True,
    )
    negative_errors = sorted(
        (uid for uid, truth, pred, score in zip(uids, y_true, y_pred, y_score, strict=True) if truth == 1 and pred == 0),
        key=lambda uid: next(
            score for u, _, _, score in zip(uids, y_true, y_pred, y_score, strict=True) if u == uid
        ),
    )

    return ErrorAnalysisReport(
        sample_count=len(uids),
        metrics=metrics,
        brier_score=brier,
        false_positive_uids=false_positive_uids,
        false_negative_uids=false_negative_uids,
        most_confident_false_positives=tuple(positive_errors[:5]),
        most_confident_false_negatives=tuple(negative_errors[:5]),
        calibration_bins=_calibration_bins(y_true, y_score, calibration_bins),
    )
