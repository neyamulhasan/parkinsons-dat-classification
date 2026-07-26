"""Feature extraction for baseline classical models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Configuration for image feature extraction."""

    histogram_bins: int = 16


def _safe_percentile(image: np.ndarray, percentile: float) -> float:
    return float(np.percentile(image, percentile)) if image.size else 0.0


def extract_volume_features(image: np.ndarray, *, histogram_bins: int = 16) -> np.ndarray:
    """Extract a compact numeric representation from a medical image volume.

    The baseline uses global statistics rather than deep features so the current
    training slice can run in a lightweight environment while still operating on
    the real DaT scan arrays.
    """

    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot extract features from an empty image")

    flat = array.reshape(-1)
    minimum = float(np.min(flat))
    maximum = float(np.max(flat))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    median = float(np.median(flat))
    percentiles = [_safe_percentile(flat, value) for value in (5, 25, 75, 95)]
    energy = float(np.mean(np.square(flat)))
    abs_mean = float(np.mean(np.abs(flat)))
    nonzero_fraction = float(np.count_nonzero(flat) / flat.size)
    shape_features = [float(dimension) for dimension in array.shape]

    if maximum > minimum:
        histogram, _ = np.histogram(flat, bins=histogram_bins, range=(minimum, maximum), density=True)
    else:
        histogram = np.zeros(histogram_bins, dtype=np.float32)

    features = np.array(
        [
            *shape_features,
            minimum,
            maximum,
            mean,
            std,
            median,
            *percentiles,
            energy,
            abs_mean,
            nonzero_fraction,
            *histogram.tolist(),
        ],
        dtype=np.float32,
    )
    return features


def stack_feature_rows(rows: Iterable[np.ndarray]) -> np.ndarray:
    """Stack a sequence of feature vectors into a 2D matrix."""

    feature_rows = [np.asarray(row, dtype=np.float32).reshape(1, -1) for row in rows]
    if not feature_rows:
        raise ValueError("At least one feature row is required")
    return np.vstack(feature_rows)
