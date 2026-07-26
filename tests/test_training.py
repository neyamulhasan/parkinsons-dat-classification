from __future__ import annotations

from pathlib import Path

import numpy as np

from parkinsons_dat_classification.data.dataset import build_samples
from parkinsons_dat_classification.training.baseline import train_baseline_classifier
from parkinsons_dat_classification.training.features import extract_volume_features
from parkinsons_dat_classification.training.metrics import compute_binary_classification_metrics


def test_extract_volume_features_produces_fixed_length_vector() -> None:
    image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    features = extract_volume_features(image, histogram_bins=8)

    assert features.ndim == 1
    assert features.size == 3 + 12 + 8


def test_compute_binary_classification_metrics_handles_basic_case() -> None:
    metrics = compute_binary_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])

    assert metrics.true_negative == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_train_baseline_classifier_runs_on_synthetic_loader() -> None:
    repo_root = Path("/home/linux/parkinsons-dat-classification")
    samples = build_samples(
        repo_root / "data/raw/train_labels_JNDlMjr.csv",
        repo_root / "data/raw/niftis/niftis_utCGpHE",
    )[:30]

    def fake_loader(path: Path) -> np.ndarray:
        seed = sum(ord(char) for char in path.stem)
        rng = np.random.default_rng(seed)
        return rng.random((4, 4, 4), dtype=np.float32)

    result = train_baseline_classifier(
        samples,
        image_loader=fake_loader,
        validation_fraction=0.2,
        histogram_bins=8,
        max_iter=200,
        random_state=11,
    )

    assert result.train_sample_count == 24
    assert result.validation_sample_count == 6
    assert result.feature_count == 3 + 12 + 8
    assert 0.0 <= result.train_metrics.accuracy <= 1.0
    assert 0.0 <= result.validation_metrics.accuracy <= 1.0
