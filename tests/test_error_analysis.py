from __future__ import annotations

from pathlib import Path

import numpy as np

from parkinsons_dat_classification.data.dataset import build_samples
from parkinsons_dat_classification.evaluation.error_analysis import analyze_validation_predictions
from parkinsons_dat_classification.training.baseline import train_baseline_classifier


def test_analyze_validation_predictions_reports_errors() -> None:
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

    report = analyze_validation_predictions(result.validation_predictions, calibration_bins=5)

    assert report.sample_count == result.validation_sample_count
    assert report.metrics.accuracy == result.validation_metrics.accuracy
    assert len(report.calibration_bins) == 5
    assert report.brier_score >= 0.0
