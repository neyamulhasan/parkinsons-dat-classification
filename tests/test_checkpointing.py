from __future__ import annotations

from pathlib import Path

import numpy as np

from parkinsons_dat_classification.data.dataset import build_samples
from parkinsons_dat_classification.training.baseline import train_baseline_classifier
from parkinsons_dat_classification.training.checkpointing import (
    load_baseline_checkpoint,
    save_baseline_checkpoint,
)


def test_baseline_checkpoint_round_trip(tmp_path: Path) -> None:
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

    model_path, metadata_path = save_baseline_checkpoint(result, tmp_path)
    loaded_model, loaded_metadata = load_baseline_checkpoint(tmp_path)

    assert model_path.exists()
    assert metadata_path.exists()
    assert loaded_metadata.feature_count == result.feature_count
    assert loaded_metadata.validation_sample_count == result.validation_sample_count
    assert np.allclose(
        loaded_model.predict_proba(np.zeros((1, result.feature_count), dtype=np.float32)),
        loaded_model.predict_proba(np.zeros((1, result.feature_count), dtype=np.float32)),
    )
