from __future__ import annotations

from pathlib import Path

import numpy as np

from parkinsons_dat_classification.data.dataset import build_samples
from parkinsons_dat_classification.submission.pipeline import (
    build_submission_predictions,
    write_submission_csv,
)
from parkinsons_dat_classification.training.baseline import train_baseline_classifier
from parkinsons_dat_classification.training.checkpointing import save_baseline_checkpoint


def test_submission_pipeline_writes_competition_format(tmp_path: Path) -> None:
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
    checkpoint_dir = tmp_path / "checkpoints"
    save_baseline_checkpoint(result, checkpoint_dir)

    predictions = build_submission_predictions(
        checkpoint_dir=checkpoint_dir,
        scan_root=repo_root / "data/smoke_test/smoke_test_data_2gePzfM/niftis",
        histogram_bins=8,
        image_loader=fake_loader,
    )

    output_path = write_submission_csv(predictions, tmp_path / "submission.csv")
    content = output_path.read_text(encoding="utf-8").splitlines()

    assert len(predictions) == 20
    assert content[0] == "uid,is_pathologic"
    assert len(content) == 21
    assert all(0.0 <= prediction.is_pathologic <= 1.0 for prediction in predictions)
