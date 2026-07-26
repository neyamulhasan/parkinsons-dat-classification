from __future__ import annotations

from pathlib import Path

import numpy as np

from parkinsons_dat_classification.data.dataset import (
    DaTDataset,
    build_dataloader,
    build_samples,
    class_distribution,
    dataset_labels,
    dataset_uids,
    split_samples,
)


def test_build_samples_matches_labels_and_paths() -> None:
    repo_root = Path("/home/linux/parkinsons-dat-classification")
    samples = build_samples(
        repo_root / "data/raw/train_labels_JNDlMjr.csv",
        repo_root / "data/raw/niftis/niftis_utCGpHE",
    )

    assert len(samples) == 1362
    assert samples[0].scan_path.name == f"{samples[0].uid}.nii.gz"
    assert {sample.label for sample in samples} == {0, 1}


def test_dataset_and_loader_work_with_custom_loader() -> None:
    repo_root = Path("/home/linux/parkinsons-dat-classification")
    samples = build_samples(
        repo_root / "data/raw/train_labels_JNDlMjr.csv",
        repo_root / "data/raw/niftis/niftis_utCGpHE",
    )[:7]

    dataset = DaTDataset(samples, image_loader=lambda path: np.ones((2, 2), dtype=np.float32))
    loader = build_dataloader(dataset, batch_size=3, shuffle=False)

    batches = list(loader)

    assert len(dataset) == 7
    assert len(loader) == 3
    assert len(batches) == 3
    assert batches[0][0].image.shape == (2, 2)
    assert dataset_uids(dataset) == [sample.uid for sample in samples]
    assert dataset_labels(dataset) == [sample.label for sample in samples]
    assert class_distribution(dataset_labels(dataset))[0] + class_distribution(dataset_labels(dataset))[1] == 7


def test_split_samples_is_deterministic() -> None:
    repo_root = Path("/home/linux/parkinsons-dat-classification")
    samples = build_samples(
        repo_root / "data/raw/train_labels_JNDlMjr.csv",
        repo_root / "data/raw/niftis/niftis_utCGpHE",
    )[:10]

    train_a, val_a = split_samples(samples, validation_fraction=0.2, seed=123)
    train_b, val_b = split_samples(samples, validation_fraction=0.2, seed=123)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == 10
