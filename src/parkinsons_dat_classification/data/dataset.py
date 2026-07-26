"""Dataset primitives for the DrivenData DaT Parkinson's Challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
import csv
import logging
import random

import numpy as np

LOGGER = logging.getLogger(__name__)

NiftiLoader = Callable[[Path], np.ndarray]


@dataclass(frozen=True, slots=True)
class DaTSample:
    """A single labeled DaT SPECT scan."""

    uid: str
    scan_path: Path
    label: int


@dataclass(frozen=True, slots=True)
class DaTExample:
    """A fully loaded dataset example."""

    uid: str
    scan_path: Path
    label: int
    image: np.ndarray


def load_nifti_array(scan_path: Path) -> np.ndarray:
    """Load a NIfTI scan into a numpy array.

    nibabel is intentionally imported lazily so the rest of the repository can be
    developed and tested in environments where the medical-imaging dependency is
    not yet installed.
    """

    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError(
            "nibabel is required to load NIfTI scans. Install project dependencies first."
        ) from exc

    image = nib.load(str(scan_path))
    return np.asarray(image.get_fdata(), dtype=np.float32)


def _read_label_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Label file is empty: {labels_path}")
    if reader.fieldnames is None or {"uid", "is_pathologic"} - set(reader.fieldnames):
        raise ValueError(f"Label file must contain uid and is_pathologic columns: {labels_path}")
    return rows


def _build_scan_path(scan_root: Path, uid: str) -> Path:
    scan_path = scan_root / f"{uid}.nii.gz"
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing scan for uid {uid}: {scan_path}")
    return scan_path


def build_samples(labels_path: Path, scan_root: Path) -> list[DaTSample]:
    """Build the labeled sample manifest for the training split."""

    rows = _read_label_rows(labels_path)
    samples = [
        DaTSample(
            uid=row["uid"],
            scan_path=_build_scan_path(scan_root, row["uid"]),
            label=int(float(row["is_pathologic"])),
        )
        for row in rows
    ]

    LOGGER.info("Built %s samples from %s", len(samples), labels_path)
    return samples


class DaTDataset:
    """Manifest-backed dataset for DaT SPECT scans."""

    def __init__(self, samples: Iterable[DaTSample], *, image_loader: NiftiLoader | None = None) -> None:
        self._samples = list(samples)
        self._image_loader = image_loader or load_nifti_array

        if not self._samples:
            raise ValueError("DaTDataset requires at least one sample")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DaTExample:
        sample = self._samples[index]
        image = self._image_loader(sample.scan_path)
        return DaTExample(uid=sample.uid, scan_path=sample.scan_path, label=sample.label, image=image)

    @property
    def samples(self) -> tuple[DaTSample, ...]:
        return tuple(self._samples)


class SimpleDataLoader:
    """Small deterministic batch iterator used until torch DataLoader is introduced."""

    def __init__(
        self,
        dataset: DaTDataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed

    def __iter__(self) -> Iterator[list[DaTExample]]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.Random(self._seed).shuffle(indices)

        for start in range(0, len(indices), self._batch_size):
            batch_indices = indices[start : start + self._batch_size]
            if len(batch_indices) < self._batch_size and self._drop_last:
                continue
            yield [self._dataset[index] for index in batch_indices]

    def __len__(self) -> int:
        dataset_size = len(self._dataset)
        if self._drop_last:
            return dataset_size // self._batch_size
        return (dataset_size + self._batch_size - 1) // self._batch_size


def build_dataloader(
    dataset: DaTDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> SimpleDataLoader:
    """Build a deterministic batch iterator for the dataset."""

    return SimpleDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def dataset_labels(dataset: DaTDataset) -> list[int]:
    """Extract labels from the dataset manifest."""

    return [sample.label for sample in dataset.samples]


def dataset_uids(dataset: DaTDataset) -> list[str]:
    """Extract UIDs from the dataset manifest."""

    return [sample.uid for sample in dataset.samples]


def class_distribution(labels: Iterable[int]) -> dict[int, int]:
    """Count class occurrences in a label sequence."""

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def split_samples(
    samples: Iterable[DaTSample],
    *,
    validation_fraction: float,
    seed: int = 42,
) -> tuple[list[DaTSample], list[DaTSample]]:
    """Split samples deterministically into train and validation subsets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("split_samples requires at least one sample")

    indices = list(range(len(sample_list)))
    random.Random(seed).shuffle(indices)

    validation_size = max(1, int(round(len(sample_list) * validation_fraction)))
    validation_indices = set(indices[:validation_size])

    train_samples = [sample for index, sample in enumerate(sample_list) if index not in validation_indices]
    validation_samples = [sample for index, sample in enumerate(sample_list) if index in validation_indices]

    if not train_samples:
        raise ValueError("validation_fraction leaves no training samples")

    return train_samples, validation_samples
