"""Dataset EDA and validation helpers for the DrivenData DaT challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


@dataclass(frozen=True, slots=True)
class LabelDistribution:
    """Class balance summary for a binary target."""

    negative_count: int
    positive_count: int

    @property
    def total(self) -> int:
        return self.negative_count + self.positive_count

    @property
    def positive_rate(self) -> float:
        return self.positive_count / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class DatasetPartitionSummary:
    """Summary of a dataset partition stored on disk."""

    root: Path
    scan_count: int
    example_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetEDAReport:
    """Structured report for repository-local dataset inspection."""

    train_labels_path: Path
    train_partition: DatasetPartitionSummary
    smoke_test_partition: DatasetPartitionSummary
    label_distribution: LabelDistribution
    label_row_count: int
    matching_scan_count: int
    missing_scan_uids: tuple[str, ...]
    unexpected_scan_uids: tuple[str, ...]

    @property
    def all_labels_match_scans(self) -> bool:
        return not self.missing_scan_uids and not self.unexpected_scan_uids


def _load_label_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Label file is empty: {labels_path}")
    if reader.fieldnames is None or {"uid", "is_pathologic"} - set(reader.fieldnames):
        raise ValueError(
            f"Label file must contain uid and is_pathologic columns: {labels_path}"
        )
    return rows


def _scan_uids(scan_root: Path) -> set[str]:
    if not scan_root.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_root}")
    if not scan_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {scan_root}")

    return {
        path.name[:-7]
        for path in scan_root.glob("*.nii.gz")
        if path.is_file() and path.name.endswith(".nii.gz")
    }


def _summarize_partition(root: Path, example_limit: int = 5) -> DatasetPartitionSummary:
    scan_files = sorted(
        path.name for path in root.glob("*.nii.gz") if path.is_file() and path.name.endswith(".nii.gz")
    )
    return DatasetPartitionSummary(root=root, scan_count=len(scan_files), example_files=tuple(scan_files[:example_limit]))


def _count_labels(rows: Iterable[dict[str, str]]) -> LabelDistribution:
    positive_count = 0
    negative_count = 0

    for row in rows:
        value = float(row["is_pathologic"])
        if value >= 0.5:
            positive_count += 1
        else:
            negative_count += 1

    return LabelDistribution(negative_count=negative_count, positive_count=positive_count)


def build_dataset_eda_report(
    *,
    train_labels_path: Path,
    train_scan_root: Path,
    smoke_test_root: Path,
) -> DatasetEDAReport:
    """Build a structured report from the repository-local dataset layout."""

    label_rows = _load_label_rows(train_labels_path)
    labeled_uids = [row["uid"] for row in label_rows]
    scan_uids = _scan_uids(train_scan_root)

    missing_scan_uids = tuple(sorted(uid for uid in labeled_uids if uid not in scan_uids))
    unexpected_scan_uids = tuple(sorted(uid for uid in scan_uids if uid not in set(labeled_uids)))

    return DatasetEDAReport(
        train_labels_path=train_labels_path,
        train_partition=_summarize_partition(train_scan_root),
        smoke_test_partition=_summarize_partition(smoke_test_root),
        label_distribution=_count_labels(label_rows),
        label_row_count=len(label_rows),
        matching_scan_count=len(labeled_uids) - len(missing_scan_uids),
        missing_scan_uids=missing_scan_uids,
        unexpected_scan_uids=unexpected_scan_uids,
    )
