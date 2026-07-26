"""Offline inference pipeline for creating competition submission files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging
from typing import Callable

import numpy as np

from parkinsons_dat_classification.training.baseline import BaselineBinaryClassifier
from parkinsons_dat_classification.training.checkpointing import load_baseline_checkpoint
from parkinsons_dat_classification.training.features import extract_volume_features, stack_feature_rows

LOGGER = logging.getLogger(__name__)
ImageLoader = Callable[[Path], np.ndarray]


@dataclass(frozen=True, slots=True)
class SubmissionPrediction:
    """A single submission row."""

    uid: str
    is_pathologic: float


def load_smoke_test_uids(scan_root: Path) -> list[str]:
    """Load scan UIDs from the offline inference directory."""

    if not scan_root.exists():
        raise FileNotFoundError(f"Smoke-test directory does not exist: {scan_root}")
    if not scan_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {scan_root}")

    uids = sorted(
        path.name[:-7]
        for path in scan_root.glob("*.nii.gz")
        if path.is_file() and path.name.endswith(".nii.gz")
    )
    if not uids:
        raise ValueError(f"No NIfTI scans found in smoke-test directory: {scan_root}")
    return uids


def _load_scan_image(scan_path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "nibabel is required for offline inference. Install project dependencies first."
        ) from exc

    return np.asarray(nib.load(str(scan_path)).get_fdata(), dtype=np.float32)


def _build_feature_matrix(
    scan_paths: list[Path],
    *,
    histogram_bins: int,
    image_loader: ImageLoader,
) -> np.ndarray:
    feature_rows = [
        extract_volume_features(image_loader(scan_path), histogram_bins=histogram_bins)
        for scan_path in scan_paths
    ]
    return stack_feature_rows(feature_rows)


def build_submission_predictions(
    *,
    checkpoint_dir: Path,
    scan_root: Path,
    histogram_bins: int,
    image_loader: ImageLoader | None = None,
) -> list[SubmissionPrediction]:
    """Build competition-format predictions from a saved baseline checkpoint."""

    model, metadata = load_baseline_checkpoint(checkpoint_dir)
    if not isinstance(model, BaselineBinaryClassifier):
        raise TypeError(f"Expected BaselineBinaryClassifier, got {type(model)!r}")

    uids = load_smoke_test_uids(scan_root)
    scan_paths = [scan_root / f"{uid}.nii.gz" for uid in uids]
    resolved_loader = image_loader or _load_scan_image
    feature_matrix = _build_feature_matrix(
        scan_paths,
        histogram_bins=histogram_bins,
        image_loader=resolved_loader,
    )

    probabilities = model.predict_proba(feature_matrix)
    if len(probabilities) != len(uids):
        raise RuntimeError("Prediction count does not match smoke-test UID count")

    predictions = [
        SubmissionPrediction(uid=uid, is_pathologic=float(probability))
        for uid, probability in zip(uids, probabilities, strict=True)
    ]

    LOGGER.info(
        "Built %s submission predictions using checkpoint metadata created at %s",
        len(predictions),
        metadata.created_at,
    )
    return predictions


def write_submission_csv(predictions: list[SubmissionPrediction], output_path: Path) -> Path:
    """Write a submission file in the DrivenData competition format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "is_pathologic"])
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({"uid": prediction.uid, "is_pathologic": prediction.is_pathologic})

    return output_path
