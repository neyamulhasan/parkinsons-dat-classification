"""Persistence helpers for baseline training artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import pickle

from parkinsons_dat_classification.training.baseline import (
    BaselineBinaryClassifier,
    BaselineTrainingResult,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineCheckpointMetadata:
    """Human-readable metadata stored alongside a baseline checkpoint."""

    created_at: str
    model_type: str
    feature_count: int
    train_sample_count: int
    validation_sample_count: int
    train_accuracy: float
    validation_accuracy: float
    train_roc_auc: float
    validation_roc_auc: float


def _checkpoint_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "baseline_model.pkl", directory / "baseline_metadata.json"


def save_baseline_checkpoint(
    result: BaselineTrainingResult,
    directory: Path,
) -> tuple[Path, Path]:
    """Persist the trained baseline model and a JSON metadata sidecar."""

    directory.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = _checkpoint_paths(directory)

    with model_path.open("wb") as model_handle:
        pickle.dump(result.model, model_handle)

    metadata = BaselineCheckpointMetadata(
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        model_type=type(result.model).__name__,
        feature_count=result.feature_count,
        train_sample_count=result.train_sample_count,
        validation_sample_count=result.validation_sample_count,
        train_accuracy=result.train_metrics.accuracy,
        validation_accuracy=result.validation_metrics.accuracy,
        train_roc_auc=result.train_metrics.roc_auc,
        validation_roc_auc=result.validation_metrics.roc_auc,
    )

    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        json.dump(asdict(metadata), metadata_handle, indent=2, sort_keys=True)

    LOGGER.info("Saved baseline checkpoint to %s", directory)
    return model_path, metadata_path


def load_baseline_checkpoint(directory: Path) -> tuple[BaselineBinaryClassifier, BaselineCheckpointMetadata]:
    """Load a previously saved baseline checkpoint."""

    model_path, metadata_path = _checkpoint_paths(directory)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")

    with model_path.open("rb") as model_handle:
        model = pickle.load(model_handle)

    if not isinstance(model, BaselineBinaryClassifier):
        raise TypeError(f"Unexpected checkpoint type: {type(model)!r}")

    with metadata_path.open("r", encoding="utf-8") as metadata_handle:
        raw_metadata = json.load(metadata_handle)

    metadata = BaselineCheckpointMetadata(**raw_metadata)
    return model, metadata
