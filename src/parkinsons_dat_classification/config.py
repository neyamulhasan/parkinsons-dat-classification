"""Configuration loading and path resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved project paths used by the training pipeline."""

    data_root: Path
    raw_data: Path
    processed_data: Path
    smoke_test: Path
    outputs: Path
    logs: Path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime settings that affect reproducibility and data loading."""

    seed: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training controls for the baseline and future model stacks."""

    validation_fraction: float
    batch_size: int
    random_state: int


@dataclass(frozen=True, slots=True)
class TorchTrainingConfig:
    """PyTorch specific training hyperparameters."""

    epochs: int
    learning_rate: float
    weight_decay: float
    device: str
    amp: bool


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Baseline model hyperparameters."""

    max_iter: int
    class_weight: str
    histogram_bins: int


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Repository-local artifact directories."""

    checkpoint_dir: Path
    report_dir: Path


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Offline inference settings."""

    submission_filename: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration object for the repository."""

    project_name: str
    paths: ProjectPaths
    runtime: RuntimeConfig
    training: TrainingConfig
    torch_training: TorchTrainingConfig
    baseline: BaselineConfig
    artifacts: ArtifactConfig
    inference: InferenceConfig


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {config_path}")
    return data


def _resolve_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_app_config(config_path: Path, project_root: Path | None = None) -> AppConfig:
    """Build a strongly typed config object from a YAML file."""

    project_root = config_path.parent.parent.resolve() if project_root is None else project_root
    raw_config = load_config(config_path)

    project_section = raw_config.get("project", {})
    paths_section = raw_config.get("paths", {})
    runtime_section = raw_config.get("runtime", {})
    training_section = raw_config.get("training", {})
    torch_training_section = raw_config.get("torch_training", {})
    baseline_section = raw_config.get("baseline", {})
    artifacts_section = raw_config.get("artifacts", {})
    inference_section = raw_config.get("inference", {})

    if not isinstance(project_section, dict):
        raise ValueError("'project' section must be a mapping")
    if not isinstance(paths_section, dict):
        raise ValueError("'paths' section must be a mapping")
    if not isinstance(runtime_section, dict):
        raise ValueError("'runtime' section must be a mapping")
    if not isinstance(training_section, dict):
        raise ValueError("'training' section must be a mapping")
    if not isinstance(torch_training_section, dict):
        raise ValueError("'torch_training' section must be a mapping")
    if not isinstance(baseline_section, dict):
        raise ValueError("'baseline' section must be a mapping")
    if not isinstance(artifacts_section, dict):
        raise ValueError("'artifacts' section must be a mapping")
    if not isinstance(inference_section, dict):
        raise ValueError("'inference' section must be a mapping")

    project_name = str(project_section.get("name", "parkinsons-dat-classification"))

    paths = ProjectPaths(
        data_root=_resolve_path(project_root, paths_section.get("data_root", "data")),
        raw_data=_resolve_path(project_root, paths_section.get("raw_data", "data/raw")),
        processed_data=_resolve_path(
            project_root, paths_section.get("processed_data", "data/processed")
        ),
        smoke_test=_resolve_path(project_root, paths_section.get("smoke_test", "data/smoke_test")),
        outputs=_resolve_path(project_root, paths_section.get("outputs", "outputs")),
        logs=_resolve_path(project_root, paths_section.get("logs", "logs")),
    )

    runtime = RuntimeConfig(
        seed=int(runtime_section.get("seed", 42)),
        num_workers=int(runtime_section.get("num_workers", 4)),
    )

    training = TrainingConfig(
        validation_fraction=float(training_section.get("validation_fraction", 0.2)),
        batch_size=int(training_section.get("batch_size", 8)),
        random_state=int(training_section.get("random_state", 42)),
    )

    torch_training = TorchTrainingConfig(
        epochs=int(torch_training_section.get("epochs", 10)),
        learning_rate=float(torch_training_section.get("learning_rate", 1e-3)),
        weight_decay=float(torch_training_section.get("weight_decay", 1e-4)),
        device=str(torch_training_section.get("device", "auto")),
        amp=bool(torch_training_section.get("amp", True)),
    )

    baseline = BaselineConfig(
        max_iter=int(baseline_section.get("max_iter", 1000)),
        class_weight=str(baseline_section.get("class_weight", "balanced")),
        histogram_bins=int(baseline_section.get("histogram_bins", 16)),
    )

    artifacts = ArtifactConfig(
        checkpoint_dir=_resolve_path(
            project_root, artifacts_section.get("checkpoint_dir", "outputs/checkpoints")
        ),
        report_dir=_resolve_path(project_root, artifacts_section.get("report_dir", "outputs/reports")),
    )

    inference = InferenceConfig(
        submission_filename=str(inference_section.get("submission_filename", "submission.csv")),
    )

    return AppConfig(
        project_name=project_name,
        paths=paths,
        runtime=runtime,
        training=training,
        torch_training=torch_training,
        baseline=baseline,
        artifacts=artifacts,
        inference=inference,
    )
