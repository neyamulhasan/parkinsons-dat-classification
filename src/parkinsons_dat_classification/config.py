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
class AppConfig:
    """Top-level configuration object for the repository."""

    project_name: str
    paths: ProjectPaths
    runtime: RuntimeConfig


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

    if not isinstance(project_section, dict):
        raise ValueError("'project' section must be a mapping")
    if not isinstance(paths_section, dict):
        raise ValueError("'paths' section must be a mapping")
    if not isinstance(runtime_section, dict):
        raise ValueError("'runtime' section must be a mapping")

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

    return AppConfig(project_name=project_name, paths=paths, runtime=runtime)
