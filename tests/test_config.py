from __future__ import annotations

from pathlib import Path

from parkinsons_dat_classification.config import build_app_config, load_config


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: example\n", encoding="utf-8")

    data = load_config(config_path)

    assert data["project"]["name"] == "example"


def test_build_app_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: example",
                "paths:",
                "  data_root: data",
                "  raw_data: data/raw",
                "  processed_data: data/processed",
                "  smoke_test: data/smoke_test",
                "  outputs: outputs",
                "  logs: logs",
                "runtime:",
                "  seed: 7",
                "  num_workers: 2",
            ]
        ),
        encoding="utf-8",
    )

    config = build_app_config(config_path=config_path, project_root=tmp_path)

    assert config.project_name == "example"
    assert config.paths.raw_data == (tmp_path / "data/raw").resolve()
    assert config.runtime.seed == 7
    assert config.runtime.num_workers == 2
