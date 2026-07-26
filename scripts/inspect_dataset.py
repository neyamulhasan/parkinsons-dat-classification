"""Inspect the local competition dataset layout."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from parkinsons_dat_classification.config import build_app_config
from parkinsons_dat_classification.data.discovery import inspect_directory
from parkinsons_dat_classification.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run a dataset inspection against the configured raw data directory."""

    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "default.yaml"
    app_config = build_app_config(config_path=config_path, project_root=project_root)

    summary = inspect_directory(app_config.paths.raw_data)
    payload = {
        "root": str(summary.root),
        "directory_count": summary.directory_count,
        "file_count": summary.file_count,
        "directories": [str(entry.path) for entry in summary.directories],
        "files": [str(entry.path) for entry in summary.files],
    }

    LOGGER.info("Dataset inspection complete for %s", summary.root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
