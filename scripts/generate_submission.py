"""Generate an offline submission CSV from a saved baseline checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

from parkinsons_dat_classification.config import build_app_config
from parkinsons_dat_classification.submission.pipeline import (
    build_submission_predictions,
    write_submission_csv,
)
from parkinsons_dat_classification.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Create a submission CSV using the saved checkpoint directory."""

    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = build_app_config(project_root / "configs" / "default.yaml", project_root=project_root)

    predictions = build_submission_predictions(
        checkpoint_dir=config.artifacts.checkpoint_dir,
        scan_root=config.paths.smoke_test / "smoke_test_data_2gePzfM" / "niftis",
        histogram_bins=config.baseline.histogram_bins,
    )
    output_path = write_submission_csv(
        predictions,
        config.paths.outputs / config.inference.submission_filename,
    )

    LOGGER.info("Submission written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
