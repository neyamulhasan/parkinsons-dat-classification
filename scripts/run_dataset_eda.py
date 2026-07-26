"""Generate a structured EDA report for the local DaT dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from parkinsons_dat_classification.config import build_app_config
from parkinsons_dat_classification.data.eda import build_dataset_eda_report
from parkinsons_dat_classification.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Load the repo configuration and print a JSON EDA report."""

    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = build_app_config(project_root / "configs" / "default.yaml", project_root=project_root)

    report = build_dataset_eda_report(
        train_labels_path=config.paths.raw_data / "train_labels_JNDlMjr.csv",
        train_scan_root=config.paths.raw_data / "niftis" / "niftis_utCGpHE",
        smoke_test_root=config.paths.smoke_test / "smoke_test_data_2gePzfM" / "niftis",
    )

    payload = {
        "train_labels_path": str(report.train_labels_path),
        "train_partition": {
            "root": str(report.train_partition.root),
            "scan_count": report.train_partition.scan_count,
            "example_files": list(report.train_partition.example_files),
        },
        "smoke_test_partition": {
            "root": str(report.smoke_test_partition.root),
            "scan_count": report.smoke_test_partition.scan_count,
            "example_files": list(report.smoke_test_partition.example_files),
        },
        "label_distribution": {
            "negative_count": report.label_distribution.negative_count,
            "positive_count": report.label_distribution.positive_count,
            "positive_rate": report.label_distribution.positive_rate,
        },
        "label_row_count": report.label_row_count,
        "matching_scan_count": report.matching_scan_count,
        "missing_scan_uids": list(report.missing_scan_uids),
        "unexpected_scan_uids": list(report.unexpected_scan_uids),
        "all_labels_match_scans": report.all_labels_match_scans,
    }

    LOGGER.info(
        "EDA complete: %s labels, %s positive, %s smoke-test scans",
        report.label_row_count,
        report.label_distribution.positive_count,
        report.smoke_test_partition.scan_count,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
