from __future__ import annotations

from pathlib import Path

from parkinsons_dat_classification.data.eda import build_dataset_eda_report


def test_build_dataset_eda_report_detects_clean_alignment() -> None:
    repo_root = Path("/home/linux/parkinsons-dat-classification")
    report = build_dataset_eda_report(
        train_labels_path=repo_root / "data/raw/train_labels_JNDlMjr.csv",
        train_scan_root=repo_root / "data/raw/niftis/niftis_utCGpHE",
        smoke_test_root=repo_root / "data/smoke_test/smoke_test_data_2gePzfM/niftis",
    )

    assert report.label_row_count == 1362
    assert report.train_partition.scan_count == 1362
    assert report.smoke_test_partition.scan_count == 20
    assert report.all_labels_match_scans
    assert report.label_distribution.total == 1362
    assert report.label_distribution.positive_count > 0
