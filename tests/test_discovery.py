from __future__ import annotations

from pathlib import Path

import pytest

from parkinsons_dat_classification.data.discovery import inspect_directory


def test_inspect_directory_counts_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "class_a").mkdir()
    (tmp_path / "class_a" / "sample.nii.gz").write_text("x", encoding="utf-8")
    (tmp_path / "class_b").mkdir()

    summary = inspect_directory(tmp_path)

    assert summary.root == tmp_path
    assert summary.directory_count == 2
    assert summary.file_count == 1


def test_inspect_directory_requires_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        inspect_directory(missing)
