"""Dataset inspection helpers that do not assume a fixed dataset layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """A single filesystem entry discovered during inspection."""

    path: Path
    is_directory: bool


@dataclass(frozen=True, slots=True)
class DirectorySummary:
    """Summary of a directory tree for EDA and dataset discovery."""

    root: Path
    directories: tuple[DirectoryEntry, ...]
    files: tuple[DirectoryEntry, ...]

    @property
    def directory_count(self) -> int:
        return len(self.directories)

    @property
    def file_count(self) -> int:
        return len(self.files)


def inspect_directory(root: Path) -> DirectorySummary:
    """Inspect a directory tree without making assumptions about structure."""

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    directories: list[DirectoryEntry] = []
    files: list[DirectoryEntry] = []

    for entry in sorted(root.rglob("*")):
        directory_entry = DirectoryEntry(path=entry, is_directory=entry.is_dir())
        if entry.is_dir():
            directories.append(directory_entry)
        else:
            files.append(directory_entry)

    return DirectorySummary(root=root, directories=tuple(directories), files=tuple(files))
