"""Scan a derivative directory for EEG files.

- Standard BIDS (raw)   → pybids BIDSLayout, queried by suffix='eeg'
- MBP-style BIDS        → glob .fif filtered by proc-{proc} in filename
- Non-BIDS dirs         → recursive glob on known EEG extensions
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions recognised by _ios.py
_NON_BIDS_EEG_EXTENSIONS = {".fif", ".vhdr", ".eeg", ".npy", ".csv", ".set", ".bdf", ".edf"}

# BIDS entry-point extensions (one file per recording, avoids duplicates from .eeg/.vmrk pairs)
_BIDS_EEG_EXTENSIONS = [".vhdr", ".fif", ".set", ".bdf", ".edf", ".eeg"]


def scan_bids(root: Path, proc: str | None = None) -> list[Path]:
    """Return EEG file paths from a BIDS dataset.

    - proc=None  → standard BIDS raw data, queried via pybids (suffix='eeg')
    - proc=str   → MBP-style derivative, glob .fif filtered by 'proc-{proc}' in filename
    """
    if proc is not None:
        return _scan_bids_proc(root, proc)

    from bids import BIDSLayout  # type: ignore

    layout = BIDSLayout(str(root), validate=False)
    files = layout.get(suffix="eeg", extension=_BIDS_EEG_EXTENSIONS)

    # Deduplicate: prefer .vhdr over companion .eeg when both exist
    paths: list[Path] = []
    seen_stems: set[str] = set()
    for bf in sorted(files, key=lambda f: _BIDS_EEG_EXTENSIONS.index(Path(f.path).suffix)
                     if Path(f.path).suffix in _BIDS_EEG_EXTENSIONS else 99):
        stem = Path(bf.path).stem
        if stem not in seen_stems:
            seen_stems.add(stem)
            paths.append(Path(bf.path))
            logger.info("  found: %s", bf.path)

    return paths


def _scan_bids_proc(root: Path, proc: str) -> list[Path]:
    """Glob .fif files whose name contains 'proc-{proc}_'  (MNE-BIDS-Pipeline convention)."""
    pattern = f"proc-{proc}_"
    paths: list[Path] = []
    for p in sorted(root.rglob("*.fif")):
        if pattern in p.name:
            logger.info("  found: %s", p)
            paths.append(p)
    return paths


def scan_non_bids(root: Path) -> list[Path]:
    """Return EEG file paths from a non-BIDS directory via recursive glob."""
    found: list[Path] = []
    seen_stems: set[str] = set()

    # Iterate extensions in priority order to deduplicate .vhdr/.eeg pairs
    priority = [".vhdr", ".fif", ".set", ".bdf", ".edf", ".eeg", ".npy", ".csv"]
    all_files: list[Path] = sorted(root.rglob("*"))

    for ext in priority:
        for p in all_files:
            if p.suffix.lower() == ext and p.stem not in seen_stems:
                seen_stems.add(p.stem)
                found.append(p)
                logger.info("  found: %s", p)

    # Catch any remaining extensions not in the priority list
    for p in all_files:
        if p.suffix.lower() in _NON_BIDS_EEG_EXTENSIONS and p.stem not in seen_stems:
            seen_stems.add(p.stem)
            found.append(p)
            logger.info("  found: %s", p)

    return found


def scan_derivative(root: Path, bids: bool, proc: str | None = None) -> list[Path]:
    """Scan a derivative root and return all EEG file paths found."""
    if bids:
        return scan_bids(root, proc=proc)
    return scan_non_bids(root)
