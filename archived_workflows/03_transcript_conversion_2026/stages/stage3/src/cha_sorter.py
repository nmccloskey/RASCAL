"""Sort and rename DIAAD-generated CHAT files for Stage 3.

This script searches DIAAD outputs under::

    data/02_chat_files/<cycle>/<SITE>/

for ``*.cha`` files. For each file, it extracts study ID, timepoint, and
narrative from DIAAD's metadata-rich filename and writes a compactly named copy
under::

    data/03_sorted_files/<cycle>/<SITE>/<timepoint_folder>/<study_id>/

For example::

    C2_TU_5_2_TU_TU58_Maint_CATGrandpa.cha

becomes::

    TU58_Maint_CATGrandpa.cha
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .tt_processor import CYCLES, SITES
except ImportError:  # pragma: no cover - direct script fallback.
    CYCLES = ("cycle2", "cycle3")
    SITES = ("AC", "BU", "TU")


def get_project_root() -> Path:
    """Return the Stage 3 project root, assuming this file lives in ``src/``."""

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
INPUT_ROOT = PROJECT_ROOT / "data" / "02_chat_files"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "03_sorted_files"

TIMEPOINT_MAP = {
    "Pre": "01_PreTx",
    "Post": "02_PostTx",
    "Maint": "03_Maintenance",
}

CHAT_NAME_PATTERN = re.compile(
    r"(?P<study_id>(?:AC|BU|TU)\d+)_"
    r"(?P<timepoint>Pre|Post|Maint)_"
    r"(?P<narrative>[^_]+)$"
)

MOVE_FILES = False
OVERWRITE = False
RECURSIVE = True


@dataclass(frozen=True)
class ParsedChatFilename:
    """Compact metadata parsed from a CHAT filename."""

    study_id: str
    timepoint: str
    narrative: str

    @property
    def compact_filename(self) -> str:
        """Return the compact CHAT filename for sorted outputs."""

        return f"{self.study_id}_{self.timepoint}_{self.narrative}.cha"


@dataclass(frozen=True)
class SortSummary:
    """Counts summarizing a sorting run."""

    copied_or_moved: int = 0
    skipped: int = 0
    failed: int = 0
    partitions: int = 0


def parse_chat_filename(path: Path) -> ParsedChatFilename | None:
    """Parse a DIAAD CHAT filename into compact naming components."""

    if path.suffix.lower() != ".cha":
        return None

    match = CHAT_NAME_PATTERN.search(path.stem)
    if match is None:
        return None

    return ParsedChatFilename(
        study_id=match.group("study_id"),
        timepoint=match.group("timepoint"),
        narrative=match.group("narrative"),
    )


def iter_cycle_site_dirs(
    input_root: Path,
    cycles: Iterable[str] = CYCLES,
    sites: Iterable[str] = SITES,
) -> Iterable[tuple[str, str, Path]]:
    """Yield existing ``cycle/site`` input directories in stable order."""

    for cycle in cycles:
        for site in sites:
            input_dir = input_root / cycle / site
            if input_dir.exists() and input_dir.is_dir():
                yield cycle, site, input_dir
            else:
                print(f"[SKIP] Missing input directory: {input_dir}")


def find_chat_files(input_dir: Path, recursive: bool = RECURSIVE) -> list[Path]:
    """Return sorted CHAT files in ``input_dir``."""

    pattern = "**/*.cha" if recursive else "*.cha"
    return sorted(input_dir.glob(pattern))


def sort_chat_file(
    cha_file: Path,
    output_dir: Path,
    move_files: bool = MOVE_FILES,
    overwrite: bool = OVERWRITE,
) -> bool:
    """Sort one CHAT file into timepoint/study folders with a compact filename."""

    parsed = parse_chat_filename(cha_file)
    if parsed is None:
        print(f"[SKIP] Could not parse filename: {cha_file.name}")
        return False

    timepoint_folder = TIMEPOINT_MAP.get(parsed.timepoint)
    if timepoint_folder is None:
        print(f"[SKIP] Unrecognized timepoint in {cha_file.name}: {parsed.timepoint}")
        return False

    destination_dir = output_dir / timepoint_folder / parsed.study_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / parsed.compact_filename

    if destination_path.exists() and not overwrite:
        print(f"[SKIP] Destination already exists: {destination_path}")
        return False

    if move_files:
        if overwrite and destination_path.exists():
            destination_path.unlink()
        shutil.move(str(cha_file), str(destination_path))
        action = "MOVED"
    else:
        shutil.copy2(cha_file, destination_path)
        action = "COPIED"

    print(f"[{action}] {cha_file.name} -> {destination_path}")
    return True


def sort_chat_files_for_partition(
    input_dir: Path,
    output_dir: Path,
    move_files: bool = MOVE_FILES,
    overwrite: bool = OVERWRITE,
    recursive: bool = RECURSIVE,
) -> SortSummary:
    """Sort all CHAT files from one cycle/site input directory."""

    print(f"\nProcessing input:  {input_dir}")
    print(f"Processing output: {output_dir}")

    cha_files = find_chat_files(input_dir, recursive=recursive)
    if not cha_files:
        print(f"[WARNING] No .cha files found in: {input_dir}")
        return SortSummary(skipped=0, partitions=1)

    sorted_count = 0
    skipped_count = 0
    failed_count = 0

    for cha_file in cha_files:
        try:
            did_sort = sort_chat_file(
                cha_file=cha_file,
                output_dir=output_dir,
                move_files=move_files,
                overwrite=overwrite,
            )
        except OSError as exc:
            print(f"[ERROR] Failed to sort {cha_file}: {exc}")
            failed_count += 1
            continue

        if did_sort:
            sorted_count += 1
        else:
            skipped_count += 1

    print(
        f"Partition done. Sorted {sorted_count} file(s); "
        f"skipped {skipped_count}; failed {failed_count}."
    )

    return SortSummary(
        copied_or_moved=sorted_count,
        skipped=skipped_count,
        failed=failed_count,
        partitions=1,
    )


def sort_all_chat_files(
    input_root: Path | str = INPUT_ROOT,
    output_root: Path | str = OUTPUT_ROOT,
    cycles: Sequence[str] = CYCLES,
    sites: Sequence[str] = SITES,
    move_files: bool = MOVE_FILES,
    overwrite: bool = OVERWRITE,
    recursive: bool = RECURSIVE,
) -> SortSummary:
    """Sort CHAT files for all existing cycle/site DIAAD output directories."""

    input_root = Path(input_root)
    output_root = Path(output_root)

    print("Starting CHAT file sorting.")
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Mode: {'move' if move_files else 'copy'}")

    total_sorted = 0
    total_skipped = 0
    total_failed = 0
    partition_count = 0

    for cycle, site, input_dir in iter_cycle_site_dirs(input_root, cycles=cycles, sites=sites):
        output_dir = output_root / cycle / site
        summary = sort_chat_files_for_partition(
            input_dir=input_dir,
            output_dir=output_dir,
            move_files=move_files,
            overwrite=overwrite,
            recursive=recursive,
        )
        partition_count += summary.partitions
        total_sorted += summary.copied_or_moved
        total_skipped += summary.skipped
        total_failed += summary.failed

    print("\nCHAT sorting complete.")
    print(f"Partitions processed: {partition_count}")
    print(f"Files sorted:         {total_sorted}")
    print(f"Files skipped:        {total_skipped}")
    print(f"Files failed:         {total_failed}")

    return SortSummary(
        copied_or_moved=total_sorted,
        skipped=total_skipped,
        failed=total_failed,
        partitions=partition_count,
    )


def main() -> None:
    """CLI entry point for direct script execution."""

    sort_all_chat_files()


if __name__ == "__main__":
    main()
