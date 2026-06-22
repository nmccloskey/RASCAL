"""Orchestrate DIAAD Stage 3 transcript conversion.

Pipeline order
--------------
1. Reformat proto transcript tables:
   ``data/00_original_proto_TTs`` -> ``data/01_reformatted_TTs``
2. Run DIAAD transcript CHAT export:
   ``diaad transcripts chats`` over each cycle/site folder.
3. Sort and compactly rename generated ``.cha`` files:
   ``data/02_chat_files`` -> ``data/03_sorted_files``

Run from the project root with::

    python -m src.main

Useful examples::

    python -m src.main --dry-run-diaad
    python -m src.main --skip-tt
    python -m src.main --skip-diaad
    python -m src.main --skip-sort
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import cha_sorter, diaad_runner, tt_processor


def get_project_root() -> Path:
    """Return the Stage 3 project root, assuming this file lives in ``src/``."""

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

ORIGINAL_TT_ROOT = DATA_DIR / "00_original_proto_TTs"
REFORMATTED_TT_ROOT = DATA_DIR / "01_reformatted_TTs"
CHAT_ROOT = DATA_DIR / "02_chat_files"
SORTED_ROOT = DATA_DIR / "03_sorted_files"


def parse_args() -> argparse.Namespace:
    """Parse Stage 3 orchestration command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the DIAAD Stage 3 conversion pipeline.")

    parser.add_argument("--skip-tt", action="store_true", help="Skip transcript-table reformatting.")
    parser.add_argument("--skip-diaad", action="store_true", help="Skip DIAAD CHAT export.")
    parser.add_argument("--skip-sort", action="store_true", help="Skip CHAT sorting/renaming.")

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch-style stages after per-partition failures where supported.",
    )
    parser.add_argument(
        "--dry-run-diaad",
        action="store_true",
        help="Print DIAAD commands without running them.",
    )
    parser.add_argument(
        "--move-chat-files",
        action="store_true",
        help="Move .cha files during sorting instead of copying them.",
    )
    parser.add_argument(
        "--overwrite-sorted",
        action="store_true",
        help="Overwrite existing sorted .cha files.",
    )
    parser.add_argument(
        "--nonrecursive-sort",
        action="store_true",
        help="Only search directly inside each 02_chat_files cycle/site folder.",
    )

    return parser.parse_args()


def run_stage3(args: argparse.Namespace) -> None:
    """Run selected Stage 3 pipeline steps."""

    print("\n=== DIAAD Stage 3 pipeline ===")
    print(f"Project root: {PROJECT_ROOT}")

    if not args.skip_tt:
        print("\n=== Step 1/3: Reformat transcript tables ===")
        tt_processor.process_all_workbooks(
            input_root=ORIGINAL_TT_ROOT,
            output_root=REFORMATTED_TT_ROOT,
            continue_on_error=args.continue_on_error,
        )
    else:
        print("\n=== Step 1/3: Reformat transcript tables SKIPPED ===")

    if not args.skip_diaad:
        print("\n=== Step 2/3: Run DIAAD transcript CHAT export ===")
        diaad_runner.run_diaad(
            input_root=REFORMATTED_TT_ROOT,
            config_dir=CONFIG_DIR,
            output_root=CHAT_ROOT,
            cwd=PROJECT_ROOT,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run_diaad,
        )
    else:
        print("\n=== Step 2/3: Run DIAAD transcript CHAT export SKIPPED ===")

    if not args.skip_sort:
        print("\n=== Step 3/3: Sort and rename CHAT files ===")
        cha_sorter.sort_all_chat_files(
            input_root=CHAT_ROOT,
            output_root=SORTED_ROOT,
            move_files=args.move_chat_files,
            overwrite=args.overwrite_sorted,
            recursive=not args.nonrecursive_sort,
        )
    else:
        print("\n=== Step 3/3: Sort and rename CHAT files SKIPPED ===")

    print("\n=== Stage 3 pipeline finished ===")


def main() -> None:
    """CLI entry point."""

    run_stage3(parse_args())


if __name__ == "__main__":
    main()
