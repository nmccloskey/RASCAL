from __future__ import annotations

import argparse
from pathlib import Path

from .file_cleaning import clean_files
from .diaad_running import run_diaad


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare POWERS coding files and run DIAAD."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean raw POWERS coding files and write revisions/aggregate files.",
    )
    clean_parser.add_argument("--continue-on-error", action="store_true", default=True)
    clean_parser.add_argument("--no-aggregate", action="store_true")
    clean_parser.add_argument(
        "--force-write-cleaned",
        action="store_true",
        help=(
            "Write cleaned files even when the revisions workbook has rows. "
            "By default, revisions block cleaned outputs."
        ),
    )

    diaad_parser = subparsers.add_parser(
        "run-diaad",
        help="Run DIAAD on cleaned POWERS files.",
    )
    diaad_parser.add_argument("--partitions", action="store_true")
    diaad_parser.add_argument("--aggregates", action="store_true")
    diaad_parser.add_argument("--dry-run", action="store_true")
    diaad_parser.add_argument("--continue-on-error", action="store_true")

    return parser


def main() -> None:
    """
    Orchestrate POWERS data prep for DIAAD processing.

    Designed for a semi-automated and iterative workflow:
    data cleaning -> revision -> re-cleaning -> DIAAD processing

    CLI commands:
    python -m src.main clean
    # review/fix powers_cleaning_revisions.xlsx
    python -m src.main clean
    python -m src.main run-diaad

    Other examples with args:
    python -m src.main clean
    python -m src.main clean --no-aggregate
    python -m src.main run-diaad --dry-run
    python -m src.main run-diaad --aggregates
    python -m src.main run-diaad --partitions
    """
    args = build_parser().parse_args()
    project_root = get_project_root()

    raw_dir = project_root / "data" / "00_raw_files"
    cleaned_dir = project_root / "data" / "01_cleaned_files"
    diaad_out_dir = project_root / "data" / "02_diaad_output"
    revisions_dir = project_root / "revisions"
    log_dir = project_root / "logs"
    config_dir = project_root / "config"

    if args.command == "clean":
        clean_files(
            input_dir=raw_dir,
            output_dir=cleaned_dir,
            log_dir=log_dir,
            revisions_dir=revisions_dir,
            continue_on_error=args.continue_on_error,
            aggregate=not args.no_aggregate,
            force_write_cleaned=args.force_write_cleaned,
        )

    elif args.command == "run-diaad":
        run_partitions = args.partitions
        run_aggregates = args.aggregates

        # If neither is specified, run both.
        if not run_partitions and not run_aggregates:
            run_partitions = True
            run_aggregates = True

        run_diaad(
            input_dir=cleaned_dir,
            config_dir=config_dir,
            output_root=diaad_out_dir,
            run_partitions=run_partitions,
            run_aggregates=run_aggregates,
            log_dir=log_dir,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
