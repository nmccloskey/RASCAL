#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from rascal.utils.support_funcs import as_path, load_config, find_transcript_tables
from rascal.run_wrappers import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples,
    run_reselect_transcription_reliability_samples,
    run_analyze_transcription_reliability,
    run_make_transcript_tables, run_make_cu_coding_files,
    run_analyze_cu_reliability,
    run_analyze_cu_coding, run_reselect_cu_reliability,
    run_make_word_count_files, run_analyze_word_count_reliability,
    run_reselect_wc_reliability, run_summarize_cus, run_run_corelex
)

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------
# Omnibus mappings
# -------------------------------------------------------------
OMNIBUS_MAP = {
    "1": ["1a"],
    "4": ["4a", "4b"],
    "7": ["7a", "7b"],
    "10": ["10a", "10b"],
}

COMMAND_MAP = {
    "1a": "transcripts select",
    "3a": "transcripts evaluate",
    "3b": "transcripts reselect",
    "4a": "transcripts make",
    "4b": "cus make",
    "6a": "cus evaluate",
    "6b": "cus reselect",
    "7a": "cus analyze",
    "7b": "words make",
    "9a": "words evaluate",
    "9b": "words reselect",
    "10a": "cus summarize",
    "10b": "corelex analyze",
}

# -------------------------------------------------------------
# CLI setup utilities
# -------------------------------------------------------------
def build_arg_parser():
    """Construct and return the argument parser used by both main.py and cli.py."""
    parser = argparse.ArgumentParser(
        description=(
            "RASCAL command-line interface.\n\n"
            "Examples:\n"
            "  rascal 3b\n"
            "  rascal transcripts reselect\n"
            "  rascal 4\n"
            "  rascal 4a,4b\n"
            "  rascal utterances make, cus make, timesheets make\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="+",
        help="Command(s) to run (comma-separated or space-separated)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )

    # ---- Help text for expansions ----
    help_lines = ["\nAvailable Commands:\n"]
    for short, long in COMMAND_MAP.items():
        help_lines.append(f"  {short:<4}  →  {long}")
    help_lines.append("\nOmnibus Commands:\n")
    for omni, subs in OMNIBUS_MAP.items():
        expansions = [f"{s} ({COMMAND_MAP[s]})" for s in subs]
        help_lines.append(f"  {omni:<4}  →  {', '.join(expansions)}")
    parser.epilog = "\n".join(help_lines)

    return parser

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main(args):
    """Main function to process input arguments and execute appropriate steps."""
    config = load_config(args.config)
    input_dir = as_path(config.get("input_dir", "rascal_data/input"))
    output_dir = as_path(config.get("output_dir", "rascal_data/output"))
    frac = config.get("reliability_fraction", 0.2)
    coders = config.get("coders", []) or []
    cu_paradigms = config.get("cu_paradigms", []) or []
    exclude_participants = config.get("exclude_participants", []) or []
    strip_clan = config.get("strip_clan", True)
    prefer_correction = config.get("prefer_correction", True)
    lowercase = config.get("lowercase", True)

    input_dir.mkdir(parents=True, exist_ok=True)
    tiers = run_read_tiers(config.get("tiers", {})) or {}

    # ---------------------------------------------------------
    # Expand omnibus & comma-separated commands
    # ---------------------------------------------------------
    if isinstance(args.command, list):
        args.command = " ".join(args.command)
    raw_commands = [c.strip() for c in args.command.split(",") if c.strip()]

    # Standardize to succinct abbreviations
    rev_cmap = {v: k for k, v in COMMAND_MAP.items()}
    converted = []
    for c in raw_commands:
        # Direct succinct match
        if c in COMMAND_MAP:
            converted.append(c)
        # Expanded (e.g. "transcripts select")
        elif c in COMMAND_MAP.values():
            converted.append(rev_cmap[c])
        # Omnibus --> succinct
        elif c in OMNIBUS_MAP:
            converted.extend(OMNIBUS_MAP[c])
        else:
            logging.warning(f"Command {c} not recognized - skipping")

    if not converted:
        logging.error("No valid commands recognized — exiting.")
        return

    logging.info(f"Executing standardized command(s): {', '.join(converted)}")

    # Timestamped output folder
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    out_dir = output_dir / f"rascal_{'_'.join(converted)}_output_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load .cha if required
    chats = None
    if any(c in ["1a", "4a"] for c in converted):
        chats = run_read_cha_files(input_dir)

    # Prepare utterance files if needed
    if "4a" not in converted and any(c in ["4b", "7b", "10b"] for c in converted):
        transcript_tables = find_transcript_tables(input_dir, out_dir)
        if not transcript_tables:
            logging.info("No input transcript tables detected — creating them automatically.")
            chats = chats or run_read_cha_files(input_dir)
            run_make_transcript_tables(tiers, chats, out_dir)

    # ---------------------------------------------------------
    # Dispatch dictionary
    # ---------------------------------------------------------
    dispatch = {
        "1a": lambda: run_select_transcription_reliability_samples(tiers, chats, frac, out_dir),
        "3a": lambda: run_analyze_transcription_reliability(
            tiers, input_dir, out_dir, exclude_participants, strip_clan, prefer_correction, lowercase
        ),
        "3b": lambda: run_reselect_transcription_reliability_samples(input_dir, out_dir, frac),
        "4a": lambda: run_make_transcript_tables(tiers, chats, out_dir),
        "4b": lambda: run_make_cu_coding_files(
            tiers, frac, coders, input_dir, out_dir, cu_paradigms, exclude_participants
        ),
        "6a": lambda: run_analyze_cu_reliability(tiers, input_dir, out_dir, cu_paradigms),
        "6b": lambda: run_reselect_cu_reliability(tiers, input_dir, out_dir, "CU", frac),
        "7a": lambda: run_analyze_cu_coding(tiers, input_dir, out_dir, cu_paradigms),
        "7b": lambda: run_make_word_count_files(tiers, frac, coders, input_dir, out_dir),
        "9a": lambda: run_analyze_word_count_reliability(tiers, input_dir, out_dir),
        "9b": lambda: run_reselect_wc_reliability(tiers, input_dir, out_dir, "WC", frac),
        "10a": lambda: run_summarize_cus(tiers, input_dir, out_dir),
        "10b": lambda: run_run_corelex(tiers, input_dir, out_dir, exclude_participants),
    }

    # ---------------------------------------------------------
    # Execute all requested commands
    # ---------------------------------------------------------
    executed = []
    for cmd in converted:
        func = dispatch.get(cmd)
        if func:
            func()
            executed.append(cmd)
        else:
            logging.error(f"Unknown command: {cmd}")

    if executed:
        logging.info(f"Completed: {', '.join(executed)}")

# -------------------------------------------------------------
# Direct execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
