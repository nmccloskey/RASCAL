#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
from rascal.utils.logger import (
    get_root,
    set_root,
    logger,
    initialize_logger,
    terminate_logger,
)
from rascal.utils.auxiliary import (
    project_path,
    load_config,
    find_corresponding_file,
    OMNIBUS_MAP,
    COMMAND_MAP,
    build_arg_parser)
from rascal.run_wrappers import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples,
    run_reselect_transcription_reliability_samples,
    run_evaluate_transcription_reliability,
    run_make_transcript_tables, run_make_cu_coding_files,
    run_evaluate_cu_reliability,
    run_analyze_cu_coding, run_reselect_cu_reliability,
    run_make_word_count_files, run_evaluate_word_count_reliability,
    run_reselect_wc_reliability, run_summarize_cus, run_run_corelex
)


def main(args):
    """Main function to process input arguments and execute appropriate steps."""
    try:
        start_time = datetime.now()
        set_root(Path.cwd())

        # -----------------------------------------------------------------
        # Configuration and directories
        # -----------------------------------------------------------------
        config_path = project_path(args.config or "config.yaml")
        config = load_config(config_path)

        input_dir = project_path(config.get("input_dir", "rascal_data/input"))
        if not input_dir.is_relative_to(get_root()):
            logger.warning(f"Input directory {input_dir} is outside the project root.")
        output_dir = project_path(config.get("output_dir", "rascal_data/output"))

        timestamp = start_time.strftime("%y%m%d_%H%M")
        out_dir = (output_dir / f"rascal_output_{timestamp}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------
        # Initialize logger once output folder is ready
        # -----------------------------------------------------------------
        initialize_logger(start_time, out_dir, "RASCAL")
        logger.info("Logger initialized and early logs flushed.")

        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", []) or []
        cu_paradigms = config.get("cu_paradigms", []) or []
        exclude_participants = config.get("exclude_participants", []) or []
        strip_clan = config.get("strip_clan", True)
        prefer_correction = config.get("prefer_correction", True)
        lowercase = config.get("lowercase", True)

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
                logger.warning(f"Command {c} not recognized - skipping")

        if not converted:
            logger.error("No valid commands recognized — exiting.")
            return

        logger.info(f"Executing command(s): {', '.join(converted)}")

        # Load .cha if required
        chats = None
        if any(c in ["1a", "4a"] for c in converted):
            chats = run_read_cha_files(input_dir)

        # Prepare utterance files if needed
        if "4a" not in converted and any(c in ["4b", "7b", "10b"] for c in converted):
            transcript_tables = find_corresponding_file(directories=[input_dir, out_dir],
                                                        search_base="transcript_tables")
            if not transcript_tables:
                logger.info("No input transcript tables detected — creating them automatically.")
                chats = chats or run_read_cha_files(input_dir)
                run_make_transcript_tables(tiers, chats, out_dir)

        # ---------------------------------------------------------
        # Dispatch dictionary
        # ---------------------------------------------------------
        dispatch = {
            "1a": lambda: run_select_transcription_reliability_samples(tiers, chats, frac, out_dir),
            "3a": lambda: run_evaluate_transcription_reliability(
                tiers, input_dir, out_dir, exclude_participants, strip_clan, prefer_correction, lowercase
            ),
            "3b": lambda: run_reselect_transcription_reliability_samples(input_dir, out_dir, frac),
            "4a": lambda: run_make_transcript_tables(tiers, chats, out_dir),
            "4b": lambda: run_make_cu_coding_files(
                tiers, frac, coders, input_dir, out_dir, cu_paradigms, exclude_participants
            ),
            "6a": lambda: run_evaluate_cu_reliability(tiers, input_dir, out_dir, cu_paradigms),
            "6b": lambda: run_reselect_cu_reliability(tiers, input_dir, out_dir, "CU", frac),
            "7a": lambda: run_analyze_cu_coding(tiers, input_dir, out_dir, cu_paradigms),
            "7b": lambda: run_make_word_count_files(tiers, frac, coders, input_dir, out_dir),
            "9a": lambda: run_evaluate_word_count_reliability(tiers, input_dir, out_dir),
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
                logger.error(f"Unknown command: {cmd}")

        if executed:
            logger.info(f"Completed: {', '.join(executed)}")

    except Exception as e:
        logger.error(f"RASCAL execution failed: {e}", exc_info=True)
        raise

    finally:
        # Always finalize logging and metadata
        terminate_logger(input_dir, out_dir, config_path, config, start_time, "RASCAL")

# -------------------------------------------------------------
# Direct execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
