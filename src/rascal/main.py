#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from rascal.utils.support_funcs import *


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """Main function to process input arguments and execute appropriate steps."""
    config = load_config(args.config)
    input_dir = config.get('input_dir', 'rascal_data/input')
    output_dir = config.get('output_dir', 'rascal_data/output')
    
    frac = config.get('reliability_fraction', 0.2)
    coders = config.get('coders', []) or []
    CU_paradigms = config.get('CU_paradigms', []) or []

    exclude_participants = config.get('exclude_participants', []) or []
    strip_clan =  config.get('strip_clan', True)
    prefer_correction =  config.get('prefer_correction', True)
    lowercase =  config.get('lowercase', True)

    input_dir = as_path(config.get('input_dir', 'rascal_data/input'))
    output_dir = as_path(config.get('output_dir', 'rascal_data/output'))

    input_dir.mkdir(parents=True, exist_ok=True)

    tiers = run_read_tiers(config.get('tiers', {})) or {}

    steps_to_run = args.step

    # --- Timestamped output folder ---
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_dir = output_dir / f"rascal_{steps_to_run}_output_{timestamp}"
    output_dir.mkdir(parent=True, exist_ok=True)

    # Ensure .cha files read if required
    if 'a' in steps_to_run or 'd' in steps_to_run:
        chats = run_read_cha_files(input_dir)

    # Stage 1.
    if 'a' in steps_to_run:
        run_select_transcription_reliability_samples(tiers, chats, frac, output_dir)

    # Stage 3.
    if 'b' in steps_to_run:
        run_analyze_transcription_reliability(tiers, input_dir, output_dir, exclude_participants, strip_clan, prefer_correction, lowercase)
    if 'c' in steps_to_run:
        run_reselect_transcription_reliability_samples(input_dir, output_dir, frac)

    # Stage 4.
    if 'd' in steps_to_run:
        run_prepare_utterance_dfs(tiers, chats, output_dir)
    if 'e' in steps_to_run:
        run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir, CU_paradigms, exclude_participants)
    if 'f' in steps_to_run:
        run_make_timesheets(tiers, input_dir, output_dir)

    # Stage 6.
    if 'g' in steps_to_run:
        run_analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms)
    if 'h' in steps_to_run:
        run_reselect_CU_reliability(tiers, input_dir, output_dir, "CU", frac)

    # Stage 7.
    if 'i' in steps_to_run:
        run_analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms)
    if 'j' in steps_to_run:
        run_make_word_count_files(tiers, frac, coders, input_dir, output_dir)

    # Stage 9.
    if 'k' in steps_to_run:
        run_analyze_word_count_reliability(tiers, input_dir, output_dir)
    if 'l' in steps_to_run:
        run_reselect_WC_reliability(tiers, input_dir, output_dir, "WC", frac)

    # Stage 10.
    if 'm' in steps_to_run:
        run_unblind_CUs(tiers, input_dir, output_dir)
    if 'n' in steps_to_run:
        run_run_corelex(input_dir, output_dir, exclude_participants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the step argument for main script.")
    parser.add_argument('step', type=str, help="Specify the step or function(s) (e.g., 'dn' for minimal CoreLex or 'def' for Stage 4).")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()
    main(args)
