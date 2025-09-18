#!/usr/bin/env python3
import os
import yaml
import argparse
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file):
    """Load configuration settings from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def run_read_tiers(config_tiers):
    from .utils.read_tiers import read_tiers
    tiers = read_tiers(config_tiers)
    if tiers:
        logging.info("Successfully parsed tiers from config.")
    else:
        logging.warning("Tiers are empty or malformed.")
    return tiers

def run_read_cha_files(input_dir):
    from .utils.read_cha_files import read_cha_files
    return read_cha_files(input_dir=input_dir, shuffle=True)

def run_select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    from .transcription.transcription_reliability_selector import select_transcription_reliability_samples
    select_transcription_reliability_samples(tiers=tiers, chats=chats, frac=frac, output_dir=output_dir)

def run_reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    from .transcription.transcription_reliability_selector import reselect_transcription_reliability_samples
    reselect_transcription_reliability_samples(input_dir, output_dir, frac)

def run_prepare_utterance_dfs(tiers, chats, output_dir):
    from .utterances.make_utterance_tables import prepare_utterance_dfs
    return prepare_utterance_dfs(tiers=tiers, chats=chats, output_dir=output_dir)

def run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir, CU_paradigms, exclude_participants):
    from .utterances.make_coding_files import make_CU_coding_files
    make_CU_coding_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms, exclude_participants=exclude_participants)

def run_analyze_transcription_reliability(tiers, input_dir, output_dir, exclude_participants, strip_clan, prefer_correction, lowercase):
    from .transcription.transcription_reliability_analysis import analyze_transcription_reliability
    analyze_transcription_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants, strip_clan=strip_clan, prefer_correction=prefer_correction, lowercase=lowercase)

def run_analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms):
    from .utterances.CU_analyzer import analyze_CU_reliability
    analyze_CU_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms)

def run_analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms):
    from .utterances.CU_analyzer import analyze_CU_coding
    analyze_CU_coding(tiers=tiers, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms)

def run_make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    from .utterances.make_coding_files import make_word_count_files
    make_word_count_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir)

def run_make_timesheets(tiers, input_dir, output_dir):
    from .utils.make_timesheets import make_timesheets
    make_timesheets(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_analyze_word_count_reliability(tiers, input_dir, output_dir):
    from .utterances.word_count_reliability_analyzer import analyze_word_count_reliability
    analyze_word_count_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_unblind_CUs(tiers, input_dir, output_dir):
    from .samples.unblind_CUs import unblind_CUs
    unblind_CUs(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_run_corelex(input_dir, output_dir, exclude_participants):
    from .samples.corelex import run_corelex
    run_corelex(input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants)

def run_reselect_CU_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from .utterances.make_coding_files import reselect_CU_WC_reliability
    reselect_CU_WC_reliability(tiers, input_dir, output_dir, rel_type, frac)

def run_reselect_WC_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from .utterances.make_coding_files import reselect_CU_WC_reliability
    reselect_CU_WC_reliability(tiers, input_dir, output_dir, rel_type, frac)


def main(args):
    """Main function to process input arguments and execute appropriate steps."""
    config = load_config(args.config)
    input_dir = config.get('input_dir', 'data/input')
    output_dir = config.get('output_dir', 'data/output')
    
    frac = config.get('reliability_fraction', 0.2)
    coders = config.get('coders', [])
    CU_paradigms = config.get('CU_paradigms', []) or []

    exclude_participants = config.get('exclude_participants', []) or []
    strip_clan =  config.get('strip_clan', True)
    prefer_correction =  config.get('prefer_correction', True)
    lowercase =  config.get('lowercase', True)

    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    os.makedirs(input_dir, exist_ok=True)

    tiers = run_read_tiers(config.get('tiers', {}))

    steps_to_run = args.step

    # --- Timestamped output folder ---
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_dir = os.path.join(output_dir, f"rascal_{steps_to_run}_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

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
    parser.add_argument('step', type=str, help="Specify the step or function(s) (e.g., '1' or 'abc').")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()
    main(args)
