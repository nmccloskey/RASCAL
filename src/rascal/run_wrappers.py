import logging

def run_read_tiers(config_tiers):
    from rascal.utils.read_tiers import read_tiers
    tiers = read_tiers(config_tiers)
    if tiers:
        logging.info("Successfully parsed tiers from config.")
    else:
        logging.warning("Tiers are empty or malformed.")
    return tiers

def run_read_cha_files(input_dir, shuffle=False):
    from rascal.utils.read_cha_files import read_cha_files
    return read_cha_files(input_dir=input_dir, shuffle=shuffle)

def run_select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    from rascal.transcripts.transcription_reliability_selector import select_transcription_reliability_samples
    select_transcription_reliability_samples(tiers=tiers, chats=chats, frac=frac, output_dir=output_dir)

def run_reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    from rascal.transcripts.transcription_reliability_selector import reselect_transcription_reliability_samples
    reselect_transcription_reliability_samples(input_dir, output_dir, frac)

def run_make_transcript_tables(tiers, chats, output_dir):
    from rascal.transcripts.transcript_tables import make_transcript_tables
    return make_transcript_tables(tiers=tiers, chats=chats, output_dir=output_dir)

def run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir, CU_paradigms, exclude_participants):
    from rascal.coding.make_coding_files import make_CU_coding_files
    make_CU_coding_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms, exclude_participants=exclude_participants)

def run_analyze_transcription_reliability(tiers, input_dir, output_dir, exclude_participants, strip_clan, prefer_correction, lowercase):
    from rascal.transcripts.transcription_reliability_analysis import analyze_transcription_reliability
    analyze_transcription_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants, strip_clan=strip_clan, prefer_correction=prefer_correction, lowercase=lowercase)

def run_analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms):
    from rascal.coding.CU_analyzer import analyze_CU_reliability
    analyze_CU_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms)

def run_analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms):
    from rascal.coding.CU_analyzer import analyze_CU_coding
    analyze_CU_coding(tiers=tiers, input_dir=input_dir, output_dir=output_dir, CU_paradigms=CU_paradigms)

def run_make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    from rascal.coding.make_coding_files import make_word_count_files
    make_word_count_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir)

def run_make_timesheets(tiers, input_dir, output_dir):
    from rascal.utils.make_timesheets import make_timesheets
    make_timesheets(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_analyze_word_count_reliability(tiers, input_dir, output_dir):
    from rascal.coding.word_count_reliability_analyzer import analyze_word_count_reliability
    analyze_word_count_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_unblind_CUs(tiers, input_dir, output_dir):
    from rascal.samples.unblind_CUs import unblind_CUs
    unblind_CUs(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_run_corelex(input_dir, output_dir, exclude_participants):
    from rascal.samples.corelex import run_corelex
    run_corelex(input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants)

def run_reselect_CU_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from rascal.coding.make_coding_files import reselect_CU_WC_reliability
    reselect_CU_WC_reliability(tiers, input_dir, output_dir, rel_type, frac)

def run_reselect_WC_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from rascal.coding.make_coding_files import reselect_CU_WC_reliability
    reselect_CU_WC_reliability(tiers, input_dir, output_dir, rel_type, frac)
