from rascal.utils.logger import logger


def run_read_tiers(config_tiers):
    from rascal.utils.tier_parsing import read_tiers
    tiers = read_tiers(config_tiers)
    if tiers:
        logger.info("Successfully parsed tiers from config.")
    else:
        logger.warning("Tiers are empty or malformed.")
    return tiers

def run_read_cha_files(input_dir, shuffle=False):
    from rascal.utils.cha_files import read_cha_files
    return read_cha_files(input_dir=input_dir, shuffle=shuffle)

def run_select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    from rascal.transcripts.transcription_reliability_selection import select_transcription_reliability_samples
    select_transcription_reliability_samples(tiers=tiers, chats=chats, frac=frac, output_dir=output_dir)

def run_reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    from rascal.transcripts.transcription_reliability_selection import reselect_transcription_reliability_samples
    reselect_transcription_reliability_samples(input_dir, output_dir, frac)

def run_make_transcript_tables(tiers, chats, output_dir):
    from rascal.transcripts.transcript_tables import make_transcript_tables
    return make_transcript_tables(tiers=tiers, chats=chats, output_dir=output_dir)

def run_make_cu_coding_files(tiers, frac, coders, input_dir, output_dir, cu_paradigms, exclude_participants):
    from rascal.coding.coding_files import make_cu_coding_files
    make_cu_coding_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms, exclude_participants=exclude_participants)

def run_evaluate_transcription_reliability(tiers, input_dir, output_dir, exclude_participants, strip_clan, prefer_correction, lowercase):
    from rascal.transcripts.transcription_reliability_evaluation import evaluate_transcription_reliability
    evaluate_transcription_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants, strip_clan=strip_clan, prefer_correction=prefer_correction, lowercase=lowercase)

def run_evaluate_cu_reliability(tiers, input_dir, output_dir, cu_paradigms):
    from rascal.coding.cu_analysis import evaluate_cu_reliability
    evaluate_cu_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms)

def run_analyze_cu_coding(tiers, input_dir, output_dir, cu_paradigms):
    from rascal.coding.cu_analysis import analyze_cu_coding
    analyze_cu_coding(tiers=tiers, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms)

def run_make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    from rascal.coding.coding_files import make_word_count_files
    make_word_count_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir)

def run_analyze_word_count_reliability(tiers, input_dir, output_dir):
    from rascal.coding.word_count_reliability_evaluation import analyze_word_count_reliability
    analyze_word_count_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_summarize_cus(tiers, input_dir, output_dir):
    from rascal.coding.cu_summarization import summarize_cus
    summarize_cus(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

def run_run_corelex(tiers, input_dir, output_dir, exclude_participants):
    from rascal.coding.corelex import run_corelex
    run_corelex(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants)

def run_reselect_cu_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from rascal.coding.coding_files import reselect_cu_wc_reliability
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type, frac)

def run_reselect_wc_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from rascal.coding.coding_files import reselect_cu_wc_reliability
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type, frac)
