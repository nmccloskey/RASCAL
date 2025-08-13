import os
import re
import pylangacq
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Align import PairwiseAligner
import logging


def percent_difference(value1, value2):
    """
    Calculate percent difference between two values.
    """
    return (abs(value1 - value2) / ((value1 + value2) / 2)) * 100

def _clean_clan_for_reliability(text: str) -> str:
    """
    Strip CLAN markup while preserving speech content crucial for reliability.
    - Keep fillers/disfluencies: '&um' -> 'um', '&uh' -> 'uh'
    - Remove structural markers: retracings [/], [//], [///], events <...>, comments ((...)), paralinguistic {...}
    - Remove bracket content [ ... ] *after* corrections handled, but preserve word-like content if any.
    - Drop tokens that are pure markup (e.g., =laughs), but keep speech hidden behind & or + if letter-like.

    This is intentionally milder than CoreLex reformatting: no contraction expansion, no digit->word, no stopword/filler removal.
    """
    # --- remove containers that never carry client words ---
    # Remove events <...>, comments ((...)), paralinguistic {...}
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\(\([^)]*\)\)", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)

    # Retracing markers and similar bracket codes (after correction handling):
    # [/], [//], [///], [?], [=! ...], [% ...], [& ...], etc.
    # If bracket content is purely non-letters, drop entirely.
    text = re.sub(r"\[\/*\]", " ", text)  # [/], [//], [///] variants
    text = re.sub(r"\[\s*[?%!=&][^\]]*\]", " ", text)

    # Any remaining bracketed spans (e.g., [x 2]) that aren't words → drop
    text = re.sub(r"\[\s*[^\w\]]+\s*\]", " ", text)

    # Remove standalone [*] (if any survived)
    text = re.sub(r"\[\*\]", " ", text)

    # --- convert speech-like tokens encoded as CLAN codes ---
    # &um, &uh, &erm -> um, uh, erm
    text = re.sub(r"(?<!\S)&([a-zA-Z]+)\b", r"\1", text)

    # +... variants sometimes mark pauses/continuations; if they prefix letters, keep letters.
    text = re.sub(r"(?<!\S)\++([a-zA-Z']+)\b", r"\1", text)

    # &=draws:a:cat or =laughs etc.  If token starts with non-word chars and then letters,
    # keep the tail letters; otherwise drop. (Conservative keep for speech-like tails)
    text = re.sub(r"(?<!\S)[^a-zA-Z'\s]+([a-zA-Z']+)\b", r"\1", text)

    # After the above, many pure markup tokens will reduce to nothing but punctuation; remove leftover [] explicitly
    text = re.sub(r"\[[^\]]+\]", " ", text)

    # Strip non-speech symbols but keep apostrophes and sentence punctuation .!?
    text = re.sub(r"[^\w\s'!.?]", " ", text)

    # Collapse multiple punctuation spaces like " ."
    text = re.sub(r"\s+(?=[.!?])", "", text)

    return text

def extract_cha_text(
    chat_data,
    *,
    exclude_participants=("INV",),   # exclude clinician by default
    strip_clan=True,                # keep raw CLAN if False
    prefer_correction=True,          # True => keep [: correction ] [*]; False => keep target
    lowercase=True
) -> str:
    """
    Extract a single comparison string from CHAT data for transcription reliability.

    - Minimal normalization when strip_clan=False (verbatim CLAN kept).
    - When strip_clan=True, CLAN markup is removed *but* speech-like content is preserved,
      including filled pauses (e.g., '&um' -> 'um') and disfluencies.

    Parameters
    ----------
    chat_data : pylangacq.Reader or compatible
        Must provide .utterances() yielding objects with .participant and .tiers[participant]
    exclude_participants : tuple[str]
        Participant codes to exclude (e.g., clinician 'INV').
    strip_clan : bool
        If True, return a speech-only surface (no CLAN codes). If False, keep CLAN.
    prefer_correction : bool
        Policy for accepted corrections '[: x] [*]': True keeps x, False keeps original token(s).
    lowercase : bool
        Lowercase final string for case-insensitive Levenshtein.
    """
    try:
        # 1) Collect utterances
        parts = []
        for line in chat_data.utterances():
            if line.participant in exclude_participants:
                continue
            utt = line.tiers.get(line.participant, "")
            # tighten spaces before . ! ?
            utt = re.sub(r"\s+(?=[.!?])", "", utt)
            parts.append(utt)
        text = " ".join(parts).strip()

        # 2) Normalize accepted corrections per policy
        # Patterns like: "... birbday [: birthday] [*] ..."
        if prefer_correction:
            # Keep the correction content, drop the target.
            # Also handle multiword corrections.
            text = re.sub(r"\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\1", text)
            # Remove any stray [*] that appear without '[: ...]'
            text = re.sub(r"\[\*\]", "", text)
        else:
            # Remove the correction block but keep original token(s)
            text = re.sub(r"\s*\[:\s*[^\]]+?\s*\]\s*\[\*\]", "", text)

        if strip_clan:
            text = _clean_clan_for_reliability(text)
        else:
            # Keep CLAN; just normalize whitespace lightly
            text = re.sub(r"[ \t]+", " ", text)

        # 3) Final touches: standardize whitespace/case but keep sentence punctuation and apostrophes
        text = re.sub(r"\s+", " ", text).strip()
        if lowercase:
            text = text.lower()
        return text

    except Exception as e:
        logging.error("extract_cha_text failed: %s", e)
        return ""

# Helper function to wrap lines at approximately 80 characters or based on delimiters
def wrap_text(text, width=80):
    """
    Wrap text to a specified width or based on utterance delimiters for better readability.
    """
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        # Add the word to the current line if it doesn't exceed the width limit
        if len(current_line) + len(word) + 1 <= width:
            current_line += ' ' + word
        else:
            # If the width limit is exceeded, append the current line and start a new one
            lines.append(current_line)
            current_line = word

    # Append the last line if there is any content left
    if current_line:
        lines.append(current_line)

    return lines

def write_reliability_report(transc_rel_subdf, report_path, partition_labels=None):
    """
    Write a plain-text transcription-reliability report.

    Parameters
    ----------
    transc_rel_subdf : pandas.DataFrame
        One row per sample. Must contain a numeric column
        'LevenshteinSimilarity' whose values lie in [0, 1].
    report_path : str | pathlib.Path
        Full path to the output .txt file.
    partition_labels : list[str] | None
        Optional tier / partition labels to display in the header.
    """

    try:
        # ── sanity checks ──────────────────────────────────────────────────────
        if 'LevenshteinSimilarity' not in transc_rel_subdf.columns:
            raise KeyError("'LevenshteinSimilarity' column is missing.")

        ls = transc_rel_subdf['LevenshteinSimilarity'].astype(float).dropna()
        n_samples = len(ls)
        mean_ls   = ls.mean()
        sd_ls     = ls.std()
        min_ls    = ls.min()
        max_ls    = ls.max()

        # ── similarity bands ───────────────────────────────────────────────────
        bands = {
            "Excellent (≥ .90)":        (ls >= 0.90),
            "Sufficient (.80 – .89)":   ((ls >= 0.80) & (ls < 0.90)),
            "Min. acceptable (.70 – .79)": ((ls >= 0.70) & (ls < 0.80)),
            "Below .70":               (ls < 0.70),
        }
        counts = {label: mask.sum() for label, mask in bands.items()}

        # ── compose the report text ────────────────────────────────────────────
        header = "Transcription Reliability Report"
        if partition_labels:
            header += f" for {' '.join(map(str, partition_labels))}"

        lines = [
            header,
            "=" * len(header),
            f"Number of samples: {n_samples}",
            "",
            f"Levenshtein similarity score summary stats:",
            f"  • Average: {mean_ls:.3f}",
            f"  • Standard Deviation: {sd_ls:.3f}",
            f"  • Min: {min_ls:.3f}",
            f"  • Max: {max_ls:.3f}",
            "",
            "Similarity bands:",
        ]
        for label, count in counts.items():
            pct = count / n_samples * 100 if n_samples else 0
            lines.append(f"  • {label}: {count} ({pct:.1f}%)")

        report_text = "\n".join(lines)

        # ── write to disk ──────────────────────────────────────────────────────
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logging.info("Successfully wrote transcription reliability report to %s", report_path)

    except Exception as e:
        logging.error("Failed to write transcription reliability report to %s: %s", report_path, e)
        raise

def analyze_transcription_reliability(tiers, input_dir, output_dir, test=False):
    """
    Analyze transcription reliability by comparing original and reliability CHAT files.

    Parameters:
    - tiers (dict): Dictionary of tier information for partitioning.
    - input_dir (str): Directory containing input CHAT files.
    - output_dir (str): Directory where analysis results will be saved.
    - test (bool): If True, run in test mode.
    """
    dfcols = list(tiers.keys()) + ['OrgFile', 'RelFile',
                                   'OrgNumTokens', 'RelNumTokens', 'PercDiffNumTokens',
                                   'OrgNumChars', 'RelNumChars', 'PercDiffNumChars',
                                   'LevenshteinDistance','LevenshteinSimilarity',
                                   'NeedlemanWunschDistance', 'NeedlemanWunschScore']
    transc_rel_df = pd.DataFrame(columns=dfcols)

    # Make output folder.
    transc_rel_dir = os.path.join(output_dir, 'TranscriptionReliabilityAnalysis')
    try:
        os.makedirs(transc_rel_dir, exist_ok=True)
        logging.info(f"Created directory: {transc_rel_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {transc_rel_dir}: {e}")
        return

    # Output subfolders based on partition tiers.
    partition_tiers = [t.name for t in tiers.values() if t.partition]

    # Read all .cha files in input directory.
    cha_files = [cha for cha in Path(input_dir).rglob("*.cha")]
    logging.info(f"Found {len(cha_files)} .cha files in the input directory.")
    rel_chats = [cha for cha in cha_files if 'Reliability' in cha.name]
    org_chats = [cha for cha in cha_files if 'Reliability' not in cha.name]
    
    logging.info("Analyzing transcription reliability...")
    for rel_cha in tqdm(rel_chats, desc="Analyzing reliability transcripts"):
        rel_labels = [t.match(rel_cha.name) for t in tiers.values()]
        logging.debug(f"Reliability file labels: {rel_labels}")

        for org_cha in org_chats:
            org_labels = [t.match(org_cha.name) for t in tiers.values()]
            # logging.info(f"Original file labels: {org_labels}")

            if rel_labels == org_labels:
                try:
                    org_chat_data = pylangacq.read_chat(str(org_cha))
                    rel_chat_data = pylangacq.read_chat(str(rel_cha))
                    # logging.info(f"Matching original file: {org_cha.name} with reliability file: {rel_cha.name}")
                except Exception as e:
                    logging.error(f"Failed to read CHAT files {org_cha} or {rel_cha}: {e}")
                    continue

                try:
                    # Extract text from both samples.
                    org_text = extract_cha_text(org_chat_data)
                    rel_text = extract_cha_text(rel_chat_data)

                    # Simple analysis.
                    org_num_tokens = len(org_text.split(' '))
                    rel_num_tokens = len(rel_text.split(' '))
                    pdiff_num_tokens = percent_difference(org_num_tokens, rel_num_tokens)
                    org_num_chars = len(org_text)
                    rel_num_chars = len(rel_text)
                    pdiff_num_chars = percent_difference(org_num_chars, rel_num_chars)
                    
                    # Levenshtein algorithm.
                    Ldist = distance(org_text, rel_text)
                    max_len = max(len(org_text), len(rel_text))
                    Lscore = 1 - (Ldist / max_len)

                    # Initialize the Needleman-Wunsch algorithm aligner
                    aligner = PairwiseAligner()
                    aligner.mode = 'global'
                    alignments = aligner.align(org_text, rel_text)
                    best_alignment = alignments[0]
                    best_score = best_alignment.score
                    longer_length = max(len(org_text), len(rel_text))
                    normalized_score = best_score / longer_length

                    row = rel_labels + [org_cha.name, rel_cha.name,
                                        org_num_tokens, rel_num_tokens, pdiff_num_tokens,
                                        org_num_chars, rel_num_chars, pdiff_num_chars,
                                        Ldist, Lscore, best_score, normalized_score]
                    transc_rel_df.loc[len(transc_rel_df)] = row
                    logging.debug(f"Appended row to DataFrame: {row}")

                    # Prepare the alignment output as formatted text
                    alignment_str = f"Global alignment score: {best_score}\n"
                    alignment_str += f"Normalized score (by length): {normalized_score}\n\n"

                    # Prepare the alignment text with match indicators
                    seq1 = best_alignment[0]
                    seq2 = best_alignment[1]

                    # Wrap each line to 80 characters
                    seq1_lines = wrap_text(seq1)
                    seq2_lines = wrap_text(seq2)

                    for s1, s2 in zip(seq1_lines, seq2_lines):
                        alignment_str += f"Sequence 1: {s1}\n"
                        alignment_line = ''.join(['|' if a == b else ' ' for a, b in zip(s1, s2)])
                        alignment_str += f"Alignment : {alignment_line}\n"
                        alignment_str += f"Sequence 2: {s2}\n\n"

                    # Extract partition tier info from file name.
                    partition_labels = [t.match(rel_cha.name) for t in tiers.values() if t.partition]
                    text_filename = f"{''.join(rel_labels)}_TranscriptionReliabilityAlignment.txt"
                    text_file_path = os.path.join(transc_rel_dir, *partition_labels, 'GlobalAlignments', text_filename)
                    try:
                        os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
                        with open(text_file_path, 'w') as file:
                            file.write(alignment_str)
                        # logging.info(f"Saved alignment text to: {text_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to write alignment file {text_file_path}: {e}")
                
                except Exception as e:
                    logging.error(f"Failed to analyze transcription reliability for {org_cha} and {rel_cha}: {e}.")
    
    # Store results for testing.
    results = []
    
    # Partition by designated tier(s).
    for tup, subdf in tqdm(transc_rel_df.groupby(partition_tiers), desc="Saving grouped DataFrames & reports"):
        df_filename = '_'.join(tup) + '_TranscriptionReliabilityAnalysis.xlsx'
        df_path = os.path.join(transc_rel_dir, *tup, df_filename)
        report_filename = '_'.join(tup) + '_TranscriptionReliabilityReport.txt'
        report_path = os.path.join(transc_rel_dir, *tup, report_filename)
        write_reliability_report(subdf, report_path, tup)

        try:
            subdf.to_excel(df_path, index=False)
            logging.info(f"Saved reliability analysis DataFrame to: {df_path}")
        except Exception as e:
            logging.error(f"Failed to write DataFrame to {df_path}: {e}")
        
        if test:
            results.append(subdf)
    
    if test:
        return results
