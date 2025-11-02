import re
import pylangacq
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Align import PairwiseAligner
import logging
from typing import Union, List


def percent_difference(a, b):
    try:
        a, b = float(a), float(b)
        if a == 0 and b == 0:
            return 0.0
        denom = (abs(a) + abs(b)) / 2.0
        return (abs(a - b) / denom) * 100.0 if denom != 0 else 0.0
    except Exception:
        return float("nan")

def scrub_clan(text: str) -> str:
    """
    Remove CLAN markup while keeping only speech-relevant material.

    - Keep common disfluencies like &um, &uh, &h (→ 'um', 'uh', 'h')
    - Remove gesture and non-speech codes (e.g., &=points:leg, =laughs, <...>, ((...)), {...}, [/], [//])
    - Remove any remaining bracketed or symbolic markup
    - Preserve ordinary words, punctuation (.!?), and apostrophes

    Example
    -------
    Input : "but &-um &-uh &+h hurt &=points:leg oh well"
    Output: "but um uh h hurt oh well"
    """
    # normalize speech-like tokens (&um, &-uh, &+h → um, uh, h)
    text = re.sub(r"(?<!\S)&[-+]?([a-zA-Z]+)\b", r"\1", text)
    # remove all other &-prefixed tokens
    text = re.sub(r"(?<!\S)&\S+", " ", text)

    # remove structural / paralinguistic markup
    text = re.sub(r"\(\([^)]*\)\)", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[\/*\]", " ", text)
    text = re.sub(r"\[[^]]*\]", " ", text)

    # remove =codes (e.g., =laughs)
    text = re.sub(r"(?<!\S)=[^\s]+", " ", text)

    # remove non-speech symbols except .!? and apostrophes
    text = re.sub(r"[^\w\s'!.?]", " ", text)

    # tidy whitespace
    text = re.sub(r"\s+(?=[.!?])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text

def process_corrections(text: str, prefer_correction: bool = True) -> str:
    """
    Handle CLAN correction notation ([: correction] [*]) according to preference.

    prefer_correction=True  -> replace with correction
    prefer_correction=False -> keep original (remove correction markup)
    """
    if prefer_correction:
        # Replace "orig [: corr] [*]" with "corr"
        text = re.sub(r"(\S+)\s*\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\2", text)
    else:
        # Replace "orig [: corr] [*]" with "orig"
        text = re.sub(r"(\S+)\s*\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\1", text)

    # Clean up spacing
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def extract_cha_text(
    source: Union[str, pylangacq.Reader],
    exclude_participants: List[str] = None,
) -> str:
    """
    Extract utterance text only when a pylangacq.Reader is provided.

    For RASCAL: accepts a Reader and returns concatenated utterances.
    For DIAAD: if input is already a text string, it is returned unchanged
    (no pylangacq parsing).

    Parameters
    ----------
    source : str or pylangacq.Reader
        - pylangacq.Reader → extract utterances
        - str → returned unchanged (already plain text)
    exclude_participants : list[str], optional
        Participant codes to exclude (e.g., ['INV']).
    """
    exclude_participants = exclude_participants or []

    try:
        if isinstance(source, pylangacq.Reader):
            parts = []
            for line in source.utterances():
                if line.participant in exclude_participants:
                    continue
                utt = line.tiers.get(line.participant, "")
                utt = re.sub(r"\s+(?=[.!?])", "", utt)
                parts.append(utt)
            return " ".join(parts).strip()

        elif isinstance(source, str):
            # Return string unchanged — already text
            return source.strip()

        else:
            raise TypeError(
                f"Unsupported input type for extract_cha_text: {type(source)}"
            )

    except Exception as e:
        logging.error(f"extract_cha_text failed: {e}")
        return ""

def process_utterances(
    chat_data: Union[str, pylangacq.Reader],
    *,
    exclude_participants: List[str] = None,
    strip_clan: bool = True,
    prefer_correction: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Unified utterance-processing pipeline for both RASCAL (Reader input)
    and DIAAD (plain text input).

    Behavior
    --------
    - If `chat_data` is a pylangacq.Reader, extract and process utterances.
    - If `chat_data` is already a string, skip pylangacq and process directly.
    - Optionally remove CLAN markup and/or apply correction preferences.

    Parameters
    ----------
    chat_data : str or pylangacq.Reader
        CHAT text (string) or Reader object.
    exclude_participants : list[str], optional
        Participants to omit (used only for Reader input).
    strip_clan : bool
        If True, scrub CLAN markup.
    prefer_correction : bool
        Policy for handling [: correction] [*].
    lowercase : bool
        Lowercase final output.
    """
    # 1. Extract text (Reader → concatenated utterances; str → unchanged)
    text = extract_cha_text(chat_data, exclude_participants)
    if not text:
        return ""

    # 2. Handle corrections
    text = process_corrections(text, prefer_correction)

    # 3. Optionally strip CLAN markup
    if strip_clan:
        text = scrub_clan(text)
    else:
        text = re.sub(r"[ \t]+", " ", text)

    # 4. Final normalization
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()

    return text

# Helper function to wrap lines at approximately 80 characters or based on delimiters
def _wrap_text(text, width=80):
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

# ---------- helpers: computation ----------

def _compute_simple_stats(org_text: str, rel_text: str):
    org_tokens = org_text.split()
    rel_tokens = rel_text.split()
    org_num_tokens = len(org_tokens)
    rel_num_tokens = len(rel_tokens)
    pdiff_num_tokens = percent_difference(org_num_tokens, rel_num_tokens)

    org_num_chars = len(org_text)
    rel_num_chars = len(rel_text)
    pdiff_num_chars = percent_difference(org_num_chars, rel_num_chars)

    return {
        "OrgNumTokens": org_num_tokens,
        "RelNumTokens": rel_num_tokens,
        "PercDiffNumTokens": pdiff_num_tokens,
        "OrgNumChars": org_num_chars,
        "RelNumChars": rel_num_chars,
        "PercDiffNumChars": pdiff_num_chars,
    }

def _levenshtein_metrics(org_text: str, rel_text: str):
    Ldist = distance(org_text, rel_text)
    max_len = max(len(org_text), len(rel_text)) or 1
    Lscore = 1 - (Ldist / max_len)
    return {"LevenshteinDistance": Ldist, "LevenshteinSimilarity": Lscore}

def _needleman_wunsch_global(org_text: str, rel_text: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignments = aligner.align(org_text, rel_text)
    best = alignments[0]
    best_score = best.score
    norm = best_score / (max(len(org_text), len(rel_text)) or 1)
    return {"NeedlemanWunschScore": best_score,
            "NeedlemanWunschNorm": norm,
            "alignment": best}

# ---------- helpers: alignment pretty print ----------

def _format_alignment_output(alignment, best_score: float, normalized_score: float):
    # Extract the two aligned sequences; Biopython's pairwise alignment object behaves like a 2-row alignment
    seq1 = alignment[0]
    seq2 = alignment[1]

    seq1_lines = _wrap_text(seq1)
    seq2_lines = _wrap_text(seq2)

    out = []
    out.append(f"Global alignment score: {best_score}")
    out.append(f"Normalized score (by length): {normalized_score}")
    out.append("")

    for s1, s2 in zip(seq1_lines, seq2_lines):
        out.append(f"Sequence 1: {s1}")
        align_line = "".join("|" if a == b else " " for a, b in zip(s1, s2))
        out.append(f"Alignment : {align_line}")
        out.append(f"Sequence 2: {s2}")
        out.append("")

    return "\n".join(out)

def _ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ---------- main analysis ----------

def analyze_transcription_reliability(
    tiers,
    input_dir,
    output_dir,
    exclude_participants=[],
    strip_clan=True,
    prefer_correction=True,
    lowercase=True,
    test=False
):
    """
    Analyze transcription reliability by comparing original and reliability CHAT files.

    Parameters
    ----------
    tiers : dict
        Tier objects with attributes: .name, .partition, .match(filename)->label
    input_dir : str
        Directory containing input CHAT files.
    output_dir : str
        Base directory where analysis results will be saved.
    exclude_participants, strip_clan, prefer_correction, lowercase :
        Passed through to extract_cha_text().
    test : bool
        If True, return grouped DataFrames for tests instead of None.        
    """
    # --- setup output dirs ---
    transc_rel_dir = output_dir / "TranscriptionReliabilityAnalysis"
    transc_rel_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {transc_rel_dir}")

    # Which tiers define partitions?
    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]

    # --- collect files and index originals by labels for O(1) match lookup ---
    cha_files = list(Path(input_dir).rglob("*.cha"))
    logging.info(f"Found {len(cha_files)} .cha files in the input directory.")

    rel_chats = [p for p in cha_files if "Reliability" in p.name]
    org_chats = [p for p in cha_files if "Reliability" not in p.name]

    def _labels_for(path: Path):
        return tuple(t.match(path.name) for t in tiers.values())

    org_index = {}
    for org in org_chats:
        labels = _labels_for(org)
        org_index[labels] = org

    # --- iterate reliability files and analyze ---
    records = []
    seen_rel_files = set()
    seen_org_files = set()

    for rel_cha in tqdm(rel_chats, desc="Analyzing reliability transcripts"):
        rel_labels = _labels_for(rel_cha)
        org_cha = org_index.get(rel_labels)
        if org_cha is None:
            logging.warning(f"No matching original .cha for reliability file: {rel_cha.name}")
            continue

        # --- safeguard: skip if reliability file already processed ---
        if rel_cha.name in seen_rel_files:
            logging.warning(f"Skipping duplicate reliability file: {rel_cha.name}")
            continue

        # --- safeguard: skip if original file already paired ---
        if org_cha.name in seen_org_files:
            logging.warning(
                f"Skipping reliability file {rel_cha.name} because original already used: {org_cha.name}"
            )
            continue

        # mark both as seen
        seen_rel_files.add(rel_cha.name)
        seen_org_files.add(org_cha.name)

        try:
            org_chat_data = pylangacq.read_chat(str(org_cha))
            rel_chat_data = pylangacq.read_chat(str(rel_cha))
        except Exception as e:
            logging.error(f"Failed to read CHAT files {org_cha} or {rel_cha}: {e}")
            continue

        try:
            org_text = process_utterances(
                org_chat_data,
                exclude_participants=exclude_participants,
                strip_clan=strip_clan,
                prefer_correction=prefer_correction,
                lowercase=lowercase,
            )
            rel_text = process_utterances(
                rel_chat_data,
                exclude_participants=exclude_participants,
                strip_clan=strip_clan,
                prefer_correction=prefer_correction,
                lowercase=lowercase,
            )

            # Compute metrics
            simple = _compute_simple_stats(org_text, rel_text)
            lev = _levenshtein_metrics(org_text, rel_text)
            nw = _needleman_wunsch_global(org_text, rel_text)

            # ---------- save alignment pretty-print ----------
            # Build path components from partitions (if any)
            partition_labels = [t.match(rel_cha.name) for t in tiers.values() if getattr(t, "partition", False)]
            text_filename = f"{''.join(rel_labels)}_TranscriptionReliabilityAlignment.txt"
            text_file_path = Path(transc_rel_dir, *partition_labels, "GlobalAlignments", text_filename)

            try:
                _ensure_parent_dir(text_file_path)
                alignment_str = _format_alignment_output(nw["alignment"], nw["NeedlemanWunschScore"], nw["NeedlemanWunschNorm"])
                with open(text_file_path, "w", encoding="utf-8") as fh:
                    fh.write(alignment_str)
            except Exception as e:
                logging.error(f"Failed to write alignment file {text_file_path}: {e}")

            # ---------- build record ----------
            row = {
                **{t.name: t.match(rel_cha.name) for t in tiers.values()},  # tier label cols
                "OrgFile": org_cha.name,
                "RelFile": rel_cha.name,
                **simple,
                **lev,
                "NeedlemanWunschScore": nw["NeedlemanWunschScore"],
                "NeedlemanWunschNorm": nw["NeedlemanWunschNorm"],
            }
            records.append(row)

        except Exception as e:
            logging.error(f"Failed to analyze transcription reliability for {org_cha} and {rel_cha}: {e}")

    # --- finalize DataFrame from records ---
    if not records:
        logging.warning("No transcription reliability records produced.")
        return [] if test else None

    transc_rel_df = pd.DataFrame.from_records(records)

    # --- save grouped outputs + reports ---
    results = []
    if partition_tiers:
        groups = transc_rel_df.groupby(partition_tiers, dropna=False)
        for tup, subdf in tqdm(groups, desc="Saving grouped DataFrames & reports"):
            tup_vals = (tup if isinstance(tup, tuple) else (tup,))
            base_name = "_".join(str(x) for x in tup_vals if x is not None)

            df_filename = f"{base_name}_TranscriptionReliabilityAnalysis.xlsx"
            df_path = Path(transc_rel_dir, *[str(x) for x in tup_vals if x is not None], df_filename)

            report_filename = f"{base_name}_TranscriptionReliabilityReport.txt"
            report_path = Path(transc_rel_dir, *[str(x) for x in tup_vals if x is not None], report_filename)

            try:
                _ensure_parent_dir(df_path)
                subdf.to_excel(df_path, index=False)
                logging.info(f"Saved reliability analysis DataFrame to: {df_path}")
            except Exception as e:
                logging.error(f"Failed to write DataFrame to {df_path}: {e}")

            try:
                write_reliability_report(subdf, report_path, tup_vals)
            except Exception as e:
                logging.error(f"Failed to write reliability report to {report_path}: {e}")

            if test:
                results.append(subdf.copy())
    else:
        # No partitions → save one Excel + one report directly under transc_rel_dir
        df_path = transc_rel_dir / "TranscriptionReliabilityAnalysis.xlsx"
        report_path = transc_rel_dir / "TranscriptionReliabilityReport.txt"

        try:
            transc_rel_df.to_excel(df_path, index=False)
            logging.info(f"Saved reliability analysis DataFrame to: {df_path}")
        except Exception as e:
            logging.error(f"Failed to write DataFrame to {df_path}: {e}")

        try:
            write_reliability_report(transc_rel_df, report_path, None)
        except Exception as e:
            logging.error(f"Failed to write reliability report to {report_path}: {e}")

        if test:
            results.append(transc_rel_df.copy())

    return results if test else None
