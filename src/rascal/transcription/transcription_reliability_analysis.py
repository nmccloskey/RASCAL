import os
import re
import pylangacq
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Align import PairwiseAligner
import logging


def percent_difference(a, b):
    try:
        a, b = float(a), float(b)
        if a == 0 and b == 0:
            return 0.0
        denom = (abs(a) + abs(b)) / 2.0
        return (abs(a - b) / denom) * 100.0 if denom != 0 else 0.0
    except Exception:
        return float("nan")

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
    exclude_participants=("INV",),
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
    transc_rel_dir = os.path.join(output_dir, "TranscriptionReliabilityAnalysis")
    os.makedirs(transc_rel_dir, exist_ok=True)
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
    for rel_cha in tqdm(rel_chats, desc="Analyzing reliability transcripts"):
        rel_labels = _labels_for(rel_cha)
        org_cha = org_index.get(rel_labels)
        if org_cha is None:
            logging.warning(f"No matching original .cha for reliability file: {rel_cha.name}")
            continue

        try:
            org_chat_data = pylangacq.read_chat(str(org_cha))
            rel_chat_data = pylangacq.read_chat(str(rel_cha))
        except Exception as e:
            logging.error(f"Failed to read CHAT files {org_cha} or {rel_cha}: {e}")
            continue

        try:
            org_text = extract_cha_text(
                org_chat_data,
                exclude_participants=exclude_participants,
                strip_clan=strip_clan,
                prefer_correction=prefer_correction,
                lowercase=lowercase,
            )
            rel_text = extract_cha_text(
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
            text_file_path = os.path.join(transc_rel_dir, *partition_labels, "GlobalAlignments", text_filename)

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
                # Keep both raw and normalized NW scores; distance often undefined so None
                "NeedlemanWunschDistance": nw["NeedlemanWunschDistance"],
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
    else:
        # Single “no-partition” group
        transc_rel_df["_NO_PARTITION_"] = "ALL"
        groups = transc_rel_df.groupby(["_NO_PARTITION_"])

    for tup, subdf in tqdm(groups, desc="Saving grouped DataFrames & reports"):
        tup_vals = (tup if isinstance(tup, tuple) else (tup,))
        base_name = "_".join(str(x) for x in tup_vals if x is not None and x != "_NO_PARTITION_") or "ALL"

        df_filename = f"{base_name}_TranscriptionReliabilityAnalysis.xlsx"
        df_path = os.path.join(transc_rel_dir, *[str(x) for x in tup_vals if x not in (None, "_NO_PARTITION_")], df_filename)

        report_filename = f"{base_name}_TranscriptionReliabilityReport.txt"
        report_path = os.path.join(transc_rel_dir, *[str(x) for x in tup_vals if x not in (None, "_NO_PARTITION_")], report_filename)

        try:
            _ensure_parent_dir(df_path)
            subdf.to_excel(df_path, index=False)
            logging.info(f"Saved reliability analysis DataFrame to: {df_path}")
        except Exception as e:
            logging.error(f"Failed to write DataFrame to {df_path}: {e}")

        try:
            write_reliability_report(subdf, report_path, tup_vals if partition_tiers else None)
        except Exception as e:
            logging.error(f"Failed to write reliability report to {report_path}: {e}")

        if test:
            results.append(subdf.copy())

    # Clean helper column if we created it
    if "_NO_PARTITION_" in transc_rel_df.columns:
        transc_rel_df.drop(columns=["_NO_PARTITION_"], inplace=True, errors="ignore")

    return results if test else None
