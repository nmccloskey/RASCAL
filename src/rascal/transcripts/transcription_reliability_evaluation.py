import re
import pylangacq
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Align import PairwiseAligner
from typing import Union, List
from rascal.utils.logger import logger, _rel


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
        logger.error(f"extract_cha_text failed: {e}")
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
        'levenshtein_similarity' whose values lie in [0, 1].
    report_path : str | pathlib.Path
        Full path to the output .txt file.
    partition_labels : list[str] | None
        Optional tier / partition labels to display in the header.
    """

    try:
        # ── sanity checks ──────────────────────────────────────────────────────
        if 'levenshtein_similarity' not in transc_rel_subdf.columns:
            raise KeyError("'levenshtein_similarity' column is missing.")

        ls = transc_rel_subdf['levenshtein_similarity'].astype(float).dropna()
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

        logger.info("Successfully wrote transcription reliability report to %s", _rel(report_path))

    except Exception as e:
        logger.error("Failed to write transcription reliability report to %s: %s", _rel(report_path), e)
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
        "org_num_tokens": org_num_tokens,
        "rel_num_tokens": rel_num_tokens,
        "perc_diff_num_tokens": pdiff_num_tokens,
        "org_num_chars": org_num_chars,
        "rel_num_chars": rel_num_chars,
        "perc_diff_num_chars": pdiff_num_chars,
    }

def _levenshtein_metrics(org_text: str, rel_text: str):
    Ldist = distance(org_text, rel_text)
    max_len = max(len(org_text), len(rel_text)) or 1
    Lscore = 1 - (Ldist / max_len)
    return {"levenshtein_distance": Ldist, "levenshtein_similarity": Lscore}

def _needleman_wunsch_global(org_text: str, rel_text: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignments = aligner.align(org_text, rel_text)
    best = alignments[0]
    best_score = best.score
    norm = best_score / (max(len(org_text), len(rel_text)) or 1)
    return {"needleman_wunsch_score": best_score,
            "needleman_wunsch_norm": norm,
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

def _convert_cha_names(input_dir: str | Path) -> dict[str, list[Path]]:
    """
    Detect a 'reliability' subdirectory under `input_dir` and create renamed copies
    of its .cha files (appending '_reliability' before the extension).
    The renamed files are written to a separate 'reliability/renamed' directory.

    Returns
    -------
    dict[str, list[Path]]
        {
          "renamed": [list of new .cha paths],
          "originals": [list of corresponding original paths]
        }

    Notes
    -----
    - Original filenames (and casing) are preserved except for appending
      '_reliability' before the extension.
    - Non-destructive: original files remain untouched.
    - Intended so the main function can exclude originals and include only the
      renamed copies when collecting .cha files.
    """
    # input_dir = Path(input_dir).expanduser().resolve()
    rel_dir = input_dir / "reliability"

    if not rel_dir.exists():
        logger.info("No 'reliability' subdirectory found; skipping _convert_cha_names.")
        return {"renamed": [], "originals": []}

    renamed_dir = rel_dir / "renamed"
    renamed_dir.mkdir(parents=True, exist_ok=True)

    renamed, originals = [], []
    for cha in rel_dir.rglob("*.cha"):
        try:
            if cha.name.endswith("reliability.cha"):
                continue

            new_name = f"{cha.stem}_reliability.cha"
            new_path = renamed_dir / new_name

            if new_path.exists():
                logger.warning("Renamed file already exists, skipping: %s", _rel(new_path))
                continue

            new_path.write_bytes(cha.read_bytes())
            renamed.append(new_path)
            originals.append(cha)
            logger.info("Created renamed reliability copy: %s → %s", cha.name, new_name)

        except Exception as e:
            logger.error("Failed to process reliability file %s: %s", _rel(cha), e)

    logger.info(
        "Reliability rename complete. %d file(s) copied to '%s'.",
        len(renamed),
        _rel(renamed_dir),
    )
    return {"renamed": renamed, "originals": originals}

def _save_alignment(tiers, rel_cha, rel_labels, transc_rel_dir, nw):
    """Save alignment for manual inspection"""
    partition_labels = [
        t.match(rel_cha.name) for t in tiers.values()
        if getattr(t, "partition", False)
    ]
    alignment_filename = f"{''.join(rel_labels)}_transcription_reliability_alignment.txt"
    alignment_path = Path(transc_rel_dir, *partition_labels, "global_alignments", alignment_filename)

    try:
        _ensure_parent_dir(alignment_path)
        alignment_str = _format_alignment_output(
            nw["alignment"], nw["needleman_wunsch_score"], nw["needleman_wunsch_norm"]
        )
        alignment_path.write_text(alignment_str, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write alignment file {_rel(alignment_path)}: {e}")

def _analyze_reliability_pairs(
    rel_chats: list[Path],
    org_index: dict,
    tiers: dict,
    transc_rel_dir: Path,
    exclude_participants: list[str],
    strip_clan: bool,
    prefer_correction: bool,
    lowercase: bool,
) -> list[dict]:
    """
    Iterate through reliability .cha files, compute similarity metrics,
    save alignment outputs, and return a list of result records.
    """
    records = []
    seen_rel_files, seen_org_files = set(), set()

    for rel_cha in tqdm(rel_chats, desc="Analyzing reliability transcripts"):
        try:
            # match to original
            rel_labels = tuple(t.match(rel_cha.name) for t in tiers.values())
            org_cha = org_index.get(rel_labels)
            if org_cha is None:
                logger.warning(f"No matching original .cha for reliability file: {rel_cha.name}")
                continue

            # skip duplicates
            if rel_cha.name in seen_rel_files or org_cha.name in seen_org_files:
                logger.warning(f"Skipping duplicate pairing: {rel_cha.name}")
                continue

            seen_rel_files.add(rel_cha.name)
            seen_org_files.add(org_cha.name)

            org_chat_data = pylangacq.read_chat(str(org_cha))
            rel_chat_data = pylangacq.read_chat(str(rel_cha))

            # process text
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

            # compute metrics
            simple = _compute_simple_stats(org_text, rel_text)
            lev = _levenshtein_metrics(org_text, rel_text)
            nw = _needleman_wunsch_global(org_text, rel_text)

            # save printed alignment
            _save_alignment(tiers, rel_cha, rel_labels, transc_rel_dir, nw)

            # assemble record
            record = {
                **{t.name: t.match(rel_cha.name) for t in tiers.values()},
                "original_file": org_cha.name,
                "reliability_file": rel_cha.name,
                **simple,
                **lev,
                "needleman_wunsch_score": nw["needleman_wunsch_score"],
                "needleman_wunsch_norm": nw["needleman_wunsch_norm"],
            }
            records.append(record)

        except Exception as e:
            logger.error(f"Failed to analyze {_rel(rel_cha)}: {e}")

    return records


def _save_reliability_outputs(
    transc_rel_df: pd.DataFrame,
    partition_tiers: list[str],
    transc_rel_dir: Path,
    test: bool = False,
):
    """
    Save reliability analysis outputs (Excel and text reports) grouped by partitions.
    """
    results = []

    if partition_tiers:
        groups = transc_rel_df.groupby(partition_tiers, dropna=False)
        for tup, subdf in tqdm(groups, desc="Saving grouped DataFrames & reports"):
            tup_vals = (tup if isinstance(tup, tuple) else (tup,))
            base_name = "_".join(str(x) for x in tup_vals if x is not None)

            df_filename = f"{base_name}_transcription_reliability_evaluation.xlsx"
            df_path = Path(transc_rel_dir, *[str(x) for x in tup_vals if x is not None], df_filename)

            report_filename = f"{base_name}_transcription_reliability_report.txt"
            report_path = Path(transc_rel_dir, *[str(x) for x in tup_vals if x is not None], report_filename)

            try:
                _ensure_parent_dir(df_path)
                subdf.to_excel(df_path, index=False)
                logger.info(f"Saved reliability analysis DataFrame to: {_rel(df_path)}")
            except Exception as e:
                logger.error(f"Failed to write DataFrame to {_rel(df_path)}: {e}")

            try:
                write_reliability_report(subdf, report_path, tup_vals)
            except Exception as e:
                logger.error(f"Failed to write reliability report to {_rel(report_path)}: {e}")

            if test:
                results.append(subdf.copy())
    else:
        df_path = transc_rel_dir / "transcription_reliability_evaluation.xlsx"
        report_path = transc_rel_dir / "transcription_reliability_report.txt"

        try:
            transc_rel_df.to_excel(df_path, index=False)
            logger.info(f"Saved reliability analysis DataFrame to: {_rel(df_path)}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to {_rel(df_path)}: {e}")

        try:
            write_reliability_report(transc_rel_df, report_path)
        except Exception as e:
            logger.error(f"Failed to write reliability report to {_rel(report_path)}: {e}")

        if test:
            results.append(transc_rel_df.copy())

    return results

def evaluate_transcription_reliability(
    tiers,
    input_dir,
    output_dir,
    exclude_participants=None,
    strip_clan=True,
    prefer_correction=True,
    lowercase=True,
    test=False,
):
    """
    Analyze transcription reliability by comparing original and reliability CHAT files.
    """
    exclude_participants = exclude_participants or []

    transc_rel_dir = output_dir / "transcription_reliability_evaluation"
    transc_rel_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {_rel(transc_rel_dir)}")

    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]

    # handle reliability subdir
    converted = _convert_cha_names(input_dir)
    original_reliability_files = converted["originals"]

    # collect files
    cha_files = [
        p for p in Path(input_dir).rglob("*.cha")
        if p not in original_reliability_files
    ]
    logger.info(f"Found {len(cha_files)} .cha files in the input directory.")

    rel_chats = [p for p in cha_files if "reliability" in p.name]
    org_chats = [p for p in cha_files if "reliability" not in p.name]

    def _labels_for(path: Path):
        return tuple(t.match(path.name) for t in tiers.values())

    org_index = {_labels_for(org): org for org in org_chats}

    # analyze pairs
    records = _analyze_reliability_pairs(
        rel_chats,
        org_index,
        tiers,
        transc_rel_dir,
        exclude_participants,
        strip_clan,
        prefer_correction,
        lowercase,
    )

    if not records:
        logger.warning("No transcription reliability records produced.")
        return [] if test else None

    transc_rel_df = pd.DataFrame.from_records(records)

    # save grouped outputs + reports
    results = _save_reliability_outputs(transc_rel_df, partition_tiers, transc_rel_dir, test)

    return results if test else None
