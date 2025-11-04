import re
import random
import itertools
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
from functools import lru_cache

from rascal.utils.logger import logger
from rascal.utils.support_funcs import find_transcript_tables, extract_transcript_data

stim_cols = ["narrative", "scene", "story", "stimulus"]

@lru_cache(maxsize=1)
def get_word_checker():
    import nltk

    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')

    from nltk.corpus import words
    valid_words = set(words.words())
    return lambda word: word in valid_words

def segment(x, n):
    """
    Segment a list x into n batches of roughly equal length.
    
    Parameters:
    - x (list): List to be segmented.
    - n (int): Number of segments to create.
    
    Returns:
    - list of lists: Segmented batches of roughly equal length.
    """
    segments = []
    # seg_len = math.ceil(len(x) / n)
    seg_len = int(round(len(x) / n))
    for i in range(0, len(x), seg_len):
        segments.append(x[i:i + seg_len])
    # Correct for small trailing segment.
    if len(segments) > n:
        last = segments.pop(-1)
        segments[-1] = segments[-1] + last
    return segments

def assign_coders(coders):
    """
    Assign each coder to each role (coder 1, coder 2, coder 3) in different segments.
    
    Parameters:
    - coders (list): List of coder names.
    
    Returns:
    - list of tuples: Each tuple contains an assignment of coders.
    """
    random.shuffle(coders)
    perms = list(itertools.permutations(coders))
    assignments = [perms[0]]
    for p in perms[1:]:
        newp = True
        for ass in assignments:
            if any(np.array(p) == np.array(ass)):
                newp = False
        if newp:
            assignments.append(p)
    random.shuffle(assignments)
    return assignments

def _assign_coding_columns(df, base_cols, cu_paradigms, exclude_participants):
    """Add empty coder columns, NA-prefilled for excluded participants."""
    for col in base_cols:
        df[col] = np.where(df['speaker'].isin(exclude_participants), 'NA', "")

    if len(cu_paradigms) < 2:
        return

    for prefix in ['c1', 'c2']:
        for tag in ['sv', 'rel']:
            base_col = f"{prefix}_{tag}"
            df.drop(columns=[base_col], inplace=True, errors='ignore')
            for paradigm in cu_paradigms:
                new_col = f"{prefix}_{tag}_{paradigm}"
                df[new_col] = np.where(df['speaker'].isin(exclude_participants), 'NA', "")

def _prepare_reliability_subset(cu_df, seg, ass, frac, cu_paradigms):
    """Generate the reliability subset dataframe for a given coder assignment."""
    rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
    relsegdf = cu_df[cu_df['sample_id'].isin(rel_samples)].copy()
    relsegdf.drop(columns=['c1_id', 'c1_comment'], inplace=True, errors='ignore')

    if len(cu_paradigms) >= 2:
        for tag in ['sv', 'rel']:
            for paradigm in cu_paradigms:
                old, new = f'c2_{tag}_{paradigm}', f'c3_{tag}_{paradigm}'
                if old in relsegdf:
                    relsegdf.rename(columns={old: new}, inplace=True)
                relsegdf.drop(columns=[f'c1_{tag}_{paradigm}'], inplace=True, errors='ignore')
        relsegdf.rename(columns={'c2_comment': 'c3_comment'}, inplace=True)
    else:
        renames = {'c2_sv': 'c3_sv', 'c2_rel': 'c3_rel', 'c2_comment': 'c3_comment'}
        relsegdf.rename(columns={k: v for k, v in renames.items() if k in relsegdf}, inplace=True)
        relsegdf.drop(columns=['c1_sv', 'c1_rel'], inplace=True, errors='ignore')
        for col in ['c3_sv', 'c3_rel', 'c3_comment']:
            if col not in relsegdf:
                relsegdf[col] = ""

    relsegdf.drop(columns=['c2_id'], inplace=True, errors='ignore')
    relsegdf.insert(relsegdf.columns.get_loc('c3_comment'), 'c3_id', ass[2])
    return relsegdf

def make_cu_coding_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    cu_paradigms,
    exclude_participants,
):
    """
    Build Complete Utterance (CU) coding and reliability workbooks from
    utterance tables in `input_dir` or `output_dir`.

    Two Excel files are created per input:
      1) *_cu_coding.xlsx – main coding workbook
      2) *_cu_reliability_coding.xlsx – reliability subset

    Behavior:
    - Loads all *Utterances.xlsx files, labels them by tiers.
    - Adds coder ID/comment/value columns (or paradigm variants).
    - Prefills excluded participants with 'NA'.
    - Randomly segments samples across coders and selects reliability subsets.
    - Writes outputs under {output_dir}/cu_coding/<labels>.

    Parameters
    ----------
    tiers : dict[str, Tier]
    frac : float (0–1)
    coders : list[str] (≥3 recommended)
    input_dir, output_dir : Path or str
    cu_paradigms : list[str]
    exclude_participants : list[str]
    """
    if len(coders) < 3:
        logger.warning(f"Only {len(coders)} coders given; using default ['1','2','3'].")
        coders = ['1', '2', '3']

    base_cols = ['c1_id', 'c1_sv', 'c1_rel', 'c1_comment',
                 'c2_id', 'c2_sv', 'c2_rel', 'c2_comment']
    cu_coding_dir = Path(output_dir) / "cu_coding"
    cu_coding_dir.mkdir(parents=True, exist_ok=True)

    transcript_tables = find_transcript_tables(input_dir, output_dir)
    utt_dfs = [extract_transcript_data(tt) for tt in transcript_tables]

    for file, uttdf in tqdm(zip(transcript_tables, utt_dfs), desc="Generating CU coding files"):
        try:
            labels = [t.match(file.name, return_None=True) for t in tiers.values()]
            labels = [l for l in labels if l]
            label_path = Path(cu_coding_dir, *labels)
            label_path.mkdir(parents=True, exist_ok=True)
            lab_str = "_".join(labels) + "_" if labels else ""

            # Shuffle samples
            subdfs = []
            for _, subdf in uttdf.groupby(by="sample_id"): 
              subdfs.append(subdf)
            random.shuffle(subdfs)
            shuffled_utt_df = pd.concat(subdfs, ignore_index=True)
            drop_cols = [ col for col in ['file', 'speaking_time'] \
                         + [t for t in tiers if t.lower() not in stim_cols] if col in shuffled_utt_df.columns ]
            cu_df = shuffled_utt_df.drop(columns=drop_cols).copy()

            _assign_coding_columns(cu_df, base_cols, cu_paradigms, exclude_participants)

            unique_ids = list(cu_df['sample_id'].drop_duplicates())
            segments = segment(unique_ids, n=len(coders))
            assignments = assign_coders(coders)
            rel_subsets = []

            for seg, ass in zip(segments, assignments):
                cu_df.loc[cu_df['sample_id'].isin(seg), ['c1_id', 'c2_id']] = ass[:2]
                rel_subsets.append(_prepare_reliability_subset(cu_df, seg, ass, frac, cu_paradigms))

            reldf = pd.concat(rel_subsets)
            logger.info(f"{file.name}: reliability={len(set(reldf['sample_id']))} / total={len(unique_ids)}")

            # Write outputs
            cu_df.to_excel(label_path / f"{lab_str}cu_coding.xlsx", index=False)
            reldf.to_excel(label_path / f"{lab_str}cu_reliability_coding.xlsx", index=False)

        except Exception as e:
            logger.error(f"Failed processing {file}: {e}")


def count_words(text, d):
    """
    Prepares a transcription text string for counting words.
    
    Parameters:
        text (str): Input transcription text.
        d (function): A function or callable to check if a word exists in the dictionary.
        
    Returns:
        int: Count of valid words.
    """
    # Normalize text
    text = text.lower().strip()
    
    # Handle specific contractions and patterns
    text = re.sub(r"(?<=(he|it))'s got", ' has got', text)
    text = ' '.join([contractions.fix(w) for w in text.split()])
    text = text.replace(u'\xa0', '')
    text = re.sub(r'(^|\b)(u|e)+?(h|m|r)+?(\b|$)', '', text)
    text = re.sub(r'(^|\b|\b.)x+(\b|$)', '', text)
    
    # Remove annotations and special markers
    text = re.sub(r'\[.+?\]', '', text)
    text = re.sub(r'\*.+?\*', '', text)
    
    # Convert numbers to words
    text = re.sub(r'\d+', lambda x: n2w.num2words(int(x.group(0))), text)
    
    # Remove non-word characters and clean up spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bcl\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    # Tokenize and validate words
    tokens = [word for word in text.split() if d(word)]
    return len(tokens)

def _read_cu_file(file: Path) -> pd.DataFrame:
    """Read and shuffle CU coding file by sample_id."""
    cu_df = pd.read_excel(file)
    subdfs = [g for _, g in cu_df.groupby("sample_id")]
    random.shuffle(subdfs)
    shuffled = pd.concat(subdfs, ignore_index=True)
    logger.info(f"Read and shuffled {file.name}")
    return shuffled

def _prepare_wc_df(df: pd.DataFrame, d) -> pd.DataFrame:
    """Add coder and word_count columns; drop CU-specific ones."""
    df = df.copy()
    df["c1_id"] = ""
    df["wc_comment"] = ""
    df["word_count"] = df.apply(
        lambda r: count_words(r["utterance"], d) if not pd.isna(r.get("c2_cu")) else "NA",
        axis=1
    )
    drop_cols = [c for c in df if c.startswith(("c2_sv", "c2_rel", "c2_cu", "c2_comment", "agreement"))]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df

def _assign_wc_coders(df: pd.DataFrame, coders: list[str], frac: float):
    """Assign coders and build reliability subset."""
    assignments = assign_coders(coders)
    unique_ids = list(df["sample_id"].drop_duplicates())
    segments = segment(unique_ids, n=len(coders))

    rel_subsets = []
    for seg, ass in zip(segments, assignments):
        df.loc[df["sample_id"].isin(seg), "c1_id"] = ass[0]
        rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
        relsegdf = df[df["sample_id"].isin(rel_samples)].copy()
        relsegdf.rename(columns={"c1_id": "c2_id", "wc_comment": "wc_rel_com"}, inplace=True)
        relsegdf["c2_id"] = ass[1]
        rel_subsets.append(relsegdf)

    wc_rel_df = pd.concat(rel_subsets)
    logger.info(f"Reliability subset: {len(wc_rel_df)} utterances")
    return df, wc_rel_df

def _write_wc_outputs(wc_df, wc_rel_df, word_count_dir, labels):
    """Write word count and reliability files to disk."""
    lab_str = "_".join(labels) + "_" if labels else ""
    out_dir = Path(word_count_dir, *labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        f"{lab_str}word_counting.xlsx": wc_df,
        f"{lab_str}word_counting_reliability.xlsx": wc_rel_df,
    }
    for fname, df in files.items():
        fpath = out_dir / fname
        try:
            df.to_excel(fpath, index=False)
            logger.info(f"Wrote {fpath}")
        except Exception as e:
            logger.error(f"Failed to write {fpath}: {e}")

def make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    """
    Create word-count coding and reliability workbooks from CU utterance tables.

    Each input *cu_coding_by_utterance*.xlsx produces:
      - *_word_counting.xlsx* → all samples
      - *_word_counting_reliability.xlsx* → subset (~frac) for reliability

    Steps:
      1. Locate CU coding files in `input_dir` and `output_dir`.
      2. Read each file, shuffle samples, drop CU-specific columns.
      3. Compute `word_count` per utterance using `count_words`.
      4. Assign coders and sample reliability subsets.
      5. Write outputs under `{output_dir}/word_counts/<labels>/`.

    Parameters
    ----------
    tiers : dict[str, Tier]
    frac : float
    coders : list[str]
    input_dir, output_dir : Path or str
    """
    word_count_dir = Path(output_dir) / "word_counts"
    word_count_dir.mkdir(parents=True, exist_ok=True)
    d = get_word_checker()
    cu_files = list(Path(input_dir).rglob("*cu_coding_by_utterance*.xlsx")) + \
               list(Path(output_dir).rglob("*cu_coding_by_utterance*.xlsx"))

    for file in tqdm(cu_files, desc="Generating word count files"):
        try:
            wc_df = _read_cu_file(file)
            labels = [t.match(file.name, return_None=True) for t in tiers.values() if t.match(file.name, return_None=True)]
            wc_df = _prepare_wc_df(wc_df, d)
            wc_df, wc_rel_df = _assign_wc_coders(wc_df, coders, frac)
            _write_wc_outputs(wc_df, wc_rel_df, word_count_dir, labels)
        except Exception as e:
            logger.error(f"Failed processing {file}: {e}")


def reselect_cu_wc_reliability(
    tiers,
    input_dir,
    output_dir,
    rel_type: str = "CU",
    frac: float = 0.2,
    rng_seed: int = 88,
):
    """
    Reselect reliability samples for either Conversation Units (CU) or Word Counting (WC),
    excluding any sample_id already present in existing reliability files that match the
    same tier labels.

    Operation
    ---------
    - Recursively finds original coder-2 tables and their paired reliability tables under
      `input_dir`. Matching is by tier labels derived from filenames via `tiers`
      (tries `t.match(name)` per tier); if `tiers` is empty, falls back to a simple
      stem-based label.
    - Unions all `sample_id`s already present across matched reliability files, subtracts
      them from the original table’s unique `sample_id`s, randomly selects a fraction
      of the total (`max(1, round(len(all_ids)*frac))`), and writes a new reliability
      workbook containing just the newly selected rows.
    - The new sheet’s “post-comment” columns are templated from the matched reliability
      files. If multiple reliability files disagree on columns after 'comment', the
      intersection is used (warning logged). All missing template columns are added as NaN.

    Parameters
    ----------
    tiers : dict[str, Tier]
        Tier objects used to derive filename labels for matching originals to reliability.
        Only `t.match(name)` is required here.
    input_dir, output_dir : str | os.PathLike
        Root directory to search; base output directory. Files are written to
        `<output_dir>/reselected_<rel_type>_reliability/`.
    rel_type : {"CU","WC"}, default "CU"
        Determines filename patterns:
          - CU: finds "*cu_coding.xlsx" with paired "*cu_reliability_coding.xlsx"
          - WC: finds "*word_counting.xlsx" with "*word_counting_reliability.xlsx"
    frac : float in (0,1], default 0.2
        Target fraction of *all* unique `sample_id`s to select (before excluding used).
        If available unused samples are fewer than target, all available are used.
    rng_seed : int, default 88
        Seed for reproducible selection.

    Returns
    -------
    None
        Writes one file per original coder-2 table:
        `<base>_reselected_{cu_reliability_coding|word_counting_reliability}.xlsx`.

    Notes
    -----
    - Requires a `sample_id` column in both original and reliability tables.
    - Skips an original file if no matched reliability file exists (schema safety).
    - CU branch ensures presence of `c3_id` and `c3_comment` columns (left as NaN unless
      already present). Suffixed c3 fields may be wiped by downstream code if needed.
    """

    rel_type = (rel_type or "CU").upper().strip()
    if rel_type not in {"CU", "WC"}:
        logger.error(f"Invalid rel_type '{rel_type}'. Must be 'CU' or 'WC'.")
        return

    random.seed(rng_seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / f"reselected_{rel_type}_reliability"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {out_dir}")
    except Exception as e:
        logger.error(f"Failed to create directory {out_dir}: {e}")
        return

    # --- discover original coder-2 files and their reliability mates ---
    if rel_type == "CU":
        coding_glob = "*cu_coding.xlsx"
        rel_glob = "*cu_reliability_coding.xlsx"
        out_rel_name = "cu_reliability_coding"
    else:  # "WC"
        coding_glob = "*word_counting.xlsx"
        rel_glob = "*word_counting_reliability.xlsx"
        out_rel_name = "word_counting_reliability"
        d = get_word_checker()

    coding_files = list(input_dir.rglob(coding_glob))
    rel_files = list(input_dir.rglob(rel_glob))

    if not coding_files:
        logger.warning(f"No original {rel_type} files found under {input_dir} (pattern: {coding_glob}).")
        return

    # --- helpers ---
    def _label_one(tier_obj, fname: str):
        try:
            if hasattr(tier_obj, "match"):
                return tier_obj.match(fname)
        except Exception:
            pass
        # Fallback: no label for this tier
        return None

    def _labels_for(path: Path):
        if not tiers:
            # If tiers are not provided, fallback to just the stem so pairs can still match
            return (path.stem,)
        labels = []
        for t in tiers.values():
            try:
                labels.append(_label_one(t, path.name))
            except Exception:
                labels.append(None)
        return tuple(labels)

    # Build map from original -> list of matching reliability files (by tier labels)
    org_rel_matches: dict[Path, list[Path]] = {}
    rel_labels_cache = {p: _labels_for(p) for p in rel_files}

    for org in coding_files:
        org_labels = _labels_for(org)
        matches = [p for p, labs in rel_labels_cache.items() if labs == org_labels]
        if not matches:
            logger.warning(f"[{rel_type}] No reliability files found for {org.name}. Skipping.")
        org_rel_matches[org] = matches

    # --- process each original coding file ---
    desc = f"Reselecting {rel_type} reliability samples"
    for org_file in tqdm(coding_files, desc=desc):
        rel_mates = org_rel_matches.get(org_file, [])
        if not rel_mates:
            continue  # already warned

        try:
            df_org = pd.read_excel(org_file)
        except Exception as e:
            logger.error(f"Failed reading original coding file {org_file}: {e}")
            continue

        # Collect reliability DataFrames & validate 'sample_id'
        rel_dfs = []
        for rf in rel_mates:
            try:
                rel_df = pd.read_excel(rf)
                rel_dfs.append(rel_df)
            except Exception as e:
                logger.warning(f"Failed reading reliability file {rf}: {e}")

        if not rel_dfs:
            logger.warning(f"[{rel_type}] All matched reliability files failed to read for {org_file.name}. Skipping.")
            continue

        # Confirm 'sample_id' columns exist
        if "sample_id" not in df_org.columns:
            logger.warning(f"[{rel_type}] 'sample_id' missing in {org_file.name}. Skipping.")
            continue
        missing_sid = [rf for rf, rdf in zip(rel_mates, rel_dfs) if "sample_id" not in rdf.columns]
        if missing_sid:
            logger.warning(f"[{rel_type}] 'sample_id' missing in reliability file(s): {[p.name for p in missing_sid]}. Skipping {org_file.name}.")
            continue

        # Compute used sample_ids across all matched reliability files
        used_sample_ids: set = set()
        for rdf in rel_dfs:
            used_sample_ids.update(set(rdf["sample_id"].dropna().astype(str).unique()))

        all_sample_ids = set(df_org["sample_id"].dropna().astype(str).unique())
        available_ids = list(all_sample_ids - used_sample_ids)

        if len(available_ids) == 0:
            logger.warning(f"[{rel_type}] No available samples to reselect for {org_file.name}. Skipping.")
            continue

        num_to_select = max(1, round(len(all_sample_ids) * float(frac)))
        if len(available_ids) < num_to_select:
            logger.warning(
                f"[{rel_type}] Not enough unused samples in {org_file.name}. "
                f"Selecting {len(available_ids)} instead of target {num_to_select}."
            )
            num_to_select = len(available_ids)

        # Sample deterministically (within run) from available ids
        random.seed(rng_seed)  # reset per file to make selection reproducible if code reruns
        reselected_ids = set(random.sample(available_ids, k=num_to_select))

        # --- Build new reliability frame ---
        # Determine the template (columns) from reliability files.
        # Use columns up to 'comment' + any columns after; if multiple have different
        # post-'comment' sets, take intersection to reduce mismatch risk.
        def _cols_to_comment(df):
            if "comment" in df.columns:
                idx = df.columns.get_loc("comment")
                return list(df.columns[: idx + 1])
            return list(df.columns)

        # "Head" columns (up through 'comment') will come from original coding table, not from reliability.
        head_cols = _cols_to_comment(df_org)

        # Post-comment columns from reliability (template)
        post_sets = []
        if "comment" in rel_dfs[0].columns:
            start = rdf.columns.get_loc("comment") + 1
            post_cols = rdf.columns[start:]
        else:
            logger.error(f"No 'comment' column found in {rel_mates[0]}.")
            post_sets.append(rdf.columns)

        # Subset original coding rows for the chosen sample_ids
        sub = df_org[df_org["sample_id"].astype(str).isin(reselected_ids)].copy()
        if sub.empty:
            logger.warning(f"[{rel_type}] Resolved 0 rows after filtering by reselected sample_ids for {org_file.name}. Skipping.")
            continue

        # Keep up through 'comment' if present
        if "comment" in sub.columns:
            end_idx = sub.columns.get_loc("comment")
            sub = sub.iloc[:, : end_idx + 1]
        # else: keep all (we'll align columns later)

        # Add post-'comment' reliability columns (NaN by default)
        for col in post_cols:
            if col not in sub.columns:
                sub[col] = ""

        # CU-specific adjustments
        if rel_type == "CU":
            for col in ["c3_id","c3_comment"]:
                if col not in sub.columns:
                    sub[col] = ""

        else:  # "WC"
            # Ensure c3_id exists if present in template; set it to coder3
            if "c2_id" not in sub.columns:
                sub["c2_id"] = ""
            # Add word count column and pull neutrality from original.
            org_sub = df_org[df_org['sample_id'].isin(reselected_ids)].copy()
            sub['word_count'] = org_sub.apply(lambda row: count_words(row['utterance'], d) if not np.isnan(row['word_count']) else 'NA', axis=1)
            # If a 'c3_comment' exists in the template, wipe it
            if "wc_rel_com" not in sub.columns:
                sub["wc_rel_com"] = ""

        # Order columns: head-cols first (as they exist), then post-cols in template order
        ordered_cols = [c for c in head_cols if c in sub.columns] + [c for c in post_cols if c in sub.columns and c not in head_cols]
        sub = sub.loc[:, ordered_cols]

        # --- Write output file ---
        stem = org_file.stem
        if rel_type == "CU" and stem.endswith("cu_coding"):
            base = stem[: -len("cu_coding")].rstrip("_")
        elif rel_type == "WC" and stem.endswith("word_counting"):
            base = stem[: -len("word_counting")].rstrip("_")
        else:
            base = stem

        out_path = out_dir / f"{base}_reselected_{out_rel_name}.xlsx".lstrip("_")
        try:
            sub.to_excel(out_path, index=False)
            logger.info(f"[{rel_type}] Saved reselected reliability file: {out_path}")
        except Exception as e:
            logger.error(f"[{rel_type}] Failed writing reselected file {out_path}: {e}")
