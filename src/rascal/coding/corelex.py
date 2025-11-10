import re
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
from datetime import datetime
from scipy.stats import percentileofscore

from rascal.utils.logger import logger, _rel
from rascal.coding.coding_files import stim_cols
from rascal.utils.auxiliary import find_files, extract_transcript_data


urls = {
    "BrokenWindow": {
        "accuracy": "https://docs.google.com/spreadsheets/d/12SAkAG8VCAkhCFv4ceJiqgRZ7U9-P9bEcet--hDeW2s/export?format=csv&gid=1059193656",
        "efficiency": "https://docs.google.com/spreadsheets/d/12SAkAG8VCAkhCFv4ceJiqgRZ7U9-P9bEcet--hDeW2s/export?format=csv&gid=1542250565"
    },
    "RefusedUmbrella": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1oYiwnUdO0dOsFVTmdZBCxkAQc5Ui-71GhUSchK_YY44/export?format=csv&gid=1670315041",
        "efficiency": "https://docs.google.com/spreadsheets/d/1oYiwnUdO0dOsFVTmdZBCxkAQc5Ui-71GhUSchK_YY44/export?format=csv&gid=1362214973"
    },
    "CatRescue": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1sTvSX0Ws0kPTw-5HHyY8JO2CubqWVgEzDvE5BuGSefc/export?format=csv&gid=1916867784",
        "efficiency": "https://docs.google.com/spreadsheets/d/1sTvSX0Ws0kPTw-5HHyY8JO2CubqWVgEzDvE5BuGSefc/export?format=csv&gid=1346760459"
    },
    "Cinderella": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1fpDq7aTrKVkfjdv8ka7BS5_iHEJ8HHI-q9nJI6wDAEA/export?format=csv&gid=280451139",
        "efficiency": "https://docs.google.com/spreadsheets/d/1fpDq7aTrKVkfjdv8ka7BS5_iHEJ8HHI-q9nJI6wDAEA/export?format=csv&gid=285651009"
    },
    "Sandwich": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1o29bBQbyNlmtL05kkTuLV6z5auz1msDeLSxIO1p_3EA/export?format=csv&gid=342443913",
        "efficiency": "https://docs.google.com/spreadsheets/d/1o29bBQbyNlmtL05kkTuLV6z5auz1msDeLSxIO1p_3EA/export?format=csv&gid=2140143611"
    }
}


# Define tokens for each scene
scene_tokens = {
    'BrokenWindow': [
        "a", "and", "ball", "be", "boy", "break", "go", "he", "in", "it", 
        "kick", "lamp", "look", "of", "out", "over", "play", "sit", "soccer", 
        "the", "through", "to", "up", "window"
    ],
    'CatRescue': [
        "a", "and", "bark", "be", "call", "cat", "climb", "come",
        "department", "dog", "down", "father", "fire", "fireman", "get",
        "girl", "go", "have", "he", "in", "ladder", "little", "not", "out",
        "she", "so", "stick", "the", "their", "there", "to", "tree", "up", "with"
    ],
    'RefusedUmbrella': [
        "a", "and", "back", "be", "boy", "do", "get", "go", "have", "he", "home",
        "i", "in", "it", "little", "mother", "need", "not", "out", "rain",
        "say", "school", "she", "so", "start", "take", "that", "the", "then",
        "to", "umbrella", "walk", "wet", "with", "you"
    ],
    'Cinderella': [
        "a", "after", "all", "and", "as", "at", "away", "back", "ball", "be",
        "beautiful", "because", "but", "by", "cinderella", "clock", "come", "could",
        "dance", "daughter", "do", "dress", "ever", "fairy", "father", "find", "fit",
        "foot", "for", "get", "girl", "glass", "go", "godmother", "happy", "have",
        "he", "home", "horse", "house", "i", "in", "into", "it", "know", "leave",
        "like", "little", "live", "look", "lose", "make", "marry", "midnight",
        "mother", "mouse", "not", "of", "off", "on", "one", "out", "prince",
        "pumpkin", "run", "say", "'s", "she", "shoe", "sister", "slipper", "so", "strike",
        "take", "tell", "that", "the", "then", "there", "they", "this", "time",
        "to", "try", "turn", "two", "up", "very", "want", "well", "when", "who",
        "will", "with"
    ],
    'Sandwich': [
        "a", "and", "bread", "butter", "get", "it", "jelly", "knife", "of", "on",
        "one", "other", "out", "peanut", "piece", "put", "slice", "spread", "take",
        "the", "then", "to", "together", "two", "you"
    ]
}


lemma_dict = {
    # Pronouns and reflexives
    "its": "it", "itself": "it",
    "your": "you", "yours": "you", "yourself": "you",
    "him": "he", "himself": "he", "his": "he",
    "her": "she", "herself": "she",
    "them": "they", "themselves": "they", "their": "they", "theirs": "they",
    "me": "i", "my": "i", "mine": "i", "myself": "i",

    # Forms of "be"
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be",
    "being": "be", "been": "be", "bein": "be",

    # Parental variations
    "daddy": "father", "dad": "father", "papa": "father", "pa": "father",
    "mommy": "mother", "mom": "mother", "mama": "mother", "ma": "mother",

    # "-in" participles (casual speech)
    "breakin": "break", "goin": "go", "kickin": "kick", "lookin": "look",
    "playin": "play", "barkin": "bark", "callin": "call", "climbin": "climb",
    "comin": "come", "gettin": "get", "havin": "have", "stickin": "stick",
    "doin": "do", "needin": "need", "rainin": "rain", "sayin": "say",
    "startin": "start", "takin": "take", "walkin": "walk",

    # Additional verb forms and common variants
    "goes": "go", "gone": "go", "went": "go", "going": "go",
    "gets": "get", "got": "get", "getting": "get",
    "says": "say", "said": "say", "saying": "say",
    "takes": "take", "took": "take", "taking": "take",
    "looks": "look", "looked": "look", "looking": "look",
    "starts": "start", "started": "start", "starting": "start",
    "plays": "play", "played": "play", "playing": "play",

    # Noun variants
    "boys": "boy", "girls": "girl", "shoes": "shoe", "sisters": "sister",
    "trees": "tree", "windows": "window", "cats": "cat", "dogs": "dog",
    "pieces": "piece", "slices": "slice", "sandwiches": "sandwich",
    "fires": "fire", "ladders": "ladder", "balls": "ball",

    # Misc fix-ups
    "wanna": "want", "gonna": "go", "gotta": "get",
    "yall": "you", "aint": "not", "cannot": "could",

    # Additional verb forms
    "wants": "want", "wanted": "want", "wanting": "want",
    "finds": "find", "found": "find", "finding": "find",
    "makes": "make", "made": "make", "making": "make",
    "tries": "try", "tried": "try", "trying": "try",
    "tells": "tell", "told": "tell", "telling": "tell",
    "runs": "run", "ran": "run", "running": "run",
    "sits": "sit", "sat": "sit", "sitting": "sit",
    "knows": "know", "knew": "know", "knowing": "know",
    "walks": "walk", "walked": "walk", "walking": "walk",
    "leaves": "leave", "left": "leave", "leaving": "leave",
    "comes": "come", "came": "come", "coming": "come",
    "calls": "call", "called": "call", "calling": "call",
    "climbs": "climb", "climbed": "climb", "climbing": "climb",
    "breaks": "break", "broke": "break", "breaking": "break",
    "starts": "start", "started": "start", "starting": "start",
    "turns": "turn", "turned": "turn", "turning": "turn",
    "puts": "put", "putting": "put",  # 'put' is same for present/past

    # Copula contractions (useful if splitting fails elsewhere)
    "'m": "be", "'re": "be",  # context-dependent, but may help

    # More noun plurals
    "slippers": "slipper", "daughters": "daughter", "sons": "son",
    "knives": "knife", "pieces": "piece", "sticks": "stick",

    # Pronoun common errors
    "themself": "they", "our": "we", "ours": "we", "ourselves": "we", "we're": "we",

    # Additional contractions and speech forms
    "didnt": "did", "couldnt": "could", "wouldnt": "would", "shouldnt": "should",
    "wasnt": "was", "werent": "were", "isnt": "is", "aint": "not", "havent": "have",
    "hasnt": "have", "hadnt": "have", "dont": "do", "doesnt": "do", "didnt": "do",
    "did": "do", "does": "do", "doing": "do",

    # Articles
    "da": "the", "an": "a",

    # Spoken reductions
    "lemme": "let", "gimme": "give", "cmon": "come", "outta": "out",
    "inna": "in", "coulda": "could", "shoulda": "should", "woulda": "would",
}


base_columns = [
    "sample_id", "narrative", "speaking_time", "num_tokens",
    "num_core_words", "num_core_word_tokens", "lexicon_coverage", "core_words_per_min",
    "core_words_pwa_percentile", "core_words_control_percentile",
    "cwpm_pwa_percentile", "cwpm_control_percentile"
]

_UNINTELLIGIBLE = {"xxx", "yyy", "www"}  # common CHAT placeholders


def reformat(text: str) -> str:
    """
    Prepares a transcription text string for CoreLex analysis.

    - Expands contractions (keeps possessive 's / ’s).
    - Converts digits to words.
    - Preserves replacements like '[: dogs] [*]' → 'dogs'.
    - Removes other CHAT/CLAN annotations (repetitions, comments, gestures, events).
    - Removes tokens that START with punctuation (e.g., &=draws:a:cat), except standalone "'s"/"’s".
    """
    try:
        text = text.lower().strip()

        # 1) Handle specific pattern: "(he|it)'s got" → "he has got" / "it has got"
        text = re.sub(r"\b(he|it)'s got\b", r"\1 has got", text)

        # 2) Expand contractions while keeping possessive 's approximately
        tokens = text.split()
        expanded = []
        for tok in tokens:
            # If looks like possessive 's or ’s, keep as-is (don't expand to "is")
            if re.fullmatch(r"\w+'s", tok) or re.fullmatch(r"\w+’s", tok):
                expanded.append(tok)
            else:
                expanded.append(contractions.fix(tok))
        text = " ".join(expanded)

        # 3) Convert standalone digits to words
        text = re.sub(r"\b\d+\b", lambda m: n2w.num2words(int(m.group())), text)

        # 4) Preserve accepted clinician replacement: "[: dogs] [*]" → "dogs"
        text = re.sub(r'\[:\s*([^\]]+?)\s*\]\s*\[\*\]', r'\1', text)

        # 5) Remove ALL other square-bracketed content (e.g., [//], [?], [% ...], [& ...])
        text = re.sub(r'\[[^\]]+\]', ' ', text)

        # 6) Remove other common CLAN containers: <...> events, ((...)) comments, {...} paralinguistic
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\(\([^)]*\)\)', ' ', text)
        text = re.sub(r'\{[^}]+\}', ' ', text)

        # 7) Remove tokens that START with punctuation (gesture/dep. tiers), e.g. &=draws:a:cat, +/., =laughs
        #    Keep the standalone possessive token "'s"/"’s" if it appears.
        #    (?<!\S)  -> start of token (preceded by start or whitespace)
        #    (?!'s\b)(?!’s\b)  -> DO NOT match if the token is exactly 's/’s
        #    [^\w\s']\S*  -> a non-word, non-space, non-apostrophe first char, then the rest of the token
        text = re.sub(r"(?<!\S)(?!'s\b)(?!’s\b)[^\w\s']\S*", ' ', text)

        # 8) Remove non-word characters except apostrophes (keeps possessives like cinderella’s)
        text = re.sub(r"[^\w\s']", ' ', text)

        # 9) Token-level cleanup: drop CHAT placeholders like 'xxx', 'yyy', 'www'
        toks = [t for t in text.split() if t not in _UNINTELLIGIBLE]

        # 10) Collapse whitespace and return
        return " ".join(toks).strip()

    except Exception as e:
        logger.error(f"An error occurred while reformatting: {e}")
        return ""


def id_core_words(scene_name: str, reformatted_text: str) -> dict:
    """
    Identifies and quantifies core words in a narrative sample.

    Args:
        scene_name (str): The narrative scene name.
        reformatted_text (str): Preprocessed transcript text.

    Returns:
        dict: {
            "num_tokens": int,
            "num_core_words": int,
            "num_cw_tokens": int,
            "lexicon_coverage": float,
            "token_sets": dict[str, set[str]]
        }
    """
    tokens = reformatted_text.split()
    token_sets = {}
    num_cw_tokens = 0

    for token in tokens:
        lemma = lemma_dict.get(token, token)

        if lemma in scene_tokens.get(scene_name, []):
            num_cw_tokens += 1
            if lemma in token_sets:
                token_sets[lemma].add(token)
            else:
                token_sets[lemma] = {token}

    if scene_name.lower() == "cinderella" and "'s" in tokens:
        token_sets["'s"] = {"'s"}
        num_cw_tokens += 1

    num_tokens = len(tokens)
    num_core_words = len(token_sets)
    total_lexicon_size = len(scene_tokens.get(scene_name, []))
    lexicon_coverage = num_core_words / total_lexicon_size if total_lexicon_size > 0 else 0.0

    return {
        "num_tokens": num_tokens,
        "num_core_words": num_core_words,
        "num_cw_tokens": num_cw_tokens,
        "lexicon_coverage": lexicon_coverage,
        "token_sets": token_sets
    }


def load_corelex_norms_online(stimulus_name: str, metric: str = "accuracy") -> pd.DataFrame:
    try:
        url = urls[stimulus_name][metric]
        return pd.read_csv(url)
    except KeyError:
        raise ValueError(f"Unknown stimulus '{stimulus_name}' or metric '{metric}'")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from URL: {e}")


def preload_corelex_norms(present_narratives: set) -> dict:
    """
    Preloads accuracy and efficiency CoreLex norms for all narratives in current batch of samples.

    Args:
        present_narratives (set): Set of narratives present in the input batch.

    Returns:
        dict: Dictionary of dictionaries {scene_name: {accuracy: df, efficiency: df}}
    """
    norm_data = {}

    for scene in present_narratives:
        try:
            norm_data[scene] = {
                "accuracy": load_corelex_norms_online(scene, "accuracy"),
                "efficiency": load_corelex_norms_online(scene, "efficiency")
            }
            logger.info(f"Loaded CoreLex norms for: {scene}")
        except Exception as e:
            logger.warning(f"Failed to load norms for {scene}: {e}")
            norm_data[scene] = {"accuracy": None, "efficiency": None}

    return norm_data


def get_percentiles(score: float, norm_df: pd.DataFrame, column: str) -> dict:
    """
    Computes percentile rank of a score relative to both control and PWA distributions.

    Args:
        score (float): The participant's score.
        norm_df (pd.DataFrame): DataFrame with 'Aphasia' and score column.
        column (str): Name of the column containing scores (e.g., 'CoreLex Score', 'CoreLex/min').

    Returns:
        dict: {
            "control_percentile": float,
            "pwa_percentile": float
        }
    """
    control_scores = norm_df[norm_df['Aphasia'] == 0][column]
    pwa_scores = norm_df[norm_df['Aphasia'] == 1][column]

    return {
        "control_percentile": percentileofscore(control_scores, score, kind="weak"),
        "pwa_percentile": percentileofscore(pwa_scores, score, kind="weak")
    }


def _read_excel_safely(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        logger.warning(f"Failed reading {_rel(path)}: {e}")
        return None


def find_corelex_inputs(input_dir: str, output_dir: str) -> dict | None:
    """
    Locate valid CoreLex input files in priority order and load them.

    Search order:
      1. *unblind_utterance_data*.xlsx — preferred single summary file.
      2. *transcript_tables*.xlsx — fallback (may be multiple; concatenated).
      Optionally merges *speaking_times*.xlsx when available.

    Parameters
    ----------
    input_dir, output_dir : str | Path
        Directories to search for candidate files.

    Returns
    -------
    tuple[str, pd.DataFrame]
        ("unblind", utt_df) or ("transcripts", utt_df)
        Returns None if no usable file is found.
    """
    search_dirs = [Path(output_dir), Path(input_dir)]

    # 1) Preferred: unblind summary
    unblind_matches = []
    for d in search_dirs:
        unblind_matches += list(d.rglob("*unblind_utterance_data*.xlsx"))
    if unblind_matches:
        p = unblind_matches[0]
        df = _read_excel_safely(p)
        if df is not None:
            logger.info(f"Using unblind utterance data: {_rel(p)}")
            return "unblind", df

    # 2) Fallback: transcript tables
    transcript_tables = find_files(directories=[input_dir, output_dir],
                                                search_base="transcript_tables")
    if not transcript_tables:
        logger.error(
            "No CoreLex input files found (*unblind_utterance_data*.xlsx "
            "or *transcript_tables*.xlsx)."
        )
        return None, None

    utt_frames = [extract_transcript_data(tt) for tt in transcript_tables if tt]
    if not utt_frames:
        logger.error("Transcript tables found but none could be loaded.")
        return None, None

    utt_df = pd.concat(utt_frames, ignore_index=True, sort=False)
    logger.info(f"Loaded and concatenated {len(transcript_tables)} transcript table(s).")
    return "transcripts", utt_df


def generate_token_columns(present_narratives):
    """
    Build column names for surface-form lists per narrative × lemma.

    Parameters
    ----------
    present_narratives : iterable[str]
        Names of narratives appearing in the dataset.

    Returns
    -------
    list[str]
        Column names like ["San_eat", "Bro_throw", ...].
    """
    token_cols = [
        f"{scene[:3]}_{token}"
        for scene in present_narratives
        for token in scene_tokens.get(scene, [])
    ]
    return token_cols


def _col(df, candidates):
    """
    Return the first matching column from a list of possible variants.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns to inspect.
    candidates : list[str]
        Possible column name variants.

    Returns
    -------
    str | None
        The first existing column name, or None if none found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prepare_corelex_inputs(input_dir, output_dir, exclude_participants):
    """
    Load and normalize utterance-level CoreLex inputs.

    Determines whether input is unblind or transcript-table mode,
    filters by valid narratives, excludes unwanted speakers, and
    normalizes column names for downstream analysis.

    Parameters
    ----------
    input_dir, output_dir : Path
        Source and destination directories.
    exclude_participants : set[str]
        Speaker IDs to exclude (e.g., {"INV"}).

    Returns
    -------
    tuple[pd.DataFrame, set[str]]
        Normalized utterance DataFrame and the set of valid narratives.
        Returns (None, None) if loading fails.
    """
    try:
        mode, utt_df = find_corelex_inputs(input_dir, output_dir)
        if utt_df is None:
            return None, None

        if mode == "unblind":
            narr_col = _col(utt_df, stim_cols)
            utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]
            cu_col = next((c for c in utt_df.columns if c.startswith("c2_cu")), None)
            wc_col = "word_count" if "word_count" in utt_df.columns else None
            filter_col = cu_col or wc_col
            if filter_col:
                utt_df = utt_df[~np.isnan(utt_df[filter_col])]
            else:
                logger.warning("No c2_cu or word_count column; continuing unfiltered.")
            present_narratives = set(utt_df[narr_col].dropna().unique())

        else:  # transcript table mode
            narr_col = _col(utt_df, stim_cols)
            utt_col = _col(utt_df, ["utterance", "text", "tokens"])
            time_col = _col(
                utt_df,
                [
                    "speaking_time",
                    "client_time",
                    "speech_time",
                    "time_s",
                    "time_sec",
                    "time_seconds",
                ],
            )

            if not all([narr_col, utt_col, time_col]):
                logger.error("Required columns missing in transcript table input.")
                return None, None

            utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]
            if "speaker" in utt_df.columns and exclude_participants:
                utt_df = utt_df[~utt_df["speaker"].isin(exclude_participants)]

            present_narratives = set(utt_df[narr_col].dropna().unique())
            utt_df = utt_df.rename(columns={narr_col: "narrative"})
            if utt_col != "utterance":
                utt_df = utt_df.rename(columns={utt_col: "utterance"})
            if time_col and time_col != "speaking_time":
                utt_df = utt_df.rename(columns={time_col: "speaking_time"})

        return utt_df, present_narratives

    except Exception as e:
        logger.error(f"Failed to prepare CoreLex inputs: {e}")
        return None, None


def _compute_corelex_for_sample(sample_df, norm_lookup, partition_tiers, tup):
    """
    Compute CoreLex coverage, diversity, and percentile metrics for one sample.

    Parameters
    ----------
    sample_df : pd.DataFrame
        All utterances from one participant × narrative.
    norm_lookup : dict
        Preloaded CoreLex norms for each narrative.
    partition_tiers : list[str]
        Tier names to attach as grouping metadata.
    tup : tuple
        Corresponding partition-tier values.

    Returns
    -------
    dict
        Row dictionary containing CoreLex metrics and percentile results.
    """
    try:
        sample = sample_df["sample_id"].iloc[0]
        scene_name = sample_df["narrative"].iloc[0]
        speaking_time = sample_df.get("speaking_time", np.nan)
        speaking_time = speaking_time.iloc[0] if hasattr(speaking_time, "iloc") else speaking_time

        text = " ".join([u for u in sample_df["utterance"].astype(str) if u.strip()])
        reformatted_text = reformat(text)
        core_stats = id_core_words(scene_name, reformatted_text)

        minutes = (
            float(speaking_time) / 60.0
            if pd.notnull(speaking_time) and speaking_time > 0
            else np.nan
        )
        cwpm = (
            core_stats["num_core_words"] / minutes
            if pd.notnull(minutes) and minutes > 0
            else np.nan
        )

        acc_df = norm_lookup[scene_name]["accuracy"]
        acc_pcts = get_percentiles(core_stats["num_core_words"], acc_df, "CoreLex Score")

        if pd.notnull(cwpm):
            eff_df = norm_lookup[scene_name]["efficiency"]
            eff_pcts = get_percentiles(cwpm, eff_df, "CoreLex/min")
            cwpm_pwa, cwpm_ctrl = eff_pcts["pwa_percentile"], eff_pcts["control_percentile"]
        else:
            cwpm_pwa = cwpm_ctrl = np.nan

        row = {
            "sample_id": sample,
            **{pt: val for pt, val in zip(partition_tiers, tup)},
            "narrative": scene_name,
            "speaking_time": speaking_time,
            "num_tokens": core_stats["num_tokens"],
            "num_core_words": core_stats["num_core_words"],
            "num_core_word_tokens": core_stats["num_cw_tokens"],
            "lexicon_coverage": core_stats["lexicon_coverage"],
            "core_words_per_min": cwpm,
            "core_words_pwa_percentile": acc_pcts["pwa_percentile"],
            "core_words_control_percentile": acc_pcts["control_percentile"],
            "cwpm_pwa_percentile": cwpm_pwa,
            "cwpm_control_percentile": cwpm_ctrl,
        }

        for lemma, surfaces in core_stats["token_sets"].items():
            row[f"{scene_name[:3]}_{lemma}"] = ", ".join(sorted(surfaces))

        return row

    except Exception as e:
        logger.error(f"Failed to compute CoreLex metrics for sample: {e}")
        return {}


def run_corelex(tiers, input_dir, output_dir, exclude_participants=None):
    """
    Execute CoreLex analysis on aphasia narratives and export results.

    Combines utterance-level text and timing data, computes lexical
    coverage and efficiency against CoreLex norms, attaches percentiles,
    and saves to `<output_dir>/core_lex/core_lex_data_<timestamp>.xlsx`.

    Parameters
    ----------
    tiers : dict
        Tier objects defining partitions.
    input_dir, output_dir : Path
        Paths for input and output directories.
    exclude_participants : set[str] | None
        Speaker codes to exclude (e.g., {"INV"}).

    Returns
    -------
    None
        Results written to Excel; logs warnings for missing data.
    """
    exclude_participants = set(exclude_participants or [])
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    corelex_dir = output_dir / "core_lex"
    corelex_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CoreLex output directory: {_rel(corelex_dir)}")

    utt_df, present_narratives = _prepare_corelex_inputs(input_dir, output_dir, exclude_participants)
    if utt_df is None:
        return

    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]
    norm_lookup = preload_corelex_norms(present_narratives)
    token_columns = generate_token_columns(present_narratives)
    all_columns = [base_columns[0]] + partition_tiers + base_columns[1:] + token_columns
    rows = []

    # --- Handle no partition tiers gracefully ---
    if partition_tiers:
        grouped = utt_df.groupby(by=partition_tiers)
    else:
        grouped = [((), utt_df)]  # single group representing all data

    for tup, subdf in grouped:
        for sample in tqdm(sorted(subdf["sample_id"].dropna().unique()), desc="Computing CoreLex"):
            sample_df = subdf[subdf["sample_id"] == sample]
            if sample_df.empty:
                continue
            row = _compute_corelex_for_sample(sample_df, norm_lookup, partition_tiers, tup)
            if row:
                rows.append(row)

    if not rows:
        logger.warning("No CoreLex rows produced; no output written.")
        return

    corelex_df = pd.DataFrame(rows, columns=all_columns)
    output_file = corelex_dir / f"core_lex_data_{timestamp}.xlsx"
    try:
        corelex_df.to_excel(output_file, index=False)
        logger.info(f"CoreLex results written to {_rel(output_file)}")
    except Exception as e:
        logger.error(f"Failed to write CoreLex results: {e}")

    logger.info("CoreLex processing complete.")
