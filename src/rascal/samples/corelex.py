import os
import re
import logging
import numpy as np
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
from datetime import datetime
from scipy.stats import percentileofscore


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
    "'m": "be", "'re": "be", "'s": "be",  # context-dependent, but may help

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

    # Determiners and articles (in case spoken strangely)
    "th": "the", "da": "the", "uh": "a", "an": "a",

    # Spoken reductions
    "lemme": "let", "gimme": "give", "cmon": "come", "outta": "out",
    "inna": "in", "coulda": "could", "shoulda": "should", "woulda": "would",
}

base_columns = [
    "sample_id", "narrative", "speakingTime", "numTokens",
    "numCoreWords", "numCoreWordTokens", "lexiconCoverage", "coreWordsPerMinute",
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
        logging.error(f"An error occurred while reformatting: {e}")
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
            logging.info(f"Loaded CoreLex norms for: {scene}")
        except Exception as e:
            logging.warning(f"Failed to load norms for {scene}: {e}")
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
        logging.warning(f"Failed reading {path}: {e}")
        return None

def find_corelex_inputs(input_dir: str, output_dir: str) -> dict:
    """
    Find available inputs for CoreLex in priority order:
      1) *unblindUtteranceData.xlsx (best)
      2) *_Utterances.xlsx            (fallback)
    Optionally: *_SpeakingTimes.xlsx  (merge with fallback)
    
    Returns dict:
      {
        "mode": "unblind" | "utterances",
        "utt_df": pd.DataFrame,
        "times_df": pd.DataFrame | None,
        "paths": {"utt": <Path|list>, "times": <list>}
      }
    or None if nothing usable is found.
    """
    search_dirs = [Path(output_dir), Path(input_dir)]

    # 1) Look for unblindUtteranceData.xlsx
    unblind_matches = []
    for d in search_dirs:
        unblind_matches += list(d.rglob("*unblindUtteranceData.xlsx"))
    if unblind_matches:
        p = unblind_matches[0]
        df = _read_excel_safely(p)
        if df is not None:
            logging.info(f"Using unblind utterance data: {p}")
            return {"mode": "unblind", "utt_df": df, "times_df": None, "paths": {"utt": p, "times": []}}

    # 2) Fallback to *_Utterances.xlsx (may be multiple; concat)
    utt_files = []
    for d in search_dirs:
        utt_files += list(d.rglob("*_Utterances.xlsx"))
    if not utt_files:
        logging.error("No utterance files found (neither *unblindUtteranceData.xlsx nor *_Utterances.xlsx).")
        return None

    utt_frames = [df for f in utt_files if (df := _read_excel_safely(f)) is not None]
    if not utt_frames:
        logging.error("Utterance files were found but none could be read.")
        return None
    utt_df = pd.concat(utt_frames, ignore_index=True, sort=False)
    logging.info(f"Using concatenated utterances from {len(utt_files)} file(s).")

    # Optional speaking times: *_SpeakingTimes.xlsx (concat)
    time_files = []
    for d in search_dirs:
        time_files += list(d.rglob("*_SpeakingTimes.xlsx"))
    times_df = None
    if time_files:
        time_frames = [df for f in time_files if (df := _read_excel_safely(f)) is not None]
        if time_frames:
            times_df = pd.concat(time_frames, ignore_index=True, sort=False)
            logging.info(f"Loaded speaking times from {len(time_files)} file(s).")
        else:
            logging.warning("Speaking time files found but none could be read; proceeding without times.")

    return {"mode": "utterances", "utt_df": utt_df, "times_df": times_df, "paths": {"utt": utt_files, "times": time_files}}

def generate_token_columns(present_narratives):
    token_cols = [f"{scene[:3]}_{token}"
                  for scene in present_narratives
                  for token in scene_tokens.get(scene, [])]
    return token_cols

def run_corelex(input_dir, output_dir, exclude_participants=None):
    """
    Run CoreLex analysis on aphasia narrative samples, producing lexical diversity
    and coverage metrics against published CoreLex norms.

    Workflow
    --------
    1. Load utterance-level input:
       - If mode = "unblind" (determined with find_corelex_inputs): expects a single "unblindUtteranceData.xlsx"
         containing columns: ['sample_id','narrative','utterance','client_time','c2CU',...].
       - Else: expects "*_Utterances.xlsx" and "*_SpeakingTimes.xlsx" files.
         * Utterances are filtered to exclude speakers in `exclude_participants`
           (e.g., {"INV"} to drop investigators).
    2. For each sample x narrative:
       - Tokenize utterances, normalize, and match against the CoreLex lists.
       - Count:
           * numTokens: total tokens
           * numCoreWords: distinct core lemmas
           * numCoreWordTokens: total core-word tokens
       - Merge with speaking time, compute:
           * coreWordsPerMinute (cwpm)
    3. Preload CoreLex norms for the relevant narratives and compute percentiles:
           * core_words_control_percentile, core_words_pwa_percentile
           * cwpm_control_percentile, cwpm_pwa_percentile
    4. Write results to:
           "<output_dir>/CoreLex/CoreLexData_<timestamp>.xlsx"

    Parameters
    ----------
    input_dir : str | os.PathLike
        Directory containing input files (unblinded summary OR utterance/time files).
    output_dir : str | os.PathLike
        Directory where the "CoreLex/" subdirectory and results file are created.
    exclude_participants : set[str] | None, default None
        Speaker codes (e.g., {"INV"}) to exclude when reading utterance-level data
        in fallback mode (`unblind=False`).

    Outputs
    -------
    Excel file:
      "<output_dir>/CoreLex/CoreLexData_<timestamp>.xlsx"

    Returns
    -------
    None
        Results are written to disk.

    Notes
    -----
    - Narrative values in the data must match keys available in CoreLex norms
      (e.g., "Sandwich", "BrokenWindow", "CatRescue").
    - Speaking time is required for cwpm calculations; in unblind mode this must
      be a 'client_time' column, in fallback mode it comes from *_SpeakingTimes.xlsx.
    - Logs warnings if norms or percentiles cannot be computed.
    """
    exclude_participants = set(exclude_participants or [])
    logging.info("Starting CoreLex processing.")
    timestamp = datetime.now().strftime('%y%m%d_%H%M')

    corelex_dir = os.path.join(output_dir, 'CoreLex')
    os.makedirs(corelex_dir, exist_ok=True)
    logging.info(f"Output directory created: {corelex_dir}")

    # ---- Find inputs (logic gates) ----
    inputs = find_corelex_inputs(input_dir, output_dir)
    if inputs is None:
        return

    mode = inputs["mode"]
    utt_df = inputs["utt_df"].copy()
    times_df = inputs["times_df"]

    # ---- Normalize column names to a common expectation where possible ----
    # We expect at least: sample_id, narrative, utterance
    # Try to auto-fix common variants in fallback mode
    def _col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    if mode == "unblind":
        narr_col   = _col(utt_df, ["narrative", "scene", "story", "stimulus"])
        # Filter to relevant narratives present in urls
        utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]

        # Inclusion: prefer CU indicator if present; else wordCount
        cu_col = next((c for c in utt_df.columns if c.startswith("c2CU")), None)
        na_col = cu_col or ('wordCount' if 'wordCount' in utt_df.columns else None)
        if na_col:
            utt_df = utt_df[~np.isnan(utt_df[na_col])]
        else:
            logging.warning("No CU/wordCount column found; proceeding without that filter in unblind mode.")

        # Present narratives
        present_narratives = set(utt_df[narr_col].dropna().unique())

    else:  # mode == 'utterances'
        # Map common column name variants
        sample_col = _col(utt_df, ["sampleID", "sample_id", "sample"])
        narr_col   = _col(utt_df, ["narrative", "scene", "story", "stimulus"])
        utt_col    = _col(utt_df, ["utterance", "text", "tokens"])
        speak_col  = _col(utt_df, ["client_time", "speaking_time", "speech_time", "time_s", "time_sec", "time_seconds"])

        missing = [name for name, c in {
            "sample_id": sample_col, "narrative": narr_col, "utterance": utt_col
        }.items() if c is None]
        if missing:
            logging.error(f"Required columns missing in *_Utterances.xlsx mode: {missing}")
            return

        # Filter by narratives in urls
        utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]

        # Exclude by speaker if available
        if "speaker" in utt_df.columns and exclude_participants:
            utt_df = utt_df[~utt_df["speaker"].isin(exclude_participants)]

        # Merge speaking times if not present inline
        if speak_col is None and times_df is not None:
            # Find time columns in times_df
            t_sample = _col(times_df, ["sampleID", "sample_id", "sample"])
            t_time   = _col(times_df, ["client_time", "speaking_time", "speech_time", "time_s", "time_sec", "time_seconds"])
            if t_sample and t_time:
                times_small = times_df[[t_sample, t_time]].dropna()
                times_small = times_small.rename(columns={t_sample: sample_col, t_time: "client_time"})
                utt_df = utt_df.merge(times_small, on=sample_col, how="left")
                speak_col = "client_time"
                logging.info("Merged speaking times into utterances data.")
            else:
                logging.warning("SpeakingTimes file found but required columns missing; proceeding without times.")
                speak_col = None

        # Present narratives
        present_narratives = set(utt_df[narr_col].dropna().unique())

        # For downstream code, normalize a few names explicitly
        utt_df = utt_df.rename(columns={sample_col: "sample_id", narr_col: "narrative"})
        if utt_col != "utterance":
            utt_df = utt_df.rename(columns={utt_col: "utterance"})
        if speak_col and speak_col != "client_time":
            utt_df = utt_df.rename(columns={speak_col: "client_time"})

    # ---- Preload all needed norms ----
    norm_lookup = preload_corelex_norms(present_narratives)

    # ---- Prepare output columns ----
    token_columns = generate_token_columns(present_narratives)
    all_columns = base_columns + token_columns
    rows = []

    # ---- Iterate by sample ----
    for sample in tqdm(sorted(utt_df['sample_id'].dropna().unique())):
        subdf = utt_df[utt_df['sample_id'] == sample]
        if subdf.empty:
            continue

        scene_name = subdf['narrative'].iloc[0]

        # speaking_time present in unblind OR merged into utterances mode
        speaking_time = subdf['client_time'].iloc[0] if 'client_time' in subdf.columns else np.nan
        # Concatenate participant's utterances (already filtered speaker if applicable)
        text = ' '.join([u for u in subdf['utterance'].astype(str) if u.strip()])

        reformatted_text = reformat(text)
        core_stats = id_core_words(scene_name, reformatted_text)

        num_tokens = core_stats["num_tokens"]
        num_core_words = core_stats["num_core_words"]
        num_cw_tokens = core_stats["num_cw_tokens"]
        lexicon_coverage = core_stats["lexicon_coverage"]
        token_sets = core_stats["token_sets"]

        # CWPM only if speaking_time is available
        minutes = (float(speaking_time) / 60.0) if (pd.notnull(speaking_time) and speaking_time > 0) else np.nan
        cwpm = (num_core_words / minutes) if (pd.notnull(minutes) and minutes > 0) else np.nan

        # Percentiles: always compute accuracy; efficiency only if cwpm is finite
        acc_df = norm_lookup[scene_name]["accuracy"]
        acc_percentiles = get_percentiles(num_core_words, acc_df, "CoreLex Score")

        if pd.notnull(cwpm):
            eff_df = norm_lookup[scene_name]["efficiency"]
            eff_percentiles = get_percentiles(cwpm, eff_df, "CoreLex/min")
            cwpm_pwa = eff_percentiles["pwa_percentile"]
            cwpm_ctrl = eff_percentiles["control_percentile"]
        else:
            cwpm_pwa = np.nan
            cwpm_ctrl = np.nan

        row = {
            "sample_id": sample,
            "narrative": scene_name,
            "speakingTime": speaking_time if pd.notnull(speaking_time) else np.nan,
            "numTokens": num_tokens,
            "numCoreWords": num_core_words,
            "numCoreWordTokens": num_cw_tokens,
            "lexiconCoverage": lexicon_coverage,
            "coreWordsPerMinute": cwpm,
            "core_words_pwa_percentile": acc_percentiles["pwa_percentile"],
            "core_words_control_percentile": acc_percentiles["control_percentile"],
            "cwpm_pwa_percentile": cwpm_pwa,
            "cwpm_control_percentile": cwpm_ctrl
        }

        # Add per-lemma surface forms as columns
        for lemma, surface_forms in token_sets.items():
            col_name = f"{scene_name[:3]}_{lemma}"
            if col_name in token_columns:
                row[col_name] = ', '.join(sorted(surface_forms))

        rows.append(row)

    corelexdf = pd.DataFrame(rows, columns=all_columns)

    output_file = os.path.join(corelex_dir, f'CoreLexData_{timestamp}.xlsx')
    corelexdf.to_excel(output_file, index=False)
    logging.info(f"Saved: {output_file}")
    logging.info("CoreLex processing complete.")
