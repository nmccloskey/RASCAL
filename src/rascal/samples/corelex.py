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
    "sampleID", "participantID", "narrative", "speakingTime", "numTokens",
    "numCoreWords", "numCoreWordTokens", "lexiconCoverage", "coreWordsPerMinute",
    "core_words_pwa_percentile", "core_words_control_percentile",
    "cwpm_pwa_percentile", "cwpm_control_percentile"
]

def reformat(text: str) -> str:
    """
    Prepares a transcription text string for CoreLex analysis.

    - Expands most contractions (e.g., "it's" → "it is", but keeps possessive 's).
    - Converts numeric digits to words.
    - Removes most CHAT annotations except accepted corrections like [: dogs] [*].
    - Removes disfluencies, fragments, gestures, etc.
    
    Args:
        text (str): The transcription text to be formatted.
        
    Returns:
        str: Cleaned and normalized text.
    """
    try:
        text = text.lower().strip()

        # Handle "he's got" and "it's got" → "he has got", etc.
        text = re.sub(r"\b(he|it)'s got\b", r"\1 has got", text)

        # Expand contractions while preserving possessive 's (approximate strategy)
        tokens = text.split()
        expanded = []
        for token in tokens:
            # Skip possessive 's (approximated by looking for preceding nouns)
            if re.match(r"\w+'s\b", token) and not contractions.fix(token).startswith("it is"):
                expanded.append(token)
            else:
                expanded.append(contractions.fix(token))
        text = ' '.join(expanded)

        # Convert digits to words (e.g., 3 → three)
        text = re.sub(r'\b\d+\b', lambda m: n2w.num2words(int(m.group())), text)

        # Preserve accepted correction format: [: word] [*]
        text = re.sub(r'\[: ([^\]]+)\] \[\*\]', r'\1', text)

        # Remove all other square-bracketed content (e.g., [//], [?], [% ...], [&])
        text = re.sub(r'\[[^\]]+\]', '', text)

        # Remove all non-word characters except apostrophes (keep possessives)
        text = re.sub(r"[^\w\s']", ' ', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

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

def find_utterance_file(input_dir: str, output_dir: str) -> str:
    """
    Searches for the unblindUtteranceData.xlsx file in input/output subdirectories.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.

    Returns:
        str: Path to the found file, or None if not found.
    """
    for base_dir in [output_dir, input_dir]:
        matches = list(Path(base_dir).rglob('unblindUtteranceData.xlsx'))
        if matches:
            logging.info(f"Found utterance file: {matches[0]}")
            return str(matches[0])

    logging.error("unblindUtteranceData.xlsx not found in input or output directories.")
    return None

def generate_token_columns(present_narratives):
    token_cols = [f"{scene[:3]}_{token}"
                  for scene in present_narratives
                  for token in scene_tokens.get(scene, [])]
    return token_cols

def run_corelex(input_dir, output_dir):
    """
    Runs CoreLex analysis on utterance data and saves the results.

    Args:
        input_dir (str): The directory where input files may be located.
        output_dir (str): The directory where output files will be saved.
    """
    logging.info("Starting CoreLex processing.")
    timestamp = datetime.now().strftime('%y%m%d_%H%M')

    corelex_dir = os.path.join(output_dir, 'CoreLex')
    os.makedirs(corelex_dir, exist_ok=True)
    logging.info(f"Output directory created: {corelex_dir}")

    # Read in utterance file
    utt_data_path = find_utterance_file(input_dir, output_dir)
    if utt_data_path is None:
        return

    utt_df = pd.read_excel(utt_data_path)
    utt_df = utt_df[utt_df['narrative'].isin(urls.keys())]
    utt_df = utt_df[~np.isnan(utt_df['wordCount'])]

    # Preload all needed norms
    present_narratives = set(utt_df['narrative'].unique())
    norm_lookup = preload_corelex_norms(present_narratives)

    # Prepare token columns
    token_columns = generate_token_columns(present_narratives)
    all_columns = base_columns + token_columns

    rows = []  # Will hold each row as a dictionary

    for sample in tqdm(set(utt_df['sampleID'])):
        subdf = utt_df[utt_df['sampleID'] == sample]
        scene_name = subdf['narrative'].iloc[0]
        pID = subdf['participantID'].iloc[0]
        speaking_time = subdf['client_time'].iloc[0]
        text = ' '.join(subdf['utterance'])

        reformatted_text = reformat(text)
        core_stats = id_core_words(scene_name, reformatted_text)

        num_tokens = core_stats["num_tokens"]
        num_core_words = core_stats["num_core_words"]
        num_cw_tokens = core_stats["num_cw_tokens"]
        lexicon_coverage = core_stats["lexicon_coverage"]
        token_sets = core_stats["token_sets"]

        # Core words per minute
        minutes = speaking_time / 60 if speaking_time and speaking_time > 0 else np.nan
        cwpm = num_core_words / minutes if minutes else np.nan

        # Load and calculate percentiles
        acc_df = norm_lookup[scene_name]["accuracy"]
        eff_df = norm_lookup[scene_name]["efficiency"]
        acc_percentiles = get_percentiles(num_core_words, acc_df, "CoreLex Score")
        eff_percentiles = get_percentiles(cwpm, eff_df, "CoreLex/min")

        row = {
            "sampleID": sample,
            "participantID": pID,
            "narrative": scene_name,
            "speakingTime": speaking_time,
            "numTokens": num_tokens,
            "numCoreWords": num_core_words,
            "numCoreWordTokens": num_cw_tokens,
            "lexiconCoverage": lexicon_coverage,
            "coreWordsPerMinute": cwpm,
            "core_words_pwa_percentile": acc_percentiles["pwa_percentile"],
            "core_words_control_percentile": acc_percentiles["control_percentile"],
            "cwpm_pwa_percentile": eff_percentiles["pwa_percentile"],
            "cwpm_control_percentile": eff_percentiles["control_percentile"]
        }

        for lemma, surface_forms in token_sets.items():
            col_name = f"{scene_name[:3]}_{lemma}"
            row[col_name] = ', '.join(sorted(surface_forms))

        rows.append(row)

    corelexdf = pd.DataFrame(rows, columns=all_columns)

    output_file = os.path.join(corelex_dir, f'CoreLexData_{timestamp}.xlsx')
    corelexdf.to_excel(output_file, index=False)
    logging.info(f"Saved: {output_file}")
    logging.info("CoreLex processing complete.")
