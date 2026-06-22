from __future__ import annotations

import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from num2words import num2words


# ---------------------------------------------------------------------
# Config-like constants
# ---------------------------------------------------------------------

# Conservative list: only convert bracketed/starred spans if one of these
# action words appears. Expand as needed.
NONVERBAL_ACTION_WORDS = {
    "point", "pointing", "points",
    "gesture", "gestures", "gesturing",
    "laugh", "laughs", "laughing",
    "sigh", "sighs", "sighing",
    "yawn", "yawns", "yawning",
    "indicate", "indicates", "indicating",
    "write", "writes", "writing",
    "imitate", "imitates", "imitating",
    "shrug", "shrugs", "shrugging",
    "nod", "nods", "nodding",
    "shake", "shakes", "shaking",
    "cry", "cries", "crying",
    "smile", "smiles", "smiling",
    "grunt", "grunts", "grunting",
    "cough", "coughs", "coughing",
    "whisper", "whispers", "whispering",
    "sound", "sounds", "makes", "making",
    "wave", "waves", "waving",
}

# Prefer shortened CHAT-ish stem for "gesturing" family
GESTURE_WORDS = {"gesture", "gestures", "gesturing"}

# Simple disfluencies for automatic &- tagging
DISFLUENCY_RE = re.compile(r"^(uh+|um+|er+|erm+|eh+)$", re.IGNORECASE)

# Legacy unintelligible tokens
UNINTELLIGIBLE_RE = re.compile(r"^(?:x{2,}|X{2,})$")

# IPA chunks delimited by slashes, e.g. /sə/
IPA_SLASH_RE = re.compile(r"(?<!\[= )/([^/\n]+)/")

# Repairs like: ricycle {tricycle}
CORRECTION_RE = re.compile(r"\b([^\s{}]+)\s*\{([^{}]+)\}")

# Candidate nonverbal chunks in *, (), or []
NONVERBAL_CHUNK_RE = re.compile(
    r"(\*[^*]+\*|\([^()]+\)|\[[^\[\]]+\])"
)

# Proper utterance enders
TERMINAL_PUNCT_RE = re.compile(r"[.!?]$")

# Excel / XML illegal control chars:
# 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F
ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

CLINICIAN_RE = re.compile(r"^\s*(?:\*?(?:Clinician|CL)|\*?Cl)\s*:\s*", re.IGNORECASE)
REMOVAL_RE = re.compile(
    r"(?:\b\d+\s*seconds?$|\bTIME:|\bTime:\s*\d|\bSTART:|\bEND:|\bCL=)",
    re.IGNORECASE,
)

# For number to word expansion
NUMBER_TOKEN_RE = re.compile(r"^-?\d+(?:\.\d+)?%?$")


# ---------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------

@dataclass
class TransformResult:
    raw: str
    transformed: str
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------
# General dataframe helper
# ---------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source.name}: {missing}"
        )

# ---------------------------------------------------------------------
# General text helpers
# ---------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim ends."""
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenization for conservative transformations."""
    return text.split()


def _contains_nonverbal_action(text: str) -> bool:
    """Return True if text contains one of the approved action words."""
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return any(tok in NONVERBAL_ACTION_WORDS for tok in toks)


def _normalize_nonverbal_payload(text: str) -> str:
    """
    Convert inner nonverbal text to CHAT-style &= payload.

    Examples
    --------
    "pointing to fish bowl" -> "pointing:to:fish:bowl"
    "writes something" -> "writes:something"
    "gesturing falling down" -> "ges:falling:down"
    "gesturing, imitating angry mother" -> "ges:imitating:angry:mother"
    """
    text = text.strip().lower()

    # Remove outer punctuation noise and convert separators to spaces first
    text = re.sub(r"[,;/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = re.findall(r"[a-z']+", text)
    if not words:
        return ""

    # Optional abbreviation for gesture-family forms
    if words[0] in GESTURE_WORDS:
        words[0] = "ges"

    return ":".join(words)

# ---------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------

def convert_nonverbals(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert likely nonverbal descriptions in *, (), or [] to CHAT-style &= codes,
    but only when an approved action word is present.

    Conservative behavior:
    - does NOT convert every parenthetical/bracketed chunk
    - allows multiple nonverbals in one utterance
    """
    flags: list[str] = []
    notes: list[str] = []

    def repl(match: re.Match) -> str:
        chunk = match.group(0)
        inner = chunk[1:-1].strip()

        if not _contains_nonverbal_action(inner):
            return chunk

        payload = _normalize_nonverbal_payload(inner)
        if not payload:
            return chunk

        flags.append("nonverbal_auto")
        notes.append(f"Converted nonverbal: {chunk} -> &={payload}")
        return f"&={payload}"

    new_text = NONVERBAL_CHUNK_RE.sub(repl, text)
    new_text = _normalize_whitespace(new_text)
    return new_text, flags, notes


def convert_corrections(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert:
        ricycle {tricycle}
    to:
        ricycle [: tricycle] [*]
    """
    flags: list[str] = []
    notes: list[str] = []

    def repl(match: re.Match) -> str:
        original = match.group(1)
        target = match.group(2).strip()
        flags.append("correction_auto")
        notes.append(f"Converted correction: {original} {{{target}}}")
        return f"{original} [: {target}] [*]"

    new_text = CORRECTION_RE.sub(repl, text)
    return new_text, flags, notes


def regularize_unintelligibles(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert xx / XX / xxx / XXXX ... to XXX tokenwise.
    """
    flags: list[str] = []
    notes: list[str] = []

    toks = []
    changed = False
    for tok in _tokenize_simple(text):
        if UNINTELLIGIBLE_RE.fullmatch(tok):
            toks.append("XXX")
            changed = True
        else:
            toks.append(tok)

    if changed:
        flags.append("unintelligible_auto")
        notes.append("Regularized xx/xxx-style unintelligibles to XXX.")

    return " ".join(toks), flags, notes


def regularize_disfluency_case(text: str) -> tuple[str, list[str], list[str]]:
    """
    Lowercase tokens that already begin with &- .
    """
    flags: list[str] = []
    notes: list[str] = []

    toks = []
    changed = False
    for tok in _tokenize_simple(text):
        if tok.startswith("&-"):
            lowered = tok.lower()
            toks.append(lowered)
            changed = changed or (lowered != tok)
        else:
            toks.append(tok)

    if changed:
        flags.append("disfluency_case_auto")
        notes.append("Lowercased existing &- disfluency tokens.")

    return " ".join(toks), flags, notes


def autotag_disfluencies(text: str) -> tuple[str, list[str], list[str]]:
    """
    Add &- to plain filler tokens such as uh, um, er, erm, eh.

    Conservative tokenwise behavior only.
    """
    flags: list[str] = []
    notes: list[str] = []

    toks = []
    changed = False
    for tok in _tokenize_simple(text):
        bare = tok.strip(".,!?;:")
        if DISFLUENCY_RE.fullmatch(bare) and not tok.startswith("&-"):
            punct_prefix = tok[: len(tok) - len(tok.lstrip("([{\"'"))]
            punct_suffix = tok[len(bare):] if tok.endswith(tuple(".,!?;:")) else ""
            toks.append(f"&-{bare.lower()}{punct_suffix}")
            changed = True
        else:
            toks.append(tok)

    if changed:
        flags.append("disfluency_auto")
        notes.append("Auto-tagged simple disfluency tokens with &-.")

    return " ".join(toks), flags, notes


def convert_ipa_slashes(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert slash-delimited IPA snippets:
        /sə/ /əf/
    to:
        [= /sə/] [= /əf/]

    Conservative:
    - wraps slash-delimited chunks
    - does not inspect phonological validity
    """
    flags: list[str] = []
    notes: list[str] = []

    def repl(match: re.Match) -> str:
        inner = match.group(1)
        flags.append("ipa_auto")
        notes.append(f"Wrapped slash IPA: /{inner}/")
        return f"[= /{inner}/]"

    new_text = IPA_SLASH_RE.sub(repl, text)
    return new_text, flags, notes


def convert_phonological_fragments(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert phonological fragments conservatively.

    Handles:
        h-home      -> &+h home
        n-n-not     -> &+n &+n not
        pr-presents -> &+pr presents
        s-s-stuck.  -> &+s &+s stuck.
        g-g-girl?   -> &+g &+g girl?
        wri-        -> &+wri
        S-          -> &+s

    Conservative:
    - accepts mixed/upper/lower input but normalizes fragment codes to lowercase
    - preserves trailing punctuation
    - leaves acronymic forms like B-S-E untouched
    """
    flags: list[str] = []
    notes: list[str] = []

    trailing_punct_re = re.compile(r"([.!?,;:]+)$")

    def convert_token(token: str) -> str:
        # Separate trailing punctuation so e.g. s-s-stuck. can be analyzed
        punct = ""
        m = trailing_punct_re.search(token)
        if m:
            punct = m.group(1)
            core = token[: m.start()]
        else:
            core = token

        if not core:
            return token

        # Case 1: standalone trailing fragment, e.g. "wri-" or "S-"
        if core.endswith("-"):
            frag = core[:-1]
            if re.fullmatch(r"[A-Za-z]{1,10}", frag):
                return f"&+{frag.lower()}{punct}"
            return token

        # Case 2: chained fragment(s) + target, e.g. h-home / n-n-not / g-g-girl
        parts = core.split("-")
        if len(parts) < 2:
            return token

        prefixes = parts[:-1]
        final = parts[-1]

        # Final target must be alphabetic-ish
        if not re.fullmatch(r"[A-Za-z][A-Za-z']*", final):
            return token

        # Prefixes must be short alphabetic fragments
        if not all(re.fullmatch(r"[A-Za-z]{1,3}", p) for p in prefixes):
            return token

        # Avoid converting acronymic things like B-S-E
        if final.isupper() and all(p.isupper() for p in prefixes):
            return token

        converted = " ".join([*(f"&+{p.lower()}" for p in prefixes), final.lower()])
        return converted + punct

    toks = []
    changed = False
    for tok in _tokenize_simple(text):
        new_tok = convert_token(tok)
        toks.append(new_tok)
        changed = changed or (new_tok != tok)

    if changed:
        flags.append("phon_frag_auto")
        notes.append(
            "Converted phonological fragments, including chained and trailing-hyphen forms."
        )

    return " ".join(toks), flags, notes


def ensure_terminal_punct(text: str) -> tuple[str, list[str], list[str]]:
    """
    Ensure utterance ends with sentence-final punctuation.

    Conservative:
    - only checks final character
    - appends " ." if missing
    """
    flags: list[str] = []
    notes: list[str] = []

    if not text or not str(text).strip():
        return "", flags, notes

    text = str(text).rstrip()

    if TERMINAL_PUNCT_RE.search(text):
        return text, flags, notes

    flags.append("terminal_punct_auto")
    notes.append("Added terminal punctuation placeholder for CLAN compatibility.")
    return text + " .", flags, notes


def detect_clinician(utt: str) -> bool:
    """Return True if utterance appears to be a clinician turn label."""
    if pd.isna(utt):
        return False
    return bool(CLINICIAN_RE.match(str(utt)))


def flag_for_removal(utt: str) -> int:
    """Return 1 if utterance looks like timing/meta text that should be removed."""
    if pd.isna(utt):
        return 0
    return int(bool(REMOVAL_RE.search(str(utt))))


def strip_excel_illegal_chars(text: str) -> tuple[str, int]:
    """
    Remove control characters that Excel worksheets cannot store.

    Returns
    -------
    cleaned_text : str
    n_removed : int
    """
    if pd.isna(text):
        return "", 0

    text = str(text)
    matches = ILLEGAL_EXCEL_CHARS_RE.findall(text)
    cleaned = ILLEGAL_EXCEL_CHARS_RE.sub("", text)
    return cleaned, len(matches)

def convert_numbers_to_words(text: str) -> tuple[str, list[str], list[str]]:
    """
    Convert simple numeric tokens to words.

    Examples
    --------
    2 -> two
    14 -> fourteen
    3.5 -> three point five
    50% -> fifty percent

    Conservative:
    - token-based only
    - does not try to parse dates, times, fractions, ordinals, ranges, or mixed alphanumerics
    - preserves trailing punctuation
    """
    flags: list[str] = []
    notes: list[str] = []

    if not text or not str(text).strip():
        return "", flags, notes

    trailing_punct_re = re.compile(r"([.,!?;:]+)$")

    def convert_token(token: str) -> str:
        punct = ""
        m = trailing_punct_re.search(token)
        if m:
            punct = m.group(1)
            core = token[: m.start()]
        else:
            core = token

        if not core:
            return token

        if not NUMBER_TOKEN_RE.fullmatch(core):
            return token

        try:
            if core.endswith("%"):
                num_part = core[:-1]
                if "." in num_part:
                    spoken = num2words(float(num_part))
                else:
                    spoken = num2words(int(num_part))
                return f"{spoken} percent{punct}"

            if "." in core:
                spoken = num2words(float(core))
            else:
                spoken = num2words(int(core))

            return f"{spoken}{punct}"

        except Exception:
            return token

    toks = []
    changed = False
    for tok in _tokenize_simple(text):
        new_tok = convert_token(tok)
        toks.append(new_tok)
        changed = changed or (new_tok != tok)

    if changed:
        flags.append("num2words_auto")
        notes.append("Converted simple numeric tokens to words.")

    return " ".join(toks), flags, notes

# ---------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------

def first_pass_transform_utterance(raw_text: str) -> TransformResult:
    """
    Conservative first-pass automation for legacy utterances.

    Returns both transformed text and a review trail.
    """
    if raw_text is None:
        raw_text = ""

    text = str(raw_text)
    flags: list[str] = []
    notes: list[str] = []

    steps = [
        convert_nonverbals,
        convert_corrections,
        regularize_unintelligibles,
        convert_ipa_slashes,
        convert_phonological_fragments,
        autotag_disfluencies,
        regularize_disfluency_case,
        ensure_terminal_punct,
    ]

    for func in steps:
        text, step_flags, step_notes = func(text)
        flags.extend(step_flags)
        notes.extend(step_notes)

    text = _normalize_whitespace(text)

    return TransformResult(
        raw=str(raw_text),
        transformed=text,
        flags=flags,
        notes=notes,
    )
