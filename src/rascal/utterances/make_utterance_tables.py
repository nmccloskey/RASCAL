from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


# Expected Tier API:
#   .name: str
#   .partition: bool
#   .match(filename: str) -> str
#
# Expected Chat API (from pylangacq.read_chat):
#   .utterances() -> iterable of Utterance, each with:
#       .participant: str
#       .tiers: Dict[str, str]  (e.g., {"PAR": "...", "%com": "..."})


def _build_utterance_df(
    tiers: Dict[str, object],
    chats: Dict[str, object],
) -> Tuple[pd.DataFrame, List[str]]:
    """Pure: build the utterances dataframe and return partition tier names."""
    dfcols = ["file"] + list(tiers.keys()) + ["sample_id", "speaker", "utterance", "comment"]
    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]
    rows: List[List[str]] = []

    # Deterministic file order for stable outputs/tests
    for i, chat_file in enumerate(sorted(chats.keys())):
        labels_all = [t.match(chat_file) for t in tiers.values()]
        partition_labels = [t.match(chat_file) for t in tiers.values() if getattr(t, "partition", False)]
        sample_id = f"{''.join(partition_labels)}S{i}"

        chat_data = chats[chat_file]
        for line in chat_data.utterances():
            speaker = line.participant
            utterance = line.tiers.get(speaker, "")
            comment = line.tiers.get("%com", None)
            if speaker not in line.tiers:
                logging.warning(
                    f"Speaker '{speaker}' not found in tiers for file '{chat_file}'."
                )
            rows.append([chat_file] + labels_all + [sample_id, speaker, utterance, comment])

    df = pd.DataFrame(rows, columns=dfcols)
    return df, partition_tiers


def _write_utterance_tables(
    df: pd.DataFrame,
    utterance_dir: str,
    partition_tiers: List[str],
) -> List[str]:
    """Side effects only: writes Excel files; returns list of filepaths written."""
    written: List[str] = []
    os.makedirs(utterance_dir, exist_ok=True)
    logging.info(f"Created directory: {utterance_dir}")

    if not partition_tiers:
        df = df.copy()
        df.insert(0, "utterance_id", "U" + df.reset_index(drop=True).index.astype(str))
        filename = os.path.join(utterance_dir, "Utterances.xlsx")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_excel(filename, index=False)
        written.append(filename)
        return written

    # Partitioned files
    for tup, subdf in tqdm(df.groupby(partition_tiers, sort=False), desc="Writing utterance files"):
        if not isinstance(tup, tuple):
            tup = (tup,)
        subdf = subdf.copy()
        subdf.insert(
            0,
            "utterance_id",
            "".join(map(str, tup)) + "U" + subdf.reset_index(drop=True).index.astype(str),
        )

        # Nest only if >1 partition tier
        output_path_parts = [utterance_dir] + (list(map(str, tup)) if len(partition_tiers) > 1 else [])
        os.makedirs(os.path.join(*output_path_parts), exist_ok=True)
        filename = os.path.join(*output_path_parts, "_".join(map(str, tup)) + "_Utterances.xlsx")
        subdf.to_excel(filename, index=False)
        written.append(filename)

    return written


def prepare_utterance_dfs(
    tiers: Dict[str, object],
    chats: Dict[str, object],
    output_dir: str,
) -> List[str]:
    """
    Process CHAT files and create DataFrames of utterances with blind codes.
    Writes Excel files under {output_dir}/Utterances by partition and
    returns the list of file paths written.
    """
    utterance_dir = os.path.join(output_dir, "Utterances")
    df, partition_tiers = _build_utterance_df(tiers, chats)
    return _write_utterance_tables(df, utterance_dir, partition_tiers)
