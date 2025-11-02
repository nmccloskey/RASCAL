from __future__ import annotations
import logging
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np


def zero_pad(num: int, lower_bound: int = 3) -> int:
    """
    Determine adaptive zero-padding width for numeric identifiers.

    Parameters
    ----------
    num : int
        Maximum number expected in the sequence.
    lower_bound : int, default=3
        Minimum padding width.

    Returns
    -------
    int
        Padding width ensuring consistent formatting.
    """
    width = max(lower_bound, len(str(max(num, 1))))
    return width


def partition_cha(chats: Dict[str, object], tiers: Dict[str, object]) -> Dict[str, List[str]]:
    """
    Partition CHAT files according to tier-defined partition labels.

    Parameters
    ----------
    chats : dict
        Mapping of file names to pylangacq.Reader objects.
    tiers : dict
        Tier definitions, each possibly specifying a 'partition' attribute.

    Returns
    -------
    dict
        Mapping of partition label strings to lists of CHAT filenames.
    """
    cha_chunks: Dict[str, List[str]] = {}
    for chat_file in sorted(chats.keys()):
        try:
            partition_labels = [
                t.match(chat_file)
                for t in tiers.values()
                if getattr(t, "partition", False)
            ] or ["NO_PARTITION_LABELS"]
            chunk_str = "_".join(map(str, partition_labels))
            cha_chunks.setdefault(chunk_str, []).append(chat_file)
        except Exception as e:
            logging.error(f"Partitioning failed for {chat_file}: {e}")
    logging.info(f"Identified {len(cha_chunks)} partition groups.")
    return cha_chunks


def make_transcript_tables(
    tiers: Dict[str, object],
    chats: Dict[str, object],
    output_dir: Path,
) -> List[str]:
    """
    Create and write transcript tables (samples + utterances) to Excel.

    Parameters
    ----------
    tiers : dict
        Tier objects defining matching and partition attributes.
    chats : dict
        CHAT file readers indexed by filename.
    output_dir : Path
        Directory to create a 'transcript_tables' subfolder within.

    Returns
    -------
    None
        All artifacts are saved to disk; the function does not return a value.

    Notes
    -----
    Each Excel file contains:
      • Sheet 'samples'  — sample-level metadata and file info
      • Sheet 'utterances' — utterance-level text data
    """
    transcript_dir = output_dir / "transcript_tables"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    cha_chunks = partition_cha(chats, tiers)
    written: List[str] = []

    sample_cols = ["sample_id", "file"] + list(tiers.keys())
    utt_cols = ["sample_id", "utterance_id", "speaker", "utterance", "comment"]

    for chunk_str, file_list in tqdm(cha_chunks.items(), desc="Building transcript tables"):
        if not file_list:
            logging.warning(f"Partition '{chunk_str}' has no files; skipping.")
            continue

        sample_rows, utt_rows = [], []
        s_pad = zero_pad(len(file_list), 3)
        partition_str = f"{chunk_str}_" if chunk_str != "NO_PARTITION_LABELS" else ""

        for i, chat_file in enumerate(sorted(file_list)):
            try:
                labels_all = [t.match(chat_file) for t in tiers.values()]
                sample_id = f"S{i+1:0{s_pad}d}"
                sample_rows.append([sample_id, chat_file] + labels_all)

                chat_data = chats[chat_file]
                utterances = getattr(chat_data, "utterances", lambda: [])()
                u_pad = zero_pad(len(utterances), 4)

                for j, line in enumerate(utterances):
                    speaker = getattr(line, "participant", None)
                    tiers_map = getattr(line, "tiers", {})
                    utterance = tiers_map.get(speaker, "")
                    comment = tiers_map.get("%com", None)
                    utt_id = f"U{j+1:0{u_pad}d}"
                    utt_rows.append([sample_id, utt_id, speaker, utterance, comment])
            except Exception as e:
                logging.error(f"Error processing {chat_file}: {e}")
                continue

        sample_df = pd.DataFrame(sample_rows, columns=sample_cols)
        sample_df["speaking_time"] = np.nan
        utt_df = pd.DataFrame(utt_rows, columns=utt_cols)

        filepath = transcript_dir.joinpath(*partition_str.strip("_").split("_"))
        filepath.mkdir(parents=True, exist_ok=True)
        filename = filepath / f"{partition_str}transcript_tables.xlsx"

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                sample_df.to_excel(writer, sheet_name="samples", index=False)
                utt_df.to_excel(writer, sheet_name="utterances", index=False)
            written.append(str(filename))
            logging.info(f"Wrote transcript table: {filename}")
        except Exception as e:
            logging.error(f"Failed to write {filename}: {e}")

    logging.info(f"Successfully wrote {len(written)} transcript table(s).")
