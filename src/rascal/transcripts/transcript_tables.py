from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from rascal.utils.logger import logger, _rel


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
            logger.error(f"Partitioning failed for {_rel(chat_file)}: {e}")
    logger.info(f"Identified {len(cha_chunks)} partition groups.")
    return cha_chunks


def _count_utterances_in_chunk(chats: Dict[str, object], file_list: List[str]) -> int:
    """
    Count total utterances across all files in a partition chunk.
    Used to set a consistent utterance_id padding width per output file.
    """
    total = 0
    for chat_file in file_list:
        try:
            chat_data = chats[chat_file]
            utterances = getattr(chat_data, "utterances", lambda: [])()
            total += len(utterances)
        except Exception as e:
            logger.warning(f"Could not count utterances in {_rel(chat_file)}: {e}")
    return total


def make_transcript_tables(
    tiers: Dict[str, object],
    chats: Dict[str, object],
    output_dir: Path,
    *,
    shuffle: bool = False,
    random_seed: int | None = 99,
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
    shuffle: bool
        Disrupt automated file order when assigning sample identifiers.

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

    sample_cols = ["sample_id", "file", "input_order", "shuffled_order"] + list(tiers.keys())
    utt_cols = [
        "sample_id",
        "utterance_id",
        "position",
        "position_sub",
        "speaker",
        "utterance",
        "comment",
    ]

    rng = np.random.default_rng(random_seed) if shuffle else None

    for chunk_str, file_list in tqdm(cha_chunks.items(), desc="Building transcript tables"):
        if not file_list:
            logger.warning(f"Partition '{chunk_str}' has no files; skipping.")
            continue

        # Deterministic base ordering by filename for readability + reproducibility.
        file_list_sorted = sorted(file_list)

        # Build a shuffled order (only used to assign sample_ids), but keep output rows file-sorted.
        if shuffle:
            shuffled = file_list_sorted.copy()
            rng.shuffle(shuffled)
            file_to_shuffled_order = {f: i + 1 for i, f in enumerate(shuffled)}
            logger.info(
                f"Shuffling enabled for partition '{chunk_str}' (seed={random_seed})."
            )
        else:
            file_to_shuffled_order = {}

        # sample_id padding determined by number of samples in this output file.
        s_pad = zero_pad(len(file_list_sorted), 3)

        # utterance_id padding determined once per output file (concatenated utterances).
        total_utts = _count_utterances_in_chunk(chats, file_list_sorted)
        u_pad = zero_pad(total_utts, 4)

        partition_str = f"{chunk_str}_" if chunk_str != "NO_PARTITION_LABELS" else ""
        sample_rows: List[list] = []
        utt_rows: List[list] = []

        # Assign sample_ids:
        # - if shuffle=True, sample_id reflects shuffled order
        # - otherwise sample_id reflects file_list_sorted order
        if shuffle:
            # sample_id index is the shuffled_order
            file_to_sample_id = {
                f: f"S{file_to_shuffled_order[f]:0{s_pad}d}" for f in file_list_sorted
            }
        else:
            file_to_sample_id = {
                f: f"S{i + 1:0{s_pad}d}" for i, f in enumerate(file_list_sorted)
            }

        for input_idx, chat_file in enumerate(file_list_sorted, start=1):
            try:
                labels_all = [t.match(chat_file) for t in tiers.values()]
                sample_id = file_to_sample_id[chat_file]
                shuffled_order = file_to_shuffled_order.get(chat_file, np.nan)

                sample_rows.append(
                    [sample_id, chat_file, input_idx, shuffled_order] + labels_all
                )

                chat_data = chats[chat_file]
                utterances = getattr(chat_data, "utterances", lambda: [])()

                # position resets per sample (1..n). position_sub starts at 0.
                for j, line in enumerate(utterances, start=1):
                    speaker = getattr(line, "participant", None)
                    tiers_map = getattr(line, "tiers", {}) or {}
                    utterance_text = tiers_map.get(speaker, "")
                    comment = tiers_map.get("%com", None)

                    utt_id = f"U{j:0{u_pad}d}"
                    position = j
                    position_sub = 0

                    utt_rows.append(
                        [
                            sample_id,
                            utt_id,
                            position,
                            position_sub,
                            speaker,
                            utterance_text,
                            comment,
                        ]
                    )
            except Exception as e:
                logger.error(f"Error processing {_rel(chat_file)}: {e}")
                continue

        sample_df = pd.DataFrame(sample_rows, columns=sample_cols)
        sample_df["speaking_time"] = np.nan  # preserve your existing placeholder
        utt_df = pd.DataFrame(utt_rows, columns=utt_cols)

        # Write artifacts
        filepath = transcript_dir.joinpath(*partition_str.strip("_").split("_"))
        filepath.mkdir(parents=True, exist_ok=True)
        filename = filepath / f"{partition_str}transcript_tables.xlsx"

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                sample_df.to_excel(writer, sheet_name="samples", index=False)
                utt_df.to_excel(writer, sheet_name="utterances", index=False)
            written.append(str(filename))
            logger.info(f"Wrote transcript table: {_rel(filename)}")
        except Exception as e:
            logger.error(f"Failed to write {_rel(filename)}: {e}")

    logger.info(f"Successfully wrote {len(written)} transcript table(s).")
    return written
