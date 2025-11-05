import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rascal.utils.logger import logger, _rel


def select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    """
    Select random CHAT files for transcription reliability and save both sampled
    and full sets. Each partition (if defined by tier.partition=True) produces:
      - Blank `_reliability.cha` files with headers only.
      - Excel file with two sheets: 'reliability_selection' and 'all_transcripts'.

    Steps:
      1. Group CHAT files by partition tiers (or single group if none).
      2. Randomly select a fraction (≥1) per group.
      3. Save blank headers and Excel summaries to
         `output_dir/transcription_reliability/<partition>/`.

    Parameters
    ----------
    tiers : dict[str, Tier]
        Tier objects with `.match()` and `.partition` attributes.
    chats : dict[str, ChatFile]
        Mapping of paths to parsed CHAT objects (with `.to_strs()`).
    frac : float
        Fraction of files to select (0 < frac ≤ 1).
    output_dir : Path
        Directory where reliability files and Excel outputs are written.
    """
    logger.info("Starting transcription reliability sample selection.")
    has_partition = any(t.partition for t in tiers.values())
    partitions = {}

    for cha_file in chats:
        if has_partition:
            keys = [t.match(cha_file) for t in tiers.values() if t.partition]
            keys = [k for k in keys if k is not None]
            if not keys:
                logger.warning(f"No partition tiers matched for '{_rel(cha_file)}', skipping.")
                continue
            key = tuple(keys)
        else:
            key = tuple()
        partitions.setdefault(key, []).append(cha_file)

    transc_rel_dir = Path(output_dir) / "transcription_reliability"
    transc_rel_dir.mkdir(parents=True, exist_ok=True)
    columns = ["file"] + list(tiers.keys())

    for partition_tiers, cha_files in tqdm(partitions.items(), desc="Selecting reliability subsets"):
        rows_all, rows_subset = [], []
        partition_path = Path(transc_rel_dir, *partition_tiers) if partition_tiers else transc_rel_dir
        partition_path.mkdir(parents=True, exist_ok=True)

        subset_size = max(1, round(frac * len(cha_files)))
        subset = random.sample(cha_files, k=subset_size)
        logger.info(f"Selected {subset_size} files for partition {partition_tiers or 'root'}.")

        for cha_file in cha_files:
            labels = [t.match(cha_file) for t in tiers.values()]
            row = [cha_file] + labels
            rows_all.append(row)
            if cha_file not in subset:
                continue

            rows_subset.append(row)
            try:
                chat_data = chats[cha_file]
                strs = next(chat_data.to_strs())
                strs = ["@Begin"] + strs.split("\n") + ["@End"]
                filepath = partition_path / (Path(cha_file).stem + "_reliability.cha")
                with filepath.open("w") as f:
                    for line in strs:
                        if line.startswith("@"):
                            f.write(line + "\n")
                logger.info(f"Written blank CHAT header: {_rel(filepath)}")
            except Exception as e:
                logger.error(f"Failed to write blank CHAT for {_rel(cha_file)}: {e}")

        try:
            df_all = pd.DataFrame(rows_all, columns=columns)
            df_subset = pd.DataFrame(rows_subset, columns=columns)
            suffix = "_".join(partition_tiers) if partition_tiers else "transcription_reliability_samples"
            df_filepath = partition_path / f"{suffix}.xlsx"
            with pd.ExcelWriter(df_filepath) as writer:
                df_subset.to_excel(writer, sheet_name="reliability_selection", index=False)
                df_all.to_excel(writer, sheet_name="all_transcripts", index=False)
            logger.info(f"Reliability Excel saved to: {_rel(df_filepath)}")
        except Exception as e:
            logger.error(f"Failed to write Excel for partition {partition_tiers}: {e}")


def reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    """
    Reselect new transcription reliability samples excluding prior ones.

    Steps:
      - Locate `*transcription_reliability_samples.xlsx` files in input_dir.
      - For each file, reload sheets, exclude prior selections,
        and draw n = max(1, round(frac * len(all_transcripts))) from remaining.
      - Save to `output_dir/reselected_transcription_reliability/`.

    Parameters
    ----------
    input_dir : Path
        Directory with existing reliability Excel files.
    output_dir : Path
        Destination for reselected outputs.
    frac : float
        Fraction of files to select (0 < frac ≤ 1).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    reselect_dir = output_dir / "reselected_transcription_reliability"
    reselect_dir.mkdir(parents=True, exist_ok=True)

    transc_sel_files = list(input_dir.rglob("*transcription_reliability_samples.xlsx"))
    if not transc_sel_files:
        logger.warning(f"No reliability transcription files found in {_rel(input_dir)}")
        return

    for filepath in transc_sel_files:
        try:
            xls = pd.ExcelFile(filepath)
            if not {"all_transcripts", "reliability_selection"} <= set(xls.sheet_names):
                logger.warning(f"Skipping {_rel(filepath)}: missing sheets.")
                continue

            df_all = pd.read_excel(filepath, sheet_name="all_transcripts")
            df_rel = pd.read_excel(filepath, sheet_name="reliability_selection")
            used_files = set(df_rel["file"])
            candidates = df_all[~df_all["file"].isin(used_files)]
            if candidates.empty:
                logger.info(f"No remaining candidates in {_rel(filepath)}, skipping.")
                continue

            n_samples = max(1, round(frac * len(df_all)))
            n_samples = min(n_samples, len(candidates))
            sample_df = candidates.sample(n=n_samples)
            outpath = reselect_dir / f"reselected_{filepath.name}"
            sample_df.to_excel(outpath, index=False, sheet_name="reselected_reliability")
            logger.info(f"Reselected {n_samples} files → {_rel(outpath)}")
        except Exception as e:
            logger.error(f"Failed to reselect samples for {_rel(filepath)}: {e}")
