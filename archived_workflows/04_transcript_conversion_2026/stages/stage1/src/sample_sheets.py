"""
env: c2c3tt

Convert old CU-coding metadata sheets into DIAAD-style transcript-table sample sheets.

Expected project structure
--------------------------
OneDrive - Temple University/SLABlab/Projects/c2c3_transcript_conversion/
│   sample_sheets.py
│
└───original_cu_files_copied_2603111/
    ├───c2_coding_files/
    │       AC_c2_utterances.xlsx
    │       BU_c2_utterances.xlsx
    │       TU_c2_utterances.xlsx
    │
    ├───c3_coding_files/
    │       AC_c3_utterances.xlsx
    │       BU_c3_utterances.xlsx
    │       TU_c3_utterances.xlsx
    │
    └───metadata/
            c2_metadata.xlsx
            c3_metadata.xlsx
"""

from pathlib import Path
from helpers import require_columns

import pandas as pd

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

TEST_ID_MAP = {
    1: "Pre",
    2: "Post",
    3: "Maint",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _format_site_code(site_value) -> str:
    """Normalize site code to a stripped string."""
    return str(site_value).strip()

def _format_study_id(site: str, participant_no) -> str:
    """Build DIAAD-style study_id: SITE + 2-digit participant number."""
    return f"{site}{int(participant_no):02d}"

def _format_sample_id(site: str, sample_no) -> str:
    """Build DIAAD-style sample_id: SITE + 3-digit sample number."""
    return f"{site}{int(sample_no):03d}"

# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def prep_sample_sheets(meta_path: Path, out_dir: Path, cycle: str) -> None:
    """
    Partition metadata by site and write one transcript-table workbook per site.

    Output:
        proto_transcript_tables/
            cycle2/
                AC_transcript_tables.xlsx
                BU_transcript_tables.xlsx
                TU_transcript_tables.xlsx

    Rules:
    - study_id = site + participant_no (zero-padded to 2 digits)
    - sample_id = site + sample_no (zero-padded to 3 digits)
    - test numeric codes are mapped as:
        1 -> Pre
        2 -> Post
        3 -> Maint
    """
    meta_df = pd.read_excel(meta_path)

    required_cols = ["site", "sample_no", "participant_no", "test"]
    require_columns(meta_df, required_cols, meta_path)

    cycle_out_dir = out_dir / f"cycle{cycle}"
    cycle_out_dir.mkdir(parents=True, exist_ok=True)

    for site, site_df in meta_df.groupby("site", dropna=False):
        site = _format_site_code(site)
        site_df = site_df.copy()

        site_df["sample_id"] = site_df["sample_no"].apply(
            lambda x: _format_sample_id(site, x)
        )
        site_df["study_id"] = site_df["participant_no"].apply(
            lambda x: _format_study_id(site, x)
        )
        site_df["test"] = site_df["test_id"].map(TEST_ID_MAP)

        site_df = site_df.loc[:, ~site_df.columns.str.contains('^Unnamed')]

        out_file = cycle_out_dir / f"{site}_transcript_tables.xlsx"
        site_df.to_excel(out_file, sheet_name="samples", index=False)

        print(f"Wrote: {out_file}")
