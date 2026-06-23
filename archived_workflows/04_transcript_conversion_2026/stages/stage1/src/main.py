from pathlib import Path
from datetime import datetime
from sample_sheets import prep_sample_sheets
from proto_utterances import prep_utterance_sheets


# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------

start_time = datetime.now()
timestamp = start_time.strftime("%y%m%d_%H%M")

CWD = Path.cwd()
IN_DIR = CWD.parent / "stage1_in" / "cu_files_modified_260330"
OUT_DIR = CWD.parent / "stage1_out" / f"proto_transcript_tables_{timestamp}"
OUT_DIR.mkdir(exist_ok=True)

C2_META = IN_DIR / "metadata" / "c2_metadata.xlsx"
C3_META = IN_DIR / "metadata" / "c3_metadata.xlsx"

C2_UTTS = IN_DIR / "c2_coding_files"
C3_UTTS = IN_DIR / "c3_coding_files"

# ---------------------------------------------------------------------
# Task orchestrator
# ---------------------------------------------------------------------

def prep_proto_transcript_tables():

    prep_sample_sheets(C2_META, OUT_DIR, "2")
    prep_sample_sheets(C3_META, OUT_DIR, "3")

    prep_utterance_sheets(C2_UTTS, OUT_DIR, "2")
    prep_utterance_sheets(C3_UTTS, OUT_DIR, "3")


if __name__ == "__main__":
    prep_proto_transcript_tables()
