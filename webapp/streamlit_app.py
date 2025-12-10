import yaml
import zipfile
import tempfile
from io import BytesIO
import streamlit as st
from pathlib import Path
import random, numpy as np
from datetime import datetime
from config_builder import build_config_ui

start_time = datetime.now()

# ------------------------------------------------------------------
# Path setup and zip utility
# ------------------------------------------------------------------
def add_src_to_sys_path():
    import sys
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_path))
add_src_to_sys_path()

def zip_folder(folder_path: Path) -> BytesIO:
    """Compress the given folder into an in-memory ZIP buffer."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


# ------------------------------------------------------------------
# RASCAL imports
# ------------------------------------------------------------------
from rascal.utils.auxiliary import find_files
from rascal.utils.logger import (
    logger,
    initialize_logger,
    terminate_logger,
    early_log,
)
from rascal.run_wrappers import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples,
    run_reselect_transcription_reliability_samples,
    run_evaluate_transcription_reliability,
    run_make_transcript_tables, run_make_cu_coding_files,
    run_evaluate_cu_reliability,
    run_analyze_cu_coding, run_reselect_cu_reliability,
    run_make_word_count_files, run_evaluate_word_count_reliability,
    run_reselect_wc_reliability, run_summarize_cus, run_run_corelex
)
from rascal import __version__


# ------------------------------------------------------------------
# Streamlit header
# ------------------------------------------------------------------
st.title("RASCAL Web App")
st.subheader("Resources for Analyzing Speech in Clinical Aphasiology Labs")

# ------------------------------------------------------------------
# Instruction manual toggle
# ------------------------------------------------------------------
manual_path = Path(__file__).resolve().parent.parent / "manual" / "rascal_manual.md"
if manual_path.exists():
    with st.expander("📘 Show / Hide Instruction Manual"):
        st.markdown(manual_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------
# PART 1: CONFIG
# ---------------------------------------------------------------
if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False

st.header("Part 1: Create or upload config file")
config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])
config = None

if config_file:
    st.session_state.confirmed_config = False
    config = yaml.safe_load(config_file)
    st.success("✅ Config file uploaded")
else:
    with st.expander("No config uploaded? Build one here"):
        config = build_config_ui()
        if st.button("✅ Use this built config"):
            st.session_state.confirmed_config = True
            st.success("Built config confirmed.")


# ---------------------------------------------------------------
# PART 2: INPUT FILES
# ---------------------------------------------------------------
st.header("Part 2: Upload input files")
cha_files = st.file_uploader("Upload input files", type=["cha", "xlsx"], accept_multiple_files=True)

if (config_file or st.session_state.confirmed_config) and cha_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir).resolve()
        early_log("info", f"RASCAL web run initialized (temporary root: {root_dir})")

        input_dir = (root_dir / "input").resolve()
        output_dir = (root_dir / "output").resolve()
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Save uploaded input files
        for file in cha_files:
            file_path = input_dir / file.name
            with file_path.open("wb") as f:
                f.write(file.read())

        # Create timestamped output folder and logger
        timestamp = start_time.strftime("%y%m%d_%H%M")
        out_dir = (output_dir / f"rascal_output_{timestamp}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        initialize_logger(start_time, out_dir, program_name="RASCAL", version=__version__)

        # ------------------------------------------------------------------
        # Load config parameters
        # ------------------------------------------------------------------
        random_seed = config.get("random_seed", 8) or 8
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"Random seed set to {random_seed}")

        tiers = run_read_tiers(config.get("tiers", {})) or {}
        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", []) or []
        CU_paradigms = config.get("CU_paradigms", []) or []
        exclude_participants = config.get("exclude_participants", []) or []
        strip_clan = config.get("strip_clan", True)
        prefer_correction = config.get("prefer_correction", True)
        lowercase = config.get("lowercase", True)

        # ---------------------------------------------------------------
        # PART 3: FUNCTION SELECTION
        # ---------------------------------------------------------------
        st.header("Part 3: Select function(s) to run")
        all_functions = [
            "1a. Select transcription reliability samples",
            "3a. Evaluate transcription reliability",
            "3b. Reselect transcription reliability samples",
            "4a. Make transcript tables",
            "4b. Make CU coding & reliability files",
            "6a. Evaluate CU reliability",
            "6b. Reselect CU reliability samples",
            "7a. Analyze CU coding",
            "7b. Make word count files",
            "9a. Evaluate word count reliability",
            "9b. Reselect word count reliability samples",
            "10a. Summarize CU samples",
            "10b. Run CoreLex analysis"
        ]

        selected_funcs = st.multiselect("Select functions", all_functions)

        if st.button("Run selected functions"):
            if not selected_funcs:
                st.warning("Please select at least one function.")
                st.stop()
            logger.info(f"Running selected functions: {selected_funcs}")

            # --- Read .cha if needed ---
            chats = None
            if any(f.startswith(("1a", "4a")) for f in selected_funcs):
                chats = run_read_cha_files(input_dir)

            # --- Prepare utterance files if needed ---
            needs_utt = any(f.startswith(x) for x in ("4b", "7b", "10b") for f in selected_funcs)
            if needs_utt and not any(f.startswith("4a") for f in selected_funcs):
                transcript_tables = find_files(directories=[input_dir, out_dir],
                                               search_base="transcript_tables")
                if not transcript_tables:
                    logger.info("No utterance files detected — creating automatically.")
                    chats = chats or run_read_cha_files(input_dir)
                    run_make_transcript_tables(tiers, chats, out_dir)

            # --- Execute selected functions ---
            for func in selected_funcs:
                if func.startswith("1a."):
                    run_select_transcription_reliability_samples(tiers, chats, frac, out_dir)
                elif func.startswith("3a."):
                    run_evaluate_transcription_reliability(
                        tiers, input_dir, out_dir,
                        exclude_participants, strip_clan, prefer_correction, lowercase
                    )
                elif func.startswith("3b."):
                    run_reselect_transcription_reliability_samples(input_dir, out_dir, frac)
                elif func.startswith("4a."):
                    run_make_transcript_tables(tiers, chats, out_dir)
                elif func.startswith("4b."):
                    run_make_cu_coding_files(
                        tiers, frac, coders, input_dir, out_dir,
                        CU_paradigms, exclude_participants
                    )
                elif func.startswith("6a."):
                    run_evaluate_cu_reliability(tiers, input_dir, out_dir, CU_paradigms)
                elif func.startswith("6b."):
                    run_reselect_cu_reliability(tiers, input_dir, out_dir, "CU", frac)
                elif func.startswith("7a."):
                    run_analyze_cu_coding(tiers, input_dir, out_dir, CU_paradigms)
                elif func.startswith("7b."):
                    run_make_word_count_files(tiers, frac, coders, input_dir, out_dir)
                elif func.startswith("9a."):
                    run_evaluate_word_count_reliability(tiers, input_dir, out_dir)
                elif func.startswith("9b."):
                    run_reselect_wc_reliability(tiers, input_dir, out_dir, "WC", frac)
                elif func.startswith("10a."):
                    run_summarize_cus(tiers, input_dir, out_dir)
                elif func.startswith("10b."):
                    run_run_corelex(tiers, input_dir, out_dir, exclude_participants)

            st.success("✅ All selected functions completed successfully!")
            terminate_logger(
                input_dir=input_dir,
                output_dir=out_dir,
                config=config,
                start_time=start_time,
                program_name="RASCAL",
                version=__version__,
            )

            # --- Create timestamped ZIP for download ---
            timestamp = start_time.strftime("%y%m%d_%H%M")
            zip_buffer = zip_folder(out_dir)
            st.download_button(
                label="⬇️ Download Results ZIP",
                data=zip_buffer,
                file_name=f"rascal_web_output_{timestamp}.zip",
                mime="application/zip"
            )


def main():
    """Launch this file as a Streamlit app."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
