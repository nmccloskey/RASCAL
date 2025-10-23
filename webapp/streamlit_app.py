import yaml
import zipfile
import tempfile
from io import BytesIO
import streamlit as st
from pathlib import Path
from datetime import datetime
from config_builder import build_config_ui


def add_src_to_sys_path():
    import sys
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_path))

add_src_to_sys_path()

from rascal.utils.support_funcs import as_path
from rascal.run_wrappers import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples,
    run_reselect_transcription_reliability_samples,
    run_analyze_transcription_reliability,
    run_prepare_utterance_dfs, run_make_CU_coding_files,
    run_make_timesheets, run_analyze_CU_reliability,
    run_analyze_CU_coding, run_reselect_CU_reliability,
    run_make_word_count_files, run_analyze_word_count_reliability,
    run_reselect_WC_reliability, run_unblind_CUs, run_run_corelex
)

st.title("RASCAL Web App")
st.caption("Resources for Analyzing Speech in Clinical Aphasiology Labs")

if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False


# --- Utility: Zip entire output folder ---
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


st.header("Part 1: Create or upload config file")

# Upload config or build it
config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])
config = None

if config_file:
    st.session_state.confirmed_config = False  # reset if new file uploaded
    config = yaml.safe_load(config_file)
    st.success("✅ Config file uploaded")
else:
    with st.expander("No config uploaded? Build one here"):
        config = build_config_ui()
        if st.button("✅ Use this built config"):
            st.session_state.confirmed_config = True
            st.success("Built config confirmed.")

st.header("Part 2: Upload input files")

# Upload .cha or .xlsx files
cha_files = st.file_uploader("Upload input files", type=["cha", ".xlsx"], accept_multiple_files=True)

if (config_file or st.session_state.confirmed_config) and cha_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = as_path(tmpdir)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Save uploaded input files
        for file in cha_files:
            file_path = input_dir / file.name
            with file_path.open("wb") as f:
                f.write(file.read())

        # Read config values
        tiers = run_read_tiers(config.get("tiers", {})) or {}
        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", []) or []
        CU_paradigms = config.get("CU_paradigms", []) or []
        exclude_participants = config.get("exclude_participants", []) or []
        strip_clan = config.get("strip_clan", True)
        prefer_correction = config.get("prefer_correction", True)
        lowercase = config.get("lowercase", True)

        # --- Available functions ---
        all_functions = [
            "a. Select transcription reliability samples",
            "b. Analyze transcription reliability",
            "c. Reselect transcription reliability samples",
            "d. Prepare utterance tables",
            "e. Make CU coding files",
            "f. Make timesheets",
            "g. Analyze CU reliability",
            "h. Reselect CU reliability samples",
            "i. Analyze CU coding",
            "j. Make word count files",
            "k. Analyze word count reliability",
            "l. Reselect WC reliability samples",
            "m. Unblind CU samples",
            "n. Run CoreLex"
        ]

        st.header("Part 3: Select functions to run")
        selected_funcs = st.multiselect("Select functions", all_functions)

        if st.button("Run selected functions"):
            # Only read chats if needed
            if any(f[0] in ["a", "d"] for f in selected_funcs):
                chats = run_read_cha_files(input_dir)
            else:
                chats = None

            for func in selected_funcs:
                if func.startswith("a."):
                    run_select_transcription_reliability_samples(tiers, chats, frac, output_dir)
                elif func.startswith("b."):
                    run_analyze_transcription_reliability(
                        tiers, input_dir, output_dir,
                        exclude_participants, strip_clan, prefer_correction, lowercase
                    )
                elif func.startswith("c."):
                    run_reselect_transcription_reliability_samples(input_dir, output_dir, frac)
                elif func.startswith("d."):
                    run_prepare_utterance_dfs(tiers, chats, output_dir)
                elif func.startswith("e."):
                    run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir,
                                             CU_paradigms, exclude_participants)
                elif func.startswith("f."):
                    run_make_timesheets(tiers, input_dir, output_dir)
                elif func.startswith("g."):
                    run_analyze_CU_reliability(tiers, input_dir, output_dir, CU_paradigms)
                elif func.startswith("h."):
                    run_reselect_CU_reliability(tiers, input_dir, output_dir, "CU", frac)
                elif func.startswith("i."):
                    run_analyze_CU_coding(tiers, input_dir, output_dir, CU_paradigms)
                elif func.startswith("j."):
                    run_make_word_count_files(tiers, frac, coders, input_dir, output_dir)
                elif func.startswith("k."):
                    run_analyze_word_count_reliability(tiers, input_dir, output_dir)
                elif func.startswith("l."):
                    run_reselect_WC_reliability(tiers, input_dir, output_dir, "WC", frac)
                elif func.startswith("m."):
                    run_unblind_CUs(tiers, input_dir, output_dir)
                elif func.startswith("n."):
                    run_run_corelex(input_dir, output_dir, exclude_participants)

            st.success("Functions completed!")

            # --- Create timestamped ZIP for download ---
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            func_str = "".join([f[0] for f in selected_funcs])
            zip_buffer = zip_folder(output_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name=f"rascal_{func_str}_output_{timestamp}.zip",
                mime="application/zip"
            )


def main():
    """Launch this file as a Streamlit app."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
