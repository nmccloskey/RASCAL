import streamlit as st
import yaml
import os
import tempfile
import zipfile
from io import BytesIO
from config_builder import build_config_ui
from datetime import datetime

def add_src_to_sys_path():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
add_src_to_sys_path()

from rascal.main import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples, run_prepare_utterance_dfs,
    run_make_CU_coding_files, run_analyze_transcription_reliability,
    run_analyze_CU_reliability, run_analyze_CU_coding,
    run_make_word_count_files, run_make_timesheets,
    run_analyze_word_count_reliability, run_unblind_CUs,
    run_run_corelex, run_reselect_CU_reliability,
    run_reselect_transcription_reliability_samples,
    run_reselect_WC_reliability
)

st.title("RASCAL Web App")

if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False

def zip_folder(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
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

# Upload .cha files
cha_files = st.file_uploader("Upload input files", type=["cha", ".xlsx"], accept_multiple_files=True)

if (config_file or st.session_state.confirmed_config) and cha_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        config["input_dir"] = input_dir
        config["output_dir"] = output_dir

        # Save uploaded .cha files
        for file in cha_files:
            with open(os.path.join(input_dir, file.name), "wb") as f:
                f.write(file.read())
        # Read config values
        tiers = run_read_tiers(config.get("tiers", {}))
        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", [])
        CU_paradigms = config.get("CU_paradigms", []) or []
        blind_columns = config.get("blind_columns", [])

        exclude_participants = config.get('exclude_participants', [])
        strip_clan = config.get('strip_clan', True)
        prefer_correction = config.get('prefer_correction', True)
        lowercase = config.get('lowercase', True)

        # --- List all functions a–n ---
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
            if any(f[0] in ['a', 'd'] for f in selected_funcs):
                chats = run_read_cha_files(input_dir)
            else:
                chats = None

            for func in selected_funcs:
                if func.startswith("a."):
                    run_select_transcription_reliability_samples(tiers, chats, frac, output_dir)
                elif func.startswith("b."):
                    run_analyze_transcription_reliability(tiers, input_dir, output_dir,
                                                         exclude_participants, strip_clan,
                                                         prefer_correction, lowercase)
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

            # --- Timestamped ZIP filename ---
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            func_str = ''.join([f[0] for f in selected_funcs])
            zip_buffer = zip_folder(output_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name=f"rascal_{func_str}_output_{timestamp}.zip",
                mime="application/zip"
            )


def main():
    import subprocess
    import sys
    # Launch this file with streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
