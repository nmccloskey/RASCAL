import streamlit as st
import yaml
import os
import sys
import tempfile
import zipfile
from io import BytesIO
from webapp.config_builder import build_config_ui

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
    run_run_corelex
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

st.header("Step 1: Provide config and input files")

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

# Upload .cha files
cha_files = st.file_uploader("Upload .cha files", type=["cha"], accept_multiple_files=True)

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

        steps = st.multiselect("Select RASCAL steps to run", [
            "a. Select transcription reliability samples",
            "b. Prepare utterance tables",
            "c. Make CU coding files",
            "d. Analyze transcription reliability",
            "e. Analyze CU reliability",
            "f. Analyze CU coding",
            "g. Make word count files",
            "h. Make timesheets",
            "i. Analyze word count reliability",
            "j. Unblind CU samples",
            "k. Run CoreLex"
        ])

        if st.button("Run selected steps"):
            if "a. Select transcription reliability samples" in steps or \
               "b. Prepare utterance tables" in steps:
                chats = run_read_cha_files(input_dir)
            else:
                chats = None

            if "a. Select transcription reliability samples" in steps:
                run_select_transcription_reliability_samples(tiers, chats, frac, output_dir)
            if "b. Prepare utterance tables" in steps:
                run_prepare_utterance_dfs(tiers, chats, output_dir)
            if "c. Make CU coding files" in steps:
                run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir)
            if "d. Analyze transcription reliability" in steps:
                run_analyze_transcription_reliability(tiers, input_dir, output_dir)
            if "e. Analyze CU reliability" in steps:
                run_analyze_CU_reliability(tiers, input_dir, output_dir)
            if "f. Analyze CU coding" in steps:
                run_analyze_CU_coding(tiers, input_dir, output_dir)
            if "g. Make word count files" in steps:
                run_make_word_count_files(tiers, frac, coders, output_dir)
            if "h. Make timesheets" in steps:
                run_make_timesheets(tiers, input_dir, output_dir)
            if "i. Analyze word count reliability" in steps:
                run_analyze_word_count_reliability(tiers, input_dir, output_dir)
            if "j. Unblind CU samples" in steps:
                run_unblind_CUs(tiers, input_dir, output_dir)
            if "k. Run CoreLex" in steps:
                run_run_corelex(input_dir, output_dir)

            st.success("Steps completed!")

            # Create download link
            zip_buffer = zip_folder(output_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name="rascal_output.zip",
                mime="application/zip"
            )
