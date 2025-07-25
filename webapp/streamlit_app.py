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
    run_digital_convo_turns_analyzer
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

        step_mapping = {
            "Step 1 (abc)": ['a. Select transcription reliability samples',
                            'b. Prepare utterance tables',
                            'c. Make CU coding files'],
            "Step 3 (defgh)": ['d. Analyze transcription reliability',
                            'e. Analyze CU reliability',
                            'f. Analyze CU coding',
                            'g. Make word count files',
                            'h. Make timesheets'],
            "Step 5 (ijk)": ['i. Analyze word count reliability',
                            'j. Unblind CU samples',
                            'k. Run CoreLex']
        }

        all_functions = [
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
            "k. Run CoreLex",
            "l. Reselect CU reliability samples",
            "m. Analyze digital conversation turns"
        ]

        st.header("Part 3: Select steps or functions")

        # --- Dropdowns ---
        step = st.selectbox("Choose a predefined step", ["(None)"] + list(step_mapping.keys()))
        funcs = st.multiselect("Or individually select functions to run", all_functions)

        # --- Resolve full list of selected functions ---
        selected_funcs = funcs or (step_mapping[step] if step != "(None)" else [])

        if st.button("Run selected functions"):
            if "a. Select transcription reliability samples" in selected_funcs or \
            "b. Prepare utterance tables" in selected_funcs:
                chats = run_read_cha_files(input_dir)
            else:
                chats = None

            if "a. Select transcription reliability samples" in selected_funcs:
                run_select_transcription_reliability_samples(tiers, chats, frac, output_dir)
            if "b. Prepare utterance tables" in selected_funcs:
                run_prepare_utterance_dfs(tiers, chats, output_dir)
            if "c. Make CU coding files" in selected_funcs:
                run_make_CU_coding_files(tiers, frac, coders, input_dir, output_dir)
            if "d. Analyze transcription reliability" in selected_funcs:
                run_analyze_transcription_reliability(tiers, input_dir, output_dir)
            if "e. Analyze CU reliability" in selected_funcs:
                run_analyze_CU_reliability(tiers, input_dir, output_dir)
            if "f. Analyze CU coding" in selected_funcs:
                run_analyze_CU_coding(tiers, input_dir, output_dir)
            if "g. Make word count files" in selected_funcs:
                run_make_word_count_files(tiers, frac, coders, output_dir)
            if "h. Make timesheets" in selected_funcs:
                run_make_timesheets(tiers, input_dir, output_dir)
            if "i. Analyze word count reliability" in selected_funcs:
                run_analyze_word_count_reliability(tiers, input_dir, output_dir)
            if "j. Unblind CU samples" in selected_funcs:
                run_unblind_CUs(tiers, input_dir, output_dir)
            if "k. Run CoreLex" in selected_funcs:
                run_run_corelex(input_dir, output_dir)
            if "l. Reselect CU reliability samples" in selected_funcs:
                coder3 = coders[2] or '3'
                run_reselect_CU_reliability(input_dir, output_dir, coder3=coder3, frac=frac)
            if "m. Analyze digital conversation turns" in selected_funcs:
                run_digital_convo_turns_analyzer(input_dir, output_dir)

            st.success("Functions completed!")

            # --- Timestamped ZIP filename ---
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            zip_buffer = zip_folder(output_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name=f"rascal_output_{timestamp}.zip",
                mime="application/zip"
            )
