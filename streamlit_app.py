import streamlit as st
import yaml
import os
import tempfile
from src.rascal.main import (
    run_read_tiers, run_read_cha_files,
    run_select_transcription_reliability_samples, run_prepare_utterance_dfs,
    run_make_CU_coding_files, run_analyze_transcription_reliability,
    run_analyze_CU_reliability, run_analyze_CU_coding,
    run_make_word_count_files, run_make_timesheets,
    run_analyze_word_count_reliability, run_unblind_CUs,
    run_run_corelex
)

st.title("RASCAL Web App")

# Upload config
config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])

# Upload .cha files
cha_files = st.file_uploader("Upload .cha files", type=["cha"], accept_multiple_files=True)

if config_file and cha_files:
    config = yaml.safe_load(config_file)
    input_dir = tempfile.mkdtemp()
    output_dir = os.path.abspath(config.get("output_dir", "data/output"))
    os.makedirs(output_dir, exist_ok=True)

    for file in cha_files:
        with open(os.path.join(input_dir, file.name), "wb") as f:
            f.write(file.read())

    config["input_dir"] = input_dir
    config["output_dir"] = output_dir

    # Process tiers
    tiers = run_read_tiers(config.get("tiers", {}))
    frac = config.get("reliability_fraction", 0.2)
    coders = config.get("coders", [])

    # Step selection
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

    # Run selected steps
    if st.button("Run selected steps"):
        chats = run_read_cha_files(input_dir)
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

        st.success("Steps completed! Check the output directory.")
