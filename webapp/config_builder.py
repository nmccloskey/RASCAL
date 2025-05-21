import streamlit as st
import yaml
from io import BytesIO

def build_config_ui():
    st.subheader("🔧 Create RASCAL Config")

    input_dir = st.text_input("Input directory", value="data/input")
    output_dir = st.text_input("Output directory", value="data/output")
    reliability_fraction = st.number_input("Reliability fraction", min_value=0.0, max_value=1.0, value=0.2)
    coders = st.text_area("Coders (comma-separated)", value="XX, YY, ZZ")
    coder_list = [c.strip() for c in coders.split(",") if c.strip()]

    # --- Tier Builder ---
    st.subheader("📐 Tier Definitions")

    if "tiers" not in st.session_state:
        st.session_state.tiers = [
            {"label": "site", "values": "AC, BU, TU", "is_partition": True},
            {"label": "test", "values": "Pre, Post, Maint", "is_partition": False},
            {"label": "participantID", "values": "site##", "is_partition": False},
            {"label": "narrative", "values": "CATGrandpa, BrokenWindow, RefusedUmbrella, CatRescue, BirthdayScene", "is_partition": False}
        ]

    for i, tier in enumerate(st.session_state.tiers):
        cols = st.columns([2, 4, 1])
        tier["label"] = cols[0].text_input(f"Tier name {i+1}", value=tier["label"], key=f"tier_label_{i}")
        tier["values"] = cols[1].text_input(f"Values (comma-separated) - {tier['label']}", value=tier["values"], key=f"tier_values_{i}")
        tier["is_partition"] = cols[2].checkbox("Partition?", value=tier["is_partition"], key=f"tier_partition_{i}")

    col1, col2 = st.columns([1, 1])
    if col1.button("➕ Add Tier"):
        st.session_state.tiers.append({"label": "", "values": "", "is_partition": False})
    if col2.button("➖ Remove Last Tier"):
        if st.session_state.tiers:
            st.session_state.tiers.pop()

    # --- YAML Assembly ---
    tiers_dict = {}
    for tier in st.session_state.tiers:
        key = f"*{tier['label']}" if tier["is_partition"] else tier["label"]
        value_list = [v.strip() for v in tier["values"].split(",") if v.strip()]
        if key:
            tiers_dict[key] = value_list if len(value_list) > 1 else value_list[0]

    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "reliability_fraction": reliability_fraction,
        "coders": coder_list,
        "tiers": tiers_dict
    }

    yaml_config = yaml.dump(config, sort_keys=False, allow_unicode=True)
    yaml_bytes = BytesIO(yaml_config.encode("utf-8"))

    st.subheader("📄 YAML Config Preview")
    st.code(yaml_config, language="yaml")
    st.download_button("📥 Download config.yaml", data=yaml_bytes, file_name="config.yaml", mime="application/x-yaml")

    return config
