import streamlit as st
import yaml
from io import BytesIO

def build_config_ui():
    st.subheader("🔧 Create RASCAL Config")

    input_dir = st.text_input("Input directory", value="data/input")
    output_dir = st.text_input("Output directory", value="data/output")
    reliability_fraction = st.number_input("Reliability fraction", min_value=0.0, max_value=1.0, value=0.2)
    coders = st.text_area("Coders (comma-separated)", value="XX, YY")
    coder_list = [c.strip() for c in coders.split(",") if c.strip()]

    site_values = st.text_area("*site values (comma-separated)", value="site1, site2, site3")
    test_values = st.text_area("test values (comma-separated)", value="time1, time2, time3")
    participant_id_pattern = st.text_input("Participant ID pattern", value="site##")

    narratives = st.text_area("Narratives (one per line)", value="\n".join([
        "CATGrandpa", "BrokenWindow", "RefusedUmbrella", "CatRescue", "BirthdayScene"
    ]))
    narrative_list = [n.strip() for n in narratives.splitlines() if n.strip()]

    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "reliability_fraction": reliability_fraction,
        "coders": coder_list,
        "tiers": {
            "*site": [s.strip() for s in site_values.split(",") if s.strip()],
            "test": [t.strip() for t in test_values.split(",") if t.strip()],
            "participantID": participant_id_pattern,
            "narrative": narrative_list
        }
    }

    yaml_config = yaml.dump(config, sort_keys=False, allow_unicode=True)
    yaml_bytes = BytesIO(yaml_config.encode("utf-8"))

    st.code(yaml_config, language="yaml")
    st.download_button("📥 Download config.yaml", data=yaml_bytes, file_name="config.yaml", mime="application/x-yaml")

    return config
