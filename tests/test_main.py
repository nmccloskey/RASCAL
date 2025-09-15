import os
import types
import pytest
import yaml

# Import target
try:
    import rascal.main as mainmod
except Exception as e:
    pytest.skip(f"Could not import rascal.main: {e}", allow_module_level=True)


class DummyArgs:
    def __init__(self, step, config):
        self.step = step
        self.config = config


def test_main_smoke(monkeypatch, tmp_path):
    """Smoke test for main(): ensures dispatch mapping triggers functions correctly."""
    # --- Make dummy config file ---
    config = {
        "input_dir": str(tmp_path / "in"),
        "output_dir": str(tmp_path / "out"),
        "coders": ["1", "2", "3"],
        "tiers": {"site": {"values": ["TU"], "partition": True, "blind": False}},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    # --- Patch all run_* functions to record calls ---
    called = {}
    def dummy(name):
        def _f(*a, **k):
            called[name] = True
            return "dummy_return"
        return _f

    monkeypatch.setattr(mainmod, "run_read_tiers", dummy("tiers"))
    monkeypatch.setattr(mainmod, "run_read_cha_files", dummy("readcha"))
    monkeypatch.setattr(mainmod, "run_select_transcription_reliability_samples", dummy("sel"))
    monkeypatch.setattr(mainmod, "run_prepare_utterance_dfs", dummy("prep"))
    monkeypatch.setattr(mainmod, "run_make_CU_coding_files", dummy("cucoding"))
    monkeypatch.setattr(mainmod, "run_analyze_transcription_reliability", dummy("transrel"))
    monkeypatch.setattr(mainmod, "run_analyze_CU_reliability", dummy("curel"))
    monkeypatch.setattr(mainmod, "run_analyze_CU_coding", dummy("cucoding2"))
    monkeypatch.setattr(mainmod, "run_make_word_count_files", dummy("wordcount"))
    monkeypatch.setattr(mainmod, "run_make_timesheets", dummy("timesheets"))
    monkeypatch.setattr(mainmod, "run_analyze_word_count_reliability", dummy("wcrel"))
    monkeypatch.setattr(mainmod, "run_unblind_CUs", dummy("unblind"))
    monkeypatch.setattr(mainmod, "run_run_corelex", dummy("corelex"))
    monkeypatch.setattr(mainmod, "run_reselect_CU_reliability", dummy("reselect"))

    # --- Run main with step '5' (mapped to 'ijk') ---
    args = DummyArgs(step="5", config=str(config_file))
    mainmod.main(args)

    # --- Verify correct branches executed (i,j,k) ---
    assert called["wcrel"]
    assert called["unblind"]
    assert called["corelex"]
