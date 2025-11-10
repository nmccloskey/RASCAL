import types
import pytest
from pathlib import Path
from datetime import datetime

import rascal.main as main_module


@pytest.fixture
def mock_run_functions(monkeypatch):
    """Patch all run_* functions to lightweight stubs that record calls."""
    called = {}

    def make_stub(name):
        def stub(*args, **kwargs):
            called[name] = {"args": args, "kwargs": kwargs}
            return name
        return stub

    run_names = [
        "run_read_tiers",
        "run_read_cha_files",
        "run_select_transcription_reliability_samples",
        "run_reselect_transcription_reliability_samples",
        "run_evaluate_transcription_reliability",
        "run_make_transcript_tables",
        "run_make_cu_coding_files",
        "run_evaluate_cu_reliability",
        "run_analyze_cu_coding",
        "run_reselect_cu_reliability",
        "run_make_word_count_files",
        "run_evaluate_word_count_reliability",
        "run_reselect_wc_reliability",
        "run_summarize_cus",
        "run_run_corelex",
    ]
    for name in run_names:
        monkeypatch.setattr(main_module, name, make_stub(name))

    return called


@pytest.fixture
def mock_utils(monkeypatch, tmp_path):
    """Patch file/config/logging utilities to behave deterministically."""
    # Root setup
    monkeypatch.setattr(main_module, "set_root", lambda p: None)
    monkeypatch.setattr(main_module, "get_root", lambda: tmp_path)
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda *a, **k: None)

    # Logger replacement (with dummy info/error/warning)
    class DummyLogger:
        def __getattr__(self, name):
            return lambda *a, **k: None
    main_module.logger = DummyLogger()

    # Stub project_path to always return tmp_path / requested subdir
    monkeypatch.setattr(
        main_module,
        "project_path",
        lambda *parts: (tmp_path / Path(*parts)).resolve()
    )

    # Stub load_config to return a minimal valid config
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda path: {
            "input_dir": str(tmp_path / "input"),
            "output_dir": str(tmp_path / "output"),
            "random_seed": 1,
            "reliability_fraction": 0.5,
        },
    )

    # Stub find_files to simulate no transcript tables (forces run_make_transcript_tables)
    monkeypatch.setattr(main_module, "find_files", lambda **k: [])

    return tmp_path


def test_main_executes_basic_command(mock_utils, mock_run_functions, tmp_path):
    """Test that main() runs a simple command and dispatches correct run_* functions."""
    args = types.SimpleNamespace(command=["4a"], config=None)
    main_module.main(args)

    # run_read_tiers and run_make_transcript_tables should have been called
    assert "run_read_tiers" in mock_run_functions
    assert "run_make_transcript_tables" in mock_run_functions

    tiers_args = mock_run_functions["run_read_tiers"]["args"]
    assert isinstance(tiers_args[0], dict)

    make_args = mock_run_functions["run_make_transcript_tables"]["args"]
    assert len(make_args) == 3
    assert isinstance(make_args[2], Path)
    assert make_args[2].name.startswith("rascal_output_")


def test_main_handles_expanded_command(mock_utils, mock_run_functions):
    """Ensure expanded command names map correctly to their succinct codes."""
    args = types.SimpleNamespace(command=["transcripts make"], config=None)
    main_module.main(args)
    assert "run_make_transcript_tables" in mock_run_functions


def test_main_handles_omnibus(mock_utils, mock_run_functions):
    """Omnibus command (e.g., '4') expands into subcommands."""
    args = types.SimpleNamespace(command=["4"], config=None)
    main_module.main(args)
    # OMNIBUS '4' expands to ['4a', '4b']
    assert "run_make_transcript_tables" in mock_run_functions
    assert "run_make_cu_coding_files" in mock_run_functions


def test_main_skips_unrecognized_command(mock_utils, mock_run_functions):
    """Unrecognized commands should trigger a warning and not crash."""
    args = types.SimpleNamespace(command=["xyz"], config=None)
    # Should exit cleanly without raising
    main_module.main(args)

    # run_read_tiers is always called early; no other dispatch functions should run
    called = set(mock_run_functions.keys())
    assert called == {"run_read_tiers"}


def test_main_executes_multiple_commands(mock_utils, mock_run_functions):
    """Comma-separated commands should run sequentially."""
    args = types.SimpleNamespace(command=["4a,10a"], config=None)
    main_module.main(args)

    # Both commands should be executed
    assert "run_make_transcript_tables" in mock_run_functions
    assert "run_summarize_cus" in mock_run_functions
