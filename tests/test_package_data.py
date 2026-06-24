from __future__ import annotations

import builtins
import importlib
import sys
import tomllib
from importlib import resources
from pathlib import Path


def test_packaged_profile_yaml_files_are_available():
    profile_root = resources.files("rascal.data.profiles")

    for filename in (
        "lab_common.yaml",
        "lab_monolog.yaml",
        "lab_dialog.yaml",
        "lab_full.yaml",
    ):
        assert profile_root.joinpath(filename).is_file()


def test_packaged_stage_yaml_files_are_available():
    stage_root = resources.files("rascal.data.stages")

    for filename in ("common.yaml", "monolog.yaml", "dialog.yaml"):
        assert stage_root.joinpath(filename).is_file()


def test_cli_entry_point_module_imports_cleanly():
    module = importlib.import_module("rascal.cli")

    assert callable(module.main)


def test_core_package_import_does_not_require_optional_pydub(monkeypatch):
    sys.modules.pop("pydub", None)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pydub" or name.startswith("pydub."):
            raise ImportError("pydub blocked for core import smoke test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.reload(importlib.import_module("rascal.asr_utils"))
    importlib.reload(importlib.import_module("rascal.cli"))


def test_pyproject_metadata_matches_wrapper_architecture():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"
    assert pyproject["project"]["scripts"] == {"rascal": "rascal.cli:main"}
    assert pyproject["project"]["dependencies"] == ["diaad>=0.3.1,<0.4.0"]
    assert "pydub>=0.25.1" in pyproject["project"]["optional-dependencies"]["asr"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert pyproject["tool"]["setuptools"]["package-data"]["rascal"] == [
        "data/profiles/*.yaml",
        "data/stages/*.yaml",
    ]
