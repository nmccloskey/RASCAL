from __future__ import annotations

import gzip
import importlib

import pytest

import rascal.asr_utils as asr_utils
from rascal.asr_utils import AsrError, combine_chat_parts, split_audio


class FakeAudio:
    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms

    def __len__(self):
        return self.duration_ms

    def __getitem__(self, item):
        return FakeAudio(item.stop - item.start)

    @classmethod
    def from_wav(cls, path):
        durations = {
            "short.wav": 30_000,
            "long.wav": 130_000,
        }
        return cls(durations[path.name])

    def export(self, path, format):
        path.write_text(f"{format}:{self.duration_ms}", encoding="utf-8")


def test_chat_parts_combine_in_numeric_part_order(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample_part10.cha").write_text(
        "*PAR:\tthird .\n@End\n",
        encoding="utf-8",
    )
    (input_dir / "sample_part2.cha").write_text(
        "*PAR:\tsecond .\n@End\n",
        encoding="utf-8",
    )
    (input_dir / "sample_part1.cha").write_text(
        "@Begin\n*PAR:\tfirst .\n@End\n",
        encoding="utf-8",
    )

    results = combine_chat_parts(input_dir, output_dir)

    assert len(results) == 1
    assert [path.name for path in results[0].part_paths] == [
        "sample_part1.cha",
        "sample_part2.cha",
        "sample_part10.cha",
    ]
    output = (output_dir / "sample_combined.cha").read_text(encoding="utf-8")
    assert output.index("first") < output.index("second") < output.index("third")


def test_chat_combination_removes_end_and_media_lines(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample_part1.cha").write_text(
        "@Begin\n@Media:\tsample\n*PAR:\tone .\n@End\n",
        encoding="utf-8",
    )

    combine_chat_parts(input_dir, output_dir)

    output = (output_dir / "sample_combined.cha").read_text(encoding="utf-8")
    assert "@Media" not in output
    assert output.count("@End") == 1
    assert output.endswith("\n@End\n")


def test_standalone_digits_convert_only_on_speaker_tiers(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample_part1.cha").write_text(
        "*PAR:\tI saw 2 birds and 10 fish .\n"
        "%mor:\t2 should stay numeric here .\n"
        "@End\n",
        encoding="utf-8",
    )

    combine_chat_parts(input_dir, output_dir)

    output = (output_dir / "sample_combined.cha").read_text(encoding="utf-8")
    assert "*PAR:\tI saw two birds and ten fish ." in output
    assert "%mor:\t2 should stay numeric here ." in output


def test_gzipped_chat_parts_are_supported(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    with gzip.open(input_dir / "sample_part1.cha.gz", "wt", encoding="utf-8") as outfile:
        outfile.write("@Begin\n*PAR:\t1 thing .\n@End\n")

    combine_chat_parts(input_dir, output_dir)

    output = (output_dir / "sample_combined.cha").read_text(encoding="utf-8")
    assert "*PAR:\tone thing ." in output


def test_missing_input_directory_gives_useful_error(tmp_path):
    with pytest.raises(AsrError, match="Input directory not found"):
        combine_chat_parts(tmp_path / "missing", tmp_path / "output")


def test_audio_splitting_uses_wav_files_only_and_fake_audio(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "short.wav").write_bytes(b"fake")
    (input_dir / "long.wav").write_bytes(b"fake")
    (input_dir / "skip.mp3").write_bytes(b"fake")

    results = split_audio(
        input_dir,
        output_dir,
        max_seconds=60,
        audio_segment_cls=FakeAudio,
    )

    assert [result.input_path.name for result in results] == ["long.wav", "short.wav"]
    assert [path.name for path in results[0].output_paths] == [
        "long_part1.wav",
        "long_part2.wav",
        "long_part3.wav",
    ]
    assert [path.name for path in results[1].output_paths] == ["short_part1.wav"]


def test_audio_splitting_rejects_invalid_duration(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with pytest.raises(AsrError, match="max_seconds"):
        split_audio(input_dir, output_dir, max_seconds=0, audio_segment_cls=FakeAudio)


def test_helper_module_does_no_filesystem_work_at_import(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    importlib.reload(asr_utils)

    assert not (tmp_path / "vid_split").exists()
    assert not (tmp_path / "transcripts_combined").exists()
