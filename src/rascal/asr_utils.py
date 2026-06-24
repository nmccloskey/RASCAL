"""Stage 0 ASR helper utilities for RASCAL."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from num2words import num2words


class AsrError(ValueError):
    """Raised when an ASR helper cannot complete."""


@dataclass(frozen=True)
class ChatCombinationResult:
    """Summary for one combined CHAT file."""

    base_name: str
    output_path: Path
    part_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AudioSplitResult:
    """Summary for one split audio file."""

    input_path: Path
    output_paths: tuple[Path, ...]


def _require_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise AsrError(f"{label} directory not found: {resolved}")
    return resolved


def convert_digits_to_words(line: str) -> str:
    """Convert standalone digits to words on CHAT speaker tiers only."""

    if not line.startswith("*"):
        return line
    return re.sub(r"\b\d+\b", lambda match: num2words(int(match.group()), lang="en"), line)


def _read_chat_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as infile:
            return infile.readlines()
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        with gzip.open(path, "rt", encoding="utf-8") as infile:
            return infile.readlines()


def _part_pattern(pattern: str | re.Pattern[str]) -> re.Pattern[str]:
    return re.compile(pattern) if isinstance(pattern, str) else pattern


def _group_chat_parts(
    input_dir: Path,
    pattern: re.Pattern[str],
) -> dict[str, list[tuple[int, Path]]]:
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        try:
            base_name = match.group("base")
            part_num = int(match.group("part"))
        except IndexError as exc:
            raise AsrError("CHAT part pattern must expose 'base' and 'part' groups.") from exc
        grouped.setdefault(base_name, []).append((part_num, path))
    return grouped


def combine_chat_parts(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    part_pattern: str | re.Pattern[str] = r"(?P<base>.+?)_part(?P<part>\d+)\.cha(?:\.gz)?",
) -> tuple[ChatCombinationResult, ...]:
    """Combine CHAT part files into deterministic complete CHAT files."""

    source_dir = _require_directory(input_dir, "Input")
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_chat_parts(source_dir, _part_pattern(part_pattern))
    results: list[ChatCombinationResult] = []

    for base_name, parts in sorted(grouped.items()):
        sorted_parts = sorted(parts, key=lambda item: item[0])
        output_path = destination_dir / f"{base_name}_combined.cha"
        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            for index, (part_num, path) in enumerate(sorted_parts):
                lines = _read_chat_lines(path)
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith(("@End", "@Media"))
                ]
                if index == 0:
                    outfile.writelines(convert_digits_to_words(line) for line in lines)
                else:
                    outfile.write(f"%com:\t--- Start of part {part_num} ---\n")
                    content_lines = [
                        line
                        for line in lines
                        if line.startswith("*") or line.startswith("%")
                    ]
                    outfile.writelines(convert_digits_to_words(line) for line in content_lines)
            outfile.write("\n@End\n")
        results.append(
            ChatCombinationResult(
                base_name=base_name,
                output_path=output_path,
                part_paths=tuple(path for _, path in sorted_parts),
            )
        )
    return tuple(results)


def _audio_segment_class(audio_segment_cls: Any | None = None) -> Any:
    if audio_segment_cls is not None:
        return audio_segment_cls
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise AsrError(
            "Audio splitting requires pydub. Install the ASR optional dependency "
            "or pass an audio segment implementation for testing."
        ) from exc
    return AudioSegment


def split_audio(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    max_seconds: int = 60,
    audio_segment_cls: Any | None = None,
) -> tuple[AudioSplitResult, ...]:
    """Split .wav files into max-duration chunks."""

    if max_seconds <= 0:
        raise AsrError("max_seconds must be greater than zero.")

    source_dir = _require_directory(input_dir, "Input")
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    audio_segment = _audio_segment_class(audio_segment_cls)
    max_duration_ms = max_seconds * 1000
    results: list[AudioSplitResult] = []

    for input_path in sorted(source_dir.rglob("*.wav")):
        audio = audio_segment.from_wav(input_path)
        duration_ms = len(audio)
        output_paths: list[Path] = []
        if duration_ms <= max_duration_ms:
            output_path = destination_dir / f"{input_path.stem}_part1.wav"
            audio.export(output_path, format="wav")
            output_paths.append(output_path)
        else:
            num_chunks = (duration_ms + max_duration_ms - 1) // max_duration_ms
            chunk_duration = duration_ms // num_chunks
            for index in range(num_chunks):
                start = index * chunk_duration
                end = (index + 1) * chunk_duration if index < num_chunks - 1 else duration_ms
                chunk = audio[start:end]
                output_path = destination_dir / f"{input_path.stem}_part{index + 1}.wav"
                chunk.export(output_path, format="wav")
                output_paths.append(output_path)
        results.append(
            AudioSplitResult(
                input_path=input_path,
                output_paths=tuple(output_paths),
            )
        )
    return tuple(results)
