from pathlib import Path
from pydub import AudioSegment

input_folder = Path("vid_orig")
output_folder = Path("vid_split")
output_folder.mkdir(parents=True, exist_ok=True)

# Max length for each chunk in milliseconds
MAX_CHUNK_DURATION_MS = 1 * 60 * 1000  # adjust as needed

input_files = list(input_folder.rglob("*.wav"))

for filepath in input_files:
    audio = AudioSegment.from_wav(filepath)
    duration_ms = len(audio)

    if duration_ms <= MAX_CHUNK_DURATION_MS:
        # Copy original if no need to split
        output_path = output_folder / f"{filepath.stem}_part1.wav"
        audio.export(output_path, format="wav")
        print(f"[✓] '{filepath.name}' not split (under max duration)")
        continue

    # Compute number of chunks needed (ceiling division)
    num_chunks = (duration_ms + MAX_CHUNK_DURATION_MS - 1) // MAX_CHUNK_DURATION_MS
    chunk_duration = duration_ms // num_chunks

    for i in range(num_chunks):
        start = i * chunk_duration
        end = (i + 1) * chunk_duration if i < num_chunks - 1 else duration_ms
        chunk = audio[start:end]
        output_path = output_folder / f"{filepath.stem}_part{i+1}.wav"
        chunk.export(output_path, format="wav")

    print(f"[✓] '{filepath.name}' split into {num_chunks} parts")
