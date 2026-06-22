from pathlib import Path
import re
import gzip
from num2words import num2words

input_folder = Path("transcripts_split")
output_folder = Path("transcripts_combined")
output_folder.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"(.+?)_part(\d+)\.cha")
grouped_files = {}

# Group by basename
for filepath in input_folder.iterdir():
    if filepath.suffix == ".cha":
        match = pattern.match(filepath.name)
        if match:
            base_name = match.group(1)
            part_num = int(match.group(2))
            grouped_files.setdefault(base_name, []).append((part_num, filepath))

def convert_digits_to_words(line: str) -> str:
    """Convert standalone digits to words in speaker tiers."""
    if line.startswith("*"):
        return re.sub(r"\b\d+\b", lambda m: num2words(int(m.group()), lang="en"), line)
    return line

for base_name, parts in grouped_files.items():
    parts.sort()  # sort by part number
    output_path = output_folder / f"{base_name}_combined.cha"

    with output_path.open("w", encoding="utf-8") as outfile:
        for i, (part_num, filepath) in enumerate(parts):
            # Try UTF-8, fall back to gzip
            try:
                lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                try:
                    with gzip.open(filepath, "rt", encoding="utf-8") as infile:
                        lines = infile.readlines()
                    print(f"[!] Decompressed gzipped file: {filepath.name}")
                except Exception as e:
                    print(f"[✗] Failed to read {filepath.name}: {e}")
                    continue

            # Remove @End if present
            lines = [line for line in lines if not line.strip().startswith(("@End", "@Media"))]

            if i == 0:
                # Write header + converted speaker lines
                lines = [convert_digits_to_words(line) for line in lines]
                outfile.writelines(lines)
            else:
                outfile.write(f"%com:\t--- Start of part {part_num} ---\n")
                content_lines = [line for line in lines if line.startswith("*") or line.startswith("%")]
                content_lines = [convert_digits_to_words(line) for line in content_lines]
                outfile.writelines(content_lines)

        outfile.write("\n@End\n")

    print(f"[✓] Merged {len(parts)} parts into '{output_path.name}'")
