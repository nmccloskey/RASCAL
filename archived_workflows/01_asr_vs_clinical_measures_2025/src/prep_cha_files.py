from pathlib import Path
import re
import shutil


def rename_chat_reliability_files(root_dir, dry_run=True):
    """
    Recursively goes through a folder and renames all .cha files
    to include '_Reliability' before the extension.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to search through.
    dry_run : bool, default=True
        If True, only prints changes without renaming.
        If False, actually renames files.
    """
    root_dir = Path(root_dir)

    for file in root_dir.rglob("*.cha"):
        new_name = re.sub(r"\.cha$", "_Reliability.cha", str(file))
        new_path = Path(new_name)

        if file == new_path:
            continue  # already renamed

        with open(file) as f:
            text = f.read()
            if "INV:" in text:
                print(f"INV speaks in {file} - skipping")
                continue

        if dry_run:
            print(f"Would rename: {file} -> {new_path}")
        else:
            shutil.move(file, new_path)
            print(f"Renamed: {file} -> {new_path}")


if __name__ == "__main__":
    # Change this to your transcripts directory
    rename_chat_reliability_files("transcriptions/final", dry_run=False)
