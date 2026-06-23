# save as tools/cex_to_cha.py
from pathlib import Path
import shutil
import logging
import argparse

def replace_cha_with_cex(root_dir, backup=True, dry_run=True, encoding="utf-8"):
    """
    Replace .cha files with the debulletized content from paired .chstr.cex files.

    Pairing rule:
      BU75PostTxBrokenWindow.chstr.cex  ->  BU75PostTxBrokenWindow.cha

    Parameters
    ----------
    root_dir : str | Path
        Directory to traverse recursively.
    backup : bool
        If True and a .cha exists, move it to .cha.bak before overwrite.
    dry_run : bool
        If True, only print the actions it would take.
    encoding : str
        Text encoding used to read/write CHAT files.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(root_dir)

    try:
        # optional sanity check: ensure pylangacq is present
        from pylangacq import read_chat  # noqa: F401
        has_pylangacq = True
    except Exception:
        logging.warning("pylangacq not found or import failed; proceeding without parse sanity checks.")
        has_pylangacq = False

    # Gather *.cex candidates and filter to those ending with .chstr.cex (case-insensitive)
    cex_files = [p for p in root.rglob("*.cex") if p.name.lower().endswith(".chstr.cex")]

    if not cex_files:
        logging.info("No '.chstr.cex' files found under %s", root.resolve())
        return

    for cex in cex_files:
        # Build target .cha path by removing the trailing ".chstr.cex" and adding ".cha"
        stem_no_suffix = cex.name[:-len(".chstr.cex")]  # remove the exact suffix length
        cha_path = cex.with_name(stem_no_suffix + ".cha")

        # Optional sanity check: try parsing with pylangacq (will raise if badly formatted)
        if has_pylangacq:
            try:
                from pylangacq import read_chat
                _ = read_chat(str(cex))  # parse; we don't use the object, just validating
            except Exception as e:
                logging.warning("Parse check failed for %s: %s. Will still copy raw text.", cex, e)

        # Read raw debulletized text from .cex
        try:
            text = cex.read_text(encoding=encoding, errors="ignore")
        except Exception as e:
            logging.error("Failed reading %s: %s", cex, e)
            continue

        if dry_run:
            action = f"Would {'backup & ' if backup and cha_path.exists() else ''}overwrite {cha_path} with {cex}"
            logging.info(action)
            continue

        # Backup existing .cha if requested
        if backup and cha_path.exists():
            bak = cha_path.with_suffix(cha_path.suffix + ".bak")
            # Avoid overwriting an existing .bak
            i = 1
            while bak.exists():
                bak = cha_path.with_suffix(cha_path.suffix + f".bak{i}")
                i += 1
            shutil.move(str(cha_path), str(bak))
            logging.info("Backed up %s -> %s", cha_path.name, bak.name)

        # Write debulletized content to .cha
        try:
            cha_path.write_text(text, encoding=encoding, errors="ignore")
            logging.info("Wrote debulletized CHAT to %s", cha_path)
        except Exception as e:
            logging.error("Failed writing %s: %s", cha_path, e)


def _cli():
    ap = argparse.ArgumentParser(description="Replace .cha with paired debulletized .chstr.cex content.")
    ap.add_argument("root_dir", help="Root directory to scan")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create .bak of existing .cha")
    ap.add_argument("--apply", dest="dry_run", action="store_false", help="Apply changes (default is dry-run)")
    ap.set_defaults(backup=True, dry_run=True)
    args = ap.parse_args()
    replace_cha_with_cex(args.root_dir, backup=args.backup, dry_run=args.dry_run)

if __name__ == "__main__":
    _cli()
