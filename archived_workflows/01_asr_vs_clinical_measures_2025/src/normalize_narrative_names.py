# save as tools/normalize_narrative_names.py
from pathlib import Path
import re
import argparse
import logging

def normalize_catpicdesc_to_catgrandpa(root_dir, dry_run=True):
    """
    Recursively rename *.cha files whose *file name* contains 'CATPicDesc'
    (any casing) to use 'CATGrandpa' exactly, leaving everything else intact.

    Examples:
      AC01Pre_CATPicDesc.cha  ->  AC01Pre_CATGrandpa.cha
      ac01pre_catpicdesc.cha  ->  ac01pre_CATGrandpa.cha
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(root_dir)
    pattern = re.compile(r"catpicdesc", flags=re.IGNORECASE)

    changed = 0
    for cha in root.rglob("*.cha"):
        old_name = cha.name
        if not pattern.search(old_name):
            continue

        new_name = pattern.sub("CATGrandpa", old_name)
        new_path = cha.with_name(new_name)

        if new_path == cha:
            continue

        if new_path.exists():
            logging.warning("Target exists, skipping: %s -> %s", cha, new_path)
            continue

        if dry_run:
            logging.info("Would rename: %s -> %s", cha, new_path)
        else:
            cha.rename(new_path)
            logging.info("Renamed: %s -> %s", cha, new_path)
            changed += 1

    if dry_run:
        logging.info("Dry-run complete.")
    else:
        logging.info("Done. Files renamed: %d", changed)


def _cli():
    ap = argparse.ArgumentParser(description="Rename CATPicDesc -> CATGrandpa in .cha filenames (case-insensitive).")
    ap.add_argument("root_dir", help="Root directory to scan")
    ap.add_argument("--apply", dest="dry_run", action="store_false", help="Apply changes (default is dry-run)")
    ap.set_defaults(dry_run=True)
    args = ap.parse_args()
    normalize_catpicdesc_to_catgrandpa(args.root_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    _cli()
