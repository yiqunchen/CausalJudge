#!/usr/bin/env python3
"""
Cleanup helper: removes archived/unused directories and intermediate artifacts.

By default, removes:
- archive/, archive_unused/, raw_results/
- results/* (predictions, metrics, checkpoints)
- figures/* (keeps compress_prior_work.drawio.png)
- stray .DS_Store files

Dry-run by default. Pass --apply to actually delete.
"""

import argparse
import os
import shutil
from pathlib import Path


def rm(path: Path, apply: bool):
    if not path.exists():
        return
    if apply:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(("DEL " if apply else "DRY ") + str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run)")
    args = p.parse_args()

    root = Path(".")

    # Remove archived/unused dirs
    for d in [root/"archive", root/"archive_unused", root/"raw_results"]:
        rm(d, args.apply)

    # Clean results
    results = root/"results"
    if results.exists():
        for child in results.iterdir():
            rm(child, args.apply)

    # Clean figures but keep preview image
    figures = root/"figures"
    keep = {figures/"compress_prior_work.drawio.png"}
    if figures.exists():
        for child in figures.iterdir():
            if child in keep:
                continue
            rm(child, args.apply)

    # Remove stray .DS_Store files
    for ds in root.rglob(".DS_Store"):
        rm(ds, args.apply)

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()

