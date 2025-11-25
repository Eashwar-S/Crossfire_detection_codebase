#!/usr/bin/env python3
"""copy_images.py

Small utility to copy all image files from a source directory to a
destination directory. Preserves filenames and timestamps.

Usage:
    python copy_images.py /path/to/src /path/to/dest

Options:
    --recursive / -r   : walk subdirectories
    --dry-run  / -n    : show what would be copied without copying
    --exts   / -e      : comma-separated extensions to include (default: jpg,jpeg,png)
"""

import argparse
import math
import os
import shutil
from pathlib import Path

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description="Copy image files from source to destination")
    p.add_argument("src", help="Source directory containing images")
    p.add_argument("dst", help="Destination directory to copy images into")
    p.add_argument("-r", "--recursive", action="store_true", help="Walk subdirectories")
    p.add_argument("-n", "--dry-run", action="store_true", help="Show files that would be copied")
    p.add_argument("-e", "--exts", default=','.join(x.lstrip('.') for x in DEFAULT_EXTS),
                   help="Comma-separated extensions to include (default: jpg,jpeg,png,...)")
    return p.parse_args()


def gather_images(src: Path, exts, recursive: bool):
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    if recursive:
        for p in src.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def main():
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.exists() or not src.is_dir():
        print(f"Source directory does not exist: {src}")
        raise SystemExit(1)

    dst.mkdir(parents=True, exist_ok=True)

    exts = [x.strip() for x in args.exts.split(',') if x.strip()]

    files = list(gather_images(src, exts, args.recursive))
    if not files:
        print("No image files found to copy.")
        return


    print(f"Found {len(files)} files to copy from {src} to {dst}")
    for p in files:
        if math.random(1, len(files)) == 1:
            rel = p.relative_to(src)

            target = dst.joinpath(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                print(f"DRY RUN: {p} -> {target}")
            else:
                try:
                    shutil.copy2(p, target)
                    print(f"Copied: {p} -> {target}")
                except Exception as e:
                    print(f"Failed to copy {p} -> {target}: {e}")


if __name__ == '__main__':
    main()
