#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List


def iter_csvs(start_dir: Path, max_depth: int) -> Iterable[Path]:
    start_dir = start_dir.resolve()
    for root, dirs, files in os.walk(start_dir):
        root_path = Path(root)
        # Compute depth relative to start_dir
        try:
            rel_parts = root_path.relative_to(start_dir).parts
            depth = len(rel_parts)
        except Exception:
            depth = 0

        # Stop descending if we reached max depth
        if depth >= max_depth:
            dirs[:] = []

        for f in files:
            if f.lower().endswith(".csv"):
                yield root_path / f


def run_build_on_csv(csv_path: Path, build_script: Path, extra_args: List[str]) -> int:
    csv_path = csv_path.resolve()

    cmd: List[str] = [
        sys.executable,
        str(build_script),
        str(csv_path),
        *extra_args,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[ERROR] {csv_path}: {proc.stderr}\n")
    else:
        # Forward concise summary from build script
        sys.stdout.write(proc.stdout.rstrip() + "\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively run build_full_dataset.py on all CSV files within a directory "
            "(up to a maximum depth). Extra args are forwarded to build_full_dataset.py."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to search for CSVs (raw data root).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to traverse from input_dir (default: 3).",
    )

    # Accept any additional args and forward them unchanged
    args, extra_args = parser.parse_known_args()

    start_dir: Path = args.input_dir
    if not start_dir.exists() or not start_dir.is_dir():
        sys.stderr.write(f"Input directory does not exist or is not a directory: {start_dir}\n")
        return 2

    # Locate build_full_dataset.py relative to this script (same directory)
    build_script = (Path(__file__).parent / "build_full_dataset.py").resolve()
    if not build_script.exists():
        sys.stderr.write(f"build_full_dataset.py not found at: {build_script}\n")
        return 2

    exit_code = 0
    for csv_file in iter_csvs(start_dir, args.max_depth):
        rc = run_build_on_csv(csv_file, build_script, extra_args)
        exit_code = exit_code or rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
