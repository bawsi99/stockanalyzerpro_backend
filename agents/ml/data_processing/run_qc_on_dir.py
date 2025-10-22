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


def run_qc_on_csv(csv_path: Path, qc_script: Path, unique_output: bool = True) -> int:
    csv_path = csv_path.resolve()
    out_json = (
        csv_path.with_name(f"{csv_path.stem}.qc.json") if unique_output else csv_path.with_name("qc_report.json")
    )

    cmd: List[str] = [
        sys.executable,
        str(qc_script),
        str(csv_path),
        "--output_json",
        str(out_json),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[ERROR] {csv_path}: {proc.stderr}\n")
    else:
        # Forward concise summary from qc script
        sys.stdout.write(proc.stdout.rstrip() + "\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively run qc_dataset.py on all CSV files within a directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to search for CSVs (processed root)."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum directory depth to traverse from input_dir (default: 5).",
    )
    parser.add_argument(
        "--use-common-name",
        action="store_true",
        help="Write results to qc_report.json (may overwrite if multiple CSVs in a folder). By default, writes <name>.qc.json",
    )

    args = parser.parse_args()

    start_dir: Path = args.input_dir
    if not start_dir.exists() or not start_dir.is_dir():
        sys.stderr.write(f"Input directory does not exist or is not a directory: {start_dir}\n")
        return 2

    # Locate qc_dataset.py relative to this script (same directory)
    qc_script = (Path(__file__).parent / "qc_dataset.py").resolve()
    if not qc_script.exists():
        sys.stderr.write(f"qc_dataset.py not found at: {qc_script}\n")
        return 2

    exit_code = 0
    for csv_file in iter_csvs(start_dir, args.max_depth):
        rc = run_qc_on_csv(csv_file, qc_script, unique_output=not args.use_common_name)
        exit_code = exit_code or rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
