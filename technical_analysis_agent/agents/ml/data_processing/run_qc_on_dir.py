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


def run_qc_on_csv(csv_path: Path, qc_script: Path, unique_output: bool = True, silence_output: bool = False) -> tuple[int, Path]:
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
        # Forward concise summary from qc script (unless silenced)
        if not silence_output:
            sys.stdout.write(proc.stdout.rstrip() + "\n")
    return proc.returncode, out_json


def _analyze_report(json_path: Path,
                     min_rows: int,
                     nan_threshold: int,
                     max_dup_index: int,
                     cls_minority_min: float) -> bool:
    import json
    try:
        with open(json_path, "r") as f:
            rep = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to read QC JSON {json_path}: {e}\n")
        return True

    issues: list[str] = []

    if not rep.get("index_monotonic_increasing", True):
        issues.append("index not monotonic increasing")

    dup_ct = int(rep.get("index_duplicate_count", 0))
    if dup_ct > max_dup_index:
        issues.append(f"duplicate index count={dup_ct} > {max_dup_index}")

    nan_counts = rep.get("nan_counts", {}) or {}
    bad_nan = {c: int(v) for c, v in nan_counts.items() if int(v) >= nan_threshold}
    if bad_nan:
        sample = ", ".join(f"{k}={v}" for k, v in list(bad_nan.items())[:5])
        more = "" if len(bad_nan) <= 5 else f" (+{len(bad_nan)-5} more)"
        issues.append(f"NaNs in columns: {sample}{more}")

    inf_counts = rep.get("inf_counts", {}) or {}
    bad_inf = {c: int(v) for c, v in inf_counts.items() if int(v) > 0}
    if bad_inf:
        sample = ", ".join(f"{k}={v}" for k, v in list(bad_inf.items())[:5])
        more = "" if len(bad_inf) <= 5 else f" (+{len(bad_inf)-5} more)"
        issues.append(f"Infs in columns: {sample}{more}")

    rows = int(rep.get("rows", 0))
    if rows < min_rows:
        issues.append(f"rows={rows} < {min_rows}")

    cls = rep.get("y_cls_distribution")
    if isinstance(cls, dict) and len(cls) >= 2:
        pcts = [float(v.get("pct", 0.0)) for v in cls.values()]
        minority = min(pcts) if pcts else 0.0
        if minority < cls_minority_min:
            issues.append(f"class imbalance minority={minority:.2f} < {cls_minority_min:.2f}")

    if issues:
        sys.stdout.write(f"[ATTENTION] {json_path}: " + "; ".join(issues) + "\n")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively run qc_dataset.py on all CSV files within a directory and flag reports needing attention."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to search for CSVs (processed or raw root)."
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
    parser.add_argument("--min-rows", type=int, default=200, help="Warn if rows < this value (default: 200)")
    parser.add_argument("--nan-threshold", type=int, default=1, help="Warn for columns with NaN count >= threshold (default: 1)")
    parser.add_argument("--max-dup-index", type=int, default=0, help="Warn if duplicate index count > this (default: 0)")
    parser.add_argument("--cls-minority-min", type=float, default=0.35, help="Warn if minority class pct < this (default: 0.35)")
    parser.add_argument("--attention-only", action="store_true", help="Only print attention logs; suppress per-file QC summaries")

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
    any_attention = False
    for csv_file in iter_csvs(start_dir, args.max_depth):
        rc, json_path = run_qc_on_csv(csv_file, qc_script, unique_output=not args.use_common_name, silence_output=args.attention_only)
        exit_code = exit_code or rc
        # Analyze report if QC succeeded
        if rc == 0 and json_path.exists():
            flagged = _analyze_report(
                json_path,
                min_rows=args.min_rows,
                nan_threshold=args.nan_threshold,
                max_dup_index=args.max_dup_index,
                cls_minority_min=args.cls_minority_min,
            )
            any_attention = any_attention or flagged

    # If attention-only and nothing flagged, give a succinct message for visibility
    if args.attention_only and not any_attention:
        sys.stdout.write("[OK] No issues found in QC reports.\n")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
