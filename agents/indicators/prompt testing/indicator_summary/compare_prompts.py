#!/usr/bin/env python3
"""
Compare two indicator_summary prompt files and report only semantic differences.
- Verifies presence of key sections (header, summary JSON, technical data JSON, levels JSON, final prompt marker, conflicts marker)
- Parses JSON blocks and compares keys/types recursively, ignoring numeric/string value differences
- Reports missing or extra keys, and type mismatches (e.g., dict vs list)

Usage:
  python compare_prompts.py \
    --old "/path/to/old/prompt_analysis_SYMBOL_YYYYMMDD_HHMMSS.txt" \
    --new "/path/to/new/prompt_SYMBOL_YYYYMMDD_HHMMSS.txt"
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

SECTION_MARKERS = {
    "summary_header": "KEY TECHNICAL INDICATORS SUMMARY:",
    "final_prompt": "FINAL PROMPT SENT TO LLM:",
    "analysis_context": "## Analysis Context:",
    "technical_data": "## Technical Data:",
    "levels": "## Levels:",
    "conflicts": "## Signal Conflicts",
}

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_marker(text: str, marker: str) -> int:
    return text.find(marker)


def extract_first_json_after(text: str, marker: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Find the first balanced JSON object that appears after the marker line.
    Returns (parsed_json, error_message)
    """
    idx = find_marker(text, marker)
    if idx == -1:
        return None, f"Marker not found: {marker}"
    # Find the first '{' after the marker
    brace_start = text.find('{', idx)
    if brace_start == -1:
        return None, f"No JSON object found after: {marker}"
    # Extract a balanced JSON object by counting braces
    brace_count = 0
    end = brace_start
    while end < len(text):
        ch = text[end]
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                # include this closing brace
                json_str = text[brace_start:end+1]
                try:
                    data = json.loads(json_str)
                    return data, None
                except Exception as ex:
                    return None, f"JSON parse error after {marker}: {ex}"
        end += 1
    return None, f"Unbalanced JSON after: {marker}"


def compare_structure(old: Any, new: Any, path: str = "$") -> List[str]:
    """Compare structures recursively and report semantic diffs (keys/types). Ignore scalar value diffs.
    Returns a list of difference messages.
    """
    diffs: List[str] = []
    type_old = type(old)
    type_new = type(new)
    if type_old != type_new:
        diffs.append(f"Type mismatch at {path}: {type_old.__name__} vs {type_new.__name__}")
        return diffs
    if isinstance(old, dict):
        keys_old = set(old.keys())
        keys_new = set(new.keys())
        missing_in_new = sorted(keys_old - keys_new)
        extra_in_new = sorted(keys_new - keys_old)
        if missing_in_new:
            diffs.append(f"Missing keys in new at {path}: {missing_in_new}")
        if extra_in_new:
            diffs.append(f"Extra keys in new at {path}: {extra_in_new}")
        for k in sorted(keys_old & keys_new):
            diffs.extend(compare_structure(old[k], new[k], f"{path}.{k}"))
    elif isinstance(old, list):
        # Lists: Only compare element types if possible; ignore lengths and values
        # Compare type of first non-null element if present
        def first_type(lst):
            for v in lst:
                if v is not None:
                    return type(v)
            return type(None)
        t_old = first_type(old)
        t_new = first_type(new)
        if t_old != t_new:
            diffs.append(f"List element type mismatch at {path}: {t_old.__name__} vs {t_new.__name__}")
    else:
        # Scalars: ignore differences (numeric/string/timestamps)
        pass
    return diffs


def compare_files(old_path: str, new_path: str) -> int:
    old_text = read_text(old_path)
    new_text = read_text(new_path)

    report: List[str] = []

    # Check presence of key markers/sections
    for name, marker in SECTION_MARKERS.items():
        present_old = find_marker(old_text, marker) != -1
        present_new = find_marker(new_text, marker) != -1
        if present_old != present_new:
            report.append(f"Section presence differs for '{name}' ({marker}): old={present_old}, new={present_new}")

    # Extract and compare JSON blocks: summary, technical data, levels
    blocks = [
        ("summary_json", SECTION_MARKERS["summary_header"]),
        ("technical_data_json", SECTION_MARKERS["technical_data"]),
        ("levels_json", SECTION_MARKERS["levels"]),
    ]

    parsed: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Optional[str]]] = {}

    for label, marker in blocks:
        old_obj, old_err = extract_first_json_after(old_text, marker)
        new_obj, new_err = extract_first_json_after(new_text, marker)
        parsed[label] = (old_obj, new_obj, old_err, new_err)
        if old_err or new_err:
            report.append(f"{label}: parse issues: old_err={old_err}, new_err={new_err}")
        elif isinstance(old_obj, dict) and isinstance(new_obj, dict):
            diffs = compare_structure(old_obj, new_obj, path=label)
            for d in diffs:
                # Filter out expected known additions we want to allow (e.g., new includes direction/confidence in trend)
                report.append(d)
        else:
            report.append(f"{label}: type mismatch or missing JSON (old={type(old_obj).__name__ if old_obj is not None else None}, new={type(new_obj).__name__ if new_obj is not None else None})")

    # Print summary
    print("Semantic comparison report")
    print("=" * 80)
    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}\n")

    if not report:
        print("No semantic differences detected (structure and keys match).")
        return 0

    for item in report:
        print(f"- {item}")
    return 1


def main():
    ap = argparse.ArgumentParser(description="Compare two prompt files semantically (ignore values)")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()

    sys.exit(compare_files(args.old, args.new))


if __name__ == "__main__":
    main()
