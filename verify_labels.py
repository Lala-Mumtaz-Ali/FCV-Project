#!/usr/bin/env python3
"""verify_labels.py

Run this BEFORE starting Stage 2 training to confirm that every .mat label
file in the dataset is being parsed correctly and maps to a known action.

Usage:
    python verify_labels.py

It will print:
  - A per-action video count table
  - Any video IDs whose label could NOT be resolved (should be zero)
  - The exact action string extracted from each unresolved file so you can debug
"""

import os
import scipy.io
import numpy as np
from collections import Counter

from data import VideoFramePairDataset

# ── Config ────────────────────────────────────────────────────────────────────
LABELS_DIR = "./dataset/Penn_Action/labels"
KNOWN_ACTIONS = set(VideoFramePairDataset.ACTION2IDX.keys())


def extract_action_string(mat_path: str) -> str:
    """Use the exact same parsing logic as data.py to extract the action string."""
    mat = scipy.io.loadmat(mat_path)
    raw_action = mat.get("action")
    if raw_action is None:
        return "<missing_key>"
    act_val = raw_action.flat[0]
    if hasattr(act_val, "flat"):
        act_val = act_val.flat[0]
    return str(act_val).strip()


def main():
    if not os.path.isdir(LABELS_DIR):
        print(f"[ERROR] Labels directory not found: {LABELS_DIR}")
        return

    mat_files = sorted(f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".mat"))
    if not mat_files:
        print("[ERROR] No .mat files found in the labels directory.")
        return

    print(f"Found {len(mat_files)} label files. Scanning...\n")

    counts = Counter()
    unresolved = []  # (video_id, raw_string)

    for fname in mat_files:
        vid_id = os.path.splitext(fname)[0]
        mat_path = os.path.join(LABELS_DIR, fname)
        try:
            act_str = extract_action_string(mat_path)
        except Exception as e:
            act_str = f"<parse_error: {e}>"

        if act_str in KNOWN_ACTIONS:
            counts[act_str] += 1
        else:
            counts["<UNKNOWN>"] += 1
            unresolved.append((vid_id, act_str))

    # ── Print per-action summary ───────────────────────────────────────────────
    total = sum(counts.values())
    print("=" * 50)
    print(f"{'Action':<25} {'Videos':>8}  {'%':>6}")
    print("-" * 50)
    for action in sorted(KNOWN_ACTIONS):
        n = counts.get(action, 0)
        pct = 100 * n / total if total else 0
        marker = ""
        if n == 0:
            marker = "  <-- WARNING: no videos found!"
        print(f"  {action:<23} {n:>8}  {pct:>5.1f}%{marker}")

    if "<UNKNOWN>" in counts:
        print(f"\n  {'<UNKNOWN>':<23} {counts['<UNKNOWN>']:>8}  <-- PROBLEM")

    print("=" * 50)
    print(f"  {'TOTAL':<23} {total:>8}")
    print()

    # ── Report unresolved files ────────────────────────────────────────────────
    if unresolved:
        print(f"[FAIL] {len(unresolved)} video(s) could NOT be mapped to a known action label.")
        # Show the distinct raw strings that failed — key for diagnosing the issue
        distinct_raws = sorted(set(raw for _, raw in unresolved))
        print(f"\nDistinct unrecognised raw strings ({len(distinct_raws)} unique):")
        for raw in distinct_raws[:20]:
            print(f"  {raw!r}")
        if len(distinct_raws) > 20:
            print(f"  ... and {len(distinct_raws) - 20} more")
        # Show first 10 example files
        print(f"\nFirst 10 affected video IDs:")
        for vid_id, raw in unresolved[:10]:
            print(f"  Video {vid_id!r:>10}  ->  raw string: {raw!r}")
        print("\nFix data.py or add these action strings to the ACTIONS list before training.")
    else:
        print("[PASS] All label files parsed correctly. Safe to start Stage 2 training!")


if __name__ == "__main__":
    main()
