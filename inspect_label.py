#!/usr/bin/env python3
"""inspect_label.py

Utility script to load a MATLAB `.mat` annotation file from the Penn Action
dataset and display its contents in a human‑readable format.

Typical usage::

    python inspect_label.py                # prints the first .mat file in the label folder
    python inspect_label.py path/to/0001.mat

The script extracts the most relevant fields (action, pose, joint locations,
etc.) and prints them nicely. It works with the `scipy.io.loadmat` format
produced by the original dataset.
"""

import argparse
import os
import sys
from pprint import pprint
import numpy as np
import scipy.io


def _decode_matlab_string(arr):
    """Convert MATLAB string arrays to a Python ``str``.
    Handles the various ways SciPy represents MATLAB strings.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind in {"U", "S", "O"}:
            if arr.size == 1:
                return str(arr.item())
            return [str(x) for x in arr.flat]
        if arr.dtype == np.uint16:
            try:
                return arr.tobytes().decode("utf-16")
            except Exception:
                return arr.tolist()
    return arr


def _process_mat_file(mat_path: str) -> dict:
    """Load a `.mat` file and return a cleaned dictionary.
    Unwrap scalar 1×1 arrays and decode strings for readability.
    """
    raw = scipy.io.loadmat(mat_path)
    data = {k: v for k, v in raw.items() if not k.startswith("__")}
    cleaned = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.shape == (1, 1):
            val = val.item()
        cleaned[key] = _decode_matlab_string(val)
    return cleaned


def _pretty_print(data: dict):
    print("=== Annotation Contents ===")
    for key, value in data.items():
        print(f"{key}:")
        if isinstance(value, np.ndarray):
            print(f"  shape: {value.shape}, dtype: {value.dtype}")
            # Show a short preview for large arrays
            if value.size > 20:
                print(f"  preview: {value.flat[:20].tolist()} ...")
            else:
                print(f"  values: {value.tolist()}")
        else:
            pprint(value, indent=2)
    print("=== End of Annotation ===")


def main():
    parser = argparse.ArgumentParser(
        description="Display Penn Action .mat label file contents"
    )
    parser.add_argument(
        "mat_path",
        nargs="?",
        default=None,
        help="Path to a .mat file. If omitted, the script picks the first file in the dataset's label folder.",
    )
    args = parser.parse_args()

    if args.mat_path:
        mat_file = args.mat_path
    else:
        # Find the first .mat file in the standard label directory
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "dataset", "Penn_Action", "labels")
        )
        candidates = [f for f in os.listdir(base_dir) if f.lower().endswith(".mat")]
        if not candidates:
            print("[ERROR] No .mat files found in the label folder.")
            sys.exit(1)
        mat_file = os.path.join(base_dir, candidates[0])
        print(f"[INFO] No path supplied – using first label file: {mat_file}")

    if not os.path.isfile(mat_file):
        print(f"[ERROR] File not found: {mat_file}")
        sys.exit(1)

    annotation = _process_mat_file(mat_file)
    _pretty_print(annotation)


if __name__ == "__main__":
    main()
