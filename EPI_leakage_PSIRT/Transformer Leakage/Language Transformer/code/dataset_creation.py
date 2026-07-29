#!/usr/bin/env python3
import os
import re
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


_RE_1 = re.compile(r"^(?P<model>.+?)\.csv_run(?P<run>\d+)\.csv$")
_RE_2 = re.compile(r"^(?P<model>.+?)_run(?P<run>\d+)\.csv$")


def parse_model_run(path: str) -> Tuple[str, int]:
    base = os.path.basename(path)
    m = _RE_1.match(base)
    if m:
        return m.group("model"), int(m.group("run"))
    m = _RE_2.match(base)
    if m:
        return m.group("model"), int(m.group("run"))
    return "", -1


def group_files(input_dir: str) -> Dict[str, List[str]]:
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for f in files:
        model, run = parse_model_run(f)
        if not model:
            continue
        groups.setdefault(model, []).append((run, f))
    return {m: [p for _, p in sorted(v, key=lambda t: t[0])] for m, v in groups.items()}


def load_model_aligned(files: List[str], epi_col: str, busy_col: str) -> pd.DataFrame:
    parts = []
    for f in files:
        df = pd.read_csv(f)
        if epi_col not in df.columns or busy_col not in df.columns:
            continue
        epi = pd.to_numeric(df[epi_col], errors="coerce")
        busy = pd.to_numeric(df[busy_col], errors="coerce")
        d = pd.DataFrame({epi_col: epi, busy_col: busy}).dropna()
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=[epi_col, busy_col])
    return pd.concat(parts, axis=0, ignore_index=True)


def build_windows(epi: np.ndarray, busy: np.ndarray, window: int, stride: int, max_windows: Optional[int]):
    n = min(epi.size, busy.size)
    if n < window:
        return []
    out = []
    count = 0
    for start in range(0, n - window + 1, stride):
        e = epi[start:start + window]
        b = busy[start:start + window]
        if e.size != window or b.size != window:
            continue
        out.append((e.astype(float).tolist(), b.astype(float).tolist()))
        count += 1
        if max_windows is not None and count >= max_windows:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".", help="Folder with run CSVs")
    ap.add_argument("--out_csv", default="dataset_seq100.csv", help="Output dataset CSV")
    ap.add_argument("--epi_col", default="epi_delta")
    ap.add_argument("--busy_col", default="busy_fraction")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10, help="Use 10 or 5 to get more samples (overlapping windows).")
    ap.add_argument("--max_windows_per_model", type=int, default=0, help="0 means no cap.")
    args = ap.parse_args()

    groups = group_files(args.input_dir)
    if not groups:
        raise SystemExit("No matching run files found (e.g., <model>.csv_run01.csv).")

    rows = []
    maxw = None if args.max_windows_per_model == 0 else args.max_windows_per_model

    for model, files in sorted(groups.items()):
        df = load_model_aligned(files, args.epi_col, args.busy_col)
        if df.empty:
            print(f"[WARN] {model}: no usable data (missing cols or all NaN).")
            continue

        epi = df[args.epi_col].to_numpy()
        busy = df[args.busy_col].to_numpy()

        inst = build_windows(epi, busy, args.window, args.stride, maxw)
        print(f"[INFO] {model}: aligned_rows={len(df)}, instances={len(inst)}")

        for e_list, b_list in inst:
            rows.append({
                "epi_delta": json.dumps(e_list),          # JSON array in CSV
                "busy_fraction": json.dumps(b_list),
                "label": model
            })

    if not rows:
        raise SystemExit("No dataset rows created. Need >=100 aligned samples per model.")

    ds = pd.DataFrame(rows)
    ds.to_csv(args.out_csv, index=False)

    print(f"[OK] Saved dataset: {args.out_csv} (rows={len(ds)}, classes={ds['label'].nunique()})")
    print("[OK] Per-class counts:")
    print(ds["label"].value_counts())


if __name__ == "__main__":
    main()
