from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, Tuple


def _norm_label(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"1", "true", "anomaly", "yes"}:
        return "1"
    if s in {"0", "false", "no"}:
        return "0"
    if s in {"undefined", "undef", ""}:
        return "undefined"
    return "undefined"


def _pct(n: int, total: int) -> float:
    return 0.0 if total <= 0 else (100.0 * n / total)


def log_label_stats_by_core(labels_csv: str, core_col: str = "num_cores", label_col: str = "label") -> None:
    """Print label counts and percentages grouped by core count."""
    if not os.path.exists(labels_csv):
        print(f"[STATS_BY_CORE] labels file not found: {labels_csv}")
        return

    buckets: Dict[int, Dict[str, int]] = defaultdict(lambda: {"1": 0, "0": 0, "undefined": 0})
    totals: Dict[int, int] = defaultdict(int)

    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        if core_col not in fields:
            print(f"[STATS_BY_CORE] missing column '{core_col}' in {labels_csv}. Available: {fields}")
            return
        if label_col not in fields:
            print(f"[STATS_BY_CORE] missing column '{label_col}' in {labels_csv}. Available: {fields}")
            return

        for r in reader:
            core_raw = (r.get(core_col) or "").strip()
            if core_raw == "":
                continue
            try:
                cores = int(float(core_raw))
            except Exception:
                continue

            lab = _norm_label(r.get(label_col, "undefined"))
            buckets[cores][lab] += 1
            totals[cores] += 1

    if not totals:
        print("[STATS_BY_CORE] no rows found (or cores missing).")
        return

    for cores in sorted(totals.keys()):
        t = totals[cores]
        c1 = buckets[cores]["1"]
        c0 = buckets[cores]["0"]
        cu = buckets[cores]["undefined"]

        print(
            f"[STATS_BY_CORE] cores={cores} | total={t} | "
            f"1: {c1} ({_pct(c1, t):.2f}%) | "
            f"0: {c0} ({_pct(c0, t):.2f}%) | "
            f"undef: {cu} ({_pct(cu, t):.2f}%)"
        )
