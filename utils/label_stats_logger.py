from __future__ import annotations

import os
import csv

from typing import Dict


def compute_label_stats(labels_csv: str) -> Dict[str, int]:
    """
    Read labels.csv and compute cumulative counts of labels.
    Returns a dict with keys: total, ones, zeros, undefined.
    """
    stats = {
        "total": 0,
        "ones": 0,
        "zeros": 0,
        "undefined": 0,
    }

    if not os.path.exists(labels_csv):
        return stats

    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "label" not in (reader.fieldnames or []):
            raise ValueError("labels.csv has no 'label' column.")

        for row in reader:
            lbl = row.get("label", "").strip()
            stats["total"] += 1
            if lbl == "1":
                stats["ones"] += 1
            elif lbl == "0":
                stats["zeros"] += 1
            else:
                stats["undefined"] += 1

    return stats


def log_label_stats(labels_csv: str, prefix: str = "[STATS]") -> None:
    """
    Compute and print cumulative label statistics.
    """
    s = compute_label_stats(labels_csv)
    total = s["total"]

    if total == 0:
        print(f"{prefix} No labels yet.")
        return

    p1 = 100.0 * s["ones"] / total
    p0 = 100.0 * s["zeros"] / total
    pu = 100.0 * s["undefined"] / total

    print(
        f"{prefix} total={total} | "
        f"1: {s['ones']} ({p1:.2f}%) | "
        f"0: {s['zeros']} ({p0:.2f}%) | "
        f"undef: {s['undefined']} ({pu:.2f}%)"
    )
