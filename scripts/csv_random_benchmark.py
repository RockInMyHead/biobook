#!/usr/bin/env python3
"""Случайный большой CSV + pandas (как в ячейке Bio-Notebook)."""
import os
import time

import numpy as np
import pandas as pd

ROWS = int(os.environ.get("CSV_BENCH_ROWS", "500_000").replace("_", ""))
OUT_DIR = os.environ.get("CSV_BENCH_DIR", "/tmp")
CSV_PATH = os.path.join(OUT_DIR, "random_large.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary_stats.csv")
TEST_PATH = os.path.join(OUT_DIR, "test_results.csv")


def main() -> None:
    print(f"pandas {pd.__version__} | rows={ROWS:,}")
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    df = pd.DataFrame(
        {
            **{f"col_{i}": rng.random(ROWS) for i in range(6)},
            "group": rng.integers(0, 40, size=ROWS),
            "label": rng.choice(["A", "B", "C", "D"], size=ROWS),
        }
    )
    df.to_csv(CSV_PATH, index=False)
    mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
    print(f"write {CSV_PATH}: {mb:.1f} MiB in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    print(f"read_csv shape={df.shape} in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    summary = df.groupby("group")[["col_0", "col_1", "col_2"]].agg(["mean", "std", "count"])
    by_label = df.groupby("label").size().rename("n")
    print(f"groupby in {time.perf_counter() - t0:.2f}s")
    print(summary.head(4).to_string())

    summary.to_csv(SUMMARY_PATH)
    by_label.to_csv(TEST_PATH)
    print(f"saved {SUMMARY_PATH}, {TEST_PATH}")
    print(df[["col_0", "col_1", "col_2"]].corr().round(4).to_string())


if __name__ == "__main__":
    main()
