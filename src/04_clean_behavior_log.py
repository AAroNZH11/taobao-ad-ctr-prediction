# Purpose : Chunked cleaning and parquet conversion for behavior_log.csv
#           behavior_log is 23 GB and cannot be loaded all at once.
#           Strategy:
#             1. Read in chunks of 2 M rows.
#             2. Per chunk: drop exact duplicates, filter invalid timestamps.
#             3. Save each cleaned chunk as snappy-compressed parquet.
#             4. Accumulate btag counts, null counts, and dedup stats.
#           If parquet output already exists (e.g. from a prior run), the script
#           reads stats from those files instead of re-scanning the raw CSV.
#
# Input   : data/behavior_log.csv  (raw, ~23 GB)
# Outputs : data/processed/behavior_log_parquet/chunk_NNNN.parquet
#           outputs/stats/behavior_log_report.json
#           outputs/plots/behavior_log_btag_pie.png
#           outputs/plots/behavior_log_distributions.png

import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PARQUET_DIR = PROCESSED_DIR / "behavior_log_parquet"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
STATS_DIR = BASE_DIR / "outputs" / "stats"

for d in [PARQUET_DIR, PLOTS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 2_000_000
# Valid timestamp window: 2017-01-01 to 2020-01-01
TS_LOW = 1_483_228_800
TS_HIGH = 1_577_836_800

# ── Check for existing parquet files ─────────────────────────────────────────
existing_parquet = sorted(PARQUET_DIR.glob("chunk_*.parquet"))

if existing_parquet:
    # ── Fast path: accumulate stats from already-cleaned parquet files ────────
    print(f"Found {len(existing_parquet)} existing parquet chunks — computing stats from them.")
    t0 = time.time()

    btag_counts: dict[str, int] = {}
    null_user = null_ts = null_btag = null_cate = null_brand = 0
    total_rows = 0

    for i, path in enumerate(existing_parquet, 1):
        chunk = pd.read_parquet(path, columns=["user", "time_stamp", "btag", "cate", "brand"])
        total_rows += len(chunk)
        null_user += int(chunk["user"].isnull().sum())
        null_ts += int(chunk["time_stamp"].isnull().sum())
        null_btag += int(chunk["btag"].isnull().sum())
        null_cate += int(chunk["cate"].isnull().sum())
        null_brand += int(chunk["brand"].isnull().sum())
        for tag, cnt in chunk["btag"].value_counts().items():
            btag_counts[tag] = btag_counts.get(tag, 0) + int(cnt)
        del chunk
        gc.collect()
        if i % 50 == 0:
            print(f"  Scanned {i}/{len(existing_parquet)} chunks …")

    # Read the cleaning metadata written by a prior full run (local stats dir only).
    cleaning_meta_path = STATS_DIR / "behavior_cleaning.json"
    if cleaning_meta_path.exists():
        meta = json.loads(cleaning_meta_path.read_text())
        total_raw = meta["total_raw"]
        dedup_removed = meta["dedup_removed"]
        bad_ts = meta["bad_ts"]
    else:
        total_raw = None
        dedup_removed = None
        bad_ts = None

    elapsed = (time.time() - t0) / 60
    parquet_gb = sum(p.stat().st_size for p in existing_parquet) / 1e9

else:
    # ── Slow path: process raw CSV in chunks ──────────────────────────────────
    print("No parquet files found — processing behavior_log.csv from scratch.")
    print("This will take ~8–15 minutes depending on hardware.")
    t0 = time.time()

    btag_counts: dict[str, int] = {}
    null_user = null_ts = null_btag = null_cate = null_brand = 0
    total_raw = 0
    dedup_removed = 0
    bad_ts = 0
    chunk_idx = 0

    for chunk in pd.read_csv(
        DATA_DIR / "behavior_log.csv",
        chunksize=CHUNKSIZE,
        dtype={"user": "int32", "time_stamp": "int64",
               "btag": "category", "cate": "int32"},
    ):
        total_raw += len(chunk)

        # Exact duplicate removal (logging artifact: all 5 columns identical)
        n_before = len(chunk)
        chunk.drop_duplicates(inplace=True)
        dedup_removed += n_before - len(chunk)

        # Invalid timestamp filter
        mask_valid = chunk["time_stamp"].between(TS_LOW, TS_HIGH)
        bad_ts += (~mask_valid).sum()
        chunk = chunk[mask_valid]

        # Accumulate null counts
        null_user += int(chunk["user"].isnull().sum())
        null_ts += int(chunk["time_stamp"].isnull().sum())
        null_btag += int(chunk["btag"].isnull().sum())
        null_cate += int(chunk["cate"].isnull().sum())
        null_brand += int(chunk["brand"].isnull().sum())

        # Accumulate btag distribution
        for tag, cnt in chunk["btag"].value_counts().items():
            btag_counts[str(tag)] = btag_counts.get(str(tag), 0) + int(cnt)

        # Save chunk as parquet
        chunk_idx += 1
        out_path = PARQUET_DIR / f"chunk_{chunk_idx:04d}.parquet"
        chunk.to_parquet(out_path, compression="snappy", index=False)

        del chunk
        gc.collect()

        if chunk_idx % 20 == 0:
            elapsed_so_far = (time.time() - t0) / 60
            print(f"  Chunk {chunk_idx:04d} done — {elapsed_so_far:.1f} min elapsed")

    elapsed = (time.time() - t0) / 60
    parquet_gb = sum(p.stat().st_size for p in sorted(PARQUET_DIR.glob("chunk_*.parquet"))) / 1e9

    # Save cleaning metadata for subsequent runs
    meta = {
        "total_raw": total_raw,
        "after_dedup": total_raw - dedup_removed,
        "dedup_removed": dedup_removed,
        "dedup_pct": round(dedup_removed / total_raw * 100, 2),
        "bad_ts": int(bad_ts),
        "final_clean": total_raw - dedup_removed - int(bad_ts),
    }
    (STATS_DIR / "behavior_cleaning.json").write_text(json.dumps(meta, indent=2))

# ── Compile final report ──────────────────────────────────────────────────────
total_btag = sum(btag_counts.values())
btag_pct = {k: round(v / total_btag * 100, 4) for k, v in btag_counts.items()}

nulls = {
    "user": null_user, "time_stamp": null_ts, "btag": null_btag,
    "cate": null_cate, "brand": null_brand,
}

report = {
    "total_rows": total_raw,
    "nulls": nulls,
    "intra_chunk_duplicates": dedup_removed,
    "btag_counts": btag_counts,
    "btag_pct": btag_pct,
    "ts_filter_low": TS_LOW,
    "ts_filter_high": TS_HIGH,
    "parquet_size_gb": round(parquet_gb, 3),
    "num_chunks": len(existing_parquet) if existing_parquet else chunk_idx,
    "processing_minutes": round(elapsed, 1),
}
(STATS_DIR / "behavior_log_report.json").write_text(json.dumps(report, indent=2))
print(f"  Report saved → {STATS_DIR / 'behavior_log_report.json'}")

# ── Btag pie chart ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
labels = [f"{k}\n({btag_pct[k]:.1f}%)" for k in btag_counts]
ax.pie(list(btag_counts.values()), labels=labels, autopct="",
       colors=["steelblue", "salmon", "mediumseagreen", "gold"],
       startangle=90, pctdistance=0.8)
ax.set_title("behavior_log — btag Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "behavior_log_btag_pie.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'behavior_log_btag_pie.png'}")

# ── Distributions bar chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
tags = list(btag_counts.keys())
counts = list(btag_counts.values())
bars = ax.bar(tags, counts, color=["steelblue", "salmon", "mediumseagreen", "gold"])
ax.set_title("behavior_log — Behavior Type Counts (after dedup)", fontsize=12)
ax.set_xlabel("Behavior Type")
ax.set_ylabel("Record Count")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{cnt/1e6:.0f}M", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "behavior_log_distributions.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'behavior_log_distributions.png'}")

print(f"Done. Parquet size: {parquet_gb:.3f} GB | {report['num_chunks']} chunks | {elapsed:.1f} min")
