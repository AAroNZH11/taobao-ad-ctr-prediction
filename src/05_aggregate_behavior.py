# Purpose : Aggregate per-user behavior counts from cleaned behavior_log parquet files.
#           Reads all parquet chunks produced by 04_clean_behavior_log.py and
#           computes per-user: pv/cart/fav/buy counts, derived rates, and a
#           purchase indicator flag.
#
# Input   : data/processed/behavior_log_parquet/*.parquet
# Output  : data/processed/user_behavior_stats.csv

import gc
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PARQUET_DIR = BASE_DIR / "data" / "processed" / "behavior_log_parquet"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_FILE = PROCESSED_DIR / "user_behavior_stats.csv"

if OUT_FILE.exists():
    print(f"Output already exists: {OUT_FILE}")
    print("Delete it and re-run if you want to recompute.")
else:
    parquet_files = sorted(PARQUET_DIR.glob("chunk_*.parquet"))
    print(f"Found {len(parquet_files)} parquet chunks.")

    # Accumulate per-user, per-btag counts by processing chunks one at a time.
    # Holding a full pivot of 1.1M users × 4 btag cols in RAM is safe (~20 MB).
    user_counts: dict[str, dict[str, int]] = {}

    for i, path in enumerate(parquet_files, 1):
        chunk = pd.read_parquet(path, columns=["user", "btag"])
        chunk_pivot = chunk.groupby(["user", "btag"], observed=True).size().unstack(fill_value=0)
        for tag in ["pv", "cart", "fav", "buy"]:
            if tag not in chunk_pivot.columns:
                chunk_pivot[tag] = 0

        for tag in ["pv", "cart", "fav", "buy"]:
            for uid, cnt in chunk_pivot[tag].items():
                if uid not in user_counts:
                    user_counts[uid] = {"pv": 0, "cart": 0, "fav": 0, "buy": 0}
                user_counts[uid][tag] += int(cnt)

        del chunk, chunk_pivot
        gc.collect()

        if i % 50 == 0:
            print(f"  Processed {i}/{len(parquet_files)} chunks …")

    print("Building output DataFrame …")
    rows = []
    for uid, counts in user_counts.items():
        pv = counts["pv"]
        cart = counts["cart"]
        fav = counts["fav"]
        buy = counts["buy"]
        total = pv + cart + fav + buy
        rows.append({
            "user": uid,
            "pv_count": pv,
            "cart_count": cart,
            "fav_count": fav,
            "buy_count": buy,
            "total_actions": total,
            "has_purchase": 1 if buy > 0 else 0,
            # Rates relative to page views; guard against zero-pv users
            "buy_rate": round(buy / pv, 6) if pv > 0 else 0.0,
            "cart_rate": round(cart / pv, 6) if pv > 0 else 0.0,
        })

    del user_counts
    gc.collect()

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(stats):,} users → {OUT_FILE}")
    print(f"  has_purchase=1 : {stats['has_purchase'].sum():,}")
    print(f"  has_purchase=0 : {(stats['has_purchase'] == 0).sum():,}")

print("Done.")
