# Purpose : Extract rich user behavior features from behavior_log parquet chunks.
#           Computes per-user stats by category (cate), brand, recency (last-action
#           timestamps), and temporal patterns (peak hour, weekday ratio).
#
# Input   : PARQUET_DIR/chunk_*.parquet  (362 snappy-compressed files)
# Output  : data/processed/user_category_stats.csv
#             cols: user, cate, pv, cart, fav, buy, cart_rate, buy_rate, has_bought
#           data/processed/user_brand_stats.csv
#             cols: user, brand, pv, cart, fav, buy
#           data/processed/user_recency_stats.csv
#             cols: user, last_pv_ts, last_cart_ts, last_fav_ts, last_buy_ts
#           data/processed/user_temporal_stats.csv
#             cols: user, peak_hour, weekday_ratio

import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR       = Path(__file__).resolve().parents[1]
FRANKLIN_DIR   = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/condescending-franklin-f538a6")
PARQUET_DIR    = FRANKLIN_DIR / "data" / "processed" / "behavior_log_parquet"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoint_phase4"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CATE_OUT     = PROCESSED_DIR / "user_category_stats.csv"
BRAND_OUT    = PROCESSED_DIR / "user_brand_stats.csv"
RECENCY_OUT  = PROCESSED_DIR / "user_recency_stats.csv"
TEMPORAL_OUT = PROCESSED_DIR / "user_temporal_stats.csv"

PROGRESS_FILE       = CHECKPOINT_DIR / "progress.pkl"
CHECKPOINT_INTERVAL = 50
BTAGS               = ["pv", "cart", "fav", "buy"]


def ckpt_path(feature: str, batch: int) -> Path:
    return CHECKPOINT_DIR / f"{feature}_batch_{batch:03d}.parquet"


def saved_batch_count(feature: str) -> int:
    return len(sorted(CHECKPOINT_DIR.glob(f"{feature}_batch_*.parquet")))


# ── per-chunk processing ────────────────────────────────────────────────────────

def process_chunk(path: Path):
    chunk = pd.read_parquet(path, columns=["user", "time_stamp", "btag", "cate", "brand"])

    # 1. category stats: pivot btag into pv/cart/fav/buy count columns
    cate_grp = (
        chunk.groupby(["user", "cate", "btag"], observed=True)
        .size()
        .unstack("btag", fill_value=0)
        .reset_index()
    )
    cate_grp.columns.name = None
    for tag in BTAGS:
        if tag not in cate_grp.columns:
            cate_grp[tag] = 0
    cate_grp = cate_grp[["user", "cate"] + BTAGS].copy()

    # 2. brand stats: same structure, skip rows with null brand
    has_brand = chunk["brand"].notna()
    if has_brand.any():
        brand_chunk = chunk.loc[has_brand]
        brand_grp = (
            brand_chunk.groupby(["user", "brand", "btag"], observed=True)
            .size()
            .unstack("btag", fill_value=0)
            .reset_index()
        )
        brand_grp.columns.name = None
        for tag in BTAGS:
            if tag not in brand_grp.columns:
                brand_grp[tag] = 0
        brand_grp = brand_grp[["user", "brand"] + BTAGS].copy()
        del brand_chunk
    else:
        brand_grp = pd.DataFrame(columns=["user", "brand"] + BTAGS)

    # 3. recency: latest time_stamp per user per btag (NaN if not observed)
    rec_grp = (
        chunk.groupby(["user", "btag"], observed=True)["time_stamp"]
        .max()
        .unstack("btag")
        .reset_index()
    )
    rec_grp.columns.name = None
    for tag in BTAGS:
        if tag not in rec_grp.columns:
            rec_grp[tag] = np.nan
    rec_grp = rec_grp[["user", "pv", "cart", "fav", "buy"]].rename(
        columns={"pv": "last_pv_ts", "cart": "last_cart_ts",
                 "fav": "last_fav_ts", "buy": "last_buy_ts"}
    )

    # 4. temporal: hour in CST (UTC+8) and weekday flag
    # avoid full datetime parsing — cheaper arithmetic on int64
    hour     = ((chunk["time_stamp"] // 3600 + 8) % 24).astype(np.int8)
    day      = (chunk["time_stamp"] + 8 * 3600) // 86400
    # 1970-01-01 is Thursday; (day + 3) % 7 gives 0=Mon…6=Sun
    is_wkday = ((day + 3) % 7 < 5).astype(np.int8)

    tmp = pd.DataFrame({
        "user":     chunk["user"].values,
        "hour":     hour.values,
        "is_wkday": is_wkday.values,
    })
    hour_grp = tmp.groupby(["user", "hour"], sort=False).size().reset_index(name="cnt")
    day_grp  = (
        tmp.groupby("user", sort=False)
        .agg(total=("is_wkday", "count"), weekday_cnt=("is_wkday", "sum"))
        .reset_index()
    )

    del chunk, tmp, hour, day, is_wkday
    gc.collect()

    return cate_grp, brand_grp, rec_grp, hour_grp, day_grp


# ── checkpoint helpers ──────────────────────────────────────────────────────────

def flush_batch(feature: str, batch_dfs: list, batch_num: int,
                group_cols: list, agg_cols: list, agg_func: str = "sum"):
    """Concat, group-reduce, and save one batch to a parquet checkpoint file."""
    if not batch_dfs:
        return
    combined = pd.concat(batch_dfs, ignore_index=True)
    if len(combined) == 0:
        merged = combined
    elif agg_func == "sum":
        merged = combined.groupby(group_cols, sort=False)[agg_cols].sum().reset_index()
    else:
        merged = combined.groupby(group_cols, sort=False)[agg_cols].max().reset_index()
    merged.to_parquet(ckpt_path(feature, batch_num), index=False)
    del combined, merged
    gc.collect()


# ── pass 2: streaming merge of all batch checkpoints ───────────────────────────

def merge_all_batches(feature: str, group_cols: list, agg_cols: list,
                      agg_func: str = "sum") -> pd.DataFrame:
    batch_files = sorted(CHECKPOINT_DIR.glob(f"{feature}_batch_*.parquet"))
    print(f"  Merging {len(batch_files)} batch files for '{feature}' ...")
    acc = None
    for bf in batch_files:
        batch = pd.read_parquet(bf)
        if acc is None:
            acc = batch
        else:
            combined = pd.concat([acc, batch], ignore_index=True)
            del acc, batch
            if agg_func == "sum":
                acc = combined.groupby(group_cols, sort=False)[agg_cols].sum().reset_index()
            else:
                acc = combined.groupby(group_cols, sort=False)[agg_cols].max().reset_index()
            del combined
            gc.collect()
    return acc if acc is not None else pd.DataFrame(columns=group_cols + agg_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — process chunks, checkpoint every 50
# ═══════════════════════════════════════════════════════════════════════════════

parquet_files = sorted(PARQUET_DIR.glob("chunk_*.parquet"))
n_total = len(parquet_files)
print(f"Found {n_total} parquet chunks.")

start_idx = 0
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, "rb") as f:
        saved = pickle.load(f)
    start_idx = saved["chunk_idx"] + 1
    batch_num = saved["batch_num"]
    print(f"Resuming from chunk {start_idx}/{n_total}, batch_num={batch_num}")
else:
    batch_num = saved_batch_count("cate")
    print(f"Starting fresh (batch_num={batch_num}).")

cate_buf, brand_buf, rec_buf, hour_buf, day_buf = [], [], [], [], []

t_start = time.time()

for i, path in enumerate(parquet_files):
    if i < start_idx:
        continue

    cate_df, brand_df, rec_df, hour_df, day_df = process_chunk(path)
    cate_buf.append(cate_df)
    brand_buf.append(brand_df)
    rec_buf.append(rec_df)
    hour_buf.append(hour_df)
    day_buf.append(day_df)

    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{n_total} chunks, elapsed {time.time() - t_start:.0f}s")

    is_last = (i == n_total - 1)
    if (i + 1) % CHECKPOINT_INTERVAL == 0 or is_last:
        print(f"  Saving batch {batch_num} (up to chunk {i + 1}) ...")
        flush_batch("cate",  cate_buf,  batch_num, ["user", "cate"],  BTAGS,                                                   "sum")
        flush_batch("brand", brand_buf, batch_num, ["user", "brand"], BTAGS,                                                   "sum")
        flush_batch("rec",   rec_buf,   batch_num, ["user"],          ["last_pv_ts", "last_cart_ts", "last_fav_ts", "last_buy_ts"], "max")
        flush_batch("hour",  hour_buf,  batch_num, ["user", "hour"],  ["cnt"],                                                  "sum")
        flush_batch("day",   day_buf,   batch_num, ["user"],          ["total", "weekday_cnt"],                                 "sum")

        cate_buf, brand_buf, rec_buf, hour_buf, day_buf = [], [], [], [], []
        batch_num += 1

        with open(PROGRESS_FILE, "wb") as f:
            pickle.dump({"chunk_idx": i, "batch_num": batch_num}, f)

        gc.collect()

print(f"\nPass 1 done in {time.time() - t_start:.0f}s. Building final outputs ...")


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — merge checkpoints and write final CSVs
# ═══════════════════════════════════════════════════════════════════════════════

# user_category_stats
if not CATE_OUT.exists():
    print("Building user_category_stats.csv ...")
    cate = merge_all_batches("cate", ["user", "cate"], BTAGS, "sum")
    cate["cart_rate"]  = np.where(cate["pv"]   > 0, cate["cart"] / cate["pv"],   0.0)
    cate["buy_rate"]   = np.where(cate["cart"] > 0, cate["buy"]  / cate["cart"],  0.0)
    cate["has_bought"] = (cate["buy"] > 0).astype(np.int8)
    cate.to_csv(CATE_OUT, index=False)
    print(f"  Saved {len(cate):,} rows → {CATE_OUT.name}")
    del cate; gc.collect()
else:
    print(f"  {CATE_OUT.name} already exists, skipping.")

# user_brand_stats
if not BRAND_OUT.exists():
    print("Building user_brand_stats.csv ...")
    brand = merge_all_batches("brand", ["user", "brand"], BTAGS, "sum")
    brand.to_csv(BRAND_OUT, index=False)
    print(f"  Saved {len(brand):,} rows → {BRAND_OUT.name}")
    del brand; gc.collect()
else:
    print(f"  {BRAND_OUT.name} already exists, skipping.")

# user_recency_stats
if not RECENCY_OUT.exists():
    print("Building user_recency_stats.csv ...")
    rec = merge_all_batches("rec", ["user"],
                            ["last_pv_ts", "last_cart_ts", "last_fav_ts", "last_buy_ts"], "max")
    rec[["last_pv_ts", "last_cart_ts", "last_fav_ts", "last_buy_ts"]] = (
        rec[["last_pv_ts", "last_cart_ts", "last_fav_ts", "last_buy_ts"]]
        .fillna(0)
        .astype(np.int64)
    )
    rec.to_csv(RECENCY_OUT, index=False)
    print(f"  Saved {len(rec):,} rows → {RECENCY_OUT.name}")
    del rec; gc.collect()
else:
    print(f"  {RECENCY_OUT.name} already exists, skipping.")

# user_temporal_stats
if not TEMPORAL_OUT.exists():
    print("Building user_temporal_stats.csv ...")
    hour = merge_all_batches("hour", ["user", "hour"], ["cnt"], "sum")
    day  = merge_all_batches("day",  ["user"], ["total", "weekday_cnt"], "sum")

    # peak_hour: hour (CST) with the highest cumulative action count per user
    hour = hour.reset_index(drop=True)
    peak_idx  = hour.groupby("user")["cnt"].idxmax()
    peak      = hour.loc[peak_idx, ["user", "hour"]].rename(
        columns={"hour": "peak_hour"}).reset_index(drop=True)
    del hour; gc.collect()

    day["weekday_ratio"] = day["weekday_cnt"] / day["total"]
    temporal = peak.merge(day[["user", "weekday_ratio"]], on="user", how="left")
    temporal.to_csv(TEMPORAL_OUT, index=False)
    print(f"  Saved {len(temporal):,} rows → {TEMPORAL_OUT.name}")
    del day, peak, temporal; gc.collect()
else:
    print(f"  {TEMPORAL_OUT.name} already exists, skipping.")

print(f"\nTotal elapsed: {time.time() - t_start:.0f}s")
print("Done.")
