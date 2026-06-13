# Purpose : Join all feature sources onto raw_sample to produce the final
#           feature matrix. Processes raw_sample in chunks to stay within memory.
#
# Inputs  : raw_sample.csv
#           user_embeddings.csv, ad_embeddings.csv    (beautiful-dhawan worktree)
#           user_profile.csv, ad_feature.csv          (DATA_DIR)
#           user_behavior_stats.csv                   (condescending-franklin worktree)
#           user_category_stats.csv, user_brand_stats.csv,
#           user_recency_stats.csv, user_temporal_stats.csv  (this worktree, Phase 4)
#
# Output  : data/processed/features_train.csv  (time_stamp < 1494633600)
#           data/processed/features_test.csv   (time_stamp >= 1494633600)

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_DIR      = Path("/Users/aaronzhong/Documents/TaoBao-Project")
FRANKLIN_DIR  = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/condescending-franklin-f538a6")
DHAWAN_DIR    = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/beautiful-dhawan-257e0a")
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = PROCESSED_DIR / "features_train.csv"
TEST_OUT  = PROCESSED_DIR / "features_test.csv"

TEST_CUTOFF = 1494633600  # 2017-05-13 00:00:00 UTC
CHUNK_SIZE  = 300_000

# pid encoding (only 2 unique values in dataset)
PID_MAP = {"430548_1007": 0, "430539_1007": 1}

SVD_DIMS = 32

# ── skip if outputs already exist ────────────────────────────────────────────
if TRAIN_OUT.exists() and TEST_OUT.exists():
    print("Both output files exist. Delete them to recompute.")
    print("Done.")
    import sys; sys.exit(0)

# ── load all lookup tables into memory ───────────────────────────────────────

print("Loading lookup tables ...")

t0 = time.time()

# user SVD embeddings: index on user_id (cast to int32 to match chunk["user"])
user_emb = pd.read_csv(
    DHAWAN_DIR / "data" / "processed" / "user_embeddings.csv",
    dtype={"user_id": "int32", **{f"svd_{i}": np.float32 for i in range(SVD_DIMS)}}
)
user_emb = user_emb.rename(columns={f"svd_{i}": f"user_svd_{i}" for i in range(SVD_DIMS)})
user_emb = user_emb.set_index("user_id")
print(f"  user_emb: {len(user_emb):,} rows  ({time.time()-t0:.1f}s)")

# ad SVD embeddings: index on adgroup_id (int32)
ad_emb = pd.read_csv(
    DHAWAN_DIR / "data" / "processed" / "ad_embeddings.csv",
    dtype={"adgroup_id": "int32", **{f"svd_{i}": np.float32 for i in range(SVD_DIMS)}}
)
ad_emb = ad_emb.rename(columns={f"svd_{i}": f"ad_svd_{i}" for i in range(SVD_DIMS)})
ad_emb = ad_emb.set_index("adgroup_id")
print(f"  ad_emb:   {len(ad_emb):,} rows  ({time.time()-t0:.1f}s)")

# ad_feature: index on adgroup_id; compute derived cols upfront
ad_feat = pd.read_csv(
    DATA_DIR / "ad_feature.csv",
    dtype={"adgroup_id": "int32", "cate_id": "int32",
           "campaign_id": "int32", "customer": "int32"},
)
ad_feat["log1p_price"]  = np.log1p(ad_feat["price"].fillna(0).astype(np.float32))
ad_feat["brand_known"]  = ad_feat["brand"].notna().astype(np.int8)
ad_feat["brand"]        = ad_feat["brand"].fillna(-1.0)  # sentinel for "no brand"
ad_feat = ad_feat.drop(columns=["price"])
ad_feat = ad_feat.set_index("adgroup_id")
print(f"  ad_feat:  {len(ad_feat):,} rows  ({time.time()-t0:.1f}s)")

# user_profile: index on userid; pvalue_known indicator; fill missing with 0
user_prof = pd.read_csv(
    DATA_DIR / "user_profile.csv",
    dtype={"userid": "int32"},
)
user_prof = user_prof.rename(columns={"userid": "user"})
user_prof["pvalue_known"] = user_prof["pvalue_level"].notna().astype(np.int8)
user_prof = user_prof.fillna(0)
for c in user_prof.select_dtypes("float64").columns:
    user_prof[c] = user_prof[c].astype(np.float32)
user_prof = user_prof.set_index("user")
print(f"  user_prof:{len(user_prof):,} rows  ({time.time()-t0:.1f}s)")

# user_behavior_stats: index on user
beh_stats = pd.read_csv(
    FRANKLIN_DIR / "data" / "processed" / "user_behavior_stats.csv",
    dtype={"user": "int32"},
)
beh_stats = beh_stats.rename(columns={
    "pv_count": "beh_pv", "cart_count": "beh_cart",
    "fav_count": "beh_fav", "buy_count": "beh_buy",
    "total_actions": "beh_total",
    "has_purchase": "beh_has_purchase",
    "buy_rate": "beh_buy_rate", "cart_rate": "beh_cart_rate",
})
beh_stats = beh_stats.set_index("user")
print(f"  beh_stats:{len(beh_stats):,} rows  ({time.time()-t0:.1f}s)")

# user_category_stats: index on (user, cate) for cross-feature lookup
cate_stats = pd.read_csv(
    PROCESSED_DIR / "user_category_stats.csv",
    dtype={"user": "int32", "cate": "int32",
           "pv": "int32", "cart": "int32", "fav": "int32", "buy": "int32",
           "has_bought": "int8"},
)
cate_stats["cart_rate"] = cate_stats["cart_rate"].astype(np.float32)
cate_stats["buy_rate"]  = cate_stats["buy_rate"].astype(np.float32)
cate_stats = cate_stats.rename(columns={
    "pv": "ucate_pv", "cart": "ucate_cart", "fav": "ucate_fav", "buy": "ucate_buy",
    "cart_rate": "ucate_cart_rate", "buy_rate": "ucate_buy_rate",
    "has_bought": "ucate_has_bought",
})
cate_stats = cate_stats.set_index(["user", "cate"])
print(f"  cate_stats:{len(cate_stats):,} rows  ({time.time()-t0:.1f}s)")

# user_brand_stats: index on (user, brand)
brand_stats = pd.read_csv(
    PROCESSED_DIR / "user_brand_stats.csv",
    dtype={"user": "int32",
           "pv": "int32", "cart": "int32", "fav": "int32", "buy": "int32"},
)
brand_stats = brand_stats.rename(columns={
    "pv": "ubrand_pv", "cart": "ubrand_cart",
    "fav": "ubrand_fav", "buy": "ubrand_buy",
})
brand_stats["brand"] = brand_stats["brand"].astype(np.float64)
brand_stats = brand_stats.set_index(["user", "brand"])
print(f"  brand_stats:{len(brand_stats):,} rows  ({time.time()-t0:.1f}s)")

# user_recency_stats: index on user
rec_stats = pd.read_csv(
    PROCESSED_DIR / "user_recency_stats.csv",
    dtype={"user": "int32",
           "last_pv_ts": "int64", "last_cart_ts": "int64",
           "last_fav_ts": "int64", "last_buy_ts": "int64"},
)
rec_stats = rec_stats.set_index("user")
print(f"  rec_stats: {len(rec_stats):,} rows  ({time.time()-t0:.1f}s)")

# user_temporal_stats: index on user
tmp_stats = pd.read_csv(
    PROCESSED_DIR / "user_temporal_stats.csv",
    dtype={"user": "int32", "peak_hour": "int8"},
)
tmp_stats["weekday_ratio"] = tmp_stats["weekday_ratio"].astype(np.float32)
tmp_stats = tmp_stats.set_index("user")
print(f"  tmp_stats: {len(tmp_stats):,} rows  ({time.time()-t0:.1f}s)")

gc.collect()
print(f"All lookup tables loaded in {time.time()-t0:.1f}s")


# ── helpers ─────────────────────────────────────────────────────────────────

def reindex_lookup(table: pd.DataFrame, keys) -> pd.DataFrame:
    """Reindex table by keys, preserving column order; missing rows become NaN."""
    return table.reindex(keys).reset_index(drop=True)


# ── open output files (write header on first write) ─────────────────────────

TRAIN_OUT.unlink(missing_ok=True)
TEST_OUT.unlink(missing_ok=True)
train_header_written = False
test_header_written  = False

# ── stream raw_sample in chunks ──────────────────────────────────────────────

print(f"\nProcessing raw_sample.csv in chunks of {CHUNK_SIZE:,} ...")

reader = pd.read_csv(
    DATA_DIR / "raw_sample.csv",
    dtype={"user": "int32", "adgroup_id": "int32"},
    chunksize=CHUNK_SIZE,
)

total_rows = 0
t_loop = time.time()
chunk_idx = 0

for chunk in reader:
    chunk_idx += 1
    n = len(chunk)
    total_rows += n

    # ── 1. drop redundant label column ──────────────────────────────────────
    chunk = chunk.drop(columns=["nonclk"])

    # ── 2. encode pid ────────────────────────────────────────────────────────
    chunk["pid"] = chunk["pid"].map(PID_MAP).fillna(0).astype(np.int8)

    # ── 3. join ad_feature → cate_id, campaign_id, customer, brand, log1p_price, brand_known
    ad_keys = chunk["adgroup_id"].values
    ad_joined = reindex_lookup(ad_feat, ad_keys)
    chunk = pd.concat([chunk.reset_index(drop=True), ad_joined], axis=1)
    del ad_joined

    # ── 4. user SVD embeddings (cold start → 0) ──────────────────────────────
    u_keys = chunk["user"].values
    u_emb_joined = reindex_lookup(user_emb, u_keys).fillna(0)
    chunk = pd.concat([chunk, u_emb_joined], axis=1)
    del u_emb_joined

    # ── 5. ad SVD embeddings (cold start → 0) ────────────────────────────────
    a_emb_joined = reindex_lookup(ad_emb, ad_keys).fillna(0)
    chunk = pd.concat([chunk, a_emb_joined], axis=1)
    del a_emb_joined

    # ── 6. user_profile (pvalue_known already added; all NaN → 0) ────────────
    prof_joined = reindex_lookup(user_prof, u_keys).fillna(0)
    chunk = pd.concat([chunk, prof_joined], axis=1)
    del prof_joined

    # ── 7. user_behavior_stats (cold start → 0) ──────────────────────────────
    beh_joined = reindex_lookup(beh_stats, u_keys).fillna(0)
    chunk = pd.concat([chunk, beh_joined], axis=1)
    del beh_joined

    # ── 8. user × category cross-feature (use candidate ad's cate_id) ────────
    cate_ids = chunk["cate_id"].values.astype(np.int32)
    cate_keys = pd.MultiIndex.from_arrays([u_keys, cate_ids])
    cate_joined = reindex_lookup(cate_stats, cate_keys).fillna(0)
    chunk = pd.concat([chunk, cate_joined], axis=1)
    del cate_joined, cate_keys

    # ── 9. user × brand cross-feature (use candidate ad's brand) ─────────────
    brand_vals = chunk["brand"].values  # -1.0 for "no brand"
    brand_keys = pd.MultiIndex.from_arrays([u_keys, brand_vals])
    brand_joined = reindex_lookup(brand_stats, brand_keys).fillna(0)
    chunk = pd.concat([chunk, brand_joined], axis=1)
    del brand_joined, brand_keys

    # ── 10. recency: convert timestamps to seconds before impression ──────────
    rec_joined = reindex_lookup(rec_stats, u_keys).fillna(0)
    ts = chunk["time_stamp"].values
    for col, new_col in [("last_pv_ts",   "recency_pv"),
                          ("last_cart_ts", "recency_cart"),
                          ("last_fav_ts",  "recency_fav"),
                          ("last_buy_ts",  "recency_buy")]:
        last_ts = rec_joined[col].values
        # delta = 0 if user never had this action, else seconds elapsed
        chunk[new_col] = np.where(last_ts > 0, ts - last_ts, 0).astype(np.int32)
    del rec_joined

    # ── 11. temporal features ─────────────────────────────────────────────────
    tmp_joined = reindex_lookup(tmp_stats, u_keys)
    tmp_joined["peak_hour"]     = tmp_joined["peak_hour"].fillna(0).astype(np.int8)
    tmp_joined["weekday_ratio"] = tmp_joined["weekday_ratio"].fillna(0.5).astype(np.float32)
    chunk = pd.concat([chunk, tmp_joined], axis=1)
    del tmp_joined

    # ── 12. split train / test and append ────────────────────────────────────
    is_test = chunk["time_stamp"] >= TEST_CUTOFF
    train_chunk = chunk[~is_test]
    test_chunk  = chunk[is_test]

    if len(train_chunk) > 0:
        train_chunk.to_csv(
            TRAIN_OUT, mode="a", header=not train_header_written, index=False
        )
        train_header_written = True

    if len(test_chunk) > 0:
        test_chunk.to_csv(
            TEST_OUT, mode="a", header=not test_header_written, index=False
        )
        test_header_written = True

    del chunk, train_chunk, test_chunk, u_keys, ad_keys, ts
    gc.collect()

    if chunk_idx % 10 == 0:
        elapsed = time.time() - t_loop
        print(f"  Chunk {chunk_idx}, total rows {total_rows:,}, elapsed {elapsed:.0f}s")

print(f"\nProcessed {total_rows:,} rows in {time.time()-t_loop:.0f}s")

# ── quick sanity check ────────────────────────────────────────────────────────
for path, name in [(TRAIN_OUT, "features_train"), (TEST_OUT, "features_test")]:
    head = pd.read_csv(path, nrows=3)
    row_count = sum(1 for _ in open(path)) - 1  # subtract header
    print(f"{name}: {row_count:,} rows × {head.shape[1]} cols  |  clk mean={head['clk'].mean():.4f}")

print("\nDone.")
