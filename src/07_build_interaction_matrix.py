# Purpose : Build a sparse user-ad interaction matrix from raw_sample (training split only).
#           Matrix shape = (n_users, n_ads); stored value = click count per (user, ad) pair.
#           No-click impressions are not stored — they are implicit zeros in the sparse format.
#           Only the first 7 days are used (2017-05-06 to 2017-05-12); day 8 is the test set.
#
# Input   : data/raw_sample.csv
# Outputs : data/processed/interaction_matrix.npz   — scipy sparse CSR matrix (float32)
#           data/processed/user_index.csv            — user_id → row_idx mapping
#           data/processed/ad_index.csv              — adgroup_id → col_idx mapping
#           outputs/stats/interaction_matrix_report.json

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
STATS_DIR = BASE_DIR / "outputs" / "stats"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = PROCESSED_DIR / "interaction_matrix.npz"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.csv"
AD_INDEX_PATH = PROCESSED_DIR / "ad_index.csv"

if MATRIX_PATH.exists() and USER_INDEX_PATH.exists() and AD_INDEX_PATH.exists():
    print("Interaction matrix already exists. Delete files and re-run to recompute.")
else:
    # 2017-05-13 00:00:00 UTC = 1494633600 — day 8 is held out as test set
    TRAIN_END_TS = 1494633600

    print("Loading raw_sample.csv …")
    df = pd.read_csv(
        DATA_DIR / "raw_sample.csv",
        usecols=["user", "time_stamp", "adgroup_id", "clk"],
        dtype={
            "user": "int32",
            "time_stamp": "int32",
            "adgroup_id": "int32",
            "clk": "int8",
        },
    )
    print(f"  Total rows: {len(df):,}")

    df = df[df["time_stamp"] < TRAIN_END_TS]
    n_train = len(df)
    print(f"  Training rows (days 1–7): {n_train:,}")

    # Only clicked rows carry signal for SVD; no-clicks are implicit zeros
    clicks = df[df["clk"] == 1][["user", "adgroup_id"]].copy()
    n_clicks = len(clicks)
    print(f"  Clicked rows: {n_clicks:,}  ({n_clicks / n_train:.2%} of training)")

    del df
    gc.collect()

    # Build consecutive integer indices so the matrix has no gaps
    unique_users = np.sort(clicks["user"].unique())
    unique_ads = np.sort(clicks["adgroup_id"].unique())

    user_to_idx = {int(u): i for i, u in enumerate(unique_users)}
    ad_to_idx = {int(a): i for i, a in enumerate(unique_ads)}

    n_users = len(unique_users)
    n_ads = len(unique_ads)
    print(f"  Unique users with ≥1 click: {n_users:,}")
    print(f"  Unique ads   with ≥1 click: {n_ads:,}")

    row = clicks["user"].map(user_to_idx).to_numpy(dtype=np.int32)
    col = clicks["adgroup_id"].map(ad_to_idx).to_numpy(dtype=np.int32)
    data = np.ones(n_clicks, dtype=np.float32)

    del clicks
    gc.collect()

    print("Building sparse CSR matrix …")
    # Duplicate (user, ad) pairs are summed → click count per pair
    matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_ads), dtype=np.float32)
    print(f"  Shape    : {matrix.shape}")
    print(f"  Non-zeros: {matrix.nnz:,}")
    print(f"  Density  : {matrix.nnz / (n_users * n_ads):.6%}")

    del row, col, data
    gc.collect()

    sp.save_npz(MATRIX_PATH, matrix)
    print(f"  Matrix saved → {MATRIX_PATH}")

    pd.DataFrame({"user_id": unique_users, "row_idx": np.arange(n_users)}).to_csv(
        USER_INDEX_PATH, index=False
    )
    pd.DataFrame({"adgroup_id": unique_ads, "col_idx": np.arange(n_ads)}).to_csv(
        AD_INDEX_PATH, index=False
    )
    print(f"  User index → {USER_INDEX_PATH}")
    print(f"  Ad index   → {AD_INDEX_PATH}")

    report = {
        "n_users_with_clicks": int(n_users),
        "n_ads_with_clicks": int(n_ads),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (n_users * n_ads)),
        "train_end_timestamp": TRAIN_END_TS,
        "n_training_rows": int(n_train),
        "n_clicked_rows": int(n_clicks),
    }
    (STATS_DIR / "interaction_matrix_report.json").write_text(json.dumps(report, indent=2))
    print(f"  Report    → {STATS_DIR / 'interaction_matrix_report.json'}")

print("Done.")
