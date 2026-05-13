# Purpose : Decompose the user-ad interaction matrix with truncated SVD (rank k=32).
#           Absorbs √Σ into both factor matrices so dot products approximate click scores:
#             user_embeddings = U_(n×k) · √Σ_(k×k)
#             ad_embeddings   = V_(m×k) · √Σ_(k×k)
#           fit_transform returns U·Σ, so user_emb = (U·Σ) / √Σ and ad_emb = Vᵀᵀ · √Σ.
#
# Input   : data/processed/interaction_matrix.npz
#           data/processed/user_index.csv
#           data/processed/ad_index.csv
# Outputs : data/processed/user_embeddings.csv   — user_id + 32 latent dims
#           data/processed/ad_embeddings.csv     — adgroup_id + 32 latent dims
#           outputs/stats/svd_report.json

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
STATS_DIR = BASE_DIR / "outputs" / "stats"

STATS_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_PATH = PROCESSED_DIR / "interaction_matrix.npz"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.csv"
AD_INDEX_PATH = PROCESSED_DIR / "ad_index.csv"
USER_EMB_PATH = PROCESSED_DIR / "user_embeddings.csv"
AD_EMB_PATH = PROCESSED_DIR / "ad_embeddings.csv"

K = 32  # latent dimensions

if USER_EMB_PATH.exists() and AD_EMB_PATH.exists():
    print("Embeddings already exist. Delete files and re-run to recompute.")
else:
    print("Loading interaction matrix …")
    matrix = sp.load_npz(MATRIX_PATH)
    print(f"  Shape: {matrix.shape},  nnz: {matrix.nnz:,}")

    print(f"Running TruncatedSVD (k={K}) …")
    svd = TruncatedSVD(n_components=K, algorithm="randomized", random_state=42)

    # fit_transform returns U·Σ, shape (n_users, k)
    U_sigma = svd.fit_transform(matrix)
    sigma = svd.singular_values_          # shape (k,)
    sigma_sqrt = np.sqrt(sigma)           # shape (k,)
    explained = float(svd.explained_variance_ratio_.sum())
    print(f"  Explained variance (top {K} components): {explained:.4%}")

    del matrix
    gc.collect()

    # user_emb = U · √Σ = (U·Σ) / √Σ
    user_emb = U_sigma / sigma_sqrt       # (n_users, k) broadcast over k

    # svd.components_ = Vᵀ, shape (k, n_ads)
    # ad_emb = V · √Σ, shape (n_ads, k)
    ad_emb = svd.components_.T * sigma_sqrt

    del U_sigma
    gc.collect()

    user_index = pd.read_csv(USER_INDEX_PATH)
    ad_index = pd.read_csv(AD_INDEX_PATH)

    emb_cols = [f"svd_{i}" for i in range(K)]

    user_df = pd.DataFrame(user_emb, columns=emb_cols)
    user_df.insert(0, "user_id", user_index["user_id"].values)
    user_df.to_csv(USER_EMB_PATH, index=False)
    print(f"  User embeddings → {USER_EMB_PATH}  ({len(user_df):,} users × {K} dims)")

    ad_df = pd.DataFrame(ad_emb, columns=emb_cols)
    ad_df.insert(0, "adgroup_id", ad_index["adgroup_id"].values)
    ad_df.to_csv(AD_EMB_PATH, index=False)
    print(f"  Ad embeddings   → {AD_EMB_PATH}  ({len(ad_df):,} ads × {K} dims)")

    report = {
        "k": K,
        "n_users": len(user_df),
        "n_ads": len(ad_df),
        "explained_variance_ratio_sum": explained,
        "singular_values_top5": sigma[:5].tolist(),
    }
    (STATS_DIR / "svd_report.json").write_text(json.dumps(report, indent=2))
    print(f"  SVD report      → {STATS_DIR / 'svd_report.json'}")

print("Done.")
