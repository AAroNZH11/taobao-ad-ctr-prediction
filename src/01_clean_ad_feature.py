# Purpose : EDA and cleaning for ad_feature.csv
# Input   : data/ad_feature.csv  (raw, ~30 MB)
# Outputs : outputs/stats/ad_feature_report.json
#           outputs/plots/ad_feature_distributions.png
#           outputs/plots/ad_feature_correlation.png

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
STATS_DIR = BASE_DIR / "outputs" / "stats"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading ad_feature.csv ...")
df = pd.read_csv(
    DATA_DIR / "ad_feature.csv",
    dtype={
        "adgroup_id": "int32",
        "cate_id": "int32",
        "campaign_id": "int32",
        "customer": "int32",
    },
)
print(f"  Shape: {df.shape}")

# ── Null analysis ─────────────────────────────────────────────────────────────
null_counts = df.isnull().sum().to_dict()
null_pct = {k: round(v / len(df) * 100, 2) for k, v in null_counts.items()}
print(f"  Nulls: {null_counts}")

# ── Deduplication ─────────────────────────────────────────────────────────────
n_before = len(df)
df.drop_duplicates(inplace=True)
n_dupes = n_before - len(df)
print(f"  Duplicates removed: {n_dupes}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert df["adgroup_id"].is_unique, "adgroup_id should be a primary key"
price_min = float(df["price"].min())
price_max = float(df["price"].max())
print(f"  Price range: [{price_min:.2f}, {price_max:.2f}]")

# ── Stats summary ─────────────────────────────────────────────────────────────
report = {
    "rows": len(df),
    "cols": df.shape[1],
    "duplicates_removed": n_dupes,
    "nulls": null_counts,
    "null_pct": null_pct,
    "price_min": price_min,
    "price_max": price_max,
    "unique_cate": int(df["cate_id"].nunique()),
    "unique_campaign": int(df["campaign_id"].nunique()),
    "unique_customer": int(df["customer"].nunique()),
}
(STATS_DIR / "ad_feature_report.json").write_text(json.dumps(report, indent=2))
print(f"  Report saved → {STATS_DIR / 'ad_feature_report.json'}")

# ── Distribution plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("ad_feature — Column Distributions", fontsize=14, fontweight="bold")

# price: log-transform because it spans many orders of magnitude
ax = axes[0, 0]
log_price = np.log1p(df["price"].dropna())
ax.hist(log_price, bins=60, color="steelblue", edgecolor="none")
ax.set_title("log1p(price)")
ax.set_xlabel("log1p(price)")
ax.set_ylabel("Count")

# brand null indicator
ax = axes[0, 1]
brand_null = df["brand"].isnull()
ax.bar(["Known", "Missing"], [brand_null.sum() == 0, brand_null.sum()],
       color=["steelblue", "salmon"])
ax.bar(["Known", "Missing"], [(~brand_null).sum(), brand_null.sum()],
       color=["steelblue", "salmon"])
ax.set_title(f"brand — missing {null_pct['brand']:.1f}%")
ax.set_ylabel("Count")

# top-20 cate_id
ax = axes[0, 2]
top_cate = df["cate_id"].value_counts().head(20)
ax.barh(top_cate.index.astype(str), top_cate.values, color="steelblue")
ax.set_title("Top-20 cate_id")
ax.set_xlabel("Ad count")
ax.invert_yaxis()

# top-20 campaign_id
ax = axes[1, 0]
top_campaign = df["campaign_id"].value_counts().head(20)
ax.barh(top_campaign.index.astype(str), top_campaign.values, color="teal")
ax.set_title("Top-20 campaign_id")
ax.set_xlabel("Ad count")
ax.invert_yaxis()

# top-20 customer
ax = axes[1, 1]
top_cust = df["customer"].value_counts().head(20)
ax.barh(top_cust.index.astype(str), top_cust.values, color="mediumseagreen")
ax.set_title("Top-20 customer")
ax.set_xlabel("Ad count")
ax.invert_yaxis()

# price box-plot per top-5 categories
ax = axes[1, 2]
top5_cate = df["cate_id"].value_counts().head(5).index.tolist()
plot_data = [df.loc[df["cate_id"] == c, "price"].dropna().values for c in top5_cate]
ax.boxplot(plot_data, labels=[str(c) for c in top5_cate], vert=True)
ax.set_yscale("log")
ax.set_title("Price distribution — top-5 categories")
ax.set_xlabel("cate_id")
ax.set_ylabel("price (log scale)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "ad_feature_distributions.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'ad_feature_distributions.png'}")

# ── Correlation heatmap ───────────────────────────────────────────────────────
numeric_cols = ["cate_id", "campaign_id", "customer", "brand", "price"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("ad_feature — Numeric Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ad_feature_correlation.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'ad_feature_correlation.png'}")

del df
gc.collect()
print("Done.")
