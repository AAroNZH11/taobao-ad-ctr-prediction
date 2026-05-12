# Purpose : Test five hypotheses about what drives CTR in the Taobao ad dataset.
#           Joins raw_sample with user_profile, user_behavior_stats, and ad_feature
#           to compute grouped CTR = total_clicks / total_impressions per group.
#
# Hypotheses
#   H1 : Higher shopping depth  → higher CTR   (reversed in practice)
#   H2 : Purchase history       → higher CTR   (reversed in practice)
#   H3 : Lower ad price         → higher CTR   (confirmed)
#   H4 : Female > Male CTR      (confirmed)
#   H5 : More active users      → higher CTR   (strongly confirmed)
#   Bonus: Age level vs CTR (U-shaped)
#
# Inputs  : data/raw_sample.csv
#           data/user_profile.csv
#           data/processed/user_behavior_stats.csv
#           data/ad_feature.csv
# Outputs : outputs/stats/hypothesis_results.json
#           outputs/plots/hypothesis_verification.png

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
STATS_DIR = BASE_DIR / "outputs" / "stats"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


def grouped_ctr(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute CTR = sum(clk) / count(*) per group, sorted by group_col."""
    return (
        df.groupby(group_col)
        .agg(impressions=("clk", "count"), clicks=("clk", "sum"))
        .assign(ctr=lambda x: x["clicks"] / x["impressions"])
        .reset_index()
        .sort_values(group_col)
    )


# ── Load raw_sample ───────────────────────────────────────────────────────────
print("Loading raw_sample.csv …")
sample = pd.read_csv(
    DATA_DIR / "raw_sample.csv",
    dtype={"user": "int32", "time_stamp": "int32",
           "adgroup_id": "int32", "pid": "category",
           "nonclk": "int8", "clk": "int8"},
)
print(f"  {len(sample):,} rows, overall CTR = {sample['clk'].mean():.4%}")

# ── Load user_profile ─────────────────────────────────────────────────────────
print("Loading user_profile.csv …")
profile = pd.read_csv(
    DATA_DIR / "user_profile.csv",
    dtype={"userid": "int32", "cms_segid": "int32", "cms_group_id": "int32",
           "final_gender_code": "int8", "age_level": "int8"},
)
profile.columns = profile.columns.str.strip()
profile = profile.rename(columns={"userid": "user"})

# ── Load user_behavior_stats ──────────────────────────────────────────────────
print("Loading user_behavior_stats.csv …")
bstats = pd.read_csv(
    PROCESSED_DIR / "user_behavior_stats.csv",
    dtype={"user": "int32", "pv_count": "int32", "cart_count": "int32",
           "fav_count": "int32", "buy_count": "int32",
           "total_actions": "int32", "has_purchase": "int8"},
)

# ── Load ad_feature ───────────────────────────────────────────────────────────
print("Loading ad_feature.csv …")
ad_feat = pd.read_csv(
    DATA_DIR / "ad_feature.csv",
    dtype={"adgroup_id": "int32", "cate_id": "int32",
           "campaign_id": "int32", "customer": "int32"},
)

# ── Build enriched sample ─────────────────────────────────────────────────────
print("Joining tables …")
df = (
    sample
    .merge(profile[["user", "final_gender_code", "shopping_level",
                     "age_level", "pvalue_level"]], on="user", how="left")
    .merge(bstats[["user", "has_purchase", "total_actions"]], on="user", how="left")
    .merge(ad_feat[["adgroup_id", "price"]], on="adgroup_id", how="left")
)

del sample, profile, bstats, ad_feat
gc.collect()
print(f"  Joined shape: {df.shape}")

# ═════════════════════════════════════════════════════════════════════════════
# H1: Higher shopping depth → higher CTR?
# ═════════════════════════════════════════════════════════════════════════════
h1 = grouped_ctr(df.dropna(subset=["shopping_level"]), "shopping_level")
print("\nH1 — Shopping Level vs CTR:")
for _, row in h1.iterrows():
    print(f"  level={row['shopping_level']:.0f}  CTR={row['ctr']:.4%}  "
          f"impressions={row['impressions']:,}")

# ═════════════════════════════════════════════════════════════════════════════
# H2: Purchase history → higher CTR?
# ═════════════════════════════════════════════════════════════════════════════
h2 = grouped_ctr(df.dropna(subset=["has_purchase"]), "has_purchase")
print("\nH2 — Purchase History vs CTR:")
for _, row in h2.iterrows():
    label = "buyers" if row["has_purchase"] == 1 else "non-buyers"
    print(f"  {label}  CTR={row['ctr']:.4%}  impressions={row['impressions']:,}")

# ═════════════════════════════════════════════════════════════════════════════
# H3: Lower ad price → higher CTR?  (quintile bins)
# ═════════════════════════════════════════════════════════════════════════════
price_data = df.dropna(subset=["price"]).copy()
price_data["price_bin"] = pd.qcut(
    price_data["price"], q=5,
    labels=["Very Low\n(0-20%)", "Low\n(20-40%)", "Mid\n(40-60%)",
            "High\n(60-80%)", "Very High\n(80-100%)"],
)
h3 = (
    price_data.groupby("price_bin", observed=True)
    .agg(
        impressions=("clk", "count"),
        clicks=("clk", "sum"),
        median_price=("price", "median"),
        n_ads=("adgroup_id", "nunique"),
    )
    .assign(ctr=lambda x: x["clicks"] / x["impressions"])
    .reset_index()
)
print("\nH3 — Price Quintile vs CTR:")
for _, row in h3.iterrows():
    print(f"  {row['price_bin'].replace(chr(10), ' ')}  "
          f"CTR={row['ctr']:.4%}  median_price={row['median_price']:.1f}")

del price_data
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# H4: Female users have higher CTR than male?
# ═════════════════════════════════════════════════════════════════════════════
h4 = grouped_ctr(df.dropna(subset=["final_gender_code"]), "final_gender_code")
print("\nH4 — Gender vs CTR:")
for _, row in h4.iterrows():
    label = "Male" if row["final_gender_code"] == 1 else "Female"
    print(f"  {label}  CTR={row['ctr']:.4%}  impressions={row['impressions']:,}")

# ═════════════════════════════════════════════════════════════════════════════
# H5: More active users → higher CTR?  (quintile bins)
# ═════════════════════════════════════════════════════════════════════════════
activity_data = df.dropna(subset=["total_actions"]).copy()
activity_data["activity_bin"] = pd.qcut(
    activity_data["total_actions"], q=5,
    labels=["Very Low\n(0-20%)", "Low\n(20-40%)", "Mid\n(40-60%)",
            "High\n(60-80%)", "Very High\n(80-100%)"],
)
h5 = (
    activity_data.groupby("activity_bin", observed=True)
    .agg(
        impressions=("clk", "count"),
        clicks=("clk", "sum"),
        median_actions=("total_actions", "median"),
        n_users=("user", "nunique"),
    )
    .assign(ctr=lambda x: x["clicks"] / x["impressions"])
    .reset_index()
)
print("\nH5 — Activity Quintile vs CTR:")
for _, row in h5.iterrows():
    print(f"  {row['activity_bin'].replace(chr(10), ' ')}  "
          f"CTR={row['ctr']:.4%}  median_actions={row['median_actions']:.0f}")

del activity_data
gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# Bonus: Age level vs CTR
# ═════════════════════════════════════════════════════════════════════════════
h_age = grouped_ctr(df.dropna(subset=["age_level"]), "age_level")
print("\nBonus — Age Level vs CTR:")
for _, row in h_age.iterrows():
    print(f"  age={row['age_level']:.0f}  CTR={row['ctr']:.4%}")

del df
gc.collect()

# ── Save JSON results ─────────────────────────────────────────────────────────
results = {
    "H1": [
        {"shopping_level": float(r["shopping_level"]),
         "impressions": int(r["impressions"]), "clicks": int(r["clicks"]),
         "ctr": float(r["ctr"])}
        for _, r in h1.iterrows()
    ],
    "H2": [
        {"has_purchase": int(r["has_purchase"]),
         "impressions": int(r["impressions"]), "clicks": int(r["clicks"]),
         "ctr": float(r["ctr"])}
        for _, r in h2.iterrows()
    ],
    "H3": [
        {"price_bin": str(r["price_bin"]).replace("\n", " "),
         "n_ads": int(r["n_ads"]),
         "median_price": float(r["median_price"]),
         "ctr": float(r["ctr"])}
        for _, r in h3.iterrows()
    ],
    "H4": [
        {"final_gender_code": int(r["final_gender_code"]),
         "gender": "Male" if r["final_gender_code"] == 1 else "Female",
         "impressions": int(r["impressions"]), "clicks": int(r["clicks"]),
         "ctr": float(r["ctr"])}
        for _, r in h4.iterrows()
    ],
    "H5": [
        {"activity_bin": str(r["activity_bin"]).replace("\n", " "),
         "n_users": int(r["n_users"]),
         "median_actions": float(r["median_actions"]),
         "ctr": float(r["ctr"])}
        for _, r in h5.iterrows()
    ],
    "Bonus_Age": [
        {"age_level": int(r["age_level"]), "ctr": float(r["ctr"])}
        for _, r in h_age.iterrows()
    ],
}
(STATS_DIR / "hypothesis_results.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved → {STATS_DIR / 'hypothesis_results.json'}")

# ── Visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Hypothesis Verification — CTR by Feature Group", fontsize=15, fontweight="bold")

BAR_COLOR = "steelblue"
HIGHLIGHT = "salmon"


def pct(v: float) -> str:
    return f"{v:.2%}"


# H1
ax = axes[0, 0]
labels = [f"Level {int(r['shopping_level'])}" for _, r in h1.iterrows()]
ctrs = [r["ctr"] * 100 for _, r in h1.iterrows()]
bars = ax.bar(labels, ctrs, color=BAR_COLOR)
bars[0].set_color(HIGHLIGHT)  # highest CTR bar
ax.set_title("H1: Shopping Depth vs CTR\n(Reversed — shallow users click more)")
ax.set_ylabel("CTR (%)")
ax.set_ylim(0, max(ctrs) * 1.25)
for bar, ctr in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{ctr:.2f}%", ha="center", va="bottom", fontsize=9)

# H2
ax = axes[0, 1]
labels = ["No Purchase\nHistory", "Has Purchase\nHistory"]
ctrs = [r["ctr"] * 100 for _, r in h2.iterrows()]
bars = ax.bar(labels, ctrs, color=BAR_COLOR)
bars[0].set_color(HIGHLIGHT)
ax.set_title("H2: Purchase History vs CTR\n(Reversed — non-buyers are more curious)")
ax.set_ylabel("CTR (%)")
ax.set_ylim(0, max(ctrs) * 1.25)
for bar, ctr in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{ctr:.2f}%", ha="center", va="bottom", fontsize=9)

# H3
ax = axes[0, 2]
bin_labels = [str(r["price_bin"]).replace(" ", "\n", 1) for _, r in h3.iterrows()]
ctrs = [r["ctr"] * 100 for _, r in h3.iterrows()]
bars = ax.bar(bin_labels, ctrs, color=BAR_COLOR)
bars[0].set_color(HIGHLIGHT)
ax.set_title("H3: Price Quintile vs CTR\n(Confirmed — cheap ads get more clicks)")
ax.set_ylabel("CTR (%)")
ax.set_ylim(0, max(ctrs) * 1.25)
for bar, ctr in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{ctr:.2f}%", ha="center", va="bottom", fontsize=8)

# H4
ax = axes[1, 0]
labels = ["Male\n(code=1)", "Female\n(code=2)"]
ctrs = [r["ctr"] * 100 for _, r in h4.iterrows()]
bars = ax.bar(labels, ctrs, color=BAR_COLOR)
bars[1].set_color(HIGHLIGHT)
ax.set_title("H4: Gender vs CTR\n(Confirmed — females click ~8% more)")
ax.set_ylabel("CTR (%)")
ax.set_ylim(0, max(ctrs) * 1.25)
for bar, ctr in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{ctr:.2f}%", ha="center", va="bottom", fontsize=9)

# H5
ax = axes[1, 1]
bin_labels = [str(r["activity_bin"]) for _, r in h5.iterrows()]
ctrs = [r["ctr"] * 100 for _, r in h5.iterrows()]
bars = ax.bar(bin_labels, ctrs, color=BAR_COLOR)
bars[-1].set_color(HIGHLIGHT)
ax.set_title("H5: Activity Level vs CTR\n(Confirmed — most active users CTR +53%)")
ax.set_ylabel("CTR (%)")
ax.set_ylim(0, max(ctrs) * 1.25)
for bar, ctr in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{ctr:.2f}%", ha="center", va="bottom", fontsize=8)

# Bonus: Age
ax = axes[1, 2]
ages = [int(r["age_level"]) for _, r in h_age.iterrows()]
ctrs = [r["ctr"] * 100 for _, r in h_age.iterrows()]
ax.plot(ages, ctrs, marker="o", color="steelblue", linewidth=2)
ax.set_title("Bonus: Age Level vs CTR\n(U-shaped — teens and seniors click more)")
ax.set_xlabel("Age Level (0=unknown, 1=old … 6=teen)")
ax.set_ylabel("CTR (%)")
ax.set_xticks(ages)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "hypothesis_verification.png", dpi=120)
plt.close()
print(f"Plot saved → {PLOTS_DIR / 'hypothesis_verification.png'}")

print("Done.")
