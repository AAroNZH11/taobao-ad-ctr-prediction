# Purpose : EDA and cleaning for user_profile.csv
# Input   : data/user_profile.csv  (raw, ~23 MB)
# Outputs : outputs/stats/user_profile_report.json
#           outputs/plots/user_profile_distributions.png
#           outputs/plots/user_profile_correlation.png
#           outputs/plots/user_profile_age_gender.png

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
STATS_DIR = BASE_DIR / "outputs" / "stats"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading user_profile.csv ...")
df = pd.read_csv(
    DATA_DIR / "user_profile.csv",
    dtype={
        "userid": "int32",
        "cms_segid": "int32",
        "cms_group_id": "int32",
        "final_gender_code": "int8",
        "age_level": "int8",
    },
)
print(f"  Shape: {df.shape}")

# The raw CSV has a trailing-space typo in one column header — strip all names
df.columns = df.columns.str.strip()

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
assert df["userid"].is_unique, "userid should be a primary key"
assert df["final_gender_code"].isin([1, 2]).all(), "gender must be 1 or 2"
assert df["age_level"].between(0, 6).all(), "age_level must be 0–6"

# ── Stats summary ─────────────────────────────────────────────────────────────
gender_dist = df["final_gender_code"].value_counts().to_dict()
age_dist = df["age_level"].value_counts().sort_index().to_dict()
pvalue_dist = df["pvalue_level"].value_counts().sort_index().to_dict()

report = {
    "rows": len(df),
    "cols": df.shape[1],
    "duplicates_removed": n_dupes,
    "nulls": {k: int(v) for k, v in null_counts.items()},
    "null_pct": null_pct,
    "gender_dist": {str(k): int(v) for k, v in gender_dist.items()},
    "age_dist": {str(k): int(v) for k, v in age_dist.items()},
    "pvalue_dist": {str(k): int(v) for k, v in pvalue_dist.items()},
}
(STATS_DIR / "user_profile_report.json").write_text(json.dumps(report, indent=2))
print(f"  Report saved → {STATS_DIR / 'user_profile_report.json'}")

# ── Distribution plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("user_profile — Column Distributions", fontsize=14, fontweight="bold")

cols_to_plot = [
    ("final_gender_code", "Gender (1=M, 2=F)"),
    ("age_level", "Age Level (0–6)"),
    ("shopping_level", "Shopping Level (1–3)"),
    ("occupation", "Occupation (0/1)"),
    ("pvalue_level", "Consumption Level"),
    ("new_user_class_level", "City Tier"),
    ("cms_segid", "Micro-segment ID"),
    ("cms_group_id", "Group ID"),
]

for ax, (col, title) in zip(axes.flat, cols_to_plot):
    data = df[col].dropna()
    counts = data.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("Users")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_profile_distributions.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'user_profile_distributions.png'}")

# ── Correlation heatmap ───────────────────────────────────────────────────────
numeric_cols = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("user_profile — Numeric Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_profile_correlation.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'user_profile_correlation.png'}")

# ── Age × Gender cross-tab ────────────────────────────────────────────────────
cross = df.groupby(["age_level", "final_gender_code"]).size().unstack(fill_value=0)
cross.columns = ["Male", "Female"]

fig, ax = plt.subplots(figsize=(9, 5))
x = cross.index
width = 0.35
ax.bar(x - width / 2, cross["Male"], width, label="Male", color="steelblue")
ax.bar(x + width / 2, cross["Female"], width, label="Female", color="salmon")
ax.set_title("User count by Age Level and Gender")
ax.set_xlabel("Age Level (0=unknown, 1=old … 6=teen)")
ax.set_ylabel("User Count")
ax.legend()
ax.set_xticks(cross.index)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "user_profile_age_gender.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'user_profile_age_gender.png'}")

del df
gc.collect()
print("Done.")
