# Purpose : EDA and cleaning for raw_sample.csv
# Input   : data/raw_sample.csv  (raw, ~1.1 GB)
# Outputs : outputs/stats/raw_sample_report.json
#           outputs/plots/raw_sample_distributions.png
#           outputs/plots/raw_sample_ctr_by_hour.png

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/aaronzhong/Documents/TaoBao-Project")
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
STATS_DIR = BASE_DIR / "outputs" / "stats"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
# dtype optimisation keeps memory well under 1 GB for this 1.1 GB file
print("Loading raw_sample.csv ...")
df = pd.read_csv(
    DATA_DIR / "raw_sample.csv",
    dtype={
        "user": "int32",
        "time_stamp": "int32",
        "adgroup_id": "int32",
        "pid": "category",
        "nonclk": "int8",
        "clk": "int8",
    },
)
print(f"  Shape: {df.shape}")

# ── Null analysis ─────────────────────────────────────────────────────────────
null_counts = df.isnull().sum().to_dict()
print(f"  Nulls: {null_counts}")

# ── Deduplication ─────────────────────────────────────────────────────────────
n_before = len(df)
df.drop_duplicates(inplace=True)
n_dupes = n_before - len(df)
print(f"  Duplicates removed: {n_dupes}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert df["pid"].nunique() == 2, "pid should have exactly 2 distinct values"
assert (df["clk"] + df["nonclk"] == 1).all(), "clk and nonclk must be complements"

# ── Datetime conversion for time-based analysis ───────────────────────────────
# time_stamp is stored as int32; cast to int64 before passing to pd.to_datetime
df["dt"] = pd.to_datetime(df["time_stamp"].astype("int64"), unit="s")
df["date"] = df["dt"].dt.date.astype(str)
df["hour"] = df["dt"].dt.hour

overall_ctr = float(df["clk"].mean())
print(f"  Overall CTR: {overall_ctr:.4%}")

# ── Per-pid statistics ────────────────────────────────────────────────────────
pid_stats = (
    df.groupby("pid", observed=True)
    .agg(impressions=("clk", "count"), clicks=("clk", "sum"))
    .assign(ctr=lambda x: x["clicks"] / x["impressions"])
)
pid_impressions = pid_stats["impressions"].to_dict()
pid_clicks = pid_stats["clicks"].to_dict()
pid_ctr = pid_stats["ctr"].to_dict()

# ── Daily statistics ──────────────────────────────────────────────────────────
daily = (
    df.groupby("date")
    .agg(impressions=("clk", "count"), clicks=("clk", "sum"))
)
daily_impressions = daily["impressions"].to_dict()
daily_clicks = daily["clicks"].to_dict()

# ── Stats summary ─────────────────────────────────────────────────────────────
report = {
    "total_rows": len(df),
    "total_clicks": int(df["clk"].sum()),
    "overall_ctr": overall_ctr,
    "nulls": {k: int(v) for k, v in null_counts.items()},
    "duplicates_removed": n_dupes,
    "pid_impressions": {str(k): int(v) for k, v in pid_impressions.items()},
    "pid_clicks": {str(k): int(v) for k, v in pid_clicks.items()},
    "pid_ctr": {str(k): float(v) for k, v in pid_ctr.items()},
    "daily_impressions": {str(k): int(v) for k, v in daily_impressions.items()},
    "daily_clicks": {str(k): int(v) for k, v in daily_clicks.items()},
}
(STATS_DIR / "raw_sample_report.json").write_text(json.dumps(report, indent=2))
print(f"  Report saved → {STATS_DIR / 'raw_sample_report.json'}")

# ── Distribution plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("raw_sample — Distributions", fontsize=14, fontweight="bold")

# Impressions per day
ax = axes[0, 0]
daily_sorted = daily.sort_index()
ax.bar(daily_sorted.index, daily_sorted["impressions"], color="steelblue")
ax.set_title("Daily Impressions")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)

# Clicks per day
ax = axes[0, 1]
ax.bar(daily_sorted.index, daily_sorted["clicks"], color="salmon")
ax.set_title("Daily Clicks")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)

# Impressions by pid
ax = axes[1, 0]
pids = list(pid_impressions.keys())
ax.bar([str(p) for p in pids], [pid_impressions[p] for p in pids], color="teal")
ax.set_title("Impressions by Ad Placement (pid)")
ax.set_ylabel("Impressions")
ax.tick_params(axis="x", rotation=20)

# CTR by pid
ax = axes[1, 1]
ax.bar([str(p) for p in pids],
       [pid_ctr[p] * 100 for p in pids], color="mediumseagreen")
ax.set_title("CTR by Ad Placement (pid)")
ax.set_ylabel("CTR (%)")
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "raw_sample_distributions.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'raw_sample_distributions.png'}")

# ── CTR by hour-of-day ────────────────────────────────────────────────────────
hourly = (
    df.groupby("hour")
    .agg(impressions=("clk", "count"), clicks=("clk", "sum"))
    .assign(ctr=lambda x: x["clicks"] / x["impressions"] * 100)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("raw_sample — Hourly Patterns", fontsize=13, fontweight="bold")

ax = axes[0]
ax.bar(hourly.index, hourly["impressions"], color="steelblue")
ax.set_title("Impressions by Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("Impressions")

ax = axes[1]
ax.plot(hourly.index, hourly["ctr"], marker="o", color="salmon", linewidth=2)
ax.set_title("CTR (%) by Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("CTR (%)")
ax.set_ylim(bottom=0)
ax.axhline(overall_ctr * 100, color="gray", linestyle="--", label=f"Overall {overall_ctr:.2%}")
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "raw_sample_ctr_by_hour.png", dpi=120)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'raw_sample_ctr_by_hour.png'}")

del df
gc.collect()
print("Done.")
