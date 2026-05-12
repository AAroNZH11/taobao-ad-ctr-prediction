# Taobao Ad CTR Prediction

Predicting whether a user will click on a displayed advertisement using the [Ali Display Ad Click dataset](https://tianchi.aliyun.com/dataset/56) from Alibaba Tianchi. The dataset covers 8 days of ad impressions across ~1.14 million users, with ~723 million rows of historical behavior logs.

---

## Dataset

| File | Size | Description |
|------|------|-------------|
| `raw_sample.csv` | 1.1 GB | 26.6M ad impression records with click labels |
| `ad_feature.csv` | 30 MB | Static attributes for 846K ads |
| `user_profile.csv` | 23 MB | Demographics for 1.06M users |
| `behavior_log.csv` | 23 GB | 723M rows of historical user behavior |

Raw files are not tracked in git. Place them in the project root before running.

---

## Pipeline

| Phase | Description | Status |
|-------|-------------|--------|
| **1 — EDA & Cleaning** | Load, deduplicate, validate, and visualize all four tables | ✅ Done |
| **2 — Hypothesis Verification** | Test five hypotheses about what drives CTR | ✅ Done |
| 3 — Interaction Matrix & SVD | Build sparse user-ad matrix, extract embeddings | Upcoming |
| 4 — Feature Enrichment | Join embeddings with profile and behavior features | Upcoming |
| 5 — CTR Model | Logistic regression baseline, evaluate AUC | Upcoming |

---

## Phase 1 — Key Findings

**Overall CTR: 5.14%** across 26.6M impressions (2017-05-06 to 2017-05-13)

**behavior_log cleaning:**
- 723M raw records → 672M after removing 7.1% exact duplicates (logging artifacts)
- 26K invalid timestamps filtered out
- Compressed to 4.9 GB parquet (362 chunks, ~5× reduction)

**Notable missing data:**
- `brand` in ad_feature: 29% missing
- `pvalue_level` in user_profile: 54% missing
- `new_user_class_level` in user_profile: 32% missing

---

## Phase 2 — Hypothesis Verification

CTR computed as total clicks / total impressions per group, joining all four tables.

| # | Hypothesis | Result | Finding |
|---|-----------|--------|---------|
| H1 | Higher shopping depth → higher CTR | ❌ Reversed | Shallow shoppers click more (5.40% vs 5.11%) — heavy shoppers search with intent, less susceptible to display ads |
| H2 | Purchase history → higher CTR | ❌ Reversed | Non-buyers click more (5.46% vs 5.12%) — exploratory users are more curious about ads |
| H3 | Lower price → higher CTR | ✅ Confirmed | Monotone decrease from 5.76% (cheapest) to 4.73% (priciest) |
| H4 | Female CTR > Male CTR | ✅ Confirmed | 5.24% vs 4.84% (+8.4%) |
| H5 | More active users → higher CTR | ✅ Strongly confirmed | 4.22% → 6.03% across activity quintiles (+43%) |

**Bonus — Age vs CTR:** U-shaped. Teenagers (level 6) and older users (level 1) have the highest CTR; middle-aged users (level 4) have the lowest.

![Hypothesis verification chart](outputs/plots/hypothesis_verification.png)

---

## Project Structure

```
├── src/
│   ├── 01_clean_ad_feature.py        # EDA + cleaning for ad_feature
│   ├── 02_clean_user_profile.py      # EDA + cleaning for user_profile
│   ├── 03_clean_raw_sample.py        # EDA + cleaning for raw_sample
│   ├── 04_clean_behavior_log.py      # Chunked 23 GB → parquet conversion
│   ├── 05_aggregate_behavior.py      # Per-user behavior aggregation
│   └── 06_hypothesis_verification.py # CTR hypothesis testing
├── outputs/
│   ├── plots/                        # All visualizations (PNG)
│   └── stats/                        # JSON summaries per table
├── data/
│   └── processed/                    # Intermediate files (not tracked)
│       ├── behavior_log_parquet/     # 362 snappy-compressed chunks
│       └── user_behavior_stats.csv  # Per-user behavior counts
├── PROJECT_BRIEF.md
├── dataset_description.md
└── requirements.txt
```

---

## Running the Scripts

Scripts are designed to run independently and in order. Each reads from files, writes to files — no shared in-memory state.

```bash
pip install -r requirements.txt

python src/01_clean_ad_feature.py
python src/02_clean_user_profile.py
python src/03_clean_raw_sample.py
python src/04_clean_behavior_log.py   # ~3 min, produces 362 parquet chunks
python src/05_aggregate_behavior.py   # reads parquet, writes user_behavior_stats.csv
python src/06_hypothesis_verification.py
```

**Memory note:** The machine used for development has 3.8 GB RAM. `behavior_log.csv` is processed in chunks of 2M rows and never loaded all at once. Each script frees memory explicitly after heavy operations.

---

## Requirements

See [requirements.txt](requirements.txt). Core dependencies: `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.
