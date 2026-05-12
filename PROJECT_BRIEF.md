# Taobao Ad CTR Prediction — Project Brief

## Background

This project uses the Ali_Display_Ad_Click dataset provided by Alibaba on the Tianchi platform. The goal is to analyze user behavior and ad features to predict whether a user will click on a displayed advertisement (CTR prediction).

Dataset source: https://tianchi.aliyun.com/dataset/56

---

## Dataset Overview

Four raw CSV files, stored in the project root directory:

### 1. `raw_sample.csv` (~1.1 GB)
Ad impression and click log — the core fact table.

| Field | Type | Description |
|-------|------|-------------|
| user | int | Anonymized user ID |
| time_stamp | int | Unix timestamp |
| adgroup_id | int | Anonymized ad unit ID |
| pid | str | Ad placement (resource position), only 2 distinct values |
| nonclk | int8 | 1 = not clicked, 0 = clicked |
| clk | int8 | 1 = clicked, 0 = not clicked (complement of nonclk) |

- ~26.56 million records
- Time range: 2017-05-06 to 2017-05-13 (8 days; first 7 for training, last 1 for testing)
- Overall CTR ≈ 5.14%

### 2. `ad_feature.csv` (~30 MB)
Static ad attributes.

| Field | Type | Description |
|-------|------|-------------|
| adgroup_id | int | Ad ID (primary key) |
| cate_id | int | Product category ID |
| campaign_id | int | Ad campaign ID |
| customer | int | Advertiser ID |
| brand | float | Brand ID (**~29% missing**) |
| price | float | Product price |

- ~846K records

### 3. `user_profile.csv` (~23 MB)
User demographic features.

| Field | Type | Description |
|-------|------|-------------|
| userid | int | User ID (primary key) |
| cms_segid | int | Micro-segment ID |
| cms_group_id | int | User group ID |
| final_gender_code | int8 | Gender (1 = male, 2 = female) |
| age_level | int8 | Age group (0–6) |
| pvalue_level | float | Consumption level (1 = low, 2 = mid, 3 = high; **54% missing**) |
| shopping_level | float | Shopping depth (1 = shallow, 2 = moderate, 3 = deep) |
| occupation | float | College student (1 = yes, 0 = no) |
| new_user_class_level | float | City tier (**32% missing**) |

- ~1.06 million records

### 4. `behavior_log.csv` (**~23 GB**)
User historical behavior log — the largest and most important table.

| Field | Type | Description |
|-------|------|-------------|
| user | int | User ID |
| time_stamp | int | Unix timestamp |
| btag | str | Behavior type: `pv` = page view, `cart` = add to cart, `fav` = favorite, `buy` = purchase |
| cate | int | Product category ID |
| brand | float | Brand ID |

- ~723 million raw records
- Covers 22 days of historical behavior for all users in raw_sample (~2017-04-14 to 2017-05-13)
- **Contains ~7.1% exact duplicate rows**: different department logging systems produce identical records when merged — must be removed
- **Contains a small number of invalid timestamps** (<0.01%): negative values or far-future dates — must be filtered

### Table Relationships
```
raw_sample.user       → user_profile.userid
raw_sample.adgroup_id → ad_feature.adgroup_id
behavior_log.user     → user_profile.userid
```

---

## Hardware Constraints (Important)

The runtime environment has only ~**3.8 GB of RAM**. Memory usage must be strictly controlled:

- `behavior_log.csv` is 23 GB — **must never be loaded all at once**, always process in chunks
- `raw_sample.csv` is 1.1 GB — use dtype optimization (int32/int8/category) when loading
- After processing each table, immediately release memory: `del df; gc.collect()`
- Each script should run independently, passing data between stages via intermediate files (not in-memory state)

---

## Recommended Project Structure

```
taobao-ad-ctr-prediction/
├── data/                          # Raw CSV files (excluded from git)
│   ├── raw_sample.csv
│   ├── ad_feature.csv
│   ├── user_profile.csv
│   └── behavior_log.csv
├── data/processed/                # Cleaned intermediate data
│   ├── behavior_log_parquet/      # Parquet chunks of behavior_log
│   └── user_behavior_stats.csv   # Per-user behavior aggregation
├── src/                           # Structured Python scripts
│   ├── 01_clean_ad_feature.py
│   ├── 02_clean_user_profile.py
│   ├── 03_clean_raw_sample.py
│   ├── 04_clean_behavior_log.py
│   ├── 05_aggregate_behavior.py
│   ├── 06_hypothesis_verification.py
│   ├── 07_build_interaction_matrix.py
│   ├── 08_matrix_factorization.py
│   ├── 09_feature_engineering.py
│   └── 10_ctr_model.py
├── outputs/
│   ├── plots/                     # All visualizations
│   └── stats/                     # JSON summaries
├── PROJECT_BRIEF.md               # This file
├── dataset_description.md
├── requirements.txt
└── README.md
```

---

## Full Project Pipeline

```
raw_sample + behavior_log
        │
        ▼
[Phase 1] EDA & Data Cleaning          ← Completed
        │
        ▼
[Phase 2] Hypothesis Verification      ← Completed
        │
        ▼
[Phase 3] Interaction Matrix & SVD     ← Next step
        │
        ▼
[Phase 4] Feature Enrichment
        │
        ▼
[Phase 5] CTR Prediction Model
        │
        ▼
        Evaluation (AUC, baseline = 0.622)
```

---

## Phase 1: EDA & Data Cleaning

### For each table

1. **Load** with appropriate dtypes to minimize memory usage
2. **Null analysis**: count and percentage of missing values per column
3. **Deduplication**: `drop_duplicates()`, log how many rows were removed
4. **Value sanity checks**: valid ranges for numeric columns, valid categories for categorical columns
5. **Histograms**: distribution plots for all meaningful columns
6. **Correlation heatmap**: correlation matrix across numeric columns
7. **Stats summary**: save results as JSON to `outputs/stats/`

### Special handling for behavior_log

Due to the 23 GB file size, use the following strategy:

**Chunked reading** (`chunksize=2_000_000`):
- Per chunk: exact deduplication, timestamp filtering, then save as parquet
- Timestamp filter: keep records where `1_483_228_800 < time_stamp < 1_577_836_800` (2017–2020)
- Save each chunk to parquet (snappy compression): 23 GB → ~4.9 GB (~5x compression)
- Accumulate statistics across chunks: btag distribution, null counts, per-user behavior counts

**Duplicate row classification** (important context):
- All 5 columns identical → logging artifact, **remove** (~7.1% of rows, ~51M records)
- Same `(user, btag, cate, brand)` but different `time_stamp` → genuine repeated behavior at different times, **keep**
- Same `(user, time_stamp)` but different content → different actions within the same second, **keep**

**Per-user behavior aggregation** — generate `user_behavior_stats.csv`:

| Field | Description |
|-------|-------------|
| user | User ID |
| pv_count | Total page views |
| cart_count | Total add-to-cart actions |
| fav_count | Total favorites |
| buy_count | Total purchases |
| total_actions | Sum of all four behavior types |
| has_purchase | Whether user has any purchase (0/1) |
| buy_rate | buy_count / pv_count |
| cart_rate | cart_count / pv_count |

---

## Phase 2: Hypothesis Verification

Five hypotheses tested using grouped CTR (= total clicks / total impressions per group), joining raw_sample with user_profile, user_behavior_stats, and ad_feature.

### H1: Higher shopping depth → higher CTR
- Variable: `shopping_level` (1/2/3)
- Expected: deeper shoppers click more
- **Result: Reversed** — shallow users have the highest CTR (5.41% > 5.17% > 5.11%)
- Interpretation: heavy shoppers have clear purchase intent and search actively; they are less susceptible to passive display ads

### H2: Users with purchase history → higher CTR
- Variable: `has_purchase` (0/1, from user_behavior_stats)
- Expected: buyers click more
- **Result: Reversed** — users with no purchase history have slightly higher CTR (5.38% vs 5.12%)
- Interpretation: non-buyers may be in an exploratory phase and are more curious about ads

### H3: Lower ad price → higher CTR
- Variable: `price` (from ad_feature), binned into quintiles
- Expected: cheaper ads get more clicks
- **Result: Confirmed** — monotonically decreasing (6.12% → 4.68%)
- Interpretation: low-cost items have a lower psychological barrier to click

### H4: Female users have higher CTR than male users
- Variable: `final_gender_code` (from user_profile)
- Expected: gender=2 has higher CTR
- **Result: Confirmed** (5.24% vs 4.84%, +8.4%)

### H5: More active users → higher CTR
- Variable: `total_actions` (from user_behavior_stats), binned into quintiles
- Expected: more active users click more
- **Result: Strongly confirmed** — CTR rises from 3.76% to 5.77% (+53%) across activity quintiles
- Interpretation: behavioral history is the strongest predictive signal for CTR in this dataset

### Bonus: Age level vs CTR
- Variable: `age_level` (0–6)
- Result: U-shaped — teenagers (level 6) and older users (level 1) have the highest CTR; middle-aged users (level 4) have the lowest

---

## Phase 3: Interaction Matrix & SVD Matrix Factorization

### Goal

Learn low-dimensional latent vector representations (embeddings) for each user and each ad using SVD, to serve as core features for CTR prediction.

### Step 1: Build the user-ad interaction matrix X̃

- Rows: users, columns: ads, values: click (1) or no-click (0)
- Extremely sparse: 1.14M users × 846K ads, each user interacts with only a tiny fraction of ads
- Use training set only (2017-05-06 to 2017-05-12); hold out test set (2017-05-13) for evaluation
- Store using `scipy.sparse.csr_matrix` — never materialize as a dense matrix

### Step 2: Truncated SVD

Decompose X̃ into low-rank factors:

```
X̃ ≈ Ũ · Ṽ + ε

Where:
  Ũ (user matrix):  shape = (n_users, k) — each row is a user's k-dimensional latent vector
  Ṽ (ad matrix):    shape = (k, n_ads)  — each column is an ad's k-dimensional latent vector
  k: rank (hyperparameter, start with k=32 or k=64)
  ε: reconstruction error
```

Full SVD: X̃ = UΣV^T. Truncated to rank k:
- Ũ = U_(u×k) · √Σ_(k×k)
- Ṽ = √Σ_(k×k) · V_(k×a)

Implementation: use `sklearn.decomposition.TruncatedSVD` (designed for sparse matrices).

### Step 3: Save embeddings

- `user_embeddings.csv`: user_id + k-dimensional latent vector
- `ad_embeddings.csv`: adgroup_id + k-dimensional latent vector

---

## Phase 4: Feature Enrichment

Concatenate SVD embeddings with original side features to build the final feature matrix.

### User features

| Source | Fields |
|--------|--------|
| SVD embedding | k-dimensional latent vector (row from Ũ) |
| user_profile | gender, age_level, shopping_level, occupation, new_user_class_level |
| user_behavior_stats | pv_count, cart_count, fav_count, buy_count, buy_rate, cart_rate |

Note: `pvalue_level` is 54% missing. Recommended approach: add a `pvalue_known` indicator column (0/1) and fill missing values with 0.

### Ad features

| Source | Fields |
|--------|--------|
| SVD embedding | k-dimensional latent vector (column from Ṽ) |
| ad_feature | cate_id, log(price), brand (29% missing — add indicator column) |

### Final sample construction

Each row in `raw_sample` becomes one training sample:

```
sample = [user_embedding (k dims) | user_profile features | user_behavior features
         | ad_embedding (k dims)  | ad_feature features]
label  = clk (0 or 1)
```

---

## Phase 5: CTR Prediction Model

### Baseline: Logistic Regression

- Input: feature matrix from Phase 4
- Output: click probability (0–1)
- Evaluation metric: **AUC** (official dataset baseline = 0.622)
- Class imbalance: CTR ≈ 5% (~19:1 ratio) — consider `class_weight='balanced'`

### Possible extensions (to be discussed with supervisor)

The supervisor's notes mention **Factorization Machines (FM)** as a natural next step:
- FM automatically learns pairwise feature interactions without manual feature crossing
- Well-suited for high-dimensional sparse CTR prediction tasks
- Can be implemented with `lightfm` or `xlearn`

---

## Coding Standards

1. **Header comment in every script**: purpose, input files, output files
2. **Annotate the "why"**, not just the "what"
3. **All paths** defined at the top of each script via `BASE_DIR`
4. **Each script runs independently**: data passed between stages via files, not in-memory state
5. **Plots** saved to `outputs/plots/`, stats summaries to `outputs/stats/` (JSON)
6. **Free memory** after heavy operations: `del df; gc.collect()`

---

## Suggested Git Commit History

```
feat: init project structure and add project brief
feat: EDA and cleaning for ad_feature
feat: EDA and cleaning for user_profile
feat: EDA and cleaning for raw_sample
feat: chunked cleaning and parquet conversion for behavior_log
feat: aggregate user behavior stats from behavior_log
feat: hypothesis verification H1-H5 with visualization
feat: build sparse user-ad interaction matrix from raw_sample
feat: SVD matrix factorization, extract user and ad embeddings
feat: enrich features with user_profile and ad_feature
feat: build final training dataset with joined features
feat: logistic regression CTR model, evaluate AUC
```

---

## Dependencies

```
pandas>=2.0
numpy
scipy          # sparse matrix operations
scikit-learn   # TruncatedSVD, LogisticRegression
pyarrow        # parquet I/O
matplotlib
seaborn
```
