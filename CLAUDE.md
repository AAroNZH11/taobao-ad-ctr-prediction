# Taobao Ad CTR Prediction — Project Context

## Goal

Predict whether a user will click on a displayed ad, given a (user, ad) pair.
Evaluation metric: **AUC**. Baseline to beat: **0.622**.

---

## Project Framework (based on supervisor's direction)

Two parallel routes, both feeding into one final model:

### Route A — Feature Engineering
Hand-craft features from behavior logs and user/ad attributes — **DONE (scripts 05, 09, 10)**

### Route B — Matrix Factorization
Decompose user-ad interaction matrix into low-dimensional embeddings — **DONE (scripts 07–08)**:
- User embedding: 32-dim SVD vector per user
- Ad embedding: 32-dim SVD vector per ad
- Explained variance: 3.08% (normal for implicit feedback matrices)

### Combined Model — Phase 6
Concatenate Route A features + Route B embeddings → train XGBoost → get AUC.

### Evaluation System — Phase 6
Simulate offline A/B test (supervisor's requirement):
- Recommended & clicked → full score (1.0)
- Recommended & not clicked → zero
- Not recommended but might have clicked → score based on that ad's average CTR in test set
Compare cumulative score: our system vs. naive baseline.

### Cohort Matching (optional enhancement — skip for now)

### ⚠️ Neural Network / Deep MLP
**Do NOT implement unless explicitly asked.**

---

## Current Status

- **Phase 1 (EDA & Cleaning):** ✅ done — scripts 01–05
- **Phase 2 (Hypothesis Verification):** ✅ done — script 06
- **Phase 3 (Interaction Matrix & SVD):** ✅ done — scripts 07–08
- **Phase 4 (Rich behavior features):** ✅ done — script 09
- **Phase 5 (Feature engineering, join everything):** ✅ done — script 10
- **Phase 6 (Model training + evaluation system):** ⬜ not started — scripts 11–12

---

## Data Locations

### Raw CSVs (not in repo)
```
/Users/aaronzhong/Documents/TaoBao-Project/
  raw_sample.csv       (1.1 GB, 26,557,961 rows)
  behavior_log.csv     (23 GB, 723M raw rows)
  user_profile.csv     (23 MB)
  ad_feature.csv       (30 MB)
```

### Processed data — split across THREE worktrees (not in repo)

```
# Phase 1 outputs — condescending-franklin worktree:
/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/condescending-franklin-f538a6/data/processed/
  behavior_log_parquet/       362 snappy parquet chunks, ~4.9 GB total
  user_behavior_stats.csv     1,136,338 rows
                              cols: user, pv_count, cart_count, fav_count, buy_count,
                                    total_actions, has_purchase, buy_rate, cart_rate

# Phase 3 outputs — beautiful-dhawan worktree:
/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/beautiful-dhawan-257e0a/data/processed/
  user_embeddings.csv         457,554 rows × 33 cols (user_id + svd_0…svd_31)
  ad_embeddings.csv           227,800 rows × 33 cols (adgroup_id + svd_0…svd_31)
  interaction_matrix.npz      sparse CSR (457,554 users × 227,800 ads)
  user_index.csv / ad_index.csv

# Phase 4 + 5 outputs — vigorous-volhard worktree (CURRENT WORKTREE):
/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/vigorous-volhard-727fcf/data/processed/
  user_category_stats.csv     57,266,514 rows, 1.7 GB
                              cols: user, cate, pv, cart, fav, buy,
                                    cart_rate, buy_rate, has_bought
  user_brand_stats.csv        147,379,055 rows, 3.0 GB
                              cols: user, brand, pv, cart, fav, buy
                              NOTE: brand is int64 (no nulls — null brands were dropped during extraction)
  user_recency_stats.csv      1,136,338 rows, 47 MB
                              cols: user, last_pv_ts, last_cart_ts, last_fav_ts, last_buy_ts
                              NOTE: 0 means "never had this action" (not Unix epoch)
  user_temporal_stats.csv     1,136,338 rows, 30 MB
                              cols: user, peak_hour (CST, 0–23), weekday_ratio (0.0–1.0)
  features_train.csv          23,709,456 rows × 109 cols, 22 GB  ← MAIN INPUT FOR PHASE 6
  features_test.csv           2,848,505 rows  × 109 cols, 2.6 GB ← MAIN INPUT FOR PHASE 6
  checkpoint_phase4/          intermediate batch parquets (safe to ignore)
```

⚠️ **Never use `/Users/aaronzhong/Documents/TaoBao-Project/analysis/`** — old exploratory session.

### Path template for new scripts (scripts 11–12)
```python
BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_DIR       = Path("/Users/aaronzhong/Documents/TaoBao-Project")
FRANKLIN_DIR   = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/condescending-franklin-f538a6")
DHAWAN_DIR     = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/beautiful-dhawan-257e0a")
VIGOROUS_DIR   = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/vigorous-volhard-727fcf")
PROCESSED_DIR  = VIGOROUS_DIR / "data" / "processed"   # Phase 4+5 outputs live here
PLOTS_DIR      = BASE_DIR / "outputs" / "plots"
STATS_DIR      = BASE_DIR / "outputs" / "stats"

# Convenience shortcuts:
TRAIN_PATH     = PROCESSED_DIR / "features_train.csv"
TEST_PATH      = PROCESSED_DIR / "features_test.csv"
```

---

## Feature Matrix — Column Reference (109 columns)

```
GROUP              COLS                                             COUNT
─────────────────────────────────────────────────────────────────────────
IDs / label        user, time_stamp, adgroup_id, pid, clk            5
Ad attributes      cate_id, campaign_id, customer, brand,            6
                   log1p_price, brand_known
                   NOTE: brand=-1.0 means no brand (sentinel)
User SVD emb       user_svd_0 … user_svd_31                         32
Ad SVD emb         ad_svd_0  … ad_svd_31                            32
User profile       cms_segid, cms_group_id, final_gender_code,       9
                   age_level, pvalue_level, shopping_level,
                   occupation, new_user_class_level, pvalue_known
                   NOTE: pvalue_known=0/1 indicator; missing→0
Basic behavior     beh_pv, beh_cart, beh_fav, beh_buy,              8
                   beh_total, beh_has_purchase,
                   beh_buy_rate, beh_cart_rate
User × category    ucate_pv, ucate_cart, ucate_fav, ucate_buy,       7
cross-feature      ucate_cart_rate, ucate_buy_rate, ucate_has_bought
                   (user's history in candidate ad's category)
User × brand       ubrand_pv, ubrand_cart, ubrand_fav, ubrand_buy    4
cross-feature      (user's history with candidate ad's brand)
Recency deltas     recency_pv, recency_cart, recency_fav, recency_buy 4
                   (seconds elapsed since that action; 0 = never done)
Temporal           peak_hour (int, CST), weekday_ratio (float)        2
─────────────────────────────────────────────────────────────────────────
TOTAL                                                               109
```

**Feature columns for model** = all 109 EXCEPT: `user`, `time_stamp`, `adgroup_id`, `clk`
→ 105 actual features fed to the model.

---

## Phase 6 — Detailed Implementation Plan

### Script 11 — `11_train_model.py`

**Goal:** Train XGBoost on features_train, predict on features_test, report AUC.

**Inputs:**
- `PROCESSED_DIR/features_train.csv` (22 GB on disk)
- `PROCESSED_DIR/features_test.csv`

**Outputs:**
- `PROCESSED_DIR/predictions_test.csv` — cols: user, adgroup_id, time_stamp, clk, pred_prob
- `PROCESSED_DIR/model.xgb` — saved XGBoost model
- `STATS_DIR/model_report.json` — AUC, top-20 feature importances

**⚠️ Critical memory strategy for loading features_train.csv:**

Do NOT load with default pandas dtypes — that would use ~20 GB. Use explicit dtypes:

```python
# Build dtype dict before read_csv
int8_cols  = ['pid', 'clk', 'brand_known', 'pvalue_known',
              'beh_has_purchase', 'ucate_has_bought']
int32_cols = ['user', 'adgroup_id', 'cate_id', 'campaign_id', 'customer',
              'recency_pv', 'recency_cart', 'recency_fav', 'recency_buy',
              'beh_pv', 'beh_cart', 'beh_fav', 'beh_buy', 'beh_total',
              'ucate_pv', 'ucate_cart', 'ucate_fav', 'ucate_buy',
              'ubrand_pv', 'ubrand_cart', 'ubrand_fav', 'ubrand_buy',
              'peak_hour']
int64_cols = ['time_stamp']
# everything else → float32 (SVD cols, rates, profile floats, brand, log1p_price, weekday_ratio)

dtype_map = {c: 'int8'  for c in int8_cols}
dtype_map.update({c: 'int32' for c in int32_cols})
dtype_map.update({c: 'int64' for c in int64_cols})
# specify float32 for all remaining float columns explicitly too

df = pd.read_csv(TRAIN_PATH, dtype=dtype_map)
```

With proper dtypes: ~10–12 GB in RAM. **After loading, immediately convert to xgb.DMatrix and delete the DataFrame to free memory before training.**

**Training approach:**
```python
import xgboost as xgb

FEATURE_COLS = [c for c in df.columns
                if c not in ('user', 'time_stamp', 'adgroup_id', 'clk')]

dtrain = xgb.DMatrix(df[FEATURE_COLS].values, label=df['clk'].values,
                     feature_names=FEATURE_COLS)
del df; gc.collect()  # free ~10 GB before training starts

params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',   # histogram-based — much faster on large data
    'learning_rate':    0.05,
    'max_depth':        6,
    'min_child_weight': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 19,       # ~1/CTR to handle class imbalance (95:5 ratio)
}
model = xgb.train(params, dtrain, num_boost_round=300,
                  evals=[(dtrain, 'train')], verbose_eval=50)
```

Then load test set (same dtype_map), build dtest DMatrix, predict, compute AUC with sklearn.

**Note on `tree_method='hist'`:** This is essential. Default `exact` method would be prohibitively slow on 23M rows. `hist` is XGBoost's fast histogram algorithm, comparable in speed to LightGBM.

### Script 12 — `12_evaluate.py`

**Goal:** Implement supervisor's A/B scoring simulation and produce comparison plots.

**Input:** `PROCESSED_DIR/predictions_test.csv` (user, adgroup_id, time_stamp, clk, pred_prob)

**Scoring rule (per-impression):**
```
score(impression) =
  if our model recommends it (pred_prob > threshold):
    clk * 1.0       # clicked → full score; not clicked → 0
  else:
    clk * ad_avg_ctr # missed click → partial credit based on ad's avg CTR in test set
```

**Implementation steps:**
1. Compute `ad_avg_ctr` = mean(clk) per adgroup_id in test set
2. Choose threshold = top-10% of pred_prob scores (i.e., recommend the 285K highest-scoring impressions)
3. Compute per-impression score under "our system" (using pred_prob ranking)
4. Compute per-impression score under "baseline system" (random ranking or uniform threshold)
5. Plot: score distribution histogram, cumulative gain curve, total score comparison
6. Save plots → `PLOTS_DIR/`, summary → `STATS_DIR/evaluation_report.json`

**Outputs:**
- `STATS_DIR/evaluation_report.json` — total scores, lift over baseline
- `PLOTS_DIR/ab_simulation.png`
- `PLOTS_DIR/gain_curve.png`

---

## Mandatory Verification Protocol

**Before starting Phase 6, spawn an Explore subagent and verify:**

| What to check | Expected |
|--------------|----------|
| `features_train.csv` exists | 23,709,456 rows, 109 cols |
| `features_test.csv` exists | 2,848,505 rows, 109 cols |
| `clk` column present with ~5% positive rate | ~4.7% in train sample |
| No all-NaN or all-zero columns | verified in Phase 5 |
| Machine RAM available | `vm_stat` — expect ≥ 12 GB free before running script 11 |

---

## Hard Rules

**Machine has 24 GB RAM** (not 3.8 GB — that was an old estimate).
- features_train.csv = 22 GB on disk → ~10–12 GB with proper dtypes → **always specify dtype_map**
- After loading and converting to lgb.Dataset, delete the raw DataFrame immediately
- `del df; gc.collect()` before every heavy operation
- Each script runs in isolation — no in-memory state passed between scripts

---

## Coding Standards

- Header comment in every script: purpose, input files, output files
- All path variables defined at the top using the template above
- Plots → `outputs/plots/`, stats summaries → `outputs/stats/` (JSON)
- `gc.collect()` after heavy operations
- Scripts numbered with two-digit prefixes: 11, 12

## Git Commit Style

Short, natural, human-sounding. No `feat:` / `fix:` prefixes.

Commit messages used so far (for reference):
- "add rich user behavior features from behavior log"
- "build final feature matrix for train and test sets"

---

## Key Numbers

- Overall CTR: 5.14% (1,366,056 clicks / 26,557,961 impressions)
- behavior_log: 723M raw rows → 7.06% duplicates removed → 362 parquet chunks
- Unique users: 1,136,338 | Unique ads: 846,811
- Training set: 2017-05-06 to 2017-05-12 (23,709,456 rows) | Test set: 2017-05-13 (2,848,505 rows)
- SVD: 457,554 users × 227,800 ads, rank=32, explained variance=3.08%
- AUC baseline to beat: **0.622**

## Key Findings (Phase 2)

- Activity level is the strongest CTR signal (most active 20% click at 6.03% vs 4.22%)
- Cheaper ads get more clicks (monotonically: 6.12% → 4.68% across price quintiles)
- Female users click more (5.24% vs 4.84%)
- Heavy shoppers and buyers click LESS (counter-intuitive — they have intent and search directly)
