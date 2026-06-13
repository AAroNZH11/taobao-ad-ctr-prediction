#!/usr/bin/env python3
"""
Script 11 — XGBoost CTR model training + test-set prediction.
Inputs:
  vigorous-volhard/data/processed/features_train.csv  (22 GB, 23.7M rows)
  vigorous-volhard/data/processed/features_test.csv   (2.6 GB, 2.8M rows)
Outputs:
  vigorous-volhard/data/processed/predictions_test.csv
  vigorous-volhard/data/processed/model.xgb
  outputs/stats/model_report.json
"""
import gc
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
VIGOROUS_DIR  = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/vigorous-volhard-727fcf")
PROCESSED_DIR = VIGOROUS_DIR / "data" / "processed"
STATS_DIR     = BASE_DIR / "outputs" / "stats"

TRAIN_PATH    = PROCESSED_DIR / "features_train.csv"
TEST_PATH     = PROCESSED_DIR / "features_test.csv"
PRED_PATH     = PROCESSED_DIR / "predictions_test.csv"
MODEL_PATH    = PROCESSED_DIR / "model.xgb"
REPORT_PATH   = STATS_DIR / "model_report.json"
DTRAIN_BUFFER = PROCESSED_DIR / "dtrain.buffer"
CKPT_DIR      = PROCESSED_DIR / "checkpoint_phase6"

STATS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dtype map (explicit types for all 109 columns) ────────────────────────────
INT8_COLS = ['pid', 'clk', 'brand_known', 'pvalue_known',
             'beh_has_purchase', 'ucate_has_bought']

INT32_COLS = ['user', 'adgroup_id', 'cate_id', 'campaign_id', 'customer',
              'recency_pv', 'recency_cart', 'recency_fav', 'recency_buy',
              'beh_pv', 'beh_cart', 'beh_fav', 'beh_buy', 'beh_total',
              'ucate_pv', 'ucate_cart', 'ucate_fav', 'ucate_buy',
              'ubrand_pv', 'ubrand_cart', 'ubrand_fav', 'ubrand_buy',
              'peak_hour']

INT64_COLS = ['time_stamp']

FLOAT32_COLS = (
    [f'user_svd_{i}' for i in range(32)]
    + [f'ad_svd_{i}'  for i in range(32)]
    + ['cms_segid', 'cms_group_id', 'final_gender_code',
       'age_level', 'pvalue_level', 'shopping_level',
       'occupation', 'new_user_class_level']
    + ['brand', 'log1p_price']
    + ['beh_buy_rate', 'beh_cart_rate']
    + ['ucate_cart_rate', 'ucate_buy_rate']
    + ['weekday_ratio']
)

DTYPE_MAP = {c: 'int8'    for c in INT8_COLS}
DTYPE_MAP.update({c: 'int32'   for c in INT32_COLS})
DTYPE_MAP.update({c: 'int64'   for c in INT64_COLS})
DTYPE_MAP.update({c: 'float32' for c in FLOAT32_COLS})

# Columns excluded from model features (IDs + label)
EXCLUDE = {'user', 'time_stamp', 'adgroup_id', 'clk'}

# ── XGBoost hyper-parameters ──────────────────────────────────────────────────
PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',   # histogram algorithm — essential for 23M rows
    'learning_rate':    0.05,
    'max_depth':        6,
    'min_child_weight': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 19,       # ~1/CTR handles 95:5 class imbalance
}
NUM_ROUNDS = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_ckpt():
    """Return path to the most recent checkpoint file, or None."""
    files = glob.glob(str(CKPT_DIR / 'xgb_ckpt_*.ubj'))
    if not files:
        return None
    # Sort by the integer suffix (xgb_ckpt_0.ubj → 0, xgb_ckpt_10.ubj → 10)
    return max(files, key=lambda f: int(Path(f).stem.split('_')[-1]))


def build_dtrain():
    """
    Return (DMatrix, feature_cols).
    Layer 3: loads from binary buffer if available, else reads CSV and saves buffer.
    """
    if DTRAIN_BUFFER.exists():
        print(f"[Layer 3] DMatrix buffer found — loading from {DTRAIN_BUFFER}")
        dm = xgb.DMatrix(str(DTRAIN_BUFFER))
        feature_cols = list(dm.feature_names)
        print(f"  Loaded: {dm.num_row():,} rows, {dm.num_col()} features")
        return dm, feature_cols

    print(f"[csv] Loading train CSV — {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, dtype=DTYPE_MAP)
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"  {len(df):,} rows — in-memory: {mem_gb:.2f} GB")

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    print(f"  Building DMatrix from DataFrame ({len(feature_cols)} features) ...")
    # Pass DataFrame directly — avoids dtype upcast that .values would trigger
    dm = xgb.DMatrix(df[feature_cols], label=df['clk'])
    del df; gc.collect()
    print("  DataFrame freed.")

    print(f"  Saving DMatrix buffer → {DTRAIN_BUFFER}")
    dm.save_binary(str(DTRAIN_BUFFER))
    print("  Buffer saved.")
    return dm, feature_cols


def train_model(dtrain, feature_cols):
    """
    Train XGBoost or resume from the latest checkpoint.
    Layer 2: saves checkpoint every 50 rounds to CKPT_DIR.
    Returns trained Booster.
    """
    callbacks = [
        xgb.callback.TrainingCheckPoint(
            directory=CKPT_DIR,
            name='xgb_ckpt',
            as_pickle=False,
            interval=50,
        )
    ]

    ckpt = find_latest_ckpt()
    if ckpt:
        # Load checkpoint to find how many rounds are already done
        tmp = xgb.Booster()
        tmp.load_model(ckpt)
        completed = tmp.num_boosted_rounds()
        del tmp
        remaining = NUM_ROUNDS - completed
        if remaining <= 0:
            print(f"[Layer 2] Checkpoint already at {completed} rounds — loading directly")
            model = xgb.Booster()
            model.load_model(ckpt)
            return model
        print(f"[Layer 2] Resuming from checkpoint: {completed} done, {remaining} to go")
        model = xgb.train(
            PARAMS, dtrain,
            num_boost_round=remaining,
            evals=[(dtrain, 'train')],
            verbose_eval=50,
            callbacks=callbacks,
            xgb_model=ckpt,
        )
    else:
        print(f"Training XGBoost from scratch — {NUM_ROUNDS} rounds")
        model = xgb.train(
            PARAMS, dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=[(dtrain, 'train')],
            verbose_eval=50,
            callbacks=callbacks,
        )

    return model


def main():
    print("=" * 60)
    print("Script 11 — XGBoost CTR Model Training")
    print("=" * 60)

    # Short-circuit: if predictions already exist, nothing to do
    if PRED_PATH.exists() and MODEL_PATH.exists() and REPORT_PATH.exists():
        print(f"All outputs already exist. Delete to re-run:")
        print(f"  {PRED_PATH}")
        sys.exit(0)

    # ── Step 1: Load / build training DMatrix (Layer 3) ──────────────────────
    dtrain, feature_cols = build_dtrain()

    # ── Step 2: Train or load model (Layer 2) ─────────────────────────────────
    if MODEL_PATH.exists():
        print(f"\n[skip] Model already saved at {MODEL_PATH} — loading")
        model = xgb.Booster()
        model.load_model(str(MODEL_PATH))
    else:
        model = train_model(dtrain, feature_cols)
        model.save_model(str(MODEL_PATH))
        print(f"\nModel saved → {MODEL_PATH}")

    del dtrain; gc.collect()

    # ── Step 3: Predict on test set ───────────────────────────────────────────
    print(f"\nLoading test CSV — {TEST_PATH}")
    dtest_df = pd.read_csv(TEST_PATH, dtype=DTYPE_MAP)
    print(f"  {len(dtest_df):,} rows")

    pred_meta = dtest_df[['user', 'adgroup_id', 'time_stamp', 'clk']].copy()
    dtest = xgb.DMatrix(dtest_df[feature_cols])
    del dtest_df; gc.collect()

    print("Predicting ...")
    pred_prob = model.predict(dtest)
    del dtest; gc.collect()

    # ── Step 4: AUC ──────────────────────────────────────────────────────────
    auc = roc_auc_score(pred_meta['clk'].values, pred_prob)
    print(f"\n{'='*60}")
    print(f"Test AUC  : {auc:.6f}")
    print(f"Baseline  : 0.622000")
    print(f"Delta     : {auc - 0.622:+.6f}  ({'BEAT' if auc > 0.622 else 'MISS'})")
    print(f"{'='*60}")

    # ── Step 5: Save predictions ──────────────────────────────────────────────
    pred_meta = pred_meta.copy()
    pred_meta['pred_prob'] = pred_prob
    pred_meta.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved → {PRED_PATH}")

    # ── Step 6: Feature importance report ────────────────────────────────────
    importance = model.get_score(importance_type='gain')
    top20 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    report = {
        'auc': float(auc),
        'baseline_auc': 0.622,
        'delta': float(auc - 0.622),
        'beats_baseline': bool(auc > 0.622),
        'num_boost_round': NUM_ROUNDS,
        'params': PARAMS,
        'top20_feature_importance_gain': {k: float(v) for k, v in top20},
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {REPORT_PATH}")

    print("\nScript 11 complete.")


if __name__ == '__main__':
    main()
