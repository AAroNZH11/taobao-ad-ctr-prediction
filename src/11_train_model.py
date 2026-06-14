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

Training strategy:
  - Temporal validation split: 2017-05-06~11 as train, 2017-05-12 as val
  - Early stopping on val AUC (prevents overfitting to training days)
  - XGBoost hist algorithm for speed on 23M rows
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
CKPT_DIR      = PROCESSED_DIR / "checkpoint_phase6"

STATS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 2017-05-12 00:00:00 CST = Unix timestamp 1494518400
# Rows >= this go to validation; rows < this go to training
VAL_START_TS = 1494518400

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

EXCLUDE = {'user', 'time_stamp', 'adgroup_id', 'clk'}

# ── XGBoost hyper-parameters ──────────────────────────────────────────────────
PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'tree_method':      'hist',
    'learning_rate':    0.05,
    'max_depth':        6,
    'min_child_weight': 50,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 19,
}
NUM_ROUNDS          = 500   # upper bound — early stopping will decide actual rounds
EARLY_STOP_ROUNDS   = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_ckpt():
    files = glob.glob(str(CKPT_DIR / 'xgb_ckpt_*.ubj'))
    if not files:
        return None
    return max(files, key=lambda f: int(Path(f).stem.split('_')[-1]))


def load_and_split():
    """
    Read features_train.csv, split by date into train/val DMatrix objects.
    Returns (dtrain, dval, feature_cols).
    Memory strategy: free each intermediate DataFrame as soon as DMatrix is built.
    """
    print(f"Loading train CSV — {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH, dtype=DTYPE_MAP)
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"  {len(df):,} rows — in-memory: {mem_gb:.2f} GB")

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    train_mask = df['time_stamp'] < VAL_START_TS

    n_train = train_mask.sum()
    n_val   = (~train_mask).sum()
    print(f"  Split: {n_train:,} train rows (05-06~11)  |  {n_val:,} val rows (05-12)")

    # Extract val subset first (smaller), then train subset, then free df
    df_val   = df[~train_mask].reset_index(drop=True)
    df_train = df[train_mask].reset_index(drop=True)
    del df; gc.collect()

    print("  Building train DMatrix ...")
    dtrain = xgb.DMatrix(df_train[feature_cols], label=df_train['clk'])
    del df_train; gc.collect()

    print("  Building val DMatrix ...")
    dval = xgb.DMatrix(df_val[feature_cols], label=df_val['clk'])
    del df_val; gc.collect()

    print("  DMatrix objects ready.")
    return dtrain, dval, feature_cols


def train_model(dtrain, dval):
    """
    Train with early stopping on dval AUC.
    Layer 2: checkpoint every 50 rounds.
    """
    callbacks = [
        xgb.callback.TrainingCheckPoint(
            directory=CKPT_DIR,
            name='xgb_ckpt',
            as_pickle=False,
            interval=50,
        )
    ]

    evals = [(dtrain, 'train'), (dval, 'val')]

    ckpt = find_latest_ckpt()
    if ckpt:
        tmp = xgb.Booster()
        tmp.load_model(ckpt)
        completed = tmp.num_boosted_rounds()
        del tmp
        remaining = NUM_ROUNDS - completed
        if remaining <= 0:
            print(f"[checkpoint] Already at {completed} rounds — loading directly")
            model = xgb.Booster()
            model.load_model(ckpt)
            return model
        print(f"[checkpoint] Resuming from round {completed}, {remaining} rounds remaining")
        model = xgb.train(
            PARAMS, dtrain,
            num_boost_round=remaining,
            evals=evals,
            verbose_eval=20,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            callbacks=callbacks,
            xgb_model=ckpt,
        )
    else:
        print(f"Training from scratch — up to {NUM_ROUNDS} rounds (early stopping @ {EARLY_STOP_ROUNDS})")
        model = xgb.train(
            PARAMS, dtrain,
            num_boost_round=NUM_ROUNDS,
            evals=evals,
            verbose_eval=20,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            callbacks=callbacks,
        )

    print(f"  Best round: {model.best_iteration}  |  Best val-AUC: {model.best_score:.6f}")
    return model


def main():
    print("=" * 60)
    print("Script 11 — XGBoost CTR Model Training (with early stopping)")
    print("=" * 60)

    if PRED_PATH.exists() and MODEL_PATH.exists() and REPORT_PATH.exists():
        print("All outputs already exist. Delete them to re-run.")
        sys.exit(0)

    # ── Step 1: Load CSV and build train/val DMatrix ──────────────────────────
    dtrain, dval, feature_cols = load_and_split()

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    if MODEL_PATH.exists():
        print(f"\n[skip] Model already at {MODEL_PATH} — loading")
        model = xgb.Booster()
        model.load_model(str(MODEL_PATH))
    else:
        model = train_model(dtrain, dval)
        model.save_model(str(MODEL_PATH))
        print(f"\nModel saved → {MODEL_PATH}")

    del dtrain, dval; gc.collect()

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

    # ── Step 6: Report ────────────────────────────────────────────────────────
    best_round = getattr(model, 'best_iteration', None)
    best_val   = getattr(model, 'best_score', None)

    importance = model.get_score(importance_type='gain')
    top20 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    report = {
        'auc': float(auc),
        'baseline_auc': 0.622,
        'delta': float(auc - 0.622),
        'beats_baseline': bool(auc > 0.622),
        'best_round': best_round,
        'best_val_auc': float(best_val) if best_val else None,
        'params': PARAMS,
        'top20_feature_importance_gain': {k: float(v) for k, v in top20},
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {REPORT_PATH}")

    print("\nScript 11 complete.")


if __name__ == '__main__':
    main()
