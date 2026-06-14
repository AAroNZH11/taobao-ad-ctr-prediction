#!/usr/bin/env python3
"""
Script 12 — Offline A/B evaluation of the CTR model.
Input:  vigorous-volhard/data/processed/predictions_test.csv
Outputs:
  outputs/stats/evaluation_report.json
  outputs/plots/ab_simulation.png
  outputs/plots/gain_curve.png

Scoring rule (per impression):
  if model recommends it (pred_prob > threshold):
      score = clk * 1.0          (clicked → 1, not clicked → 0)
  else:
      score = clk * ad_avg_ctr   (missed click → partial credit)

Baseline: random recommendation at the same 10% rate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
VIGOROUS_DIR  = Path("/Users/aaronzhong/Documents/TaoBao-Project/.claude/worktrees/vigorous-volhard-727fcf")
PROCESSED_DIR = VIGOROUS_DIR / "data" / "processed"
PLOTS_DIR     = BASE_DIR / "outputs" / "plots"
STATS_DIR     = BASE_DIR / "outputs" / "stats"

PRED_PATH     = PROCESSED_DIR / "predictions_test.csv"
REPORT_PATH   = STATS_DIR / "evaluation_report.json"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

RECOMMEND_RATE = 0.10   # top 10% of impressions are "recommended"
RANDOM_SEED    = 42


def compute_scores(df, is_recommended_col, ad_avg_ctr):
    """
    Vectorised scoring: recommended impressions get clk*1.0,
    non-recommended get clk*ad_avg_ctr.
    Returns per-impression score Series.
    """
    scores = np.where(
        df[is_recommended_col],
        df['clk'].values * 1.0,
        df['clk'].values * df['_ad_ctr'].values,
    )
    return scores


def main():
    print("=" * 60)
    print("Script 12 — Offline A/B Evaluation")
    print("=" * 60)

    if not PRED_PATH.exists():
        print(f"ERROR: predictions not found at {PRED_PATH}")
        print("Run script 11 first.")
        raise SystemExit(1)

    # ── Load predictions ──────────────────────────────────────────────────────
    print(f"Loading predictions: {PRED_PATH}")
    df = pd.read_csv(PRED_PATH,
                     dtype={'user': 'int32', 'adgroup_id': 'int32',
                            'time_stamp': 'int64', 'clk': 'int8',
                            'pred_prob': 'float32'})
    n = len(df)
    n_recommend = int(n * RECOMMEND_RATE)
    print(f"  {n:,} impressions — recommending top {n_recommend:,} ({RECOMMEND_RATE:.0%})")

    # ── Ad average CTR in the test set ───────────────────────────────────────
    ad_avg_ctr = df.groupby('adgroup_id')['clk'].mean().rename('_ad_ctr')
    df = df.join(ad_avg_ctr, on='adgroup_id')

    total_clicks = df['clk'].sum()
    overall_ctr  = df['clk'].mean()
    print(f"  Test set: {total_clicks:,} clicks, overall CTR = {overall_ctr:.4f}")

    # ── Our model: recommend top-N by pred_prob ───────────────────────────────
    threshold = df['pred_prob'].nlargest(n_recommend).min()
    df['model_recommended'] = df['pred_prob'] >= threshold

    model_scores   = compute_scores(df, 'model_recommended', ad_avg_ctr)
    model_total    = model_scores.sum()
    model_clicks   = df.loc[df['model_recommended'], 'clk'].sum()
    model_recall   = model_clicks / total_clicks if total_clicks > 0 else 0.0

    # ── Baseline: random recommendation at same rate ──────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    rand_mask = np.zeros(n, dtype=bool)
    rand_idx  = rng.choice(n, size=n_recommend, replace=False)
    rand_mask[rand_idx] = True
    df['baseline_recommended'] = rand_mask

    baseline_scores = compute_scores(df, 'baseline_recommended', ad_avg_ctr)
    baseline_total  = baseline_scores.sum()
    baseline_clicks = df.loc[df['baseline_recommended'], 'clk'].sum()
    baseline_recall = baseline_clicks / total_clicks if total_clicks > 0 else 0.0

    lift = (model_total / baseline_total - 1) * 100 if baseline_total > 0 else 0.0

    print(f"\nResults:")
    print(f"  Model total score   : {model_total:,.2f}  (clicks in recommended: {model_clicks:,}, recall: {model_recall:.3f})")
    print(f"  Baseline total score: {baseline_total:,.2f}  (clicks in recommended: {baseline_clicks:,}, recall: {baseline_recall:.3f})")
    print(f"  Lift over baseline  : {lift:+.2f}%")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        'n_test_impressions':   n,
        'n_recommended':        n_recommend,
        'recommend_rate':       RECOMMEND_RATE,
        'total_clicks':         int(total_clicks),
        'overall_ctr':          float(overall_ctr),
        'model': {
            'total_score':      float(model_total),
            'clicks_captured':  int(model_clicks),
            'click_recall':     float(model_recall),
        },
        'baseline': {
            'total_score':      float(baseline_total),
            'clicks_captured':  int(baseline_clicks),
            'click_recall':     float(baseline_recall),
        },
        'lift_pct': float(lift),
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {REPORT_PATH}")

    # ── Plot 1: Total score comparison bar chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Our XGBoost\nModel', 'Random\nBaseline'],
                  [model_total, baseline_total],
                  color=['#2196F3', '#90CAF9'], edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='{:,.1f}', padding=4, fontsize=11)
    ax.set_ylabel('Total A/B Score', fontsize=12)
    ax.set_title(f'Offline A/B Score Comparison\n(Top {RECOMMEND_RATE:.0%} recommended, lift = {lift:+.1f}%)',
                 fontsize=12)
    ax.set_ylim(0, max(model_total, baseline_total) * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / 'ab_simulation.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out}")

    # ── Plot 2: Cumulative gain curve ─────────────────────────────────────────
    # Sort by pred_prob descending; track cumulative clicks captured vs. fraction shown
    df_sorted = df.sort_values('pred_prob', ascending=False).reset_index(drop=True)
    cum_clicks_model = df_sorted['clk'].cumsum().values
    frac_shown = np.arange(1, n + 1) / n

    # Baseline: random order → expected cumulative = total_clicks * frac_shown
    cum_clicks_random = total_clicks * frac_shown

    # Perfect model: all clicks first
    perfect = np.minimum(np.arange(1, n + 1), total_clicks)

    # Downsample to 500 points for a compact plot
    step = max(1, n // 500)
    idx  = np.arange(0, n, step)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frac_shown[idx] * 100, cum_clicks_model[idx] / total_clicks * 100,
            label='XGBoost model', color='#2196F3', linewidth=2)
    ax.plot(frac_shown[idx] * 100, cum_clicks_random[idx] / total_clicks * 100,
            label='Random baseline', color='#FF7043', linewidth=1.5, linestyle='--')
    ax.plot(frac_shown[idx] * 100, perfect[idx] / total_clicks * 100,
            label='Perfect model', color='#4CAF50', linewidth=1.5, linestyle=':')
    ax.axvline(RECOMMEND_RATE * 100, color='grey', linewidth=1,
               linestyle=':', label=f'Recommend threshold ({RECOMMEND_RATE:.0%})')
    ax.set_xlabel('% of impressions shown (sorted by pred score)', fontsize=11)
    ax.set_ylabel('% of total clicks captured', fontsize=11)
    ax.set_title('Cumulative Gain Curve — CTR Model', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = PLOTS_DIR / 'gain_curve.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out}")

    print("\nScript 12 complete.")


if __name__ == '__main__':
    main()
