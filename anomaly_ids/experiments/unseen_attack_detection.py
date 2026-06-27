"""
experiments/unseen_attack_detection.py
========================================
"Can our IDS detect attacks it has NEVER been trained on?"

Experimental Design
--------------------
1.  Load the full MULTICLASS dataset (CICIDS 2017/2018).
2.  Pick N attack types to HOLD OUT (unseen during training).
3.  Train:
      • Best supervised model  (LightGBM)
      • Autoencoder            (trained on BENIGN only)
4.  Evaluate BOTH on the held-out attack flows.
5.  Visualise detection rates side-by-side.

This demonstrates the key advantage of the autoencoder approach:
it can flag novel attacks purely from deviation in traffic patterns.

Usage
-----
python -m experiments.unseen_attack_detection
    --data_dir data/raw
    --holdout_attacks "Bot" "Infiltration" "Heartbleed"
    --sample 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# ── Make sure project root is on path ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.data_loader import load_dataset, preprocess, generate_demo_data
from features.feature_engineering import FeatureEngineer
from models.lightgbm_model import LightGBMIDS
from models.autoencoder_model import AutoencoderIDS
from evaluation.evaluator import Evaluator
from visualization.visualizer import (
    plot_unseen_attack_results,
    plot_reconstruction_error_dist,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _print_banner(msg: str) -> None:
    w = 60
    print(f"\n{'='*w}")
    print(f"  {msg}")
    print(f"{'='*w}")


def pick_holdout_attacks(
    data: pd.DataFrame,
    n_holdout: int = 3,
    min_samples: int = 200,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Automatically pick attack types with enough samples to hold out.
    """
    exclude = set(exclude or [config.BENIGN_LABEL])
    counts  = data[config.LABEL_COLUMN].value_counts()
    valid   = [lbl for lbl, cnt in counts.items()
               if lbl not in exclude and cnt >= min_samples]
    if not valid:
        raise ValueError("No attack types with enough samples to hold out.")
    chosen = valid[:n_holdout]
    return chosen


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(
    data: pd.DataFrame,
    holdout_attacks: Optional[List[str]] = None,
    n_holdout: int = 3,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Core experiment logic.

    Parameters
    ----------
    data             : Full CICIDS DataFrame (multiclass labels)
    holdout_attacks  : Specific attack names to hold out; auto-chosen if None
    n_holdout        : Number of attacks to auto-choose if holdout_attacks is None

    Returns
    -------
    detection_rates : {model_name: {attack_type: detection_rate}}
    """
    lbl = config.LABEL_COLUMN

    # ── Step 1 – Choose holdout attacks ───────────────────────────────────
    if holdout_attacks is None:
        holdout_attacks = pick_holdout_attacks(data, n_holdout=n_holdout)
    _print_banner(f"Unseen Attack Detection Experiment")
    print(f"\n  Holdout attacks (never seen by supervised model):")
    for a in holdout_attacks:
        n = (data[lbl] == a).sum()
        print(f"    • {a:<30}  ({n:,} samples)")

    # ── Step 2 – Split: seen data (train) vs unseen attacks (test) ────────
    mask_holdout   = data[lbl].isin(holdout_attacks)
    data_seen      = data[~mask_holdout].copy()          # BENIGN + seen attacks
    data_unseen    = data[mask_holdout].copy()           # unseen attacks only

    print(f"\n  Training data : {len(data_seen):,} rows "
          f"(BENIGN + seen attacks)")
    print(f"  Holdout data  : {len(data_unseen):,} rows  ← unseen attacks")

    # ── Step 3 – Prepare "seen" training set ──────────────────────────────
    # Force binary mode for this experiment: BENIGN=0, any-attack=1
    config.CLASSIFICATION_MODE = "binary"
    X_tr_raw, X_te_raw, y_tr, y_te, _, feat_names, class_names = preprocess(data_seen)

    fe = FeatureEngineer()
    X_tr, y_tr = fe.fit_transform(X_tr_raw, y_tr, feat_names)
    X_te        = fe.transform(X_te_raw)

    # ── Prepare unseen test set ───────────────────────────────────────────
    num_cols  = data_unseen.select_dtypes(include=[np.number]).columns.tolist()
    X_unseen_raw = data_unseen[num_cols].values
    y_unseen     = np.ones(len(X_unseen_raw), dtype=int)  # all are attacks

    # Align columns (fe was fitted on data_seen columns)
    # Only keep the numeric columns that exist in both sets
    seen_num_cols = data_seen.select_dtypes(include=[np.number]).columns.tolist()
    common_cols   = [c for c in seen_num_cols if c in num_cols]
    col_idx_map   = {c: i for i, c in enumerate(num_cols)}
    X_unseen_raw_aligned = data_unseen[[c for c in common_cols]].values

    # Pad missing columns with zeros
    full_unseen  = np.zeros((len(X_unseen_raw_aligned), len(seen_num_cols)))
    for j, c in enumerate(common_cols):
        full_unseen[:, seen_num_cols.index(c)] = X_unseen_raw_aligned[:, j]
    X_unseen = fe.transform(full_unseen)

    # ── Step 4 – Train supervised model ───────────────────────────────────
    _print_banner("Training Supervised Model (LightGBM — seen attacks only)")
    lgbm = LightGBMIDS()
    lgbm.fit(X_tr, y_tr)

    # ── Step 5 – Train autoencoder (BENIGN only) ──────────────────────────
    _print_banner("Training Autoencoder (BENIGN traffic only)")
    X_benign = X_tr[y_tr == 0]
    ae = AutoencoderIDS(input_dim=X_tr.shape[1], threshold_percentile=95)
    ae.fit(X_benign)

    # ── Step 6 – Evaluate on held-out attack types ─────────────────────────
    _print_banner("Evaluating on Unseen Attacks")
    detection_rates: Dict[str, Dict[str, float]] = {
        "LightGBM (Supervised)": {},
        "Autoencoder (Anomaly)": {},
    }

    for attack_type in holdout_attacks:
        mask_att  = data_unseen[lbl] == attack_type
        X_att     = X_unseen[mask_att.values]
        if len(X_att) == 0:
            continue

        pred_lgbm = lgbm.predict(X_att)
        pred_ae   = ae.predict(X_att)

        y_att = np.ones(len(X_att), dtype=int)
        dr_lgbm = float(recall_score(y_att, pred_lgbm, zero_division=0))
        dr_ae   = float(recall_score(y_att, pred_ae, zero_division=0))

        detection_rates["LightGBM (Supervised)"][attack_type] = dr_lgbm
        detection_rates["Autoencoder (Anomaly)"][attack_type] = dr_ae
        print(f"\n  {attack_type}:")
        print(f"    LightGBM  → {dr_lgbm*100:.1f}% detected  "
              f"(classified as known attack)")
        print(f"    Autoencoder → {dr_ae*100:.1f}% detected  "
              f"(anomalous reconstruction)")

    # ── Step 7 – False positive rate on BENIGN ────────────────────────────
    X_benign_test = X_te[y_te == 0]
    fpr_lgbm = 1 - recall_score(
        np.zeros(len(X_benign_test)), lgbm.predict(X_benign_test), zero_division=0
    )
    fpr_ae = float(np.mean(ae.predict(X_benign_test)))
    print(f"\n  False Positive Rate on BENIGN traffic:")
    print(f"    LightGBM    → {fpr_lgbm*100:.2f}%")
    print(f"    Autoencoder → {fpr_ae*100:.2f}%")

    # ── Step 8 – Reconstruction error plot ────────────────────────────────
    errors_b  = ae.reconstruction_errors(X_benign_test[:5000])
    errors_a  = ae.reconstruction_errors(X_unseen[:5000])
    plot_reconstruction_error_dist(errors_b, errors_a, ae.threshold_)

    # ── Step 9 – Summary plot ─────────────────────────────────────────────
    plot_unseen_attack_results(detection_rates)

    # ── Summary table ─────────────────────────────────────────────────────
    _print_banner("Summary")
    df = pd.DataFrame(detection_rates).T * 100
    df = df.round(1)
    print(df.to_string())
    print()

    return detection_rates


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unseen Attack Detection Experiment for Anomaly IDS"
    )
    p.add_argument("--data_dir", default=config.DATA_DIR)
    p.add_argument("--holdout_attacks", nargs="*", default=None,
                   help="Attack type names to hold out (e.g. Bot Infiltration)")
    p.add_argument("--n_holdout", type=int, default=3,
                   help="Number of attack types to auto-select if --holdout_attacks not given")
    p.add_argument("--sample", type=float, default=0.5)
    p.add_argument("--demo", action="store_true",
                   help="Use synthetic data (no real dataset needed)")
    return p.parse_args()


def main():
    args = parse_args()
    config.APPLY_SMOTE = False   # Skip SMOTE for speed in experiments
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    if args.demo:
        print("  [Demo mode] Generating synthetic data…")
        data = generate_demo_data(n_samples=60_000)
    else:
        config.DATA_DIR = args.data_dir
        config.SAMPLE_FRACTION = args.sample
        config.CLASSIFICATION_MODE = "multiclass"  # need per-class labels
        data = load_dataset()

    run_experiment(
        data,
        holdout_attacks=args.holdout_attacks,
        n_holdout=args.n_holdout,
    )


if __name__ == "__main__":
    main()
