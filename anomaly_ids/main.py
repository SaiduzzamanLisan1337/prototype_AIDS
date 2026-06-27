"""
main.py  —  Anomaly IDS Pipeline
=========================================
Entry point that wires together the entire system.

Quick start
-----------
# Test with synthetic data (no dataset needed):
    python main.py --demo

# Binary classification (BENIGN vs ATTACK):
    python main.py --data_dir data/raw --mode binary

# Multi-class (per-attack-type labels):
    python main.py --data_dir data/raw --mode multiclass

# Use only 20% of data for a fast run:
    python main.py --data_dir data/raw --sample 0.2

# Run specific models only:
    python main.py --data_dir data/raw --models RandomForest XGBoost Autoencoder

# Run the unseen-attack experiment after training:
    python main.py --data_dir data/raw --experiment
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import config

# ── Lazy imports so missing extras give a friendly error ────────────────────
from data.data_loader import load_dataset, preprocess, generate_demo_data
from features.feature_engineering import FeatureEngineer
from models.random_forest_model import RandomForestIDS
from models.xgboost_model import XGBoostIDS
from models.lightgbm_model import LightGBMIDS
from models.svm_model import SVMIDS
from models.deep_learning_model import DeepLearningIDS
from models.autoencoder_model import AutoencoderIDS
from evaluation.evaluator import Evaluator
from visualization.visualizer import (
    plot_class_distribution,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
    plot_dl_training_history,
    plot_per_class_f1_heatmap,
    plot_per_attack_detection_rates,
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Anomaly IDS — Full ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default=config.DATA_DIR,
                   help="Folder containing CICIDS CSV files")
    p.add_argument("--mode", default=config.CLASSIFICATION_MODE,
                   choices=["binary", "multiclass"],
                   help="Classification mode")
    p.add_argument("--sample", type=float, default=config.SAMPLE_FRACTION,
                   help="Fraction of data to use (0.0–1.0)")
    p.add_argument("--models", nargs="+",
                   default=config.MODELS_TO_RUN,
                   choices=["RandomForest", "XGBoost", "LightGBM",
                            "SVM", "DeepLearning", "Autoencoder"],
                   help="Which models to train")
    p.add_argument("--no_smote", action="store_true",
                   help="Disable SMOTE oversampling")
    p.add_argument("--demo", action="store_true",
                   help="Run on synthetic data (no dataset needed — great for testing!)")
    p.add_argument("--experiment", action="store_true",
                   help="Also run the unseen-attack detection experiment")
    p.add_argument("--output_dir", default=config.OUTPUT_DIR)
    return p.parse_args()


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(name: str, input_dim: int, n_classes: int):
    registry = {
        "RandomForest": RandomForestIDS,
        "XGBoost":      XGBoostIDS,
        "LightGBM":     LightGBMIDS,
        "SVM":          SVMIDS,
        "DeepLearning": lambda: DeepLearningIDS(input_dim=input_dim, n_classes=n_classes),
    }
    if name in registry:
        return registry[name]()
    raise ValueError(f"Unknown model: {name}")


# ─── Banner ───────────────────────────────────────────────────────────────────

def _banner(msg: str, width: int = 60) -> None:
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


# ─── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    config.CLASSIFICATION_MODE = args.mode
    config.APPLY_SMOTE         = not args.no_smote
    config.SAMPLE_FRACTION     = args.sample
    config.DATA_DIR            = args.data_dir
    config.OUTPUT_DIR          = args.output_dir
    config.PLOTS_DIR           = os.path.join(args.output_dir, "plots")
    config.MODELS_DIR          = os.path.join(args.output_dir, "saved_models")

    for d in (config.OUTPUT_DIR, config.PLOTS_DIR, config.MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    _banner("Anomaly-Based IDS  —  ML Pipeline")
    print(f"  Mode      : {config.CLASSIFICATION_MODE}")
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  SMOTE     : {'enabled' if config.APPLY_SMOTE else 'disabled'}")
    print(f"  Sample    : {config.SAMPLE_FRACTION*100:.0f}% of dataset")

    # ── [1] Load data ─────────────────────────────────────────────────────
    _banner("[1] Loading Data")
    if args.demo:
        print("  ⚡  Demo mode — generating synthetic CICIDS-like data …")
        data = generate_demo_data(n_samples=60_000)
    else:
        data = load_dataset()

    # ── [2] Preprocess ────────────────────────────────────────────────────
    _banner("[2] Preprocessing")
    X_train, X_test, y_train, y_test, le, feature_names, class_names = preprocess(data)
    n_classes = len(class_names)

    # Class distribution plot
    plot_class_distribution(y_train, class_names)

    # ── [3] Feature engineering ───────────────────────────────────────────
    _banner("[3] Feature Engineering")
    fe = FeatureEngineer()
    X_train_fe, y_train_fe = fe.fit_transform(X_train, y_train, feature_names)
    X_test_fe              = fe.transform(X_test)
    input_dim              = X_train_fe.shape[1]

    # Feature importance plot (from ExtraTrees selector)
    plot_feature_importance(fe.feature_importances, fe.selected_feature_names)

    # Save the feature engineer (needed for real-time detection)
    fe_path = os.path.join(config.MODELS_DIR, "feature_engineer.pkl")
    joblib.dump(fe, fe_path)
    print(f"\n  ✓ FeatureEngineer saved → {fe_path}")

    # ── [4] Validation split (for Deep Learning) ─────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_fe, y_train_fe,
        test_size    = config.VALIDATION_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y_train_fe,
    )

    # ── [5] Train & evaluate supervised models ────────────────────────────
    _banner("[4] Training & Evaluating Models")
    evaluator   = Evaluator(class_names)
    roc_data    = {}
    sup_models  = [m for m in args.models if m != "Autoencoder"]

    for model_name in sup_models:
        _banner(f"  {model_name}")
        try:
            model = build_model(model_name, input_dim, n_classes)
        except Exception as exc:
            print(f"  ⚠  Could not build {model_name}: {exc} — skipping")
            continue

        t0 = time.time()
        try:
            if model_name == "DeepLearning":
                model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train_fe, y_train_fe)
        except Exception as exc:
            print(f"  ✗  Training failed: {exc}")
            continue

        train_time = time.time() - t0
        y_pred     = model.predict(X_test_fe)
        y_proba    = model.predict_proba(X_test_fe)

        metrics = evaluator.evaluate(
            model_name, y_test, y_pred, y_proba, training_time=train_time
        )

        # Confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, class_names, model_name)

        # ROC data (binary only)
        if n_classes == 2:
            roc = evaluator.roc_data(model_name, y_test, y_proba)
            if roc:
                roc_data[model_name] = roc

        # Deep Learning training curves
        if model_name == "DeepLearning" and model.history:
            plot_dl_training_history(model.history)

        # Save model
        model.save(config.MODELS_DIR)

    # ── [6] Autoencoder (semi-supervised) ─────────────────────────────────
    if "Autoencoder" in args.models:
        _banner("  Autoencoder  (trained on BENIGN only)")
        try:
            X_benign   = X_train_fe[y_train_fe == 0]
            ae         = AutoencoderIDS(input_dim=input_dim)
            t0         = time.time()
            ae.fit(X_benign)
            train_time = time.time() - t0

            y_pred_ae  = ae.predict(X_test_fe)
            y_proba_ae = ae.predict_proba(X_test_fe)

            # Force binary evaluation for autoencoder
            y_test_bin = (y_test > 0).astype(int) if n_classes > 2 else y_test
            ev_ae = Evaluator(["BENIGN", "ATTACK"])
            metrics_ae = ev_ae.evaluate(
                "Autoencoder", y_test_bin, y_pred_ae, y_proba_ae,
                training_time=train_time,
            )
            evaluator.results["Autoencoder"] = ev_ae.results["Autoencoder"]
            evaluator.results["Autoencoder"]["accuracy"] = ev_ae.results["Autoencoder"]["accuracy"]

            cm_ae = np.array(metrics_ae["confusion_matrix"])
            plot_confusion_matrix(cm_ae, ["BENIGN", "ATTACK"], "Autoencoder")
            roc_ae = ev_ae.roc_data("Autoencoder", y_test_bin, y_proba_ae)
            if roc_ae:
                roc_data["Autoencoder"] = roc_ae

            if ae.history:
                plot_dl_training_history(ae.history, filename="06b_autoencoder_training.png")
            ae.save(config.MODELS_DIR)

        except Exception as exc:
            print(f"  ⚠  Autoencoder failed: {exc}")

    # ── [7] Comparison plots ──────────────────────────────────────────────
    _banner("[5] Generating Comparison Plots")
    comp_df = evaluator.comparison_dataframe()
    print("\n", comp_df.to_string())

    plot_model_comparison(comp_df)

    if roc_data:
        plot_roc_curves(roc_data)

    plot_per_class_f1_heatmap(evaluator.results, class_names)
    plot_per_attack_detection_rates(evaluator.results)

    # ── [8] Save results ──────────────────────────────────────────────────
    _banner("[6] Saving Results")
    evaluator.save(config.OUTPUT_DIR)
    comp_df.to_csv(os.path.join(config.OUTPUT_DIR, "model_comparison.csv"))
    print(f"\n  All outputs saved to: {os.path.abspath(config.OUTPUT_DIR)}/")

    # ── [9] Optional experiment ───────────────────────────────────────────
    if args.experiment and not args.demo:
        _banner("[7] Unseen Attack Detection Experiment")
        try:
            from experiments.unseen_attack_detection import run_experiment
            config.CLASSIFICATION_MODE = "multiclass"
            exp_data = load_dataset()
            run_experiment(exp_data)
        except Exception as exc:
            print(f" Experiment failed: {exc}")

    _banner("Pipeline Complete!")
    print(f" Plots   → {os.path.abspath(config.PLOTS_DIR)}/")
    print(f" Results → {os.path.abspath(config.OUTPUT_DIR)}/")
    print(f" Models  → {os.path.abspath(config.MODELS_DIR)}/\n")


if __name__ == "__main__":
    main()
