"""
visualization/visualizer.py
============================
Publication-quality plots for the Anomaly IDS pipeline.

Plots generated
---------------
1.  Class distribution (bar + pie)
2.  Feature importance (top-N horizontal bar)
3.  Confusion matrices per model (raw + normalised)
4.  ROC curves – all models on one axes
5.  Precision-Recall curves
6.  Model comparison grouped bar chart
7.  Deep Learning training history (loss + accuracy)
8.  Per-class F1 heatmap by model
9.  Autoencoder reconstruction error distribution
10. Unseen attack detection summary
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import config

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)
PALETTE = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#E65100", "#00695C"]


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊  Saved → {path}")
    return path


# ─── Plot 1 – Class distribution ─────────────────────────────────────────────

def plot_class_distribution(
    y: np.ndarray, class_names: List[str],
    filename: str = "01_class_distribution.png"
) -> str:
    unique, counts = np.unique(y, return_counts=True)
    labels  = [class_names[i] for i in unique]
    colours = PALETTE[:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax1.bar(labels, counts, color=colours, edgecolor="black", linewidth=0.6)
    ax1.set_title("Sample Count per Class", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=20)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{c:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, colors=colours,
        autopct="%1.1f%%", startangle=90, pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title("Class Proportion", fontsize=13, fontweight="bold")

    fig.suptitle("Dataset Class Distribution", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 2 – Feature importance ─────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    filename: str = "02_feature_importance.png",
) -> str:
    idx  = np.argsort(importances)[::-1][:top_n]
    imp  = importances[idx]
    nms  = [feature_names[i] for i in idx]
    cmap = sns.color_palette("Blues_r", top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(nms[::-1], imp[::-1], color=cmap, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, imp[::-1]):
        ax.text(val + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5)
    ax.set_xlabel("Importance Score (ExtraTrees)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 3 – Confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    filename = filename or f"03_confusion_{model_name}.png"
    cm_norm  = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n        = len(class_names)
    annot_kw = {"fontsize": max(6, 9 - n)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5 + n // 3))
    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Raw Counts"),
        (ax2, cm_norm, ".2%",  "Row-Normalised"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.4, linecolor="lightgray",
                    annot_kws=annot_kw, square=n < 6)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"{model_name} — {title}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 4 – ROC curves ────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data: Dict[str, Tuple],
    filename: str = "04_roc_curves.png",
) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                label=f"{name}  (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 5 – Model comparison bar chart ─────────────────────────────────────

def plot_model_comparison(
    comparison_df,
    filename: str = "05_model_comparison.png",
) -> str:
    metrics  = [c for c in ("Accuracy (%)", "Precision (%)", "Recall (%)",
                             "F1-Score (%)", "ROC-AUC (%)") if c in comparison_df.columns]
    n_models = len(comparison_df)
    x        = np.arange(len(metrics))
    width    = 0.80 / n_models
    offsets  = np.linspace(-(n_models - 1) / 2 * width,
                            (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (model, row) in enumerate(comparison_df.iterrows()):
        vals = [row.get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=PALETTE[i % len(PALETTE)], edgecolor="black",
                      linewidth=0.5, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (%)", "") for m in metrics], fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 6 – Deep Learning training history ─────────────────────────────────

def plot_dl_training_history(
    history,
    filename: str = "06_dl_training_history.png",
) -> str:
    hist   = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, ylabel in [
        (ax1, "loss",     "Loss",     "Loss"),
        (ax2, "accuracy", "Accuracy", "Accuracy"),
    ]:
        ax.plot(epochs, hist[key], "b-o", markersize=4, lw=1.8, label="Train")
        if f"val_{key}" in hist:
            ax.plot(epochs, hist[f"val_{key}"], "r-s", markersize=4, lw=1.8, label="Validation")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"MLP — {title}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.35)
        ax.set_xlim([0, len(epochs) + 1])

    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 7 – Per-class F1 heatmap ───────────────────────────────────────────

def plot_per_class_f1_heatmap(
    results: dict,
    class_names: List[str],
    filename: str = "07_per_class_f1_heatmap.png",
) -> Optional[str]:
    """Parse classification_report strings → heatmap (models × classes)."""
    model_f1s: Dict[str, Dict[str, float]] = {}
    for model_name, m in results.items():
        report = m.get("classification_report", "")
        f1s: Dict[str, float] = {}
        for cls in class_names:
            pat = rf"(?:^|\n)\s*{re.escape(cls)}\s+[\d.]+\s+[\d.]+\s+([\d.]+)"
            hit = re.search(pat, report)
            if hit:
                f1s[cls] = float(hit.group(1))
        if f1s:
            model_f1s[model_name] = f1s

    if not model_f1s:
        return None

    models  = list(model_f1s.keys())
    matrix  = np.array([[model_f1s[m].get(c, 0.0) for c in class_names] for m in models])

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.4),
                                     max(5, len(models) * 0.9)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=models,
                linewidths=0.4, linecolor="white", ax=ax,
                vmin=0, vmax=1, annot_kws={"size": 9})
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title("Per-Class F1-Score by Model", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 8 – Autoencoder reconstruction error distribution ──────────────────

def plot_reconstruction_error_dist(
    errors_benign: np.ndarray,
    errors_attack: np.ndarray,
    threshold: float,
    filename: str = "08_autoencoder_error_dist.png",
) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    # Cap extreme values for readability
    cap = np.percentile(np.concatenate([errors_benign, errors_attack]), 99)

    ax.hist(np.clip(errors_benign, 0, cap), bins=100,
            alpha=0.65, color="#1565C0", density=True, label="BENIGN")
    ax.hist(np.clip(errors_attack, 0, cap), bins=100,
            alpha=0.65, color="#C62828", density=True, label="ATTACK")
    ax.axvline(threshold, color="#E65100", lw=2.0, ls="--",
               label=f"Decision Threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Autoencoder: Reconstruction Error Distributions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    # Shade regions
    x_max = min(cap, ax.get_xlim()[1])
    ax.axvspan(threshold, x_max, alpha=0.08, color="#C62828", label="Attack zone")
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 9 – Unseen attack detection bar chart ───────────────────────────────

def plot_unseen_attack_results(
    results: Dict[str, Dict[str, float]],
    filename: str = "09_unseen_attack_detection.png",
) -> str:
    """
    results : {model_name: {attack_type: detection_rate_0_to_1}}
    """
    models  = list(results.keys())
    attacks = sorted({a for m in results.values() for a in m})
    matrix  = np.array([[results[m].get(a, 0.0) * 100 for a in attacks] for m in models])

    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 1.4),
                                    max(5, len(models) * 0.8 + 2)))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=attacks, yticklabels=models,
                vmin=0, vmax=100, linewidths=0.4, linecolor="white",
                annot_kws={"size": 9}, ax=ax)
    ax.set_xlabel("Unseen Attack Type", fontsize=12)
    ax.set_title(
        "Unseen Attack Detection Rate (%)  —  Models never trained on these attacks",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    plt.tight_layout()
    return _save(fig, filename)


# ─── Plot 10 – Multiclass per-attack detection bar ───────────────────────────

def plot_per_attack_detection_rates(
    all_model_results: dict,
    filename: str = "10_per_attack_detection_rates.png",
) -> Optional[str]:
    """
    Bar chart of per-class detection rate for each model (multiclass mode).
    """
    models_with_dr = {
        name: m["per_class_detection_rate"]
        for name, m in all_model_results.items()
        if "per_class_detection_rate" in m
    }
    if not models_with_dr:
        return None

    all_classes = sorted({c for m in models_with_dr.values() for c in m})
    n_models    = len(models_with_dr)
    x           = np.arange(len(all_classes))
    width       = 0.75 / n_models
    offsets     = np.linspace(-(n_models - 1) / 2 * width,
                               (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 1.2), 7))
    for i, (model, dr_dict) in enumerate(models_with_dr.items()):
        vals = [dr_dict.get(c, 0.0) * 100 for c in all_classes]
        ax.bar(x + offsets[i], vals, width, label=model,
               color=PALETTE[i % len(PALETTE)], edgecolor="black", linewidth=0.4, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Attack-Type Detection Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    return _save(fig, filename)
