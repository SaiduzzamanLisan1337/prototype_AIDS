"""
evaluation/evaluator.py
========================
Compute, store, and compare metrics for all IDS models.

Metrics
-------
Accuracy · Precision · Recall · F1-Score · ROC-AUC · Avg Precision
Confusion Matrix · Per-class classification report
Detection Rate per attack type (multiclass)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

import config


class Evaluator:
    """
    Evaluate and compare one or many IDS models.

    Usage
    -----
    ev = Evaluator(class_names=["BENIGN", "ATTACK"])
    metrics = ev.evaluate("RandomForest", y_test, y_pred, y_proba)
    df      = ev.comparison_dataframe()
    ev.save("outputs/")
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.n_classes   = len(class_names)
        self.results: Dict[str, dict] = {}

    # ── Single model evaluation ───────────────────────────────────────────
    def evaluate(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        training_time: float = 0.0,
    ) -> dict:
        """
        Compute all metrics for one model and store internally.

        Returns the metrics dict.
        """
        avg = "binary" if self.n_classes == 2 else "weighted"

        m: dict = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "training_time_s":  round(training_time, 2),
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.class_names, zero_division=0
            ),
        }

        # ── ROC-AUC / Average Precision ────────────────────────────────────
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    m["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    m["avg_precision"] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    m["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as exc:
                print(f"  ⚠  AUC skipped for {name}: {exc}")

        # ── Per-class detection rate (multiclass) ──────────────────────────
        if self.n_classes > 2:
            cm   = np.array(m["confusion_matrix"])
            with np.errstate(divide="ignore", invalid="ignore"):
                per_class_recall = np.where(
                    cm.sum(axis=1) > 0,
                    np.diag(cm) / cm.sum(axis=1),
                    0.0,
                )
            m["per_class_detection_rate"] = {
                cls: float(r)
                for cls, r in zip(self.class_names, per_class_recall)
            }

        self.results[name] = m
        self._print(name, m)
        return m

    # ── ROC data for plotting ─────────────────────────────────────────────
    def roc_data(self, name: str, y_true: np.ndarray,
                 y_proba: np.ndarray) -> Optional[Tuple]:
        """Return (fpr, tpr, auc_value) for binary classifiers."""
        if self.n_classes != 2 or y_proba is None:
            return None
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc_val     = auc(fpr, tpr)
            return fpr, tpr, auc_val
        except Exception:
            return None

    # ── Comparison table ──────────────────────────────────────────────────
    def comparison_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame comparing all evaluated models."""
        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model":         name,
                "Accuracy (%)":  round(m["accuracy"]  * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)":    round(m["recall"]    * 100, 2),
                "F1-Score (%)":  round(m["f1"]        * 100, 2),
                "ROC-AUC (%)":   round(m.get("roc_auc", float("nan")) * 100, 2),
                "Train Time (s)": m.get("training_time_s", 0),
            })
        return pd.DataFrame(rows).set_index("Model")

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, output_dir: str = None) -> None:
        output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # JSON (exclude classification_report for readability)
        serialisable = {}
        for name, m in self.results.items():
            serialisable[name] = {
                k: v for k, v in m.items() if k != "classification_report"
            }
        json_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        # CSV comparison table
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        self.comparison_dataframe().to_csv(csv_path)

        print(f"\n  ✓ Metrics saved  → {json_path}")
        print(f"  ✓ Comparison CSV → {csv_path}")

    # ── Private ───────────────────────────────────────────────────────────
    @staticmethod
    def _print(name: str, m: dict) -> None:
        bar = "─" * 55
        print(f"\n{bar}")
        print(f"  {name}")
        print(bar)
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"):
            if k in m:
                print(f"  {k.upper().replace('_', ' '):<20}: {m[k]*100:>7.3f}%")
        print(f"\n{m['classification_report']}")
        if "per_class_detection_rate" in m:
            print("  Per-class Detection Rate:")
            for cls, dr in m["per_class_detection_rate"].items():
                print(f"    {cls:<30}: {dr*100:>6.2f}%")
