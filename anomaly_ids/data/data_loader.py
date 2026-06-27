"""
data/data_loader.py
====================
Loads and preprocesses the CICIDS 2017 dataset.

Key CICIDS quirks handled:
  - Column names often have leading/trailing whitespace
  - Flow Bytes/s and Flow Packets/s can be +/-Inf  
  - Sparse NaN values exist in several columns
  - Label column may appear as ' Label' (with space)
  - Duplicate rows are common
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names (very common in CICIDS CSVs)."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _locate_label_column(df: pd.DataFrame) -> str:
    """Find the label column regardless of surrounding whitespace."""
    # Direct match first
    if config.LABEL_COLUMN in df.columns:
        return config.LABEL_COLUMN
    # Fuzzy match (after stripping)
    for col in df.columns:
        if col.strip().lower() == config.LABEL_COLUMN.lower():
            df.rename(columns={col: config.LABEL_COLUMN}, inplace=True)
            return config.LABEL_COLUMN
    raise KeyError(
        f"Label column '{config.LABEL_COLUMN}' not found.\n"
        f"Available columns (first 10): {df.columns[:10].tolist()}"
    )


def _handle_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ±Inf with NaN, then median-impute numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    return df


def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise label strings: strip, title-case BENIGN, unify Web Attack variants."""
    lbl = config.LABEL_COLUMN
    df[lbl] = df[lbl].astype(str).str.strip()
    # Normalise BENIGN to uppercase
    df[lbl] = df[lbl].str.replace(r"(?i)^benign$", "BENIGN", regex=True)
    # Collapse Web Attack sub-types for cleaner multiclass labels
    df[lbl] = df[lbl].str.replace(r"Web Attack.*", "Web Attack", regex=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = None, sample_fraction: float = None) -> pd.DataFrame:
    """
    Recursively find and load all CSV files from *data_dir*.

    Parameters
    ----------
    data_dir        : Path to folder with CICIDS CSVs.
    sample_fraction : Fraction of rows to keep (1.0 = all).

    Returns
    -------
    Cleaned, combined DataFrame ready for preprocessing.
    """
    data_dir        = "MachineLearningCSV"
    sample_fraction = sample_fraction if sample_fraction is not None else config.SAMPLE_FRACTION

    csv_files = sorted(
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        + glob.glob(os.path.join(data_dir, "*.csv"))
    )

    if not csv_files:
        raise FileNotFoundError(
            f"\n[!] No CSV files found in '{data_dir}'.\n"
            "    Please place the CICIDS CSV files there and update DATA_DIR in config.py.\n"
            "    Tip: run with --demo flag to test with synthetic data."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        print(f"  ↳ Loading: {os.path.basename(path)}", end=" … ")
        try:
            df = pd.read_csv(path, low_memory=False)
            df = _strip_column_names(df)
            print(f"{len(df):,} rows")
            frames.append(df)
        except Exception as exc:
            print(f"SKIPPED ({exc})")

    if not frames:
        raise RuntimeError("All CSV files failed to load.")

    data = pd.concat(frames, ignore_index=True)
    print(f"\n  ✓ Total rows loaded : {len(data):,}")

    # ── Clean ─────────────────────────────────────────────────────────────
    _locate_label_column(data)       # ensure label column exists
    data = _handle_inf_nan(data)
    data = _clean_labels(data)

    # ── Optional subsample ────────────────────────────────────────────────
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=config.RANDOM_STATE)
        data = data.reset_index(drop=True)
        print(f"  ↳ Sampled to       : {len(data):,} rows ({sample_fraction*100:.0f}%)")

    return data


def preprocess(data: pd.DataFrame) -> Tuple:
    """
    Split features / label, encode targets, stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    label_encoder                     : fitted LabelEncoder (None in binary mode)
    feature_names                     : List[str]
    class_names                       : List[str]
    """
    lbl = config.LABEL_COLUMN
    print("\n[Preprocessing]")

    # ── Remove duplicates ─────────────────────────────────────────────────
    n_before = len(data)
    data = data.drop_duplicates()
    print(f"  Duplicates removed  : {n_before - len(data):,}")

    # ── Separate X and y ─────────────────────────────────────────────────
    y_raw = data[lbl].copy()
    X_raw = data.drop(columns=[lbl])

    # Keep only numeric columns (drop timestamps, IPs, etc.)
    X = X_raw.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    print(f"  Numeric features    : {len(feature_names)}")

    # ── Encode labels ─────────────────────────────────────────────────────
    le: LabelEncoder | None = None

    if config.CLASSIFICATION_MODE == "binary":
        y_enc = (y_raw != config.BENIGN_LABEL).astype(int).values
        class_names = [config.BENIGN_LABEL, "ATTACK"]
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        class_names = list(le.classes_)

    # ── Print class distribution ───────────────────────────────────────────
    print(f"\n  {'Class':<30} {'Count':>10}  {'%':>6}")
    print(f"  {'-'*50}")
    unique, counts = np.unique(y_enc, return_counts=True)
    for u, c in zip(unique, counts):
        name = class_names[u]
        print(f"  {name:<30} {c:>10,}  {c/len(y_enc)*100:>5.1f}%")

    # ── Stratified split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc,
        test_size    = config.TEST_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y_enc,
    )
    print(f"\n  Train samples       : {len(X_train):,}")
    print(f"  Test  samples       : {len(X_test):,}")

    return X_train, X_test, y_train, y_test, le, feature_names, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode: synthetic CICIDS-like data
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_data(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Generate synthetic network flow data that mimics CICIDS structure.
    Useful for testing the pipeline without the real dataset.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_features = 78
    feature_cols = [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Total Length of Fwd Packets",
        "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
        "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
        "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
        "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
        "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ]
    # Pad to n_features if needed
    while len(feature_cols) < n_features:
        feature_cols.append(f"Feature_{len(feature_cols)}")
    feature_cols = feature_cols[:n_features]

    # Benign traffic (70%)
    n_benign = int(n_samples * 0.70)
    X_benign = rng.exponential(scale=1000, size=(n_benign, n_features)).astype(np.float32)

    # Attack traffic (30%) – different distribution
    n_attack = n_samples - n_benign
    X_attack = rng.exponential(scale=300, size=(n_attack, n_features)).astype(np.float32)
    X_attack += rng.normal(0, 50, size=X_attack.shape)

    X = np.vstack([X_benign, X_attack])
    y = np.array(["BENIGN"] * n_benign + ["DoS"] * (n_attack // 2) + ["DDoS"] * (n_attack - n_attack // 2))

    df = pd.DataFrame(X, columns=feature_cols)
    df["Label"] = y

    # Shuffle
    df = df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    print(f"  [Demo] Generated {len(df):,} synthetic samples.")
    return df
