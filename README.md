[README.md](https://github.com/user-attachments/files/28996705/README.2.md)
# Prototype Anomaly-Based IDS (Intrusion Detection System)

**Author: Saiduzzaman Md.** | **Student ID: M25W7553** | **Email: st262556@m2.kcg.edu**

A machine learning pipeline for network intrusion detection using the CICIDS2017 dataset. This prototype implements binary classification (BENIGN vs. ATTACK) with multiple ensemble and classical ML models, automated feature engineering, and comprehensive evaluation metrics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)

---

## Overview

This project builds an **Anomaly-Based Intrusion Detection System (IDS)** that leverages machine learning to distinguish between normal network traffic (BENIGN) and malicious attacks. The pipeline is designed to be modular, reproducible, and production-ready, with automated preprocessing, feature selection, model training, and evaluation.

**Key Capabilities:**
- Binary classification with 99.8%+ accuracy
- Multi-model comparison (Random Forest, XGBoost, LightGBM, SVM)
- Automated feature engineering with variance, correlation, and importance filtering
- SMOTE-based class balancing for imbalanced datasets
- Comprehensive visualization and metric reporting

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Training** | Train and compare RandomForest, XGBoost, LightGBM, and SVM simultaneously |
| **Automated Preprocessing** | Duplicate removal, null handling, and feature scaling |
| **Feature Engineering** | 3-stage filtering: Variance Threshold → Correlation Analysis → ExtraTrees Importance Selection |
| **Class Balancing** | SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced attack data |
| **Performance Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision |
| **Visualization Suite** | Confusion matrices, ROC curves, feature importance plots, model comparison charts |
| **Model Persistence** | Save trained models, feature engineers, and evaluation results for deployment |

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset

**Download** — https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

The pipeline loads and merges 8 CSV files representing different days and attack scenarios:

| File | Rows | Description |
|------|------|-------------|
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | Normal traffic only |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | Normal + Brute Force attacks |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | Normal + DoS/DDoS attacks |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | Normal + Web attacks |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | Normal + Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | Normal + Botnet |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | Normal + PortScan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | Normal + DDoS |

**Total:** ~2.83 million rows | **Features:** 78 numeric | **Classes:** BENIGN (83.1%) vs. ATTACK (16.9%)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: CICIDS2017 CSVs                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [1] DATA LOADING & MERGING                                  │
│  • Load 8 CSV files → 2,830,743 total rows                  │
│  • Remove 308,381 duplicates                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [2] PREPROCESSING                                           │
│  • Handle missing values & infinities                        │
│  • Extract 78 numeric features                               │
│  • Train/Test split (80/20)                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [3] FEATURE ENGINEERING                                     │
│  • Variance Filter: 78 → 66 features                         │
│  • Correlation Filter: 66 → 48 features                      │
│  • ExtraTrees Top-30: 48 → 30 features                       │
│  • SMOTE: 2,017,889 → 3,354,374 samples (+1,336,485)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [4] MODEL TRAINING & EVALUATION                             │
│  • RandomForest  • XGBoost  • LightGBM  • SVM               │
│  • Cross-validation, confusion matrices, ROC curves         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  [5] OUTPUT GENERATION                                       │
│  • Saved models (.pkl)  • Evaluation JSON/CSV               │
│  • Plots: distributions, importance, ROC, confusion         │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/SaiduzzamanLisan1337/prototype_AIDS.git
cd prototype_AIDS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn
matplotlib, seaborn, joblib
```

---

## Usage

### Basic Usage (Binary Classification)

```bash
python main.py --mode binary --models RandomForest XGBoost LightGBM SVM
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Classification mode: `binary` or `multiclass` | `binary` |
| `--models` | Space-separated list of models to train | All models |
| `--smote` | Enable/disable SMOTE | `enabled` |
| `--sample` | Dataset sampling percentage (1-100) | `100` |

### Example Commands

```bash
# Train only XGBoost and LightGBM
python main.py --mode binary --models XGBoost LightGBM

# Quick test with 10% sample
python main.py --mode binary --models RandomForest --sample 10

# Multiclass mode (all attack types)
python main.py --mode multiclass --models XGBoost LightGBM
```

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|------------|
| **RandomForest** | **99.844%** | **99.382%** | 99.697% | **99.539%** | 99.995% | 414.9s |
| **XGBoost** | 99.820% | 99.064% | **99.878%** | 99.469% | **99.997%** | 36.5s |
| **LightGBM** | 99.802% | 98.959% | **99.878%** | 99.416% | **99.997%** | **18.9s** |
| SVM | 95.549% | 79.815% | 98.562% | 88.203% | 99.569% | 42.4s* |

\* SVM trained on subsampled data (80,000 rows) due to computational constraints

### Key Findings

- **Best Overall:** RandomForest achieves the highest F1-score (99.54%) but requires ~7× more training time than XGBoost
- **Best Efficiency:** LightGBM delivers 99.8% accuracy in just **18.9 seconds** — optimal for real-time deployment
- **Best Recall:** XGBoost and LightGBM both achieve 99.88% recall, minimizing false negatives (missed attacks)
- **SVM Limitations:** Significantly lower precision (79.8%) indicates high false-positive rate; subsampling required for scalability

### Top 5 Discriminative Features

1. **Destination Port** — Most critical for attack identification
2. **Flow Duration** — Attack flows often exhibit distinct timing patterns
3. **Total Fwd Packets** — Volume-based anomaly indicator
4. **Total Length of Fwd Packets** — Payload size signatures
5. **Fwd Packet Length Min** — Minimum packet size anomalies

---

## Project Structure

```
prototype_AIDS/
├── main.py                          # Entry point & CLI argument parsing
├── config.py                        # Configuration constants & paths
├── data_loader.py                   # CICIDS2017 dataset loading & merging
├── preprocessor.py                  # Data cleaning, scaling, train/test split
├── feature_engineer.py              # Feature selection pipeline (3-stage)
├── model_trainer.py                 # Model training & hyperparameter setup
├── evaluator.py                     # Metrics calculation & visualization
├── utils.py                         # Helper functions & logging
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/                            # CICIDS2017 CSV files (not included)
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── ...
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│
└── outputs/                         # Generated artifacts
    ├── plots/
    │   ├── 01_class_distribution.png
    │   ├── 02_feature_importance.png
    │   ├── 03_confusion_RandomForest.png
    │   ├── 03_confusion_XGBoost.png
    │   ├── 03_confusion_LightGBM.png
    │   ├── 03_confusion_SVM.png
    │   ├── 04_roc_curves.png
    │   ├── 05_model_comparison.png
    │   └── 07_per_class_f1_heatmap.png
    ├── saved_models/
    │   ├── feature_engineer.pkl
    │   ├── RandomForest.pkl
    │   ├── XGBoost.pkl
    │   ├── LightGBM.pkl
    │   └── SVM.pkl
    ├── evaluation_results.json
    └── model_comparison.csv
```

---

## Key Insights

### 1. Class Imbalance Handling
The dataset is highly imbalanced (83.1% BENIGN vs. 16.9% ATTACK). SMOTE successfully balanced the training set, generating 1.3M synthetic attack samples and improving model sensitivity to minority class attacks.

### 2. Feature Dimensionality Reduction
The 3-stage feature selection reduced dimensions from **78 → 30 features** (62% reduction) while maintaining >99.8% accuracy. This significantly reduces inference latency and model complexity.

### 3. Model Trade-offs
- **Production Recommendation:** **LightGBM** — Best balance of accuracy (99.8%) and speed (18.9s)
- **High-Security Environments:** **XGBoost** — Highest recall (99.88%) ensures minimal missed attacks
- **Research/Benchmarking:** **RandomForest** — Highest F1-score but computationally expensive

### 4. Scalability Considerations
- SVM requires subsampling for large datasets, making it unsuitable for high-volume network traffic
- Tree-based models (RF, XGB, LGBM) scale linearly with data size and handle high-dimensional features efficiently

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.8+ |
| **ML Libraries** | scikit-learn, XGBoost, LightGBM |
| **Data Processing** | pandas, NumPy |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | joblib, pickle |

---
