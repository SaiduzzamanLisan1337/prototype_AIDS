[README.md](https://github.com/user-attachments/files/29409539/README.8.md)

## Student Information

* **Name:** Saiduzzaman Md.
* **Student ID:** M25W7553
* **Email:** [st262556@m2.kcg.edu](mailto:st262556@m2.kcg.edu)

# Comparative Analysis of Machine Learning Models for Network Anomaly Detection on CICIDS2017

**Keywords:** Intrusion Detection, Anomaly Detection, Network Security, Machine Learning, CICIDS2017, SMOTE, Random Forest, Gradient Boosting, Support Vector Machines, Deep Learning

---

## Abstract

Network intrusion detection systems (IDS) are broadly classified into signature-based and anomaly-based approaches. Anomaly-based systems learn statistical patterns of normal traffic and flag deviations, allowing detection of previously unseen attack variants — a capability signature-based systems lack. This work presents an end-to-end anomaly-based intrusion detection pipeline (AIDS) trained and evaluated on the CICIDS2017 dataset. The pipeline performs data ingestion, deduplication, multi-stage feature selection, class balancing via SMOTE, and comparative training of five classifiers: Random Forest, XGBoost, LightGBM, Support Vector Machine (SVM), and a fully connected Deep Neural Network (DNN). On a held-out test set of 504,473 flow records, gradient-boosted and ensemble tree methods (Random Forest, XGBoost, LightGBM) achieve accuracy above 99.8% and ROC-AUC above 99.99%, while SVM and the DNN trail substantially, at 91.68% and 88.90% accuracy respectively. These results are reported with full per-class precision, recall, and F1-score, and are discussed in terms of the accuracy–efficiency trade-off relevant to operational deployment.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Experimental Setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Limitations](#7-limitations)
8. [Conclusion and Future Work](#8-conclusion-and-future-work)
9. [References](#9-references)
- [Appendix A — Detailed Classification Reports](#appendix-a--detailed-classification-reports)
- [Appendix B — Deep Learning Training History](#appendix-b--deep-learning-training-history)

---

## 1. Introduction

Intrusion detection systems are a core component of network defense, tasked with distinguishing malicious traffic from benign activity in real time or in retrospective analysis. Anomaly-based detection frames this as a supervised binary (or multi-class) classification problem over flow-level statistical features — packet counts, byte counts, flow duration, inter-arrival times, and related descriptors — rather than relying on known attack signatures.

This project implements a binary anomaly-based IDS pipeline (`anomaly_ids/main.py`) and evaluates it against the CICIDS2017 benchmark, a widely used dataset for network intrusion research containing both benign traffic and a representative range of modern attack types (DoS/DDoS, port scanning, infiltration, and web-based attacks) [1]. The objective of this study is twofold: (i) to establish a reproducible preprocessing and feature-engineering pipeline suitable for high-volume flow data, and (ii) to provide a controlled, like-for-like comparison of five widely used classification paradigms — bagged trees, two gradient-boosting variants, a kernel method, and a neural network — under identical data conditions.

---

## 2. Dataset

### 2.1 Source Files

**Download Link of the Dataset:**
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

The pipeline ingests all eight daily capture files distributed with CICIDS2017:

| File | Rows |
|---|---|
| Monday-WorkingHours.pcap_ISCX.csv | 529,918 |
| Tuesday-WorkingHours.pcap_ISCX.csv | 445,909 |
| Wednesday-workingHours.pcap_ISCX.csv | 692,703 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 170,366 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 288,602 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 191,033 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 286,467 |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 225,745 |
| **Total** | **2,830,743** |

**Table 1.** Per-file and aggregate row counts as loaded by the pipeline.

### 2.2 Preprocessing Summary

| Quantity | Value |
|---|---|
| Total records loaded | 2,830,743 |
| Duplicate records removed | 308,381 |
| Final sample size | 2,522,362 |
| Numeric features (pre-selection) | 78 |

**Table 2.** Effect of deduplication on dataset size.

### 2.3 Class Distribution

Attack labels are collapsed into a binary scheme for this study:

| Class | Samples | Proportion |
|---|---|---|
| BENIGN | 2,096,484 | 83.1% |
| ATTACK | 425,878 | 16.9% |

**Table 3.** Binary class distribution after deduplication (visualized in `outputs/plots/01_class_distribution.png`).

The approximately 5:1 class ratio motivates the SMOTE balancing step described in Section 3.3.

### 2.4 Train–Test Partition

| Partition | Samples |
|---|---|
| Training set | 2,017,889 |
| Test set | 504,473 |

**Table 4.** Train–test split sizes (test set composition: 419,297 BENIGN, 85,176 ATTACK).

---

## 3. Methodology

### 3.1 Data Preprocessing

Raw records are deduplicated, validated for numeric consistency across 78 candidate features, and assigned a binary label (BENIGN / ATTACK). The cleaned dataset is then partitioned into training and test subsets prior to any feature selection or balancing step, to avoid information leakage from the test set into model development.

### 3.2 Feature Engineering

Feature dimensionality is reduced in three sequential stages:

| Stage | Features Remaining | Method |
|---|---|---|
| Original | 78 | — |
| Variance filtering | 66 | Removal of near-constant / zero-variance columns |
| Correlation filtering | 48 | Removal of highly correlated, redundant feature pairs |
| Extra Trees selection | 30 | Retention of the top-30 features by impurity-based feature importance |

**Table 5.** Sequential feature reduction pipeline (visualized in `outputs/plots/02_feature_importance.png`).

The five most influential retained features, ranked by Extra Trees importance, were: *Destination Port*, *Flow Duration*, *Total Fwd Packets*, *Total Length of Fwd Packets*, and *Fwd Packet Length Min*.

### 3.3 Class Balancing (SMOTE)

To mitigate the class imbalance reported in Table 3, the Synthetic Minority Over-sampling Technique (SMOTE) [2] is applied to the training partition only. SMOTE generates synthetic minority-class (ATTACK) instances via interpolation between each minority sample and its k-nearest minority neighbors in feature space.

| Quantity | Value |
|---|---|
| Training samples before SMOTE | 2,017,889 |
| Training samples after SMOTE | 3,354,374 |
| Synthetic samples generated | 1,336,485 |

**Table 6.** Effect of SMOTE on the training partition. The test set is left untouched to preserve a realistic evaluation distribution.

### 3.4 Candidate Models

Five classifiers, spanning bagging, boosting, margin-based, and neural paradigms, are trained on the balanced 30-feature representation:

| Model | Paradigm | Reference |
|---|---|---|
| Random Forest | Bagged decision trees | Breiman (2001) [3] |
| XGBoost | Gradient-boosted trees | Chen & Guestrin (2016) [4] |
| LightGBM | Histogram-based gradient boosting | Ke et al. (2017) [5] |
| Support Vector Machine | Margin-based kernel classifier | Cortes & Vapnik (1995) [6] |
| Deep Neural Network | Fully connected feed-forward network (50,689 parameters) | — |

**Table 7.** Candidate model paradigms.

---

## 4. Experimental Setup

The pipeline was executed with the following configuration:

| Parameter | Value |
|---|---|
| Classification mode | Binary (BENIGN vs. ATTACK) |
| Models trained | RandomForest, XGBoost, LightGBM, SVM, DeepLearning |
| SMOTE | Enabled |
| Dataset sample fraction | 100% |

**Table 8.** Pipeline run configuration.

All models were trained and evaluated on identical training/test partitions and the identical 30-feature representation, with the following model-specific exceptions:

- **SVM** was automatically subsampled to 80,000 training rows to maintain tractable training time on the full 30-dimensional feature space.
- **Deep Learning** was trained for a maximum of 30 epochs with a learning-rate reduction schedule and early stopping; execution was CPU-only, as no CUDA/cuDNN runtime libraries were detected in the execution environment.

---

## 5. Results

### 5.1 Comparative Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg. Precision | Training Time |
|---|---|---|---|---|---|---|---|
| Random Forest | 99.844% | 99.382% | 99.697% | 99.539% | 99.995% | 99.976% | 252.2 s |
| XGBoost | 99.820% | 99.064% | 99.878% | 99.469% | 99.997% | 99.984% | 31.2 s |
| LightGBM | 99.802% | 98.959% | 99.878% | 99.416% | 99.997% | 99.984% | 17.6 s |
| SVM | 91.679% | 67.962% | 95.952% | 79.567% | 97.497% | 89.746% | 7.3 s |
| Deep Learning | 88.901% | 64.304% | 77.019% | 70.089% | 93.526% | 81.228% | 351.5 s |

**Table 9.** Test-set performance across all five models (n = 504,473; 419,297 BENIGN / 85,176 ATTACK). Per-model confusion matrices are saved to `outputs/plots/03_confusion_<Model>.png`; ROC curves to `outputs/plots/04_roc_curves.png`; the cross-model comparison chart to `outputs/plots/05_model_comparison.png`; and a per-class F1 heatmap to `outputs/plots/07_per_class_f1_heatmap.png`.

> **Note on timing figures.** Training times above are taken from the per-model fit timer reported during pipeline execution. The pipeline's separately generated summary table (`outputs/model_comparison.csv`) reports marginally higher values for LightGBM (18.30 s) and Deep Learning (353.98 s); this small discrepancy is most likely attributable to the latter measurement including model serialization and scoring overhead in addition to the raw `fit()` call.

### 5.2 Deep Learning Training Dynamics

The DNN converged on a training accuracy of 94.22% by epoch 9, at which point early stopping halted training (configured maximum: 30 epochs). Validation loss reached its minimum (0.3476) at epoch 4 and fluctuated non-monotonically thereafter, indicating the onset of overfitting or optimization instability beyond that point. Full per-epoch metrics are provided in Appendix B and visualized in `outputs/plots/06_dl_training_history.png`.

## 5.3 Outputs

Running the pipeline produces the following artifacts under `anomaly_ids/outputs/`:

```
outputs/
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_feature_importance.png
│   ├── 03_confusion_RandomForest.png
│   ├── 03_confusion_XGBoost.png
│   ├── 03_confusion_LightGBM.png
│   ├── 03_confusion_SVM.png
│   ├── 03_confusion_DeepLearning.png
│   ├── 04_roc_curves.png
│   ├── 05_model_comparison.png
│   ├── 06_dl_training_history.png
│   └── 07_per_class_f1_heatmap.png
├── saved_models/
│   ├── feature_engineer.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── SVM.pkl
│   └── DeepLearning.keras
├── evaluation_results.json
└── model_comparison.csv
```

---

## 5.4 Project Structure

```
prototype_AIDS/
├── .venv/                  # Python virtual environment
└── anomaly_ids/
    ├── main.py              # Pipeline entry point
    └── outputs/             # Generated plots, models, and metrics (see above)
```

---

## 5.5 Usage

```bash
# from the project root, with the virtual environment activated
python anomaly_ids/main.py
```

This runs the full pipeline end-to-end:

1. **Load Data** — read and concatenate all 8 CICIDS2017 CSV files
2. **Preprocess** — drop duplicates, clean numeric features, binarize labels, train/test split
3. **Feature Engineering** — variance filter → correlation filter → Extra Trees top-30 selection → SMOTE
4. **Train & Evaluate** — fit RandomForest, XGBoost, LightGBM, SVM, and a Deep Learning MLP; score each on the test set
5. **Compare** — generate ROC curves, confusion matrices, and a cross-model comparison table/plot
6. **Save Results** — persist trained models, plots, and metrics (JSON + CSV) to `outputs/`

---

## 5.6 Requirements

- Python 3.x
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn (SMOTE)
- tensorflow / keras
- pandas, numpy, matplotlib (for data handling and plotting)

---

## 6. Discussion

The results in Table 9 indicate a clear stratification of model performance by paradigm rather than by computational cost. The three tree-based ensembles — Random Forest, XGBoost, and LightGBM — form a tightly clustered top tier, each exceeding 99.8% accuracy and 99.99% ROC-AUC, with F1-scores within 0.12 percentage points of one another. Within this tier, LightGBM offers the most favorable accuracy-to-training-time ratio, completing training in 17.6 seconds — roughly an order of magnitude faster than Random Forest (252.2 s) — while sacrificing only 0.123 percentage points of F1-score.

SVM and the Deep Learning model form a clearly separated lower tier. SVM attains comparatively high recall (95.952%) but its precision (67.962%) implies that approximately one in three flow records flagged as ATTACK is in fact benign — a false-positive rate that would generate substantial analyst workload in an operational security-monitoring context. The Deep Learning model underperforms all other candidates on every reported metric; combined with the validation-loss instability noted in Section 5.2, this suggests that the current fully connected architecture, training configuration, or input representation is not yet well matched to this 30-feature tabular dataset, in contrast to the inherent feature-interaction modeling that tree-based splits provide on this same representation.

Critically, the data show no evidence of a strict accuracy–cost trade-off in this experiment: the two cheapest tree-based models to train (LightGBM, XGBoost) are simultaneously the two highest-accuracy models among the ensembles, while the most expensive model to train (Deep Learning, 351.5 s) is the lowest-performing of all five. This implies that, for this dataset and feature representation, additional computational investment in the Deep Learning model would need to be paired with architectural or representational changes — rather than simply additional training time — to be competitive with the tree-based methods.

---

## 7. Limitations

- The present study addresses **binary** classification only (BENIGN vs. ATTACK); attack-type granularity (e.g., DDoS vs. PortScan vs. Infiltration) is not evaluated.
- Evaluation is performed **offline** on static CSV exports; no live packet capture, streaming inference, or concept-drift evaluation is included.
- No **cross-validation** or systematic **hyperparameter optimization** was performed for any model; reported metrics reflect a single train/test split and a single hyperparameter configuration per model.
- The SVM classifier was trained on an **80,000-row subsample** rather than the full training partition, which may understate its achievable performance relative to the other models.
- The Deep Learning model's instability (Section 5.2) was not further diagnosed (e.g., via learning-rate search, regularization, or architecture variation) within the scope of this study.

---

## 8. Conclusion and Future Work

This study demonstrates that gradient-boosted and bagged tree ensembles substantially outperform both a kernel-based method and a fully connected neural network on a 30-feature, SMOTE-balanced representation of the CICIDS2017 dataset, achieving accuracy and ROC-AUC in excess of 99.8% and 99.99% respectively. LightGBM in particular offers a favorable balance of predictive performance and computational cost.

Future work should extend this pipeline toward: (i) **multi-class classification** to distinguish specific attack categories; (ii) **systematic hyperparameter tuning and k-fold cross-validation** to obtain more robust performance estimates; (iii) **architectural revision of the Deep Learning model**, potentially including regularization, alternative activation/normalization schemes, or sequence-aware architectures better suited to flow data; (iv) **explainability analysis** (e.g., SHAP or LIME) to characterize feature attributions per prediction; and (v) **real-time or streaming deployment**, moving beyond the current offline, CSV-based evaluation paradigm.

---

## 9. References

[1] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. 2018. Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018). SCITEPRESS, 108–116. https://doi.org/10.5220/0006639801080116

[2] Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. 2002. SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research 16 (2002), 321–357. https://doi.org/10.1613/jair.953

[3] Leo Breiman. 2001. Random Forests. Machine Learning 45, 1 (2001), 5–32. https://doi.org/10.1023/A:1010933404324

[4] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 785–794. https://doi.org/10.1145/2939672.2939785

[5] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017). Curran Associates, Inc., 3146–3154.

[6] Corinna Cortes and Vladimir Vapnik. 1995. Support-Vector Networks. Machine Learning 20, 3 (1995), 273–297. https://doi.org/10.1007/BF00994018

---

## Appendix A — Detailed Classification Reports

**Random Forest**
```
              precision    recall  f1-score   support

      BENIGN       1.00      1.00      1.00    419297
      ATTACK       0.99      1.00      1.00     85176

    accuracy                           1.00    504473
   macro avg       1.00      1.00      1.00    504473
weighted avg       1.00      1.00      1.00    504473
```

**XGBoost**
```
              precision    recall  f1-score   support

      BENIGN       1.00      1.00      1.00    419297
      ATTACK       0.99      1.00      0.99     85176

    accuracy                           1.00    504473
   macro avg       1.00      1.00      1.00    504473
weighted avg       1.00      1.00      1.00    504473
```

**LightGBM**
```
              precision    recall  f1-score   support

      BENIGN       1.00      1.00      1.00    419297
      ATTACK       0.99      1.00      0.99     85176

    accuracy                           1.00    504473
   macro avg       0.99      1.00      1.00    504473
weighted avg       1.00      1.00      1.00    504473
```

**SVM**
```
              precision    recall  f1-score   support

      BENIGN       0.99      0.91      0.95    419297
      ATTACK       0.68      0.96      0.80     85176

    accuracy                           0.92    504473
   macro avg       0.84      0.93      0.87    504473
weighted avg       0.94      0.92      0.92    504473
```

**Deep Learning**
```
              precision    recall  f1-score   support

      BENIGN       0.95      0.91      0.93    419297
      ATTACK       0.64      0.77      0.70     85176

    accuracy                           0.89    504473
   macro avg       0.80      0.84      0.82    504473
weighted avg       0.90      0.89      0.89    504473
```

---

## Appendix B — Deep Learning Training History

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss | Learning Rate |
|---|---|---|---|---|---|
| 1 | 0.9218 | 0.1733 | 0.7066 | 0.6122 | 1.0 × 10⁻³ |
| 2 | 0.9338 | 0.1522 | 0.8049 | 0.6695 | 1.0 × 10⁻³ |
| 3 | 0.9365 | 0.1494 | 0.8152 | 0.5203 | 1.0 × 10⁻³ |
| 4 | 0.9371 | 0.1484 | 0.8401 | **0.3476 (minimum)** | 1.0 × 10⁻³ |
| 5 | 0.9376 | 0.1480 | 0.8048 | 0.3932 | 1.0 × 10⁻³ |
| 6 | 0.9383 | 0.1477 | 0.7950 | 0.7011 | 1.0 × 10⁻³ |
| 7 | 0.9391 | 0.1466 | 0.8024 | 0.4781 | 1.0 × 10⁻³ |
| 8 | 0.9417 | 0.1404 | 0.7929 | 0.7267 | 5.0 × 10⁻⁴ |
| 9 | 0.9422 | 0.1387 | 0.8013 | 0.7536 | 5.0 × 10⁻⁴ |

**Table B.1.** Per-epoch training metrics for the Deep Learning model. The learning rate was reduced from 1.0 × 10⁻³ to 5.0 × 10⁻⁴ following epoch 7, consistent with a plateau-triggered learning-rate schedule. Training was halted after epoch 9 of a configured maximum of 30, consistent with an early-stopping criterion that did not observe further improvement in validation loss beyond epoch 4. Model parameters: 50,689. Hardware: CPU-only execution (no CUDA/cuDNN devices detected in the runtime environment).
