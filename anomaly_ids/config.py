"""
config.py – Central configuration for the Anomaly IDS Pipeline.
All paths, hyperparameters, and toggles live here.
"""
import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"                       # Folder containing CICIDS CSVs
OUTPUT_DIR  = "outputs"                        # Root for all outputs
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "saved_models")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
LABEL_COLUMN        = "Label"     # Column name in CICIDS CSVs (stripped)
BENIGN_LABEL        = "BENIGN"    # The label for normal traffic
CLASSIFICATION_MODE = "binary"    # "binary"  →  BENIGN vs ATTACK
                                  # "multiclass" → per-attack-type labels
SAMPLE_FRACTION     = 1.0         # 1.0 = full dataset; 0.3 = fast 30% run
RANDOM_STATE        = 42

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
VARIANCE_THRESHOLD    = 0.01   # Drop near-zero variance columns
CORRELATION_THRESHOLD = 0.98   # Drop highly correlated duplicate columns
TOP_N_FEATURES        = 30     # Keep top-N by ExtraTrees importance
APPLY_SMOTE           = True   # Oversample minority classes (requires imbalanced-learn)

# ─────────────────────────────────────────────────────────────────────────────
# Train / Test / Validation Split
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20   # 20% held out for final evaluation
VALIDATION_SIZE = 0.10   # 10% of training → validation (for Deep Learning)

# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 20,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    n_jobs          = -1,
    random_state    = RANDOM_STATE,
    class_weight    = "balanced",
)

XGB_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 7,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    eval_metric       = "logloss",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    tree_method       = "hist",      # fast histogram method
)

LGBM_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 7,
    learning_rate    = 0.1,
    num_leaves       = 63,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    class_weight     = "balanced",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbose          = -1,
)

SVM_PARAMS = dict(
    kernel       = "rbf",
    C            = 10.0,
    gamma        = "scale",
    probability  = True,
    class_weight = "balanced",
    max_iter     = 5000,
)

DL_PARAMS = dict(
    epochs        = 30,
    batch_size    = 512,
    learning_rate = 0.001,
    dropout_rate  = 0.30,
    patience      = 5,       # EarlyStopping patience
    l2_reg        = 1e-4,
)

# ─────────────────────────────────────────────────────────────────────────────
# Which models to run  (can be overridden via CLI)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_TO_RUN = ["RandomForest", "XGBoost", "LightGBM", "SVM", "DeepLearning"]

# Maximum rows fed to SVM (it's quadratic – subsample automatically)
SVM_MAX_TRAIN_ROWS = 80_000
