"""
models/svm_model.py
Support Vector Machine — LinearSVC for tractability on large IDS datasets.
Automatically subsamples when data exceeds SVM_MAX_TRAIN_ROWS.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import config
from models.base_model import BaseIDSModel


class SVMIDS(BaseIDSModel):
    name = "SVM"

    def __init__(self):
        # LinearSVC scales linearly; CalibratedClassifierCV adds probability support
        params = dict(getattr(config, "SVM_PARAMS", {}))
        params.pop("kernel", None)      # LinearSVC is always linear
        params.pop("gamma", None)       # Not used by linear kernel
        params.pop("probability", None) # Handled by CalibratedClassifierCV
        params.setdefault("dual", "auto")
        params.setdefault("random_state", getattr(config, "RANDOM_STATE", 42))

        base_svc = LinearSVC(**params)
        self.model = CalibratedClassifierCV(base_svc, ensemble=False, method="sigmoid", cv=3)
        self.scaler = StandardScaler()
        self.train_time: float = 0.0
        self._subsampled = False
        self.feature_names: list[str] | None = None

    def fit(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray, **kwargs) -> None:
        t0 = time.time()
        # ── Auto-subsample to keep training tractable ──────────────────
        max_rows = config.SVM_MAX_TRAIN_ROWS
        n_samples = len(X_train)
        if n_samples > max_rows:
            rng = np.random.default_rng(config.RANDOM_STATE)
            idx = rng.choice(n_samples, max_rows, replace=False)
            X_train = X_train[idx] if isinstance(X_train, np.ndarray) else X_train.iloc[idx]
            y_train = y_train[idx]
            self._subsampled = True
            print(f"  ↳ Fitting {self.name}  "
                  f"(auto-subsampled to {max_rows:,} rows for speed)…")
        else:
            print(f"  ↳ Fitting {self.name}…")

        # Store feature names & convert to DataFrame if needed
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_fit = X_train
        else:
            self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            X_fit = pd.DataFrame(X_train, columns=self.feature_names)

        # Scale features — essential for SVM convergence
        X_scaled = self.scaler.fit_transform(X_fit)

        # Fit quietly (no convergence spam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.model.fit(X_scaled, y_train)

        self.train_time = time.time() - t0
        print(f"    Done in {self.train_time:.1f}s")

    def _to_dataframe(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(self._to_dataframe(X))
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(self._to_dataframe(X))
        return self.model.predict_proba(X_scaled)