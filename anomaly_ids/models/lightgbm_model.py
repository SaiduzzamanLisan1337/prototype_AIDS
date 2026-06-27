"""
models/lightgbm_model.py
LightGBM — fastest gradient boosting, excellent on CICIDS imbalanced data.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import config
from models.base_model import BaseIDSModel


class LightGBMIDS(BaseIDSModel):
   name = "LightGBM"

   def __init__(self):
       import lightgbm as lgb
       self.model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
       self.train_time: float = 0.0
       self.feature_names: list[str] | None = None

   def fit(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray, **kwargs) -> None:
       # Store feature names if available, otherwise generate defaults
       if isinstance(X_train, pd.DataFrame):
           self.feature_names = list(X_train.columns)
           X_fit = X_train
       else:
           self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]
           X_fit = pd.DataFrame(X_train, columns=self.feature_names)

       t0 = time.time()
       print(f"  ↳ Fitting {self.name}…")
       self.model.fit(X_fit, y_train)
       self.train_time = time.time() - t0
       print(f"    Done in {self.train_time:.1f}s")

   def _to_dataframe(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
       """Ensure X is a DataFrame with matching feature names."""
       if isinstance(X, pd.DataFrame):
           return X
       return pd.DataFrame(X, columns=self.feature_names)

   def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
       return self.model.predict(self._to_dataframe(X))

   def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
       return self.model.predict_proba(self._to_dataframe(X))