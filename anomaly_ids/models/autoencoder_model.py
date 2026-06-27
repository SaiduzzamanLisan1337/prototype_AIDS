"""
models/autoencoder_model.py
============================
Autoencoder-based Anomaly Detector.

Key idea
--------
Train an encoder-decoder network ONLY on BENIGN traffic.
The network learns to reconstruct normal patterns well.
Anomalies (attacks) produce HIGH reconstruction error because they
deviate from learnt normal distributions.

This gives the system "zero-label" attack detection capability —
it can flag attack types it has NEVER seen during training.

Architecture
------------
  Input(n)
    → Dense(128, relu) → BN → Dropout
    → Dense(64,  relu) → BN            ← encoder bottleneck
    → Dense(32,  relu)                 ← latent space
    → Dense(64,  relu) → BN
    → Dense(128, relu) → BN → Dropout
    → Dense(n,   linear)               ← reconstruction
  Loss: MSE
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import config


class AutoencoderIDS:
    """
    Semi-supervised anomaly detector.

    Usage
    -----
    ae = AutoencoderIDS(input_dim=30)
    ae.fit(X_train_benign)          # train on benign only
    labels = ae.predict(X_test)     # 0=BENIGN, 1=ATTACK
    scores = ae.anomaly_scores(X_test)   # continuous [0,1]
    """
    name = "Autoencoder"

    def __init__(
        self,
        input_dim: int,
        threshold_percentile: float = 95.0,
        encoding_dim: int = 32,
    ):
        self.input_dim            = input_dim
        self.threshold_percentile = threshold_percentile
        self.encoding_dim         = encoding_dim
        self.threshold_           = None
        self.model                = None
        self.history              = None
        self.train_time: float    = 0.0

    # ── Architecture ──────────────────────────────────────────────────────
    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers

        drop = config.DL_PARAMS["dropout_rate"]
        n    = self.input_dim
        enc  = self.encoding_dim

        inp = tf.keras.Input(shape=(n,), name="input")
        # Encoder
        x = layers.Dense(128, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(enc, activation="relu", name="latent")(x)   # bottleneck
        # Decoder
        x = layers.Dense(64, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
        out = layers.Dense(n, activation="linear", name="reconstruction")(x)

        ae = tf.keras.Model(inp, out, name="autoencoder")
        ae.compile(
            optimizer=tf.keras.optimizers.Adam(config.DL_PARAMS["learning_rate"]),
            loss="mse",
        )
        return ae

    # ── Training ──────────────────────────────────────────────────────────
    def fit(self, X_benign: np.ndarray, **kwargs) -> None:
        """
        Train the autoencoder exclusively on BENIGN samples.

        Parameters
        ----------
        X_benign : Feature array of benign-only traffic
        """
        import tensorflow as tf

        self.model = self._build()
        print(f"\n  ↳ Training {self.name} "
              f"on {len(X_benign):,} BENIGN samples only…")

        # Small internal validation split (10%)
        from sklearn.model_selection import train_test_split
        X_tr, X_val = train_test_split(
            X_benign, test_size=0.10, random_state=config.RANDOM_STATE
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.DL_PARAMS["patience"],
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0,
            ),
        ]

        t0 = time.time()
        self.history = self.model.fit(
            X_tr, X_tr,             # reconstruct the input
            validation_data=(X_val, X_val),
            epochs=config.DL_PARAMS["epochs"],
            batch_size=config.DL_PARAMS["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        self.train_time = time.time() - t0
        epochs_run = len(self.history.history["loss"])
        print(f"    Done in {self.train_time:.1f}s  ({epochs_run} epochs)")

        # ── Compute anomaly threshold from training reconstruction errors ──
        errors = self._mse(X_benign)
        self.threshold_ = np.percentile(errors, self.threshold_percentile)
        print(f"    Anomaly threshold  : {self.threshold_:.6f}  "
              f"(p{self.threshold_percentile:.0f} of training errors)")

    # ── Inference ─────────────────────────────────────────────────────────
    def _mse(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample mean squared reconstruction error."""
        X_rec = self.model.predict(X, verbose=0)
        return np.mean((X - X_rec) ** 2, axis=1)

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Raw reconstruction MSE for each sample."""
        return self._mse(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Normalised anomaly score in [0, 1].
        Values near 1 → likely attack; near 0 → likely benign.
        """
        errors = self._mse(X)
        # Clip to [0, 3 × threshold] then scale to [0, 1]
        max_val = max(self.threshold_ * 3, errors.max())
        return np.clip(errors / max_val, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0 (BENIGN) or 1 (ATTACK) based on reconstruction error."""
        assert self.threshold_ is not None, "Call fit() first."
        return (self._mse(X) > self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [P(benign), P(attack)] probability matrix using anomaly score."""
        attack_prob = self.anomaly_scores(X)
        return np.column_stack([1 - attack_prob, attack_prob])

    # ── Threshold tuning ──────────────────────────────────────────────────
    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                       metric: str = "f1") -> float:
        """
        Search percentile thresholds [50–99] to maximise *metric* on a
        labelled validation set.  Useful when you have some labelled data.

        Returns the best threshold found.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        errors    = self._mse(X_val)
        best_thr  = self.threshold_
        best_val  = 0.0
        fn        = {"f1": f1_score, "precision": precision_score,
                     "recall": recall_score}[metric]

        for p in range(50, 100):
            thr  = np.percentile(errors, p)
            pred = (errors > thr).astype(int)
            score = fn(y_val, pred, zero_division=0)
            if score > best_val:
                best_val, best_thr = score, thr

        self.threshold_ = best_thr
        print(f"    Tuned threshold → {best_thr:.6f}  ({metric}={best_val:.4f})")
        return best_thr

    # ── Save / Load ───────────────────────────────────────────────────────
    def save(self, directory: str) -> str:
        import joblib
        os.makedirs(directory, exist_ok=True)
        keras_path = os.path.join(directory, "Autoencoder.keras")
        self.model.save(keras_path)
        meta = {
            "input_dim": self.input_dim,
            "threshold_percentile": self.threshold_percentile,
            "encoding_dim": self.encoding_dim,
            "threshold_": self.threshold_,
            "train_time": self.train_time,
        }
        joblib.dump(meta, os.path.join(directory, "Autoencoder_meta.pkl"))
        print(f"  Model saved → {keras_path}")
        return keras_path

    @classmethod
    def load(cls, directory: str) -> "AutoencoderIDS":
        import joblib, tensorflow as tf
        meta = joblib.load(os.path.join(directory, "Autoencoder_meta.pkl"))
        ae   = cls(input_dim=meta["input_dim"],
                   threshold_percentile=meta["threshold_percentile"],
                   encoding_dim=meta["encoding_dim"])
        ae.threshold_ = meta["threshold_"]
        ae.train_time = meta["train_time"]
        ae.model = tf.keras.models.load_model(
            os.path.join(directory, "Autoencoder.keras")
        )
        return ae
