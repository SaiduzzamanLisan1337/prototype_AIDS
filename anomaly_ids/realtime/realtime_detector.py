"""
realtime/realtime_detector.py
==============================
Real-time network flow anomaly detection.

Simulates a live stream by reading a CSV file in rolling chunks,
applying the full trained pipeline, and displaying a live dashboard.

Usage
-----
# First train and save models with main.py, then:
python realtime/realtime_detector.py \\
    --csv       data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \\
    --model     outputs/saved_models/LightGBM.pkl \\
    --fe        outputs/saved_models/feature_engineer.pkl \\
    --chunk     200 \\
    --delay     0.3

# Demo mode (no files needed):
python realtime/realtime_detector.py --demo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─── ANSI colour codes ────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


# ─── Dashboard ───────────────────────────────────────────────────────────────

class LiveDashboard:
    """
    Minimal terminal dashboard that refreshes in-place.
    Works without curses — just uses ANSI escape codes.
    """
    WINDOW = 500   # rolling window for rate calculation

    def __init__(self, model_name: str):
        self.model_name    = model_name
        self.total         = 0
        self.n_attack      = 0
        self.n_benign      = 0
        self.history       = deque(maxlen=self.WINDOW)  # (timestamp, is_attack)
        self.alert_log     = deque(maxlen=50)
        self._start        = time.time()
        self._alert_thresh = 0.30    # alert if >30% of last WINDOW flows are attacks

    def update(self, labels: np.ndarray, scores: np.ndarray) -> None:
        """Process a batch of predictions."""
        now = time.time()
        for lbl, sc in zip(labels, scores):
            self.total  += 1
            is_att       = bool(lbl)
            self.history.append((now, is_att))
            if is_att:
                self.n_attack += 1
                self.alert_log.append(
                    f"  {datetime.now():%H:%M:%S.%f}"[:-3]
                    + f"  score={sc:.3f}"
                )
            else:
                self.n_benign += 1

    @property
    def _recent_attack_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(is_att for _, is_att in self.history) / len(self.history)

    def render(self) -> None:
        elapsed = time.time() - self._start
        fps     = self.total / max(elapsed, 1)
        rate    = self._recent_attack_rate
        alert   = rate > self._alert_thresh

        # Move cursor up to overwrite previous render (20 lines)
        if self.total > 0:
            print("\033[20A\033[J", end="")

        sep = "─" * 60
        print(_col(f"\n{'═'*60}", CYAN))
        print(_col(f"  🛡   ANOMALY IDS  —  REAL-TIME MONITOR", BOLD + CYAN))
        print(_col(f"  Model: {self.model_name}", DIM))
        print(_col(sep, CYAN))

        print(f"  Flows processed  : {_col(f'{self.total:>8,}', BOLD)}")
        print(f"  BENIGN           : {_col(f'{self.n_benign:>8,}', GREEN)}")
        att_str = f"{self.n_attack:>8,}"
        print(f"  ATTACK           : {_col(att_str, RED if self.n_attack else GREEN)}")
        print(f"  Throughput       : {fps:>8.1f} flows/s")
        print(f"  Elapsed          : {elapsed:>8.1f} s")

        # ── Rolling attack rate ──────────────────────────────────────────
        bar_len  = 30
        filled   = int(rate * bar_len)
        bar_col  = RED if rate > self._alert_thresh else (YELLOW if rate > 0.1 else GREEN)
        bar_str  = "█" * filled + "░" * (bar_len - filled)
        print(f"\n  {_col(f'Attack rate (last {self.WINDOW})', BOLD)}")
        print(f"  [{_col(bar_str, bar_col)}]  {rate*100:5.1f}%")

        # ── Alert ────────────────────────────────────────────────────────
        if alert:
            print(f"\n  {_col('⚠  HIGH ATTACK RATE — POSSIBLE INCIDENT', RED + BOLD)}")
        else:
            print(f"\n  {_col('✓  Traffic appears normal', GREEN)}")

        # ── Recent alerts ────────────────────────────────────────────────
        print(f"\n  {_col('Recent Detections:', BOLD)}")
        recent = list(self.alert_log)[-5:]
        for entry in recent:
            print(f"  {_col('▶ ATTACK', RED)} {entry}")
        for _ in range(5 - len(recent)):
            print()

        print(_col(f"{'═'*60}", CYAN))
        sys.stdout.flush()


# ─── Detector ────────────────────────────────────────────────────────────────

class RealTimeDetector:
    """
    Loads a pre-trained model + feature engineer and processes flows
    from a CSV file (or a generator) in chunks to simulate streaming.
    """

    def __init__(
        self,
        model,
        feature_engineer,
        class_names: Optional[list] = None,
    ):
        self.model        = model
        self.fe           = feature_engineer
        self.class_names  = class_names or ["BENIGN", "ATTACK"]

    def _prepare_chunk(self, chunk: pd.DataFrame) -> Optional[np.ndarray]:
        """Strip labels, clean, and apply feature engineering."""
        lbl = config.LABEL_COLUMN
        if lbl in chunk.columns:
            chunk = chunk.drop(columns=[lbl])

        # Clean the same way as training
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk         = chunk.select_dtypes(include=[np.number])
        chunk.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        chunk.fillna(chunk.median(), inplace=True)

        if chunk.empty or len(chunk.columns) == 0:
            return None
        try:
            return self.fe.transform(chunk.values)
        except Exception as exc:
            print(f"  ⚠  FE transform failed: {exc}")
            return None

    def stream_csv(
        self,
        csv_path: str,
        chunk_size: int = 200,
        delay: float = 0.3,
        log_path: Optional[str] = None,
        max_chunks: Optional[int] = None,
    ) -> None:
        """
        Stream a CSV file chunk-by-chunk with a live dashboard.

        Parameters
        ----------
        csv_path   : Path to the CSV file.
        chunk_size : Number of rows per batch.
        delay      : Simulated inter-batch delay (seconds).
        log_path   : Optional path to save detection log (CSV).
        max_chunks : Stop after this many chunks (None = all).
        """
        print(f"  Streaming: {csv_path}")
        print(f"  Chunk size: {chunk_size}  |  Delay: {delay}s\n")

        model_name = getattr(self.model, "name", type(self.model).__name__)
        dash       = LiveDashboard(model_name=model_name)
        log_rows   = []

        reader = pd.read_csv(
            csv_path, chunksize=chunk_size, low_memory=False
        )

        for i, chunk in enumerate(reader):
            if max_chunks and i >= max_chunks:
                break

            X = self._prepare_chunk(chunk)
            if X is None:
                continue

            preds  = self.model.predict(X)
            probes = self.model.predict_proba(X)
            scores = probes[:, 1] if probes.shape[1] >= 2 else probes[:, 0]

            dash.update(preds, scores)
            dash.render()

            # Accumulate log
            for j, (pred, sc) in enumerate(zip(preds, scores)):
                log_rows.append({
                    "chunk":      i,
                    "flow_index": i * chunk_size + j,
                    "timestamp":  datetime.now().isoformat(),
                    "label":      self.class_names[min(int(pred), len(self.class_names) - 1)],
                    "attack_score": round(float(sc), 5),
                })

            time.sleep(delay)

        print("\n\n  ✓  Stream complete.")

        # Save log
        if log_path and log_rows:
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            print(f"  📄  Detection log → {log_path}")

    def stream_generator(
        self,
        row_generator,
        feature_names: list,
        chunk_size: int = 100,
        delay: float = 0.5,
    ) -> None:
        """
        Accept a Python generator that yields individual flow dicts.
        Buffers *chunk_size* rows, then predicts as a batch.

        Useful for integration with live packet capture (e.g. Scapy).
        """
        model_name = getattr(self.model, "name", type(self.model).__name__)
        dash       = LiveDashboard(model_name=model_name)
        buf        = []

        for row in row_generator:
            buf.append(row)
            if len(buf) < chunk_size:
                continue

            chunk = pd.DataFrame(buf, columns=feature_names)
            buf   = []
            X     = self._prepare_chunk(chunk)
            if X is None:
                continue

            preds  = self.model.predict(X)
            probes = self.model.predict_proba(X)
            scores = probes[:, 1]

            dash.update(preds, scores)
            dash.render()
            time.sleep(delay)


# ─── Demo mode (synthetic flows) ─────────────────────────────────────────────

def _synthetic_generator(n_features: int, n_flows: int = 500, attack_rate: float = 0.15):
    """Yield synthetic feature dicts with occasional attack bursts."""
    rng = np.random.default_rng(42)
    for i in range(n_flows):
        is_attack = rng.random() < attack_rate
        row = rng.exponential(300 if is_attack else 1000, size=n_features).tolist()
        yield row
        time.sleep(0.002)


def _demo_run():
    """Self-contained demo that needs no saved model or dataset."""
    from sklearn.ensemble import IsolationForest
    from features.feature_engineering import FeatureEngineer
    from data.data_loader import generate_demo_data

    print(_col("\n  [Demo] Generating synthetic training data…\n", CYAN))
    demo_data = generate_demo_data(n_samples=20_000)

    # Minimal feature engineer for demo
    from data.data_loader import preprocess
    config.CLASSIFICATION_MODE = "binary"
    X_tr, X_te, y_tr, y_te, _, feat_names, class_names = preprocess(demo_data)

    fe = FeatureEngineer()
    X_tr, y_tr = fe.fit_transform(X_tr, y_tr, feat_names)

    # Simple IsolationForest as demo model (no deep learning needed)
    class IFWrapper:
        name = "IsolationForest (Demo)"
        def __init__(self):
            self._m = IsolationForest(contamination=0.15, random_state=42)
        def predict(self, X):
            return (self._m.predict(X) == -1).astype(int)
        def predict_proba(self, X):
            scores = -self._m.score_samples(X)
            sc = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            return np.column_stack([1 - sc, sc])

    model = IFWrapper()
    model._m.fit(X_tr)

    detector = RealTimeDetector(model, fe, class_names)
    print(_col("\n  [Demo] Starting synthetic stream…\n", CYAN))

    gen = _synthetic_generator(n_features=X_tr.shape[1], n_flows=800)
    detector.stream_generator(
        gen,
        feature_names=[f"f{i}" for i in range(X_tr.shape[1])],
        chunk_size=50,
        delay=0.15,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Anomaly IDS — Real-Time Detection"
    )
    p.add_argument("--csv",     default=None,
                   help="Path to CICIDS CSV file to stream")
    p.add_argument("--model",   default="outputs/saved_models/LightGBM.pkl",
                   help="Path to saved model (.pkl)")
    p.add_argument("--fe",      default="outputs/saved_models/feature_engineer.pkl",
                   help="Path to saved FeatureEngineer (.pkl)")
    p.add_argument("--chunk",   type=int,   default=200)
    p.add_argument("--delay",   type=float, default=0.3,
                   help="Simulated inter-batch delay (seconds)")
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--log",     default="outputs/realtime_log.csv",
                   help="Path to save detection log CSV")
    p.add_argument("--demo",    action="store_true",
                   help="Run self-contained demo (no files needed)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        _demo_run()
        return

    if args.csv is None:
        print("  ✗  Provide --csv or use --demo")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"  ✗  CSV not found: {args.csv}")
        sys.exit(1)

    import joblib
    print(f"  Loading model        : {args.model}")
    model = joblib.load(args.model)
    print(f"  Loading FeatureEngineer: {args.fe}")
    fe = joblib.load(args.fe)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    detector = RealTimeDetector(model, fe)
    detector.stream_csv(
        args.csv,
        chunk_size  = args.chunk,
        delay       = args.delay,
        log_path    = args.log,
        max_chunks  = args.max_chunks,
    )


if __name__ == "__main__":
    main()
