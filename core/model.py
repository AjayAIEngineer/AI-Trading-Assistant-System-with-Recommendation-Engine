"""
core/model.py
=============
Trains the 5-model ensemble and saves model.pkl.
Uses TimeSeriesSplit (walk-forward) to avoid look-ahead bias.

Models:
  XGBoost · Random Forest · LightGBM → Soft Voting Ensemble

Usage:
    # Train from CLI
    python core/model.py --train --pair "USD/INR"

    # Train from Python
    from core.model import ModelTrainer
    trainer = ModelTrainer("USD/INR")
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.save()                     # saves models/model.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

# ── Hyperparameters ───────────────────────────────────────────────────────────

XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", random_state=42, n_jobs=-1)

RF_PARAMS  = dict(n_estimators=300, max_depth=10, min_samples_leaf=5,
                  class_weight="balanced", random_state=42, n_jobs=-1)

LGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, class_weight="balanced",
                  random_state=42, n_jobs=-1, verbose=-1)


class ModelTrainer:

    def __init__(self, pair: str = "USD/INR"):
        self.pair          = pair
        self.scaler        = StandardScaler()
        self.ensemble      = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train ensemble on training data."""
        self.feature_names = list(X.columns)
        X_sc = self.scaler.fit_transform(X)

        xgb_m = xgb.XGBClassifier(**XGB_PARAMS)
        rf_m  = RandomForestClassifier(**RF_PARAMS)
        lgb_m = lgb.LGBMClassifier(**LGB_PARAMS)

        logger.info("Training XGBoost…")
        xgb_m.fit(X_sc, y)
        logger.info("Training Random Forest…")
        rf_m.fit(X_sc, y)
        logger.info("Training LightGBM…")
        lgb_m.fit(X_sc, y)

        logger.info("Building Soft Voting Ensemble…")
        self.ensemble = VotingClassifier(
            estimators=[("xgb", xgb_m), ("rf", rf_m), ("lgb", lgb_m)],
            voting="soft",
            weights=[0.4, 0.3, 0.3],
        )
        self.ensemble.fit(X_sc, y)
        logger.success("Training complete ✅")

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate on held-out test set."""
        X_sc   = self.scaler.transform(X)
        y_pred = self.ensemble.predict(X_sc)

        self.metrics = {
            "accuracy":  round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        }
        logger.info("\n" + classification_report(y, y_pred, target_names=["SELL","BUY"]))
        return self.metrics

    # ── Cross-Validate ────────────────────────────────────────────────────────

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """Walk-forward cross-validation — no data leakage."""
        tscv   = TimeSeriesSplit(n_splits=n_splits)
        folds  = []
        for i, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            sc    = StandardScaler()
            X_trs = sc.fit_transform(X_tr)
            X_vs  = sc.transform(X_val)

            vc = VotingClassifier(
                estimators=[
                    ("xgb", xgb.XGBClassifier(**XGB_PARAMS)),
                    ("rf",  RandomForestClassifier(**RF_PARAMS)),
                    ("lgb", lgb.LGBMClassifier(**LGB_PARAMS)),
                ],
                voting="soft", weights=[0.4, 0.3, 0.3],
            )
            vc.fit(X_trs, y_tr)
            yp = vc.predict(X_vs)

            m = {"fold": i+1,
                 "acc":  round(accuracy_score(y_val, yp), 4),
                 "f1":   round(f1_score(y_val, yp, zero_division=0), 4)}
            folds.append(m)
            logger.info(f"  Fold {i+1}: acc={m['acc']:.3f}  f1={m['f1']:.3f}")

        avg = {k: round(np.mean([f[k] for f in folds]), 4) for k in ["acc","f1"]}
        logger.success(f"CV avg: {avg}")
        return {"folds": folds, "avg": avg}

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save model bundle to models/model.pkl"""
        bundle = {
            "ensemble":      self.ensemble,
            "scaler":        self.scaler,
            "feature_names": self.feature_names,
            "metrics":       self.metrics,
            "pair":          self.pair,
            "trained_at":    datetime.now().isoformat(),
        }
        joblib.dump(bundle, path)
        logger.success(f"Model saved → {path}  ({path.stat().st_size/1024:.1f} KB)")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ModelTrainer":
        """Load model bundle from models/model.pkl"""
        bundle          = joblib.load(path)
        inst            = cls(pair=bundle["pair"])
        inst.ensemble   = bundle["ensemble"]
        inst.scaler     = bundle["scaler"]
        inst.feature_names = bundle["feature_names"]
        inst.metrics    = bundle.get("metrics", {})
        logger.success(f"Model loaded ← {path}  (trained {bundle.get('trained_at','?')})")
        return inst

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return BUY probabilities for each row."""
        X_sc = self.scaler.transform(X[self.feature_names])
        return self.ensemble.predict_proba(X_sc)[:, 1]


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",  action="store_true")
    ap.add_argument("--pair",   default="USD/INR")
    ap.add_argument("--period", default="180d")
    args = ap.parse_args()

    if not args.train:
        ap.print_help()
        return

    from core.data_loader import DataLoader
    from core.features    import FeatureEngineer

    df_raw  = DataLoader().fetch(args.pair, period=args.period, interval="5m")
    df_feat = FeatureEngineer().build(df_raw)
    X, y    = FeatureEngineer().get_Xy(df_feat)

    split   = int(len(X) * 0.8)
    trainer = ModelTrainer(args.pair)
    trainer.cross_validate(X.iloc[:split], y.iloc[:split])
    trainer.train(X.iloc[:split], y.iloc[:split])
    m = trainer.evaluate(X.iloc[split:], y.iloc[split:])
    print(f"\n📊 Test results: {m}")
    trainer.save()


if __name__ == "__main__":
    main()
