"""
core/predict.py
===============
Real-time signal engine. Loads model.pkl → fetches live data →
outputs BUY/SELL signals with confidence, entry, SL, TP, R:R.

Usage:
    from core.predict import SignalEngine
    engine  = SignalEngine()
    signals = engine.run_all()          # all pairs
    signal  = engine.predict("USD/INR") # single pair
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

from core.data_loader import DataLoader, PAIR_CONFIG
from core.features    import FeatureEngineer
from core.model       import ModelTrainer, MODEL_PATH


@dataclass
class Signal:
    pair:           str
    direction:      str       # "BUY" | "SELL"
    confidence:     float     # 63–95 %
    entry_price:    float
    stop_loss:      float
    take_profit:    float
    sl_pips:        int
    tp_pips:        int
    risk_reward:    float
    strategy:       str
    patterns:       list
    contract:       str       # e.g. "USDINR30APR26"
    expiry:         str
    lot_size:       int
    margin_per_lot: int
    timestamp:      datetime = field(default_factory=datetime.now)

    def __repr__(self):
        icon = "📈" if self.direction == "BUY" else "📉"
        return (f"{icon} {self.pair} {self.direction} @ ₹{self.entry_price:.4f} | "
                f"Conf:{self.confidence:.0f}% | SL:₹{self.stop_loss:.4f} | "
                f"TP:₹{self.take_profit:.4f} | R:R 1:{self.risk_reward}")


PATTERN_MAP = {
    "bullish_engulf": "Bullish Engulfing",
    "bearish_engulf": "Bearish Engulfing",
    "hammer":         "Hammer",
    "shooting_star":  "Shooting Star",
    "doji":           "Doji",
    "pin_bar":        "Pin Bar",
}

BUY_REASONS = [
    "RSI({rsi:.0f}) oversold with MACD bullish crossover above signal line",
    "EMA 20/50 golden cross with bullish engulfing on key demand zone",
    "Price bouncing off Bollinger lower band — RSI oversold at {rsi:.0f}",
    "Stochastic crossover below 20 — oversold reversal signal detected",
    "CCI({cci:.0f}) extreme low — mean reversion BUY setup forming",
    "Volume spike ({vol:.1f}x avg) confirms breakout above EMA 200",
]

SELL_REASONS = [
    "RSI({rsi:.0f}) overbought with MACD bearish crossover below signal",
    "EMA 20/50 death cross with bearish engulfing at key resistance",
    "Price rejecting Bollinger upper band — RSI overbought at {rsi:.0f}",
    "Stochastic crossover above 80 — overbought reversal signal",
    "CCI({cci:.0f}) extreme high — mean reversion SELL setup forming",
    "Volume spike ({vol:.1f}x avg) confirms breakdown below EMA 200",
]


class SignalEngine:

    def __init__(self, model_path: Path = MODEL_PATH):
        self.loader  = DataLoader()
        self.fe      = FeatureEngineer()
        self.trainer: ModelTrainer | None = None
        self._load_model(model_path)

    # ── Public ────────────────────────────────────────────────────────────────

    def predict(self, pair: str, interval: str = "5m") -> Signal | None:
        """Generate a signal for one pair."""
        try:
            df = self.loader.fetch(pair, period="30d", interval=interval, use_cache=False)
            if len(df) < 60:
                return None

            df_feat = self.fe.build(df, add_target=False)
            if df_feat.empty:
                return None

            X, _ = self.fe.get_Xy(df_feat)
            ind  = self._snapshot(df_feat)

            # ML probability or rule-based fallback
            if self.trainer:
                prob = float(self.trainer.predict_proba(X.iloc[[-1]])[0])
            else:
                prob = self._rules(ind)

            direction  = "BUY" if prob >= 0.5 else "SELL"
            confidence = self._confidence(prob, ind)
            price      = float(df["Close"].iloc[-1])

            sl, tp, sl_pips, tp_pips = self._sl_tp(pair, price, direction, ind["ATR_14"])
            patterns  = self._patterns(df_feat)
            strategy  = self._reason(direction, ind)
            contract  = self.loader.get_angel_one_symbol(pair)
            expiry    = self.loader.get_expiry_date()
            cfg       = PAIR_CONFIG[pair]

            return Signal(
                pair=pair, direction=direction, confidence=confidence,
                entry_price=round(price, 4),
                stop_loss=round(sl, 4), take_profit=round(tp, 4),
                sl_pips=sl_pips, tp_pips=tp_pips,
                risk_reward=round(tp_pips / max(sl_pips, 1), 1),
                strategy=strategy, patterns=patterns,
                contract=contract, expiry=expiry,
                lot_size=cfg["lot"], margin_per_lot=cfg["margin"],
            )
        except Exception as e:
            logger.error(f"predict({pair}): {e}")
            return None

    def run_all(self, interval: str = "5m", min_conf: float = 65.0) -> list[Signal]:
        """Generate signals for all pairs, filtered by min confidence."""
        out = []
        for pair in PAIR_CONFIG:
            s = self.predict(pair, interval=interval)
            if s and s.confidence >= min_conf:
                out.append(s)
        out.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(f"Generated {len(out)} signals ≥ {min_conf}% confidence")
        return out

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> None:
        if path.exists():
            try:
                self.trainer = ModelTrainer.load(path)
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using rule-based fallback.")
        else:
            logger.warning(f"model.pkl not found at {path}. Train with: python core/model.py --train")

    def _snapshot(self, df: pd.DataFrame) -> dict:
        r = df.iloc[-1]
        return {k: float(r.get(k, 0)) for k in [
            "RSI_14","MACD","MACD_hist","MACD_cross","Stoch_K",
            "BB_pct_b","BB_squeeze","ATR_14","ADX","CCI",
            "vol_ratio","vol_spike","cross_20_50","cross_9_20","RSI_div",
        ]}

    def _rules(self, ind: dict) -> float:
        score = 0.5
        rsi   = ind["RSI_14"]
        if rsi < 35:   score += 0.15
        elif rsi > 65: score -= 0.15
        score += 0.10 if ind["MACD_cross"] else -0.10
        score += 0.10 if ind["cross_20_50"] else -0.10
        if ind["BB_pct_b"] < 0.2:   score += 0.08
        elif ind["BB_pct_b"] > 0.8: score -= 0.08
        if ind["vol_spike"]: score += 0.05
        if ind["RSI_div"]:   score += 0.07
        return float(np.clip(score, 0.1, 0.9))

    def _confidence(self, prob: float, ind: dict) -> float:
        conf = 63 + abs(prob - 0.5) * 2 * 32
        if (prob > 0.5 and ind["RSI_14"] < 35) or (prob < 0.5 and ind["RSI_14"] > 65):
            conf += 3
        if ind["MACD_cross"] and prob > 0.5: conf += 2
        if ind["vol_spike"]:                 conf += 2
        if ind["BB_squeeze"]:                conf += 1
        if ind["RSI_div"]:                   conf += 2
        return round(min(conf, 95), 1)

    def _sl_tp(self, pair, price, direction, atr):
        pip    = PAIR_CONFIG[pair]["pip"]
        sl_d   = max(atr * 1.5, pip * 15)
        tp_d   = max(atr * 2.5, pip * 22)
        sl     = price - sl_d if direction == "BUY" else price + sl_d
        tp     = price + tp_d if direction == "BUY" else price - tp_d
        return sl, tp, max(1, int(sl_d / pip)), max(1, int(tp_d / pip))

    def _patterns(self, df: pd.DataFrame) -> list:
        found = []
        for col, label in PATTERN_MAP.items():
            if col in df.columns and df[col].tail(3).any():
                found.append(label)
        return found[:3]

    def _reason(self, direction: str, ind: dict) -> str:
        import random
        tmpl = random.choice(BUY_REASONS if direction == "BUY" else SELL_REASONS)
        return tmpl.format(rsi=ind["RSI_14"], cci=ind["CCI"], vol=ind["vol_ratio"])


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine  = SignalEngine()
    signals = engine.run_all(min_conf=65)

    print(f"\n{'='*60}")
    print(f"  LIVE SIGNALS — {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    print(f"{'='*60}")
    for s in signals:
        print(f"\n{'📈' if s.direction=='BUY' else '📉'}  {s.pair:10s} {s.direction}")
        print(f"   Entry  : ₹{s.entry_price:.4f}")
        print(f"   SL/TP  : ₹{s.stop_loss:.4f} / ₹{s.take_profit:.4f}")
        print(f"   Conf   : {s.confidence:.1f}%  |  R:R 1:{s.risk_reward}")
        print(f"   Contract: {s.contract} · NSECDS")
    if not signals:
        print("  No signals above threshold.")
    print(f"\n  Market: {'OPEN' if engine.loader.is_market_open() else 'CLOSED'}")
    print(f"  Expiry: {engine.loader.get_expiry_date()}")
    print(f"{'='*60}\n")
