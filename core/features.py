"""
core/features.py
================
Transforms raw OHLCV data into a 25+ feature matrix for ML training.

Feature groups:
  Trend      — EMA 9/20/50/200, ADX, crossovers
  Momentum   — RSI, MACD, Stochastic, CCI, Williams %R, ROC
  Volatility — ATR, Bollinger Bands, BB squeeze, historical vol
  Volume     — OBV, VWAP, volume ratio, volume spike
  Patterns   — Engulfing, Hammer, Shooting Star, Doji, Pin Bar
  Time       — Hour, day-of-week, session flags (London open, NSE open)
  Target     — Binary label: 1=BUY if price rises ≥0.15% in next 3 bars

Usage:
    from core.features import FeatureEngineer
    fe = FeatureEngineer()
    df = fe.build(df_ohlcv)
    X, y = fe.get_Xy(df)
"""

import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:

    def __init__(self, lookahead_bars: int = 3, profit_target_pct: float = 0.0015):
        self.lookahead_bars    = lookahead_bars
        self.profit_target_pct = profit_target_pct

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, add_target: bool = True) -> pd.DataFrame:
        """Build full feature matrix. Drops NaN rows automatically."""
        logger.info(f"Building features on {len(df)} candles")
        out = df.copy()
        out = self._trend(out)
        out = self._momentum(out)
        out = self._volatility(out)
        out = self._volume(out)
        out = self._patterns(out)
        out = self._time(out)
        if add_target:
            out = self._target(out)
        out = out.dropna()
        logger.success(f"Feature matrix: {out.shape[0]} rows × {out.shape[1]} cols")
        return out

    def get_Xy(self, df: pd.DataFrame):
        """Split into feature matrix X and target y."""
        drop = {"Open", "High", "Low", "Close", "Volume", "Pair", "Target"}
        X = df[[c for c in df.columns if c not in drop]]
        y = df["Target"]
        return X, y

    # ── Trend ─────────────────────────────────────────────────────────────────

    def _trend(self, df):
        c = df["Close"]
        df["EMA_9"]   = c.ewm(span=9,   adjust=False).mean()
        df["EMA_20"]  = c.ewm(span=20,  adjust=False).mean()
        df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
        df["EMA_200"] = c.ewm(span=200, adjust=False).mean()

        df["dist_EMA20"]  = (c - df["EMA_20"])  / df["EMA_20"]
        df["dist_EMA50"]  = (c - df["EMA_50"])  / df["EMA_50"]
        df["dist_EMA200"] = (c - df["EMA_200"]) / df["EMA_200"]

        df["cross_9_20"]  = (df["EMA_9"]  > df["EMA_20"]).astype(int)
        df["cross_20_50"] = (df["EMA_20"] > df["EMA_50"]).astype(int)
        df["ADX"]         = self._adx(df, 14)
        return df

    # ── Momentum ──────────────────────────────────────────────────────────────

    def _momentum(self, df):
        c, h, lo = df["Close"], df["High"], df["Low"]

        df["RSI_14"] = self._rsi(c, 14)
        df["RSI_7"]  = self._rsi(c, 7)
        df["RSI_21"] = self._rsi(c, 21)
        df["RSI_div"] = ((df["RSI_14"].diff(3) * c.pct_change(3)) < 0).astype(int)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
        df["MACD_cross"]  = (df["MACD"] > df["MACD_signal"]).astype(int)

        low14  = lo.rolling(14).min()
        high14 = h.rolling(14).max()
        df["Stoch_K"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        tp  = (h + lo + c) / 3
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + 1e-9)

        df["Williams_R"] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - lo.rolling(14).min() + 1e-9)
        df["ROC_5"]  = c.pct_change(5)  * 100
        df["ROC_10"] = c.pct_change(10) * 100
        return df

    # ── Volatility ────────────────────────────────────────────────────────────

    def _volatility(self, df):
        c = df["Close"]
        df["ATR_14"]  = self._atr(df, 14)
        df["ATR_pct"] = df["ATR_14"] / c * 100

        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        df["BB_upper"]   = bb_mid + 2 * bb_std
        df["BB_lower"]   = bb_mid - 2 * bb_std
        df["BB_mid"]     = bb_mid
        df["BB_width"]   = (df["BB_upper"] - df["BB_lower"]) / bb_mid
        df["BB_pct_b"]   = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
        df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(50).quantile(0.2)).astype(int)
        df["HV_20"]      = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
        return df

    # ── Volume ────────────────────────────────────────────────────────────────

    def _volume(self, df):
        c, v = df["Close"], df["Volume"]
        df["OBV"]       = (np.sign(c.diff()) * v).fillna(0).cumsum()
        vol_avg         = v.rolling(20).mean()
        df["vol_ratio"] = v / (vol_avg + 1e-9)
        df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)

        tp = (df["High"] + df["Low"] + c) / 3
        df["VWAP"]      = (tp * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
        df["dist_VWAP"] = (c - df["VWAP"]) / df["VWAP"]
        return df

    # ── Candlestick Patterns ──────────────────────────────────────────────────

    def _patterns(self, df):
        o, h, lo, c = df["Open"], df["High"], df["Low"], df["Close"]
        body  = (c - o).abs()
        upper = h - df[["Open","Close"]].max(axis=1)
        lower = df[["Open","Close"]].min(axis=1) - lo
        rng   = h - lo + 1e-9

        df["bullish_engulf"] = ((c > o) & (c.shift(1) < o.shift(1)) &
                                (c > o.shift(1)) & (o < c.shift(1))).astype(int)
        df["bearish_engulf"] = ((c < o) & (c.shift(1) > o.shift(1)) &
                                (c < o.shift(1)) & (o > c.shift(1))).astype(int)
        df["doji"]           = (body / rng < 0.1).astype(int)
        df["hammer"]         = ((lower > 2 * body) & (upper < 0.5 * body) & (c > o)).astype(int)
        df["shooting_star"]  = ((upper > 2 * body) & (lower < 0.5 * body) & (c < o)).astype(int)
        df["pin_bar"]        = ((lower > 2 * body) | (upper > 2 * body)).astype(int)
        return df

    # ── Time Features ─────────────────────────────────────────────────────────

    def _time(self, df):
        idx = df.index
        df["hour"]        = idx.hour
        df["day_of_week"] = idx.dayofweek
        df["london_open"] = ((idx.hour >= 13) & (idx.hour < 16)).astype(int)
        df["nse_open"]    = ((idx.hour == 9)  & (idx.minute < 30)).astype(int)
        return df

    # ── Target Label ──────────────────────────────────────────────────────────

    def _target(self, df):
        """1 if price rises ≥ profit_target_pct within lookahead_bars."""
        c = df["Close"]
        future = c.shift(-self.lookahead_bars).rolling(self.lookahead_bars).max()
        df["Target"] = ((future - c) / c >= self.profit_target_pct).astype(int)
        return df

    # ── Indicator Math ────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(s: pd.Series, p: int) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))

    @staticmethod
    def _atr(df: pd.DataFrame, p: int) -> pd.Series:
        h, lo, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([(h - lo), (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, p: int) -> pd.Series:
        h, lo, c = df["High"], df["Low"], df["Close"]
        pdm = h.diff().clip(lower=0)
        ndm = (-lo.diff()).clip(lower=0)
        pdm[pdm < ndm] = 0
        ndm[ndm < pdm] = 0
        tr  = pd.concat([(h - lo), (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        pdi = 100 * pdm.rolling(p).mean() / (atr + 1e-9)
        ndi = 100 * ndm.rolling(p).mean() / (atr + 1e-9)
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
        return dx.rolling(p).mean()


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.data_loader import DataLoader
    df_raw  = DataLoader().fetch("USD/INR", period="30d", interval="5m")
    fe      = FeatureEngineer()
    df_feat = fe.build(df_raw)
    X, y    = fe.get_Xy(df_feat)
    print(f"Shape  : {X.shape}")
    print(f"BUY  %: {y.mean()*100:.1f}%")
    print(f"Cols   : {list(X.columns)}")
