"""
tests/test_features.py
=======================
Unit tests for the FeatureEngineer module.
Run with:  pytest tests/ -v --cov=core
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.features import FeatureEngineer


@pytest.fixture
def ohlcv():
    """300-bar synthetic OHLCV."""
    np.random.seed(42)
    n = 300
    p = 83.42 + np.cumsum(np.random.randn(n) * 0.025)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    return pd.DataFrame({
        "Open":   p * (1 - 0.0005 * np.random.rand(n)),
        "High":   p * (1 + 0.001  * np.random.rand(n)),
        "Low":    p * (1 - 0.001  * np.random.rand(n)),
        "Close":  p,
        "Volume": np.random.randint(500, 5000, n).astype(float),
        "Pair":   "USD/INR",
    }, index=idx)

@pytest.fixture
def fe():
    return FeatureEngineer()


class TestBuildOutput:

    def test_returns_dataframe(self, fe, ohlcv):
        assert isinstance(fe.build(ohlcv), pd.DataFrame)

    def test_no_nan_values(self, fe, ohlcv):
        df = fe.build(ohlcv)
        assert df.isnull().sum().sum() == 0

    def test_fewer_rows_than_input(self, fe, ohlcv):
        # NaN rows should be dropped
        assert len(fe.build(ohlcv)) < len(ohlcv)

    def test_reproducible(self, fe, ohlcv):
        pd.testing.assert_frame_equal(fe.build(ohlcv.copy()), fe.build(ohlcv.copy()))


class TestFeatureGroups:

    def test_trend_columns(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["EMA_9","EMA_20","EMA_50","EMA_200","cross_20_50","ADX"]:
            assert col in df.columns, f"Missing: {col}"

    def test_momentum_columns(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["RSI_14","MACD","MACD_signal","Stoch_K","CCI","Williams_R"]:
            assert col in df.columns, f"Missing: {col}"

    def test_volatility_columns(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["ATR_14","BB_upper","BB_lower","BB_width","BB_pct_b","BB_squeeze"]:
            assert col in df.columns, f"Missing: {col}"

    def test_volume_columns(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["OBV","vol_ratio","vol_spike","VWAP","dist_VWAP"]:
            assert col in df.columns, f"Missing: {col}"

    def test_pattern_columns_binary(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["bullish_engulf","bearish_engulf","doji","hammer","pin_bar"]:
            if col in df.columns:
                assert set(df[col].unique()).issubset({0,1}), f"{col} not binary"

    def test_time_columns(self, fe, ohlcv):
        df = fe.build(ohlcv)
        for col in ["hour","day_of_week","london_open","nse_open"]:
            assert col in df.columns, f"Missing: {col}"


class TestTarget:

    def test_target_binary(self, fe, ohlcv):
        df = fe.build(ohlcv, add_target=True)
        assert "Target" in df.columns
        assert set(df["Target"].unique()).issubset({0,1})

    def test_no_target_when_disabled(self, fe, ohlcv):
        df = fe.build(ohlcv, add_target=False)
        assert "Target" not in df.columns


class TestGetXy:

    def test_shapes_match(self, fe, ohlcv):
        df = fe.build(ohlcv)
        X, y = fe.get_Xy(df)
        assert len(X) == len(y)

    def test_target_not_in_X(self, fe, ohlcv):
        df = fe.build(ohlcv)
        X, _ = fe.get_Xy(df)
        assert "Target" not in X.columns

    def test_ohlcv_not_in_X(self, fe, ohlcv):
        df = fe.build(ohlcv)
        X, _ = fe.get_Xy(df)
        for col in ["Open","High","Low","Close","Volume"]:
            assert col not in X.columns


class TestRSI:

    def test_rsi_in_range(self, fe, ohlcv):
        df = fe.build(ohlcv)
        rsi = df["RSI_14"]
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_uptrend_high(self):
        up  = pd.Series(np.linspace(80, 100, 100))
        rsi = FeatureEngineer._rsi(up, 14).dropna()
        assert float(rsi.iloc[-1]) > 70

    def test_rsi_downtrend_low(self):
        dn  = pd.Series(np.linspace(100, 80, 100))
        rsi = FeatureEngineer._rsi(dn, 14).dropna()
        assert float(rsi.iloc[-1]) < 30
