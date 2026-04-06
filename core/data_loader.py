"""
core/data_loader.py
===================
Fetches and cleans OHLCV data for Indian currency pairs via yfinance.
Maps INR pairs to yfinance tickers and filters to NSE market hours.

Pairs:  USD/INR · EUR/INR · GBP/INR · JPY/INR
Source: Yahoo Finance (free, no API key needed)

Usage:
    from core.data_loader import DataLoader
    loader = DataLoader()
    df = loader.fetch("USD/INR", period="60d", interval="5m")
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

PAIR_TO_TICKER = {
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "JPY/INR": "JPYINR=X",
}

PAIR_CONFIG = {
    "USD/INR": {"pip": 0.0025,  "lot": 1_000,    "margin": 2_500, "sym": "USDINR"},
    "EUR/INR": {"pip": 0.0025,  "lot": 1_000,    "margin": 2_700, "sym": "EURINR"},
    "GBP/INR": {"pip": 0.0025,  "lot": 1_000,    "margin": 3_200, "sym": "GBPINR"},
    "JPY/INR": {"pip": 0.00025, "lot": 100_000,  "margin": 1_700, "sym": "JPYINR"},
}

IST            = pytz.timezone("Asia/Kolkata")
NSE_OPEN_HOUR  = 9
NSE_CLOSE_HOUR = 17


class DataLoader:
    """Fetches, cleans and caches OHLCV data for NSE currency pairs."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(
        self,
        pair: str,
        period: str = "60d",
        interval: str = "5m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a currency pair.

        Args:
            pair:      e.g. "USD/INR"
            period:    yfinance period  e.g. "30d", "60d", "6mo"
            interval:  candle size      e.g. "1m", "5m", "15m", "1h"
            use_cache: return cached result if available

        Returns:
            pd.DataFrame — Open, High, Low, Close, Volume (IST index)
        """
        if pair not in PAIR_TO_TICKER:
            raise ValueError(f"Unsupported pair '{pair}'. Use: {list(PAIR_TO_TICKER)}")

        key = f"{pair}|{period}|{interval}"
        if use_cache and key in self._cache:
            return self._cache[key]

        ticker = PAIR_TO_TICKER[pair]
        logger.info(f"Fetching {pair} ({ticker}) period={period} interval={interval}")

        raw = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)

        if raw.empty:
            raise ValueError(f"No data returned for {pair}.")

        df = self._clean(raw, pair, interval)
        self._cache[key] = df
        logger.success(f"{pair}: {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
        return df

    def fetch_all(self, period: str = "60d", interval: str = "5m") -> dict:
        """Fetch all supported pairs."""
        out = {}
        for pair in PAIR_TO_TICKER:
            try:
                out[pair] = self.fetch(pair, period=period, interval=interval)
            except Exception as e:
                logger.warning(f"Skipping {pair}: {e}")
        return out

    def get_latest_price(self, pair: str) -> float:
        df = self.fetch(pair, period="2d", interval="5m", use_cache=False)
        return float(df["Close"].iloc[-1])

    def is_market_open(self) -> bool:
        """True if NSE currency market is currently open."""
        now = datetime.now(IST)
        return now.weekday() < 5 and NSE_OPEN_HOUR <= now.hour < NSE_CLOSE_HOUR

    def get_expiry_date(self) -> str:
        """Last business day of current month (NSE currency futures expiry)."""
        now = datetime.now(IST)
        if now.month == 12:
            last = datetime(now.year + 1, 1, 1, tzinfo=IST) - timedelta(days=1)
        else:
            last = datetime(now.year, now.month + 1, 1, tzinfo=IST) - timedelta(days=1)
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        return last.strftime("%d %b %Y").upper()

    def get_angel_one_symbol(self, pair: str) -> str:
        """
        Exact search term for Angel One platform.
        Example: "USDINR10APR26"
        """
        cfg     = PAIR_CONFIG[pair]
        now     = datetime.now(IST)
        months  = ["JAN","FEB","MAR","APR","MAY","JUN",
                   "JUL","AUG","SEP","OCT","NOV","DEC"]
        expiry  = self.get_expiry_date()          # "30 APR 2026"
        day     = expiry.split()[0]               # "30"
        mo      = months[now.month - 1]           # "APR"
        yr      = str(now.year)[-2:]              # "26"
        return f"{cfg['sym']}{day}{mo}{yr}"       # "USDINR30APR26"

    # ── Private ───────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, pair: str, interval: str) -> pd.DataFrame:
        df = df.copy()

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Ensure IST timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        # Filter to NSE market hours for intraday
        if interval not in ["1d", "1wk", "1mo"]:
            df = df.between_time(f"{NSE_OPEN_HOUR:02d}:00",
                                 f"{NSE_CLOSE_HOUR - 1:02d}:59")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df.ffill()
        df["Pair"] = pair
        return df


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    print(f"Expiry : {loader.get_expiry_date()}")
    print(f"Market : {'OPEN' if loader.is_market_open() else 'CLOSED'}\n")

    for pair in PAIR_TO_TICKER:
        try:
            df  = loader.fetch(pair, period="5d", interval="5m")
            sym = loader.get_angel_one_symbol(pair)
            print(f"  {pair:10s}  ₹{df['Close'].iloc[-1]:.4f}  {sym}  ({len(df)} bars)")
        except Exception as e:
            print(f"  {pair:10s}  ERROR: {e}")
