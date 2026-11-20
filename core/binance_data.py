"""
core/binance_data.py
High-speed cached market data client for Binance USDT-M Futures.
Fetches real-time candles, funding rates, open interest, and taker buy/sell ratios.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
import pandas as pd

BASE_URL = "https://fapi.binance.com"

class BinanceDataClient:
    def __init__(self):
        self._funding_cache: Dict[str, float] = {}
        self._funding_cache_time: float = 0.0
        self._cache_ttl: float = 30.0  # 30s cache
        self._candle_cache: Dict[str, Any] = {}
        self._taker_cache: Dict[str, float] = {}
        self._taker_cache_time: Dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def get_funding_rates(self) -> Dict[str, float]:
        """Fetches and caches Binance Futures 8h funding rates"""
        now = time.time()
        if self._funding_cache and (now - self._funding_cache_time < self._cache_ttl):
            return self._funding_cache

        rates = {}
        try:
            resp = self.session.get(f"{BASE_URL}/fapi/v1/premiumIndex", timeout=3)
            if resp.status_code == 200:
                for item in resp.json():
                    sym = item.get("symbol")
                    fr = item.get("lastFundingRate")
                    if sym and fr is not None:
                        rates[sym] = float(fr)
        except Exception:
            pass

        if rates:
            self._funding_cache = rates
            self._funding_cache_time = now
        return self._funding_cache

    def get_recent_candles(self, symbol: str, interval: str = "1h", limit: int = 60) -> pd.DataFrame:
        """Fetches OHLCV candles with 15s caching and fast Keep-Alive pooling"""
        sym = symbol.upper()
        cache_key = f"{sym}_{interval}"
        now = time.time()
        if cache_key in self._candle_cache:
            c_time, c_df = self._candle_cache[cache_key]
            if now - c_time < 30.0:
                return c_df.copy()

        vision = ("https://data-api.binance.vision/api/v3/klines", {"symbol": sym, "interval": interval, "limit": limit})
        fapi = ("https://fapi.binance.com/fapi/v1/klines", {"symbol": sym, "interval": interval, "limit": limit})
        spot = ("https://api.binance.com/api/v3/klines", {"symbol": sym, "interval": interval, "limit": limit})
        # Vercel runs in US regions where fapi/api.binance.com may return 451/timeout; try binance.vision first
        urls = [vision, fapi, spot] if os.environ.get("VERCEL") else [fapi, spot, vision]
        for url, params in urls:
            try:
                r = self.session.get(url, params=params, timeout=3)
                if r.status_code == 200:
                    raw = r.json()
                    if raw and isinstance(raw, list) and len(raw) >= 10:
                        df = pd.DataFrame(raw, columns=[
                            "timestamp", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "trades", "taker_buy_base",
                            "taker_buy_quote", "ignore"
                        ])
                        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]:
                            df[col] = df[col].astype(float)
                        df["timestamp"] = df["timestamp"].astype(int)
                        self._candle_cache[cache_key] = (now, df)
                        return df
            except Exception:
                pass
        return pd.DataFrame()

    def get_ticker_24h(self, symbol: Optional[str] = None) -> Any:
        """Fetches 24h ticker price change statistics"""
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            r = self.session.get(url, params=params, timeout=4)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}

    def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetches current open interest statistics"""
        sym = symbol.upper()
        url = f"{BASE_URL}/fapi/v1/openInterest"
        try:
            r = self.session.get(url, params={"symbol": sym}, timeout=4)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {"symbol": sym, "openInterest": "0", "time": int(time.time() * 1000)}

    def get_taker_volume_ratio(self, symbol: str, period: str = "1h", limit: int = 1) -> float:
        """Fetches taker buy/sell volume ratio (CVD indicator) with Keep-Alive pooling & 30s caching"""
        sym = symbol.upper()
        now = time.time()
        if sym in self._taker_cache and (now - self._taker_cache_time.get(sym, 0) < self._cache_ttl):
            return self._taker_cache[sym]

        url = f"{BASE_URL}/futures/data/takerlongshortRatio"
        try:
            r = self.session.get(url, params={"symbol": sym, "period": period, "limit": limit}, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list):
                    ratio = float(data[-1].get("buySellRatio", 1.0))
                    self._taker_cache[sym] = ratio
                    self._taker_cache_time[sym] = now
                    return ratio
        except Exception:
            pass
        return self._taker_cache.get(sym, 1.0)
