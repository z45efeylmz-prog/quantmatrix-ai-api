"""
core/volatility_engine.py
Institutional Volatility & Microstructure Risk Analytics.
Calculates Garman-Klass Extreme Value Volatility, Parkinson High-Low Volatility,
and Dynamic Position Risk Distance.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from core.binance_data import BinanceDataClient

class VolatilityEngine:
    def __init__(self, data_client: Optional[BinanceDataClient] = None):
        self.data_client = data_client or BinanceDataClient()

    def analyze_volatility(self, symbol: str) -> Dict[str, Any]:
        """Calculates Garman-Klass, Parkinson, and Realized Volatilities"""
        sym = symbol.upper()
        df = self.data_client.get_recent_candles(sym, interval="1h", limit=48)
        if len(df) < 20:
            return {"error": f"Insufficient data for {sym}", "symbol": sym}

        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # 1. Garman-Klass (accounts for Open, High, Low, Close)
        log_hl = np.log(h / np.maximum(l, 1e-9))
        log_co = np.log(c / np.maximum(o, 1e-9))
        gk_terms = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_vol = np.sqrt(np.maximum(0, np.mean(gk_terms[-24:]))) * np.sqrt(365 * 24) * 100

        # 2. Parkinson (accounts for High and Low extremes)
        park_terms = (log_hl ** 2) / (4 * np.log(2))
        park_vol = np.sqrt(np.maximum(0, np.mean(park_terms[-24:]))) * np.sqrt(365 * 24) * 100

        # 3. Standard Close-to-Close Realized Volatility
        returns = np.diff(c) / c[:-1]
        realized_vol = np.std(returns[-24:]) * np.sqrt(365 * 24) * 100

        # 4. Volatility regime categorization
        if gk_vol < 35:
            regime = "LOW_COMPRESSION"
            multiplier = 1.2
        elif gk_vol < 70:
            regime = "NORMAL_STABLE"
            multiplier = 1.0
        else:
            regime = "HIGH_EXPLOSION"
            multiplier = 0.6  # Reduce position size in high volatility

        return {
            "symbol": sym,
            "current_price": float(c[-1]),
            "annualized_garman_klass_vol_pct": round(float(gk_vol), 2),
            "annualized_parkinson_vol_pct": round(float(park_vol), 2),
            "annualized_realized_vol_pct": round(float(realized_vol), 2),
            "volatility_regime": regime,
            "recommended_position_multiplier": multiplier,
            "timestamp": int(time.time())
        }
