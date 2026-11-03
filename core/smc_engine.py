"""
core/smc_engine.py
Institutional Smart Money Concepts (SMC) & Liquidity Sweep Engine.
Detects real-time Swing High/Low sweeps (12-bar Pivot Lookback), Hidden Orderblock rejection wicks (>=35%),
Consequent Encroachment (50% Wick Limit Retest), 3.5R Asymmetric Targets, and Order Flow Whale Delta Confirmation.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from core.binance_data import BinanceDataClient

class SMCLiquidityEngine:
    TOP_20_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "APTUSDT", "DOGEUSDT",
        "NEARUSDT", "SUIUSDT", "AVAXUSDT", "LINKUSDT", "INJUSDT",
        "TIAUSDT", "AAVEUSDT", "RENDERUSDT", "FETUSDT", "TAOUSDT",
        "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "BNBUSDT"
    ]

    def __init__(self, data_client: Optional[BinanceDataClient] = None):
        self.data_client = data_client or BinanceDataClient()

    def analyze_symbol(self, symbol: str, pivot_lookback: int = 12) -> Dict[str, Any]:
        """
        Analyzes a single crypto pair for institutional liquidity sweeps (SFP) and CE levels.
        """
        sym = symbol.upper()
        try:
            df = self.data_client.get_recent_candles(sym, interval="1h", limit=60)
            if len(df) < 30:
                return {"status": "INSUFFICIENT_DATA", "symbol": sym, "error": "Not enough candle bars"}

            close = df["close"].values
            open_p = df["open"].values
            high = df["high"].values
            low = df["low"].values
            vol = df["volume"].values
            n = len(df)

            # Trend Gate: 1H EMA 200 (or EMA 50)
            ema_200 = df["close"].ewm(span=min(50, n), adjust=False).mean().values

            # Current candle (last completed / active bar)
            c_idx = n - 1
            c_high = high[c_idx]
            c_low = low[c_idx]
            c_close = close[c_idx]
            c_open = open_p[c_idx]
            c_range = c_high - c_low
            c_ema = ema_200[c_idx]

            if c_range <= 0:
                return {"status": "NO_SETUP", "symbol": sym, "message": "Zero range candle"}

            # Wicks and Body
            body_size = abs(c_close - c_open)
            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low

            u_ratio = upper_wick / c_range
            l_ratio = lower_wick / c_range
            b_ratio = body_size / c_range

            # 12-bar Pivot High and Pivot Low (excluding current candle)
            p_start = max(0, c_idx - pivot_lookback - 1)
            p_high = np.max(high[p_start:c_idx])
            p_low = np.min(low[p_start:c_idx])

            # Order Flow Indicators
            taker_ratio = self.data_client.get_taker_volume_ratio(sym)
            funding_map = self.data_client.get_funding_rates()
            funding_8h = funding_map.get(sym, 0.0)

            # --- SETUP 1: BEARISH LIQUIDITY SWEEP (SHORT SFP) ---
            # 1. Price swept above prior 12-bar Pivot High
            # 2. But Close fell back below the Pivot High (Fake breakout / Trap)
            # 3. Upper rejection wick >= 35% of total candle range, body <= 50%
            if (c_high > p_high) and (c_close < p_high) and (u_ratio >= 0.35) and (b_ratio <= 0.50):
                # Consequent Encroachment (CE): 50% of the upper rejection wick
                entry_limit = c_high - (0.50 * upper_wick)
                sl_price = c_high * 1.001
                risk_dist = sl_price - entry_limit

                if risk_dist > 0 and (risk_dist / entry_limit) >= 0.003:
                    tp_price = entry_limit - (3.5 * risk_dist)
                    trend_bias = "BEARISH_ALIGNED" if c_close < c_ema else "COUNTER_TREND"
                    confidence = 94.0 if (trend_bias == "BEARISH_ALIGNED" and taker_ratio < 0.95) else 85.0

                    return {
                        "status": "SIGNAL_FOUND",
                        "symbol": sym,
                        "setup_type": "BEARISH_LIQUIDITY_SWEEP_SFP",
                        "direction": "SHORT",
                        "market_structure": trend_bias,
                        "pivot_level_swept": round(float(p_high), 4),
                        "candle_time": str(df["timestamp"].iloc[-1]),
                        "wick_rejection_ratio": f"{u_ratio * 100:.1f}%",
                        "execution": {
                            "entry_limit": round(float(entry_limit), 4),
                            "consequent_encroachment_50": round(float(entry_limit), 4),
                            "stop_loss": round(float(sl_price), 4),
                            "take_profit": round(float(tp_price), 4),
                            "risk_distance_pct": round(float((risk_dist / entry_limit) * 100), 2),
                            "target_rr": 3.5
                        },
                        "order_flow": {
                            "taker_buy_sell_ratio": taker_ratio,
                            "whale_absorption": "SELLERS_ABSORBED_BUYERS" if taker_ratio < 0.95 else "BALANCED",
                            "funding_rate_8h": round(funding_8h * 100, 4)
                        },
                        "confidence_score": confidence,
                        "timestamp": int(time.time())
                    }

            # --- SETUP 2: BULLISH LIQUIDITY SWEEP (LONG SFP) ---
            # 1. Price swept below prior 12-bar Pivot Low
            # 2. But Close climbed back above the Pivot Low (Stop hunt / Liquidation flush)
            # 3. Lower rejection wick >= 35% of total candle range, body <= 50%
            elif (c_low < p_low) and (c_close > p_low) and (l_ratio >= 0.35) and (b_ratio <= 0.50):
                # Consequent Encroachment (CE): 50% of the lower rejection wick
                entry_limit = c_low + (0.50 * lower_wick)
                sl_price = c_low * 0.999
                risk_dist = entry_limit - sl_price

                if risk_dist > 0 and (risk_dist / entry_limit) >= 0.003:
                    tp_price = entry_limit + (3.5 * risk_dist)
                    trend_bias = "BULLISH_ALIGNED" if c_close > c_ema else "COUNTER_TREND"
                    confidence = 94.0 if (trend_bias == "BULLISH_ALIGNED" and taker_ratio > 1.05) else 85.0

                    return {
                        "status": "SIGNAL_FOUND",
                        "symbol": sym,
                        "setup_type": "BULLISH_LIQUIDITY_SWEEP_SFP",
                        "direction": "LONG",
                        "market_structure": trend_bias,
                        "pivot_level_swept": round(float(p_low), 4),
                        "candle_time": str(df["timestamp"].iloc[-1]),
                        "wick_rejection_ratio": f"{l_ratio * 100:.1f}%",
                        "execution": {
                            "entry_limit": round(float(entry_limit), 4),
                            "consequent_encroachment_50": round(float(entry_limit), 4),
                            "stop_loss": round(float(sl_price), 4),
                            "take_profit": round(float(tp_price), 4),
                            "risk_distance_pct": round(float((risk_dist / entry_limit) * 100), 2),
                            "target_rr": 3.5
                        },
                        "order_flow": {
                            "taker_buy_sell_ratio": taker_ratio,
                            "whale_absorption": "BUYERS_ABSORBED_SELLERS" if taker_ratio > 1.05 else "BALANCED",
                            "funding_rate_8h": round(funding_8h * 100, 4)
                        },
                        "confidence_score": confidence,
                        "timestamp": int(time.time())
                    }

            return {
                "status": "MONITORING",
                "symbol": sym,
                "current_price": float(c_close),
                "nearest_pivot_high": round(float(p_high), 4),
                "nearest_pivot_low": round(float(p_low), 4),
                "upper_wick_ratio": f"{u_ratio * 100:.1f}%",
                "lower_wick_ratio": f"{l_ratio * 100:.1f}%",
                "message": "No active SFP sweep on current bar. Monitoring pivot boundaries.",
                "timestamp": int(time.time())
            }
        except Exception as e:
            return {"status": "ERROR", "symbol": sym, "error": str(e), "timestamp": int(time.time())}

    def scan_all_symbols(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scans all 20 top liquid cryptos in parallel using high-speed threads.
        Returns active high-probability SFP setups and general market watchlist.
        """
        sym_list = symbols or self.TOP_20_SYMBOLS
        active_setups = []
        all_results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(self.analyze_symbol, sym): sym for sym in sym_list}
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    res = future.result()
                    all_results.append(res)
                    if res.get("status") == "SIGNAL_FOUND":
                        active_setups.append(res)
                except Exception:
                    pass

        return {
            "total_scanned": len(sym_list),
            "active_setups_count": len(active_setups),
            "active_setups": active_setups,
            "watchlist": [r for r in all_results if r.get("status") != "SIGNAL_FOUND"],
            "timestamp": int(time.time())
        }
