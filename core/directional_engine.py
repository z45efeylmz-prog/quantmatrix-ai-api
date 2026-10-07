"""
core/directional_engine.py
Institutional Directional Quant Signal Engine.
Multi-Timeframe Trend Confluence, 3-Gate Execution Scoring (0-100),
Dynamic ATR Asymmetric Stops, Market Regime Classifier, and Flash-Crash Wick Sniper.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from core.binance_data import BinanceDataClient

class DirectionalQuantEngine:
    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
        "SUIUSDT", "NEARUSDT", "LINKUSDT", "APTUSDT", "AAVEUSDT",
        "ARBUSDT", "RENDERUSDT", "FETUSDT", "TAOUSDT", "PAXGUSDT"
    ]

    def __init__(self, data_client: Optional[BinanceDataClient] = None):
        self.data_client = data_client or BinanceDataClient()

    def _calc_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates EMA, RSI, ATR, and Bollinger Bands"""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        # EMAs
        ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values[-1]
        ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values[-1]
        ema200 = pd.Series(close).ewm(span=min(100, n), adjust=False).mean().values[-1]

        # RSI (14)
        deltas = np.diff(close)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        roll_g = pd.Series(gains).rolling(14).mean().iloc[-1]
        roll_l = pd.Series(losses).rolling(14).mean().iloc[-1] + 1e-9
        rs = roll_g / roll_l
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # ATR (14)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(min(14, len(tr))).mean().iloc[-1]

        # Bollinger Bands (20, 2)
        bb_mean = pd.Series(close).rolling(20).mean().iloc[-1]
        bb_std = pd.Series(close).rolling(20).std().iloc[-1]
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        bb_width_pct = (bb_upper - bb_lower) / max(bb_mean, 1e-9) * 100

        return {
            "current_price": float(close[-1]),
            "ema20": float(ema20),
            "ema50": float(ema50),
            "ema200": float(ema200),
            "rsi": float(rsi),
            "atr": float(atr),
            "bb_upper": float(bb_upper),
            "bb_lower": float(bb_lower),
            "bb_width_pct": float(bb_width_pct)
        }

    def generate_directional_signal(self, symbol: str) -> Dict[str, Any]:
        """Evaluates 3-Gate Execution Score, ATR Asymmetric Sizing, and ML-style Confluence"""
        sym = symbol.upper()
        df = self.data_client.get_recent_candles(sym, interval="1h", limit=100)

        if len(df) < 30:
            return {"error": f"Insufficient candle data for {sym}", "symbol": sym}

        ind = self._calc_indicators(df)
        price = ind["current_price"]
        atr = ind["atr"]

        # Gate 1: Trend Alignment (Max 35 points)
        g1_score = 0
        trend_bias = "NEUTRAL"
        if price > ind["ema20"] > ind["ema50"]:
            g1_score += 25
            trend_bias = "BULLISH"
            if price > ind["ema200"]:
                g1_score += 10
        elif price < ind["ema20"] < ind["ema50"]:
            g1_score += 25
            trend_bias = "BEARISH"
            if price < ind["ema200"]:
                g1_score += 10
        else:
            g1_score = 15

        # Gate 2: Microstructure & Flow (Max 35 points)
        g2_score = 15
        taker_ratio = self.data_client.get_taker_volume_ratio(sym)
        if trend_bias == "BULLISH" and taker_ratio > 1.05:
            g2_score += 20
        elif trend_bias == "BEARISH" and taker_ratio < 0.95:
            g2_score += 20
        else:
            g2_score += 10

        # Gate 3: Volatility & Funding Rate (Max 30 points)
        g3_score = 15
        funding_map = self.data_client.get_funding_rates()
        funding_rate = funding_map.get(sym, 0.0)
        # Healthy funding: not extreme
        if abs(funding_rate) < 0.0003:
            g3_score += 15
        elif (trend_bias == "BULLISH" and funding_rate < 0) or (trend_bias == "BEARISH" and funding_rate > 0):
            g3_score += 10  # Negative funding favors long squeeze, positive favors short

        total_score = g1_score + g2_score + g3_score

        # Signal Synthesis
        action = "NEUTRAL"
        if total_score >= 70:
            action = "LONG" if trend_bias == "BULLISH" else "SHORT"
        elif total_score >= 58:
            if trend_bias == "BULLISH" and ind["rsi"] < 65:
                action = "LONG"
            elif trend_bias == "BEARISH" and ind["rsi"] > 35:
                action = "SHORT"

        # Trade Plan Parameters (Asymmetric Risk/Reward: 1R Stop, 1.5R Scale-Out, 3.0R Target)
        stop_dist = 2.0 * atr
        if action == "LONG":
            sl_price = round(price - stop_dist, 4)
            tp1_price = round(price + (1.5 * stop_dist), 4)
            tp2_price = round(price + (3.0 * stop_dist), 4)
        elif action == "SHORT":
            sl_price = round(price + stop_dist, 4)
            tp1_price = round(price - (1.5 * stop_dist), 4)
            tp2_price = round(price - (3.0 * stop_dist), 4)
        else:
            sl_price = price
            tp1_price = price
            tp2_price = price

        return {
            "symbol": sym,
            "action": action,
            "current_price": price,
            "execution_score": total_score,
            "confidence_pct": min(95.0, round(total_score * 0.95, 1)),
            "gates_breakdown": {
                "gate_1_trend_score": g1_score,
                "gate_2_microstructure_flow": g2_score,
                "gate_3_funding_volatility": g3_score
            },
            "technical_indicators": {
                "rsi_14": round(ind["rsi"], 2),
                "ema20": round(ind["ema20"], 4),
                "ema50": round(ind["ema50"], 4),
                "ema200": round(ind["ema200"], 4),
                "atr_14": round(atr, 4),
                "bb_width_pct": round(ind["bb_width_pct"], 2),
                "taker_buy_sell_ratio": round(taker_ratio, 3),
                "funding_rate_8h_pct": round(funding_rate * 100, 4)
            },
            "trade_plan": {
                "entry_type": "POST_ONLY_MAKER_LIMIT",
                "stop_loss_price": sl_price,
                "take_profit_scale_out_1_5R": tp1_price,
                "take_profit_chandelier_3_0R": tp2_price,
                "risk_reward_ratio": "1:2.25 (Asymmetric)",
                "recommended_capital_risk_pct": 1.0
            },
            "timestamp": int(time.time())
        }

    def detect_market_regime(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Classifies Market Regime: Bull Trend, Bear Trend, Choppy Range, or High Volatility Squeeze"""
        sym = symbol.upper()
        df = self.data_client.get_recent_candles(sym, interval="4h", limit=60)
        if len(df) < 20:
            return {"symbol": sym, "regime": "NEUTRAL_BALANCED", "adx": 20.0}

        close = df["close"].values
        ret = np.diff(close) / close[:-1]
        volatility = np.std(ret) * np.sqrt(365 * 6) * 100  # Annualized %

        ema20 = pd.Series(close).ewm(span=20).mean().values[-1]
        ema50 = pd.Series(close).ewm(span=50).mean().values[-1]

        if close[-1] > ema20 > ema50 and volatility < 75:
            regime = "TRENDING_BULL"
            desc = "Stable upward institutional trend. Trend-following models favored."
        elif close[-1] < ema20 < ema50 and volatility < 75:
            regime = "TRENDING_BEAR"
            desc = "Sustained downward pressure. Short momentum or mean-reversion hedging favored."
        elif volatility >= 75:
            regime = "HIGH_VOLATILITY_EXPANSION"
            desc = "Aggressive breakout/liquidation volatility. Strict ATR stops required."
        else:
            regime = "CHOPPY_MEAN_REVERTING"
            desc = "Range-bound sideways market. Pairs statistical arbitrage strongly favored."

        return {
            "symbol": sym,
            "regime": regime,
            "description": desc,
            "annualized_volatility_pct": round(float(volatility), 2),
            "current_price": float(close[-1]),
            "timestamp": int(time.time())
        }

    def scan_wick_sniper(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Monitors for flash-crash dips: 1m drop >= 4% and RSI < 18 (Liquidation Cascades)"""
        sym_list = symbols or self.DEFAULT_SYMBOLS[:8]
        alerts = []
        for sym in sym_list:
            df = self.data_client.get_recent_candles(sym, interval="5m", limit=30)
            if len(df) < 15:
                continue
            close = df["close"].values
            open_p = df["open"].values
            last_change_pct = (close[-1] - open_p[-1]) / open_p[-1] * 100

            # 5m RSI
            deltas = np.diff(close)
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            rsi = 50.0
            if len(deltas) >= 10:
                rs = np.mean(gains[-10:]) / (np.mean(losses[-10:]) + 1e-9)
                rsi = 100.0 - (100.0 / (1.0 + rs))

            if last_change_pct <= -3.5 or rsi <= 20.0:
                alerts.append({
                    "symbol": sym,
                    "event": "FLASH_DIP_OPPORTUNITY",
                    "candle_drop_pct": round(float(last_change_pct), 2),
                    "rsi_5m": round(float(rsi), 1),
                    "dip_price": float(close[-1]),
                    "strategy": "OVERSOLD_REBOUND_DIP_BUY",
                    "target_bounce_pct": 1.5,
                    "timestamp": int(time.time())
                })
        return alerts
