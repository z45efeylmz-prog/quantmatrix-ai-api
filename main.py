"""
main.py
QuantMatrix AI: Institutional Quantitative Crypto Signals & Statistical Arbitrage API.
Production-grade FastAPI application engineered for RapidAPI Hub and B2B algorithmic traders.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure core package is resolvable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.binance_data import BinanceDataClient
from core.pairs_engine import PairsArbitrageEngine
from core.directional_engine import DirectionalQuantEngine
from core.volatility_engine import VolatilityEngine

app = FastAPI(
    title="QuantMatrix AI: Institutional Crypto Intelligence API",
    description=(
        "Enterprise-grade quantitative finance API powering algorithmic traders and crypto hedge funds. "
        "Provides real-time ML-style directional signals (3-Gate Execution, ATR Stops), "
        "statistical pairs arbitrage (ADF Cointegration, Kalman Beta, Ornstein-Uhlenbeck Half-Life), "
        "market regime classification, and microstructure analytics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Mathematical Engines
data_client = BinanceDataClient()
pairs_engine = PairsArbitrageEngine(data_client=data_client)
directional_engine = DirectionalQuantEngine(data_client=data_client)
volatility_engine = VolatilityEngine(data_client=data_client)

@app.get("/", summary="API Root & Status")
async def root():
    return {
        "api_name": "QuantMatrix AI Engine",
        "status": "ONLINE",
        "version": "1.0.0",
        "architecture": "Institutional Quantitative Finance & Statistical Arbitrage",
        "documentation": "/docs",
        "endpoints": {
            "directional_signals": "/api/v1/signals/directional?symbol=BTCUSDT",
            "market_regime": "/api/v1/signals/regime?symbol=BTCUSDT",
            "wick_sniper_dips": "/api/v1/signals/wick-sniper",
            "pairs_arbitrage_scan": "/api/v1/arbitrage/pairs/scan",
            "pairs_deep_analysis": "/api/v1/arbitrage/pairs/analyze?asset_a=ETHUSDT&asset_b=BTCUSDT",
            "microstructure_analytics": "/api/v1/analytics/microstructure?symbol=BTCUSDT",
            "volatility_analytics": "/api/v1/analytics/volatility?symbol=BTCUSDT"
        },
        "server_time": int(time.time())
    }

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}

# ─────────────────────────────────────────────────────────────
# 1. DIRECTIONAL SIGNALS & REGIMES
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/signals/directional", summary="Get Directional Signals (15 Core Cryptos)")
async def get_directional_signals(
    symbol: Optional[str] = Query(None, description="Optional symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT). If omitted, returns multi-asset scan.")
):
    """
    Returns high-conviction directional quant signals:
    - 3-Gate Execution Score (0 to 100)
    - Action: LONG, SHORT, or NEUTRAL
    - Dynamic ATR Stop-Loss and Asymmetric Targets (1.5R Scale-out, 3.0R Chandelier)
    - Technical momentum & CVD flow confirmation
    """
    if symbol:
        res = directional_engine.generate_directional_signal(symbol)
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res

    # Multi-asset batch scan
    results = []
    for sym in directional_engine.DEFAULT_SYMBOLS[:8]:
        sig = directional_engine.generate_directional_signal(sym)
        if "error" not in sig:
            results.append(sig)
    return {
        "count": len(results),
        "signals": results,
        "timestamp": int(time.time())
    }

@app.get("/api/v1/signals/regime", summary="Market Regime Classification")
async def get_market_regime(
    symbol: str = Query("BTCUSDT", description="Symbol to classify (default: BTCUSDT)")
):
    """
    Identifies market state:
    - TRENDING_BULL (Sustained upward trend)
    - TRENDING_BEAR (Downward momentum)
    - CHOPPY_MEAN_REVERTING (Ideal for statistical arbitrage)
    - HIGH_VOLATILITY_EXPANSION (Risk management strict mode)
    """
    return directional_engine.detect_market_regime(symbol)

@app.get("/api/v1/signals/wick-sniper", summary="Flash-Crash Oversold Dip Alerts")
async def get_wick_sniper():
    """
    Scans for extreme liquidation wicks and oversold flash drops (drop >= 3.5% with RSI <= 20)
    for mean-reversion dip buying.
    """
    alerts = directional_engine.scan_wick_sniper()
    return {
        "count": len(alerts),
        "alerts": alerts,
        "timestamp": int(time.time())
    }

# ─────────────────────────────────────────────────────────────
# 2. STATISTICAL PAIRS ARBITRAGE
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/arbitrage/pairs/scan", summary="Scan 14 Institutional Cointegrated Pairs")
async def scan_pairs_arbitrage():
    """
    Scans 14 institutional cryptocurrency pairs for statistical arbitrage:
    - Augmented Dickey-Fuller (ADF) Cointegration stationarity p-value
    - Dynamic Kalman Filter Hedge Ratio (Beta)
    - Residual Spread Z-Score
    - Ornstein-Uhlenbeck (OU) Mean-Reversion Half-Life ($t_{1/2}$)
    - 8h Funding Rate Carry APR
    """
    pairs = pairs_engine.scan_all_pairs()
    active_opportunities = [p for p in pairs if p.get("decision") in ("ENTER_ARBITRAGE", "TARGET_REACHED_TP")]
    return {
        "total_pairs_scanned": len(pairs),
        "active_opportunities_count": len(active_opportunities),
        "opportunities": active_opportunities,
        "all_pairs": pairs,
        "timestamp": int(time.time())
    }

@app.get("/api/v1/arbitrage/pairs/analyze", summary="Deep Cointegration Analysis for Custom Pair")
async def analyze_custom_pair(
    asset_a: str = Query(..., description="First asset (e.g. AVAXUSDT or ETHUSDT)"),
    asset_b: str = Query(..., description="Second asset (e.g. NEARUSDT or SOLUSDT)")
):
    """
    Analyzes cointegration and dynamic hedge ratio between any two custom crypto assets.
    """
    res = pairs_engine.analyze_pair(asset_a, asset_b)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

# ─────────────────────────────────────────────────────────────
# 3. MICROSTRUCTURE & VOLATILITY ANALYTICS
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/analytics/microstructure", summary="Order Flow & Microstructure Radar")
async def get_microstructure(
    symbol: str = Query("BTCUSDT", description="Crypto symbol (e.g. BTCUSDT, SOLUSDT)")
):
    """
    Fetches institutional microstructure metrics:
    - CVD Taker Buy/Sell Volume Ratio
    - 8-hour Funding Rate
    - Real-time Open Interest
    """
    sym = symbol.upper()
    taker_ratio = data_client.get_taker_volume_ratio(sym)
    funding_map = data_client.get_funding_rates()
    fr = funding_map.get(sym, 0.0)
    oi_data = data_client.get_open_interest(sym)

    return {
        "symbol": sym,
        "taker_buy_sell_ratio": taker_ratio,
        "funding_rate_8h_pct": round(fr * 100, 4),
        "annualized_funding_apr_pct": round(fr * 3 * 365 * 100, 2),
        "open_interest": oi_data.get("openInterest", "0"),
        "timestamp": int(time.time())
    }

@app.get("/api/v1/analytics/volatility", summary="Garman-Klass & Parkinson Volatility")
async def get_volatility(
    symbol: str = Query("BTCUSDT", description="Crypto symbol (e.g. BTCUSDT, SOLUSDT)")
):
    """
    Calculates extreme-value volatility models:
    - Garman-Klass Volatility (incorporating Open, High, Low, Close)
    - Parkinson Extreme Volatility (High-Low range)
    - Close-to-Close Realized Volatility
    - Dynamic position size multiplier recommendation
    """
    res = volatility_engine.analyze_volatility(symbol)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
