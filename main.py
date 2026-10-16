"""
main.py
QuantMatrix AI: Institutional Quantitative Crypto Signals & Statistical Arbitrage API.
Production-grade FastAPI application engineered for RapidAPI Hub and B2B algorithmic traders.
Features sub-150ms in-memory cache pre-warming, 7/24 keep-alive, and zero-error payload architecture.
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, Request
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
        "liquidation hunt radar, market regime classification, and sub-100ms cached microstructure analytics."
    ),
    version="1.1.0",
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

# ─────────────────────────────────────────────────────────────
# HIGH-SPEED IN-MEMORY CACHE (Sub-150ms Response Time)
# ─────────────────────────────────────────────────────────────
CACHE = {
    "directional": {},
    "batch_signals": None,
    "pairs_scan": None,
    "volatility": {},
    "regime": {},
    "microstructure": {},
    "last_warm": 0
}
CACHE_LOCK = threading.Lock()

def prewarm_directional_worker():
    """Background worker that continuously pre-calculates directional signals in parallel every 30s."""
    top_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "SUIUSDT", "NEARUSDT", "LINKUSDT"]
    while True:
        try:
            batch = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_sym = {executor.submit(directional_engine.generate_directional_signal, sym): sym for sym in top_symbols}
                for future in as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        sig = future.result()
                        if "error" not in sig:
                            with CACHE_LOCK:
                                CACHE["directional"][sym] = sig
                            batch.append(sig)
                    except Exception:
                        pass
            
            if batch:
                with CACHE_LOCK:
                    CACHE["batch_signals"] = {
                        "count": len(batch),
                        "signals": batch,
                        "timestamp": int(time.time()),
                        "cached": True
                    }
                    CACHE["last_warm"] = int(time.time())
        except Exception:
            pass
        time.sleep(30)

def prewarm_pairs_worker():
    """Background worker for pairs arbitrage scanning (runs independently every 60s)."""
    while True:
        try:
            pairs = pairs_engine.scan_all_pairs()
            active_opps = [p for p in pairs if p.get("decision") in ("ENTER_ARBITRAGE", "TARGET_REACHED_TP")]
            with CACHE_LOCK:
                CACHE["pairs_scan"] = {
                    "total_pairs_scanned": len(pairs),
                    "active_opportunities_count": len(active_opps),
                    "opportunities": active_opps,
                    "all_pairs": pairs,
                    "timestamp": int(time.time()),
                    "cached": True
                }
        except Exception:
            pass
        time.sleep(60)

# Spawn worker threads upon startup
threading.Thread(target=prewarm_directional_worker, daemon=True).start()
threading.Thread(target=prewarm_pairs_worker, daemon=True).start()


@app.get("/", summary="API Root & Status")
async def root():
    return {
        "api_name": "QuantMatrix AI Engine",
        "status": "ONLINE",
        "version": "1.1.0",
        "speed": "Ultra-Low Latency (In-Memory Cached)",
        "architecture": "Institutional Quantitative Finance & Statistical Arbitrage",
        "documentation": "/docs",
        "endpoints": {
            "directional_signals": "/api/v1/signals/directional?symbol=BTCUSDT",
            "ai_trading_plan": "/api/v1/signals/plan?symbol=BTCUSDT",
            "liquidation_radar": "/api/v1/signals/liquidation-radar",
            "market_regime": "/api/v1/signals/regime?symbol=BTCUSDT",
            "wick_sniper_dips": "/api/v1/signals/wick-sniper",
            "pairs_arbitrage_scan": "/api/v1/arbitrage/pairs/scan",
            "pairs_deep_analysis": "/api/v1/arbitrage/pairs/analyze?asset_a=ETHUSDT&asset_b=BTCUSDT",
            "microstructure_analytics": "/api/v1/analytics/microstructure?symbol=BTCUSDT",
            "volatility_analytics": "/api/v1/analytics/volatility?symbol=BTCUSDT"
        },
        "server_time": int(time.time())
    }

@app.get("/health", summary="Health & Keep-Alive Check")
async def health_check():
    return {
        "status": "healthy",
        "uptime": "active",
        "cache_warmed": CACHE["last_warm"] > 0,
        "timestamp": int(time.time())
    }

# ─────────────────────────────────────────────────────────────
# 1. DIRECTIONAL SIGNALS & AI TRADE PLANS
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/signals/directional", summary="Get Directional Signals (15 Core Cryptos)")
async def get_directional_signals(
    symbol: Optional[str] = Query(None, description="Optional symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT). If omitted, returns multi-asset scan.")
):
    """
    Returns high-conviction directional quant signals with sub-100ms response time:
    - 3-Gate Execution Score (0 to 100)
    - Action: LONG, SHORT, or NEUTRAL
    - Dynamic ATR Stop-Loss and Asymmetric Targets (1.5R Scale-out, 3.0R Chandelier)
    - Technical momentum & CVD flow confirmation
    """
    if symbol:
        sym = symbol.upper()
        with CACHE_LOCK:
            if sym in CACHE["directional"]:
                return CACHE["directional"][sym]
                
        res = directional_engine.generate_directional_signal(sym)
        if "error" in res:
            return {
                "success": False,
                "symbol": sym,
                "error": res["error"],
                "message": f"Symbol '{sym}' could not be processed. Please provide a valid Binance Futures ticker (e.g., BTCUSDT, ETHUSDT, SOLUSDT).",
                "timestamp": int(time.time())
            }
        with CACHE_LOCK:
            CACHE["directional"][sym] = res
        return res

    with CACHE_LOCK:
        if CACHE["batch_signals"]:
            return CACHE["batch_signals"]

    # Fallback multi-asset batch scan using fast parallel threads
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(directional_engine.generate_directional_signal, s) for s in directional_engine.DEFAULT_SYMBOLS[:8]]
        for f in as_completed(futures):
            try:
                sig = f.result()
                if "error" not in sig:
                    results.append(sig)
            except Exception:
                pass
    fallback_res = {
        "count": len(results),
        "signals": results,
        "timestamp": int(time.time())
    }
    with CACHE_LOCK:
        CACHE["batch_signals"] = fallback_res
    return fallback_res

@app.get("/api/v1/signals/plan", summary="AI Trade Execution Plan (Entry, Stops, Leverage, R:R)")
async def get_trade_plan(
    symbol: str = Query("BTCUSDT", description="Crypto symbol (e.g. BTCUSDT, SOLUSDT, ETHUSDT)")
):
    """
    Generates a turn-key institutional trade execution plan:
    - Recommended Side (LONG / SHORT / WAIT)
    - Confluence Score & Conviction Level
    - Precise Entry, Invalidation (Stop Loss), Target 1 (1.5R), Target 2 (3.0R)
    - Optimal Recommended Leverage based on Extreme Volatility
    """
    sym = symbol.upper()
    sig = directional_engine.generate_directional_signal(sym)
    if "error" in sig:
        return {
            "success": False,
            "symbol": sym,
            "message": f"Symbol '{sym}' not found. Please provide a valid Binance Futures symbol."
        }
    
    vol = volatility_engine.analyze_volatility(sym)
    rec_lev = vol.get("recommended_position_multiplier", 1.0)
    leverage = min(10, max(2, int(rec_lev * 5)))
    
    return {
        "symbol": sym,
        "action": sig.get("action", "NEUTRAL"),
        "confidence_score": sig.get("total_score", 50),
        "current_price": sig.get("current_price"),
        "trade_plan": {
            "entry_range": [round(sig.get("current_price", 0) * 0.999, 4), round(sig.get("current_price", 0) * 1.001, 4)],
            "stop_loss": sig.get("trade_plan", {}).get("stop_loss"),
            "take_profit_1": sig.get("trade_plan", {}).get("take_profit_1"),
            "take_profit_2": sig.get("trade_plan", {}).get("take_profit_2"),
            "risk_reward_ratio": "1 : 2.25 (Asymmetric)",
            "recommended_leverage": f"{leverage}x"
        },
        "regime": sig.get("trend_bias"),
        "timestamp": int(time.time())
    }

@app.get("/api/v1/signals/liquidation-radar", summary="Liquidation Squeeze & Hunt Radar")
async def get_liquidation_radar():
    """
    Scans crypto assets experiencing high funding rate imbalances and extreme order flow pressure,
    identifying high-probability Short/Long Squeeze liquidation targets.
    """
    funding_map = data_client.get_funding_rates()
    squeeze_targets = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "DOGEUSDT"]:
        fr = funding_map.get(sym, 0.0)
        tr = data_client.get_taker_volume_ratio(sym)
        
        # High negative funding with aggressive buying -> Short Squeeze
        if fr < -0.0002 and tr > 1.10:
            squeeze_targets.append({
                "symbol": sym,
                "squeeze_type": "SHORT_SQUEEZE_HUNT",
                "funding_8h": round(fr * 100, 4),
                "taker_buy_pressure": tr,
                "urgency": "HIGH"
            })
        elif fr > 0.0005 and tr < 0.90:
            squeeze_targets.append({
                "symbol": sym,
                "squeeze_type": "LONG_FLUSH_HUNT",
                "funding_8h": round(fr * 100, 4),
                "taker_buy_pressure": tr,
                "urgency": "HIGH"
            })
    return {
        "count": len(squeeze_targets),
        "targets": squeeze_targets if squeeze_targets else [{"status": "No extreme squeeze detected at this moment. Market is balanced."}],
        "timestamp": int(time.time())
    }

@app.get("/api/v1/signals/regime", summary="Market Regime Classification")
async def get_market_regime(
    symbol: str = Query("BTCUSDT", description="Symbol to classify (default: BTCUSDT)")
):
    sym = symbol.upper()
    with CACHE_LOCK:
        if sym in CACHE["regime"] and (time.time() - CACHE["regime"][sym].get("_cached_at", 0) < 60):
            return CACHE["regime"][sym]
    res = directional_engine.detect_market_regime(sym)
    if isinstance(res, dict) and "error" not in res:
        res["_cached_at"] = time.time()
        with CACHE_LOCK:
            CACHE["regime"][sym] = res
    return res

@app.get("/api/v1/signals/wick-sniper", summary="Flash-Crash Oversold Dip Alerts")
async def get_wick_sniper():
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
    with CACHE_LOCK:
        if CACHE["pairs_scan"] and (time.time() - CACHE["last_warm"] < 120):
            return CACHE["pairs_scan"]

    pairs = pairs_engine.scan_all_pairs()
    active_opportunities = [p for p in pairs if p.get("decision") in ("ENTER_ARBITRAGE", "TARGET_REACHED_TP")]
    res = {
        "total_pairs_scanned": len(pairs),
        "active_opportunities_count": len(active_opportunities),
        "opportunities": active_opportunities,
        "all_pairs": pairs,
        "timestamp": int(time.time())
    }
    with CACHE_LOCK:
        CACHE["pairs_scan"] = res
    return res

@app.get("/api/v1/arbitrage/pairs/analyze", summary="Deep Cointegration Analysis for Custom Pair")
async def analyze_custom_pair(
    asset_a: str = Query(..., description="First asset (e.g. AVAXUSDT or ETHUSDT)"),
    asset_b: str = Query(..., description="Second asset (e.g. NEARUSDT or SOLUSDT)")
):
    res = pairs_engine.analyze_pair(asset_a, asset_b)
    if "error" in res:
        return {
            "success": False,
            "asset_a": asset_a.upper(),
            "asset_b": asset_b.upper(),
            "error": res["error"],
            "message": "Failed to analyze pair. Ensure both symbols exist on Binance Futures with sufficient history.",
            "timestamp": int(time.time())
        }
    return res

# ─────────────────────────────────────────────────────────────
# 3. MICROSTRUCTURE & VOLATILITY ANALYTICS
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/analytics/microstructure", summary="Order Flow & Microstructure Radar")
async def get_microstructure(
    symbol: str = Query("BTCUSDT", description="Crypto symbol (e.g. BTCUSDT, SOLUSDT)")
):
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
    sym = symbol.upper()
    with CACHE_LOCK:
        if sym in CACHE["volatility"] and (time.time() - CACHE["volatility"][sym].get("_cached_at", 0) < 60):
            return CACHE["volatility"][sym]
    res = volatility_engine.analyze_volatility(sym)
    if "error" in res:
        return {
            "success": False,
            "symbol": sym,
            "error": res["error"],
            "message": "Failed to compute volatility metrics. Check symbol formatting.",
            "timestamp": int(time.time())
        }
    res["_cached_at"] = time.time()
    with CACHE_LOCK:
        CACHE["volatility"][sym] = res
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
