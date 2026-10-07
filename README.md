# QuantMatrix AI: Institutional Crypto Intelligence API

> **Enterprise-Grade Quantitative Finance, Multi-Asset Machine Learning Signals & Delta-Neutral Statistical Arbitrage Engine for Algorithmic Traders and Crypto Hedge Funds.**

---

## 🌟 Overview
**QuantMatrix AI** delivers institutional hedge fund analytics directly to algorithmic traders, bot builders, and quantitative researchers via high-performance REST APIs.

Instead of writing complex statistical cointegration math, Kalman filters, or Order Flow taker volume pipelines, developers can query **QuantMatrix AI** for real-time trade-ready execution plans.

---

## 🚀 Key Features

### 1. Multi-Timeframe Directional Quant Signals (`/api/v1/signals/directional`)
* **3-Gate Execution Engine**: Evaluates Multi-Timeframe EMA alignment, CVD Taker volume order flow, and 8h funding rate bias (0–100 score).
* **Asymmetric Risk/Reward**: Output includes entry price, 2.0x ATR Stop-Loss, 1.5R Scale-Out Take-Profit, and 3.0R Chandelier Trailing Stop.
* **Coverage**: Top 15 liquid crypto pairs (BTC, ETH, SOL, BNB, SUI, AVAX, NEAR, LINK, APT, AAVE, ARB, RENDER, FET, TAO, PAXG).

### 2. Delta-Neutral Statistical Arbitrage (`/api/v1/arbitrage/pairs/scan`)
* **Augmented Dickey-Fuller (ADF) Unit Root Test**: Real-time stationarity test on log-spread residuals.
* **Dynamic Kalman Filter Hedge Ratio ($\beta$)**: Continuously estimates state-space dynamic beta without lagging.
* **Ornstein-Uhlenbeck (OU) Mean-Reversion**: Computes mean-reversion speed ($\theta$) and half-life ($t_{1/2}$ in hours) to filter out drifting spreads.
* **Funding Rate Carry APR**: Projects 8-hour net carry APR between the hedged Long/Short legs.

### 3. Market Regime & Flash Crash Sniper
* **`/api/v1/signals/regime`**: Classifies market into `TRENDING_BULL`, `TRENDING_BEAR`, `CHOPPY_MEAN_REVERTING`, or `HIGH_VOLATILITY_EXPANSION`.
* **`/api/v1/signals/wick-sniper`**: Scans 5m/1m liquidation cascades for extreme oversold dips ($drop \ge 3.5\%$ and $RSI \le 20$).

### 4. Institutional Microstructure & Volatility
* **Garman-Klass & Parkinson Volatility**: Extreme-value volatility estimators providing 5x higher efficiency than standard close-to-close volatility.
* **CVD Taker Buy/Sell Ratio**: Tracks institutional aggressive taker order flow.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API Status & Service Catalog |
| `GET` | `/health` | Health Check (200 OK) |
| `GET` | `/api/v1/signals/directional` | Directional Signals (Filter by `?symbol=BTCUSDT` or multi-asset scan) |
| `GET` | `/api/v1/signals/regime` | Market Regime Classification (`?symbol=BTCUSDT`) |
| `GET` | `/api/v1/signals/wick-sniper` | Active Flash-Crash Dip Opportunities |
| `GET` | `/api/v1/arbitrage/pairs/scan` | Scans 14 Institutional Cointegrated Pairs |
| `GET` | `/api/v1/arbitrage/pairs/analyze` | Deep Cointegration Analysis (`?asset_a=ETHUSDT&asset_b=BTCUSDT`) |
| `GET` | `/api/v1/analytics/microstructure` | CVD Taker Ratio, 8h Funding, and Open Interest |
| `GET` | `/api/v1/analytics/volatility` | Garman-Klass, Parkinson & Realized Volatility |

---

## 💻 Code Examples

### Python (Requests)
```python
import requests

url = "https://quantmatrix-api.onrender.com/api/v1/signals/directional"
params = {"symbol": "BTCUSDT"}
headers = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
    "X-RapidAPI-Host": "quantmatrix-ai.p.rapidapi.com"
}

response = requests.get(url, params=params, headers=headers)
print(response.json())
```

### JavaScript / Node.js (Axios)
```javascript
const axios = require('axios');

async function getArbitragePairs() {
  const options = {
    method: 'GET',
    url: 'https://quantmatrix-api.onrender.com/api/v1/arbitrage/pairs/scan',
    headers: {
      'X-RapidAPI-Key': 'YOUR_RAPIDAPI_KEY',
      'X-RapidAPI-Host': 'quantmatrix-ai.p.rapidapi.com'
    }
  };

  const response = await axios.request(options);
  console.log(response.data);
}
getArbitragePairs();
```

---

## 💰 RapidAPI Pricing Tiers

* **Basic ($0 / Month)**: 50 requests/mo (Rate limit: 1 req/sec). Free developer sandbox.
* **Pro ($29 / Month)**: 2,500 requests/mo ($0.012 per extra request).
* **Ultra ($79 / Month)**: 10,000 requests/mo ($0.008 per extra request).
* **Mega / Fund ($199 / Month)**: 50,000 requests/mo ($0.004 per extra request).

---

## 🛠️ Local Development

```bash
git clone https://github.com/z45efeylmz-prog/quantmatrix-ai-api.git
cd quantmatrix-ai-api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at: `http://localhost:8000/docs`
