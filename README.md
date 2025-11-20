# DeepFinance ML Engine

AI-powered microservice for financial price prediction and risk analysis using ensemble machine learning models.

## Features

- **Price Prediction**: Ensemble predictions using XGBoost, LSTM, and Transformer models
- **Risk Analysis**: Comprehensive metrics including Beta, Volatility, Sharpe Ratio, VaR
- **Confidence Scoring**: Model confidence and consensus analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR

## Architecture

```
deepfinance-ml/
├── main.py              # Application entry point
├── config.py            # Configuration settings
├── api/
│   └── routes.py        # FastAPI endpoints
├── models/
│   ├── xgboost_model.py # XGBoost predictor
│   ├── lstm_model.py    # LSTM neural network
│   └── transformer_model.py # Transformer model
├── services/
│   ├── data_fetcher.py  # Yahoo Finance + Finnhub
│   ├── prediction_service.py # Model orchestration
│   ├── risk_service.py  # Risk calculations
│   └── confidence_engine.py # Ensemble & scoring
├── requirements.txt
└── Dockerfile
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Price Prediction
```
POST /api/v1/predict
{
  "symbol": "AAPL",
  "horizon": 5,
  "models": ["xgboost", "lstm", "transformer"]
}
```

### Risk Calculation
```
POST /api/v1/risk
{
  "symbol": "AAPL",
  "period_days": 60,
  "include_benchmark": true
}
```

## Installation

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

### Docker
```bash
docker build -t deepfinance-ml .
docker run -p 8000:8000 deepfinance-ml
```

### Railway Deployment
1. Connect your GitHub repository to Railway
2. Set environment variables:
   - `PORT=8000`
   - `FINNHUB_API_KEY=your_key` (optional)
3. Deploy

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8000 | Server port |
| DEBUG | false | Debug mode |
| FINNHUB_API_KEY | - | Finnhub API key (fallback) |
| PREDICTION_HORIZON | 5 | Days to predict |
| LOOKBACK_PERIOD | 60 | Historical data days |
| RISK_FREE_RATE | 0.05 | Annual risk-free rate |

## Response Example

```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "current_price": 175.50,
    "predictions": {
      "xgboost": {...},
      "lstm": {...},
      "transformer": {...}
    },
    "ensemble": {
      "final_price": 178.25,
      "price_change_percent": 1.57,
      "consensus": {
        "direction": "bullish",
        "strength": 0.85
      }
    },
    "confidence": {
      "overall": 0.72,
      "breakdown": {
        "level": "high",
        "description": "Good confidence with moderate uncertainty"
      }
    },
    "indicators": {
      "rsi": 55.3,
      "macd": 1.25,
      "sma_20": 173.80
    }
  }
}
```

## License

MIT
