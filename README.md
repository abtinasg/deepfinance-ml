# DeepFinance ML Engine v2

A production-ready FastAPI microservice for financial price prediction using ensemble machine learning models.

## Features

### Security
- API Key authentication (X-API-Key header)
- CORS configured for specific frontend origin
- Rate limiting (100 requests/minute per API key)
- Sanitized error responses
- Global exception handling

### ML Models
- **XGBoost**: Fast gradient boosting for quick predictions
- **LSTM**: Recurrent neural network for sequential patterns
- **Transformer**: Attention-based model for complex dependencies
- Pre-trained weights loaded at startup
- Model registry with versioning

### Performance
- Redis caching for market data (5 min TTL) and predictions (1 min TTL)
- Circuit breaker for external APIs (Yahoo Finance, Finnhub)
- Connection pooling with httpx
- Async processing throughout

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check (no auth required) |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/status` | GET | Detailed model status |
| `/api/v1/models/reload` | POST | Reload model weights |
| `/api/v1/predict` | POST | Generate predictions |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/predict/{symbol}/history` | GET | Prediction history |
| `/api/v1/risk` | POST | Risk analysis |
| `/api/v1/symbols/{symbol}` | GET | Symbol info |
| `/api/v1/metrics` | GET | Prometheus metrics |
| `/api/v1/stream` | WS | Real-time predictions |

## Quick Start

### Prerequisites
- Python 3.11+
- Redis
- Docker (optional)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd deepfinance-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and set ML_SERVICE_API_KEY

# Start Redis (if not using Docker)
redis-server

# Train models (first time only)
python scripts/train_models.py --symbol SPY --days 365

# Run the application
python -m app.main
```

### Docker Deployment

```bash
# Build and run with docker-compose
cd docker
docker-compose up --build

# Or build manually
docker build -f docker/Dockerfile -t deepfinance-ml .
docker run -p 8000:8000 \
  -e ML_SERVICE_API_KEY=your-key \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  deepfinance-ml
```

### Testing

```bash
# Run tests with coverage
pytest tests/ -v --cov=app

# Run specific test file
pytest tests/test_api/test_health.py -v
```

### Linting

```bash
# Format code
black app scripts tests
isort app scripts tests

# Check linting
flake8 app scripts tests
mypy app
```

## API Usage

### Authentication

All endpoints (except `/health` and `/metrics`) require an API key:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/models
```

### Price Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "horizon": 5,
    "models": ["xgboost", "lstm", "transformer"]
  }'
```

### Risk Analysis

```bash
curl -X POST http://localhost:8000/api/v1/risk \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "period_days": 30,
    "include_benchmark": true
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/stream?api_key=your-key');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbol: 'AAPL'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Prediction:', data);
};
```

## Project Structure

```
deepfinance-ml/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── dependencies.py      # Dependency injection
│   ├── core/
│   │   ├── auth.py          # API key authentication
│   │   ├── cache.py         # Redis wrapper
│   │   ├── logging.py       # Structured logging
│   │   └── rate_limit.py    # SlowAPI rate limiting
│   ├── api/v1/
│   │   ├── health.py        # Health & monitoring
│   │   ├── predictions.py   # Prediction endpoints
│   │   ├── risk.py          # Risk analysis
│   │   └── symbols.py       # Symbol info
│   ├── schemas/
│   │   ├── common.py        # Common schemas
│   │   ├── prediction.py    # Prediction schemas
│   │   └── risk.py          # Risk schemas
│   ├── models/
│   │   ├── base_model.py    # Abstract base
│   │   ├── xgboost_model.py # XGBoost implementation
│   │   ├── lstm_model.py    # LSTM implementation
│   │   ├── transformer_model.py # Transformer
│   │   ├── registry.py      # Model registry
│   │   └── weights/         # Pre-trained weights
│   └── services/
│       ├── data_fetcher.py  # Data acquisition
│       ├── prediction_service.py
│       ├── risk_service.py
│       ├── confidence_engine.py
│       └── cache_service.py
├── scripts/
│   ├── train_models.py      # Offline training
│   └── preload_models.py    # Weight verification
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/
│   └── ci.yml               # CI/CD pipeline
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ML_SERVICE_API_KEY` | API key for authentication | **Required** |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `FRONTEND_ORIGIN` | CORS allowed origin | `https://deepin.app` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Enable debug mode | `false` |
| `FINNHUB_API_KEY` | Finnhub API key (optional) | - |
| `DATA_CACHE_TTL` | Market data cache TTL (seconds) | `300` |
| `PREDICTION_CACHE_TTL` | Prediction cache TTL (seconds) | `60` |

## Deployment

### Railway

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main

### Fly.io

```bash
# Install flyctl and login
flyctl launch
flyctl secrets set ML_SERVICE_API_KEY=your-key
flyctl deploy
```

## Performance

- Health check: ~10ms
- Symbol info: 50-200ms (cached)
- Risk analysis: 100-500ms
- Price prediction: 200-1000ms (cached)
- First prediction (cold): 2-5s

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request
