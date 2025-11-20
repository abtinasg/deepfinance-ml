"""
Pytest fixtures for DeepFinance ML Engine tests.
"""

import asyncio
from typing import AsyncGenerator

import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set environment variables before imports
import os
os.environ["ML_SERVICE_API_KEY"] = "test-api-key-12345"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["DEBUG"] = "true"

from app.main import app
from app.config import get_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def api_key_headers():
    """Headers with valid API key."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 60

    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, n_days)
    })


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample returns for risk calculations."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 100)
