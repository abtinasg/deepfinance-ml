"""
Tests for prediction endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_predict_requires_auth(test_client: TestClient):
    """Test predict endpoint requires API key."""
    response = test_client.post(
        "/api/v1/predict",
        json={"symbol": "AAPL", "horizon": 5}
    )

    assert response.status_code == 401


def test_predict_validates_symbol(test_client: TestClient, api_key_headers: dict):
    """Test predict endpoint validates symbol format."""
    response = test_client.post(
        "/api/v1/predict",
        json={"symbol": "", "horizon": 5},
        headers=api_key_headers
    )

    assert response.status_code == 422


def test_predict_validates_horizon(test_client: TestClient, api_key_headers: dict):
    """Test predict endpoint validates horizon range."""
    # Horizon too large
    response = test_client.post(
        "/api/v1/predict",
        json={"symbol": "AAPL", "horizon": 100},
        headers=api_key_headers
    )

    assert response.status_code == 422


def test_predict_validates_models(test_client: TestClient, api_key_headers: dict):
    """Test predict endpoint validates model names."""
    response = test_client.post(
        "/api/v1/predict",
        json={"symbol": "AAPL", "horizon": 5, "models": ["invalid_model"]},
        headers=api_key_headers
    )

    assert response.status_code == 422


def test_batch_predict_validates_symbols(test_client: TestClient, api_key_headers: dict):
    """Test batch predict validates symbol list."""
    # Empty list
    response = test_client.post(
        "/api/v1/predict/batch",
        json={"symbols": []},
        headers=api_key_headers
    )

    assert response.status_code == 422

    # Too many symbols
    response = test_client.post(
        "/api/v1/predict/batch",
        json={"symbols": ["A"] * 100},
        headers=api_key_headers
    )

    assert response.status_code == 422
