"""
Tests for health and monitoring endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_health_check(test_client: TestClient):
    """Test health endpoint returns correct structure."""
    response = test_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert "service" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "healthy"


def test_root_endpoint(test_client: TestClient):
    """Test root endpoint returns service info."""
    response = test_client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "service" in data
    assert "version" in data


def test_models_endpoint_requires_auth(test_client: TestClient):
    """Test models endpoint requires API key."""
    response = test_client.get("/api/v1/models")

    assert response.status_code == 401


def test_models_endpoint_with_auth(test_client: TestClient, api_key_headers: dict):
    """Test models endpoint with valid API key."""
    response = test_client.get("/api/v1/models", headers=api_key_headers)

    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "ensemble_weights" in data


def test_invalid_api_key(test_client: TestClient):
    """Test endpoint rejects invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    response = test_client.get("/api/v1/models", headers=headers)

    assert response.status_code == 403
