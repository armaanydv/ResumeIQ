"""
Tests for Phase 1: Health check, Configuration, and Root Endpoints.
"""

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import get_settings


client = TestClient(app)


def test_get_settings():
    """Verify that configuration loads cleanly with default attributes."""
    settings = get_settings()
    assert settings.APP_NAME == "ResumeIQ"
    assert settings.APP_ENV in ["development", "staging", "production"]
    assert isinstance(settings.cors_origins_list, list)
    assert len(settings.cors_origins_list) > 0


def test_root_endpoint():
    """Verify the root endpoint returns correct application metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "ResumeIQ"
    assert data["status"] == "online"
    assert "docs" in data
    assert "api_health" in data


def test_health_check_endpoint():
    """Verify the /api/health endpoint returns healthy status and timestamp."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert "environment" in data
    assert "gemini_ready" in data


if __name__ == "__main__":
    print("Running Phase 1 manual test run...")
    test_get_settings()
    print("✓ Settings test passed.")
    test_root_endpoint()
    print("✓ Root endpoint test passed.")
    test_health_check_endpoint()
    print("✓ Health check test passed.")
    print("All Phase 1 tests passed successfully!")

