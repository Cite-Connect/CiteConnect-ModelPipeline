"""
Tests for Health API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthAPI:
    """Test suite for Health API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code in [200, 503]  # 503 if still starting up
        data = response.json()
        assert 'status' in data
