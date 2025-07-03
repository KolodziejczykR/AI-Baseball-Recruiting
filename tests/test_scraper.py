import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_scrape_vanderbilt_roster():
    payload = {
        "team": "Vanderbilt",
        "position": "INF",  # Example position from screenshot
        "class_year": 2025
    }
    response = client.post("/scrape/team-info", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "current_roster" in data
    assert "recruiting_class" in data
    assert isinstance(data["current_roster"].get("players"), list)
    # Check at least one player is returned (if the roster is up)
    assert data["current_roster"]["count"] == len(data["current_roster"]["players"])
    if data["current_roster"]["players"]:
        player = data["current_roster"]["players"][0]
        assert "name" in player
        assert "position" in player 