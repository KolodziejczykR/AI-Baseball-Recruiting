import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Infielder API Tests
def test_infielder_predict_high_performer():
    """Test infielder prediction with high-performing player data"""
    data = {
        "height": 72.0,
        "weight": 180.0,
        "hand_speed_max": 22.5,
        "bat_speed_max": 75.0,
        "rot_acc_max": 18.0,
        "sixty_time": 6.8,
        "thirty_time": 3.2,
        "ten_yard_time": 1.7,
        "run_speed_max": 22.0,
        "exit_velo_max": 88.0,
        "exit_velo_avg": 78.0,
        "distance_max": 320.0,
        "sweet_spot_p": 0.75,
        "inf_velo": 78.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "SS"
    }
    response = client.post("/infielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert "confidence" in result
    assert "model_chain" in result
    assert "d1_probability" in result
    assert "final_category" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())

def test_infielder_predict_minimal_data():
    """Test infielder prediction with minimal data"""
    data = {
        "height": 70,
        "weight": 170,
        "primary_position": "SS",
        "hitting_handedness": "R",
        "throwing_hand": "R",
        "player_region": "West",
        "exit_velo_max": 85.0,
        "inf_velo": 75.0,
        "sixty_time": 7.0
    }
    response = client.post("/infielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_infielder_features_endpoint():
    """Test infielder features endpoint"""
    response = client.get("/infielder/features")
    assert response.status_code == 200
    result = response.json()
    assert "required_features" in result
    assert "feature_info" in result
    assert isinstance(result["required_features"], list)
    assert isinstance(result["feature_info"], dict)

def test_infielder_health_endpoint():
    """Test infielder health endpoint"""
    response = client.get("/infielder/health")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert "pipeline_loaded" in result
    assert isinstance(result["status"], str)
    assert isinstance(result["pipeline_loaded"], bool)

def test_infielder_example_endpoint():
    """Test infielder example endpoint"""
    response = client.get("/infielder/example")
    assert response.status_code == 200
    result = response.json()
    assert "example_input" in result
    assert "description" in result
    assert isinstance(result["example_input"], dict)
    assert isinstance(result["description"], str)

def test_infielder_different_positions():
    """Test infielder predictions for different positions"""
    positions = ["SS", "2B", "3B", "1B"]
    base_data = {
        "exit_velo_max": 85.0,
        "inf_velo": 75.0,
        "sixty_time": 7.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "height": 72.0,
        "weight": 180.0
    }
    
    for position in positions:
        data = base_data.copy()
        data["primary_position"] = position
        response = client.post("/infielder/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "final_prediction" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

# Outfielder API Tests
def test_outfielder_predict_high_performer():
    """Test outfielder prediction with high-performing player data"""
    data = {
        "height": 73.0,
        "weight": 185.0,
        "hand_speed_max": 23.0,
        "bat_speed_max": 78.0,
        "rot_acc_max": 19.0,
        "sixty_time": 6.6,
        "thirty_time": 3.1,
        "ten_yard_time": 1.6,
        "run_speed_max": 23.5,
        "exit_velo_max": 92.0,
        "exit_velo_avg": 82.0,
        "distance_max": 350.0,
        "sweet_spot_p": 0.78,
        "of_velo": 82.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "CF"
    }
    response = client.post("/outfielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert "confidence" in result
    assert "model_chain" in result
    assert "d1_probability" in result
    assert "final_category" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())

def test_outfielder_predict_minimal_data():
    """Test outfielder prediction with minimal data"""
    data = {
        "height": 70,
        "weight": 170,
        "primary_position": "CF",
        "hitting_handedness": "R",
        "throwing_hand": "R",
        "player_region": "West",
        "exit_velo_max": 88.0,
        "of_velo": 78.0,
        "sixty_time": 7.0
    }
    response = client.post("/outfielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_outfielder_features_endpoint():
    """Test outfielder features endpoint"""
    response = client.get("/outfielder/features")
    assert response.status_code == 200
    result = response.json()
    assert "required_features" in result
    assert "feature_info" in result
    assert isinstance(result["required_features"], list)
    assert isinstance(result["feature_info"], dict)

def test_outfielder_health_endpoint():
    """Test outfielder health endpoint"""
    response = client.get("/outfielder/health")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert "pipeline_loaded" in result
    assert isinstance(result["status"], str)
    assert isinstance(result["pipeline_loaded"], bool)

def test_outfielder_example_endpoint():
    """Test outfielder example endpoint"""
    response = client.get("/outfielder/example")
    assert response.status_code == 200
    result = response.json()
    assert "example_input" in result
    assert "description" in result
    assert isinstance(result["example_input"], dict)
    assert isinstance(result["description"], str)

def test_outfielder_different_positions():
    """Test outfielder predictions for different positions"""
    positions = ["CF", "LF", "RF", "OF"]
    base_data = {
        "height": 71.0,
        "weight": 175.0,
        "sixty_time": 7.0,
        "exit_velo_max": 88.0,
        "of_velo": 78.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West"
    }
    
    for position in positions:
        data = base_data.copy()
        data["primary_position"] = position
        response = client.post("/outfielder/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "final_prediction" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_outfielder_speed_focused_player():
    """Test outfielder prediction for speed-focused player"""
    data = {
        "height": 70.0,
        "weight": 160.0,
        "sixty_time": 6.4,  # Very fast
        "thirty_time": 3.0,  # Very fast
        "ten_yard_time": 1.5,  # Very fast
        "run_speed_max": 24.0,  # Very fast
        "exit_velo_max": 82.0,
        "of_velo": 80.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "Southeast",
        "primary_position": "CF"
    }
    response = client.post("/outfielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_outfielder_power_focused_player():
    """Test outfielder prediction for power-focused player"""
    data = {
        "height": 74.0,
        "weight": 200.0,
        "sixty_time": 7.2,  # Average speed
        "exit_velo_max": 95.0,  # Very high
        "exit_velo_avg": 85.0,  # Very high
        "distance_max": 360.0,  # Very high
        "of_velo": 75.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "RF"
    }
    response = client.post("/outfielder/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

# Error handling tests
def test_infielder_invalid_data():
    """Test infielder prediction with invalid data"""
    data = {
        "exit_velo_max": "invalid",  # Should be a number
        "inf_velo": 75.0,
        "primary_position": "SS"
    }
    response = client.post("/infielder/predict", json=data)
    # Should either return 422 (validation error) or 400 (processing error)
    assert response.status_code in [400, 422]

def test_outfielder_invalid_data():
    """Test outfielder prediction with invalid data"""
    data = {
        "exit_velo_max": "invalid",  # Should be a number
        "of_velo": 78.0,
        "primary_position": "CF"
    }
    response = client.post("/outfielder/predict", json=data)
    # Should either return 422 (validation error) or 400 (processing error)
    assert response.status_code in [400, 422]

def test_infielder_empty_data():
    """Test infielder prediction with empty data"""
    data = {}
    response = client.post("/infielder/predict", json=data)
    assert response.status_code == 400  # Should return error for missing required fields

def test_outfielder_empty_data():
    """Test outfielder prediction with empty data"""
    data = {}
    response = client.post("/outfielder/predict", json=data)
    assert response.status_code == 400  # Should return error for missing required fields

# Catcher API Tests
def test_catcher_predict_high_performer():
    """Test catcher prediction with high-performing player data"""
    data = {
        "height": 72.0,
        "weight": 185.0,
        "hand_speed_max": 22.5,
        "bat_speed_max": 75.0,
        "rot_acc_max": 18.0,
        "sixty_time": 6.8,
        "thirty_time": 3.2,
        "ten_yard_time": 1.7,
        "run_speed_max": 22.0,
        "exit_velo_max": 88.0,
        "exit_velo_avg": 78.0,
        "distance_max": 320.0,
        "sweet_spot_p": 0.75,
        "c_velo": 78.0,
        "pop_time": 1.8,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "C"
    }
    response = client.post("/catcher/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert "confidence" in result
    assert "model_chain" in result
    assert "d1_probability" in result
    assert "final_category" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())

def test_catcher_predict_minimal_data():
    """Test catcher prediction with minimal data"""
    data = {
        "exit_velo_max": 85.0,
        "c_velo": 75.0,
        "primary_position": "C"
    }
    response = client.post("/catcher/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert "probabilities" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_catcher_features_endpoint():
    """Test catcher features endpoint"""
    response = client.get("/catcher/features")
    assert response.status_code == 200
    result = response.json()
    assert "required_features" in result
    assert "feature_info" in result
    assert isinstance(result["required_features"], list)
    assert isinstance(result["feature_info"], dict)

def test_catcher_health_endpoint():
    """Test catcher health endpoint"""
    response = client.get("/catcher/health")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert "pipeline_loaded" in result
    assert isinstance(result["status"], str)
    assert isinstance(result["pipeline_loaded"], bool)

def test_catcher_example_endpoint():
    """Test catcher example endpoint"""
    response = client.get("/catcher/example")
    assert response.status_code == 200
    result = response.json()
    assert "example_input" in result
    assert "description" in result
    assert isinstance(result["example_input"], dict)
    assert isinstance(result["description"], str)

def test_catcher_specific_features():
    """Test catcher prediction with catcher-specific features"""
    data = {
        "height": 70.0,
        "weight": 175.0,
        "exit_velo_max": 85.0,
        "c_velo": 75.0,
        "pop_time": 2.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "C"
    }
    response = client.post("/catcher/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "final_prediction" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_catcher_different_pop_times():
    """Test catcher predictions with different pop times"""
    pop_times = [1.8, 2.0, 2.2, 2.5]
    base_data = {
        "exit_velo_max": 85.0,
        "c_velo": 75.0,
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "C"
    }
    
    for pop_time in pop_times:
        data = base_data.copy()
        data["pop_time"] = pop_time
        response = client.post("/catcher/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "final_prediction" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_catcher_invalid_data():
    """Test catcher prediction with invalid data"""
    data = {
        "exit_velo_max": "invalid",  # Should be a number
        "c_velo": 75.0,
        "primary_position": "C"
    }
    response = client.post("/catcher/predict", json=data)
    # Should either return 422 (validation error) or 400 (processing error)
    assert response.status_code in [400, 422]

def test_catcher_empty_data():
    """Test catcher prediction with empty data"""
    data = {}
    response = client.post("/catcher/predict", json=data)
    assert response.status_code == 400  # Should return error for missing required fields 