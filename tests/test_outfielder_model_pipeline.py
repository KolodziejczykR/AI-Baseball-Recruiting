import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/ml'))
from outfielder_pipeline import OutfielderPredictionPipeline

@pytest.fixture(scope="module")
def pipeline():
    return OutfielderPredictionPipeline(models_dir=os.path.join(os.path.dirname(__file__), '../backend/ml/models'))

def test_high_performer(pipeline):
    """Test prediction for a high-performing outfielder"""
    high_performer = {
        "age": 17.5,
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
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "CF"
    }
    result = pipeline.predict(high_performer)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_average_performer(pipeline):
    """Test prediction for an average outfielder"""
    average_player = {
        "age": 17.0,
        "height": 71.0,
        "weight": 170.0,
        "hand_speed_max": 20.5,
        "bat_speed_max": 72.0,
        "rot_acc_max": 16.0,
        "sixty_time": 7.1,
        "thirty_time": 3.4,
        "ten_yard_time": 1.8,
        "run_speed_max": 21.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 290.0,
        "sweet_spot_p": 0.68,
        "of_velo": 75.0,
        "player_state": "TX",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "South",
        "primary_position": "RF"
    }
    result = pipeline.predict(average_player)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_low_performer(pipeline):
    """Test prediction for a lower-performing outfielder"""
    lower_performer = {
        "age": 16.5,
        "height": 69.0,
        "weight": 155.0,
        "hand_speed_max": 18.5,
        "bat_speed_max": 67.0,
        "rot_acc_max": 13.0,
        "sixty_time": 7.6,
        "thirty_time": 3.7,
        "ten_yard_time": 1.9,
        "run_speed_max": 19.0,
        "exit_velo_max": 78.0,
        "exit_velo_avg": 70.0,
        "distance_max": 260.0,
        "sweet_spot_p": 0.58,
        "of_velo": 70.0,
        "player_state": "FL",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "Southeast",
        "primary_position": "LF"
    }
    result = pipeline.predict(lower_performer)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_minimal_input(pipeline):
    """Test prediction with minimal input data"""
    minimal_data = {
        "age": 17.0,
        "of_velo": 78.0,
        "exit_velo_max": 88.0
    }
    result = pipeline.predict(minimal_data)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_feature_info(pipeline):
    """Test the get_feature_info method"""
    info = pipeline.get_feature_info()
    assert "numerical_features" in info
    assert "categorical_features" in info
    assert "descriptions" in info
    assert isinstance(info["numerical_features"], list)
    assert isinstance(info["categorical_features"], list)
    assert isinstance(info["descriptions"], dict)

def test_edge_cases(pipeline):
    """Test edge cases and boundary conditions"""
    
    # Test with very high values
    high_values = {
        "age": 20.0,
        "height": 78.0,
        "weight": 220.0,
        "hand_speed_max": 28.0,
        "bat_speed_max": 85.0,
        "rot_acc_max": 22.0,
        "sixty_time": 6.2,
        "thirty_time": 2.9,
        "ten_yard_time": 1.4,
        "run_speed_max": 24.0,
        "exit_velo_max": 95.0,
        "exit_velo_avg": 88.0,
        "distance_max": 380.0,
        "sweet_spot_p": 0.92,
        "of_velo": 88.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "CF"
    }
    result = pipeline.predict(high_values)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
    
    # Test with very low values
    low_values = {
        "age": 15.0,
        "height": 62.0,
        "weight": 125.0,
        "hand_speed_max": 16.0,
        "bat_speed_max": 62.0,
        "rot_acc_max": 9.0,
        "sixty_time": 8.2,
        "thirty_time": 4.0,
        "ten_yard_time": 2.2,
        "run_speed_max": 16.0,
        "exit_velo_max": 72.0,
        "exit_velo_avg": 67.0,
        "distance_max": 220.0,
        "sweet_spot_p": 0.48,
        "of_velo": 65.0,
        "player_state": "FL",
        "throwing_hand": "L",
        "hitting_handedness": "L",
        "player_region": "Southeast",
        "primary_position": "LF"
    }
    result = pipeline.predict(low_values)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_different_positions(pipeline):
    """Test predictions for different outfield positions"""
    positions = ["CF", "LF", "RF", "OF"]
    
    base_data = {
        "age": 17.0,
        "height": 71.0,
        "weight": 175.0,
        "hand_speed_max": 20.5,
        "bat_speed_max": 72.0,
        "rot_acc_max": 16.0,
        "sixty_time": 7.0,
        "thirty_time": 3.4,
        "ten_yard_time": 1.8,
        "run_speed_max": 21.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 290.0,
        "sweet_spot_p": 0.68,
        "of_velo": 75.0,
        "player_state": "TX",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "South"
    }
    
    for position in positions:
        test_data = base_data.copy()
        test_data["primary_position"] = position
        result = pipeline.predict(test_data)
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_different_regions(pipeline):
    """Test predictions for different player regions"""
    regions = ["West", "South", "Midwest", "Northeast", "Southeast"]
    
    base_data = {
        "age": 17.0,
        "height": 71.0,
        "weight": 175.0,
        "hand_speed_max": 20.5,
        "bat_speed_max": 72.0,
        "rot_acc_max": 16.0,
        "sixty_time": 7.0,
        "thirty_time": 3.4,
        "ten_yard_time": 1.8,
        "run_speed_max": 21.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 290.0,
        "sweet_spot_p": 0.68,
        "of_velo": 75.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "primary_position": "CF"
    }
    
    for region in regions:
        test_data = base_data.copy()
        test_data["player_region"] = region
        result = pipeline.predict(test_data)
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_handedness_combinations(pipeline):
    """Test different throwing and hitting handedness combinations"""
    handedness_combinations = [
        ("R", "R"), ("R", "L"), ("R", "S"),
        ("L", "R"), ("L", "L"), ("L", "S")
    ]
    
    base_data = {
        "age": 17.0,
        "height": 71.0,
        "weight": 175.0,
        "hand_speed_max": 20.5,
        "bat_speed_max": 72.0,
        "rot_acc_max": 16.0,
        "sixty_time": 7.0,
        "thirty_time": 3.4,
        "ten_yard_time": 1.8,
        "run_speed_max": 21.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 290.0,
        "sweet_spot_p": 0.68,
        "of_velo": 75.0,
        "player_state": "CA",
        "player_region": "West",
        "primary_position": "CF"
    }
    
    for throw_hand, hit_hand in handedness_combinations:
        test_data = base_data.copy()
        test_data["throwing_hand"] = throw_hand
        test_data["hitting_handedness"] = hit_hand
        result = pipeline.predict(test_data)
        assert "prediction" in result
        assert "probabilities" in result
        assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_missing_values_handling(pipeline):
    """Test how the pipeline handles missing values"""
    # Test with only essential features
    minimal_data = {
        "exit_velo_max": 88.0,
        "of_velo": 78.0,
        "primary_position": "CF"
    }
    result = pipeline.predict(minimal_data)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    
    # Test with no numerical features
    categorical_only = {
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "CF"
    }
    result = pipeline.predict(categorical_only)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_probability_distribution(pipeline):
    """Test that probabilities sum to approximately 1.0"""
    test_data = {
        "age": 17.0,
        "height": 71.0,
        "weight": 175.0,
        "hand_speed_max": 20.5,
        "bat_speed_max": 72.0,
        "rot_acc_max": 16.0,
        "sixty_time": 7.0,
        "thirty_time": 3.4,
        "ten_yard_time": 1.8,
        "run_speed_max": 21.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 290.0,
        "sweet_spot_p": 0.68,
        "of_velo": 75.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "CF"
    }
    
    result = pipeline.predict(test_data)
    probabilities = result["probabilities"]
    total_prob = sum(probabilities.values())
    
    # Allow for small floating point errors
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, expected 1.0"

def test_required_features(pipeline):
    """Test the get_required_features method"""
    required_features = pipeline.get_required_features()
    assert isinstance(required_features, list)
    assert len(required_features) > 0
    
    # Check that some expected features are present
    expected_features = ["exit_velo_max", "of_velo", "primary_position"]
    for feature in expected_features:
        assert feature in required_features

def test_speed_focused_player(pipeline):
    """Test a player with exceptional speed metrics"""
    speed_player = {
        "age": 17.0,
        "height": 70.0,
        "weight": 160.0,
        "hand_speed_max": 21.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 6.4,  # Very fast
        "thirty_time": 3.0,  # Very fast
        "ten_yard_time": 1.5,  # Very fast
        "run_speed_max": 24.0,  # Very fast
        "exit_velo_max": 82.0,
        "exit_velo_avg": 72.0,
        "distance_max": 280.0,
        "sweet_spot_p": 0.65,
        "of_velo": 80.0,
        "player_state": "FL",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "Southeast",
        "primary_position": "CF"
    }
    result = pipeline.predict(speed_player)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_power_focused_player(pipeline):
    """Test a player with exceptional power metrics"""
    power_player = {
        "age": 17.5,
        "height": 74.0,
        "weight": 200.0,
        "hand_speed_max": 24.0,
        "bat_speed_max": 80.0,
        "rot_acc_max": 18.0,
        "sixty_time": 7.2,  # Average speed
        "thirty_time": 3.5,
        "ten_yard_time": 1.9,
        "run_speed_max": 20.0,
        "exit_velo_max": 95.0,  # Very high
        "exit_velo_avg": 85.0,  # Very high
        "distance_max": 360.0,  # Very high
        "sweet_spot_p": 0.75,
        "of_velo": 75.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "RF"
    }
    result = pipeline.predict(power_player)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_balanced_player(pipeline):
    """Test a player with balanced metrics"""
    balanced_player = {
        "age": 17.0,
        "height": 72.0,
        "weight": 180.0,
        "hand_speed_max": 22.0,
        "bat_speed_max": 75.0,
        "rot_acc_max": 17.0,
        "sixty_time": 6.8,
        "thirty_time": 3.2,
        "ten_yard_time": 1.7,
        "run_speed_max": 22.0,
        "exit_velo_max": 88.0,
        "exit_velo_avg": 78.0,
        "distance_max": 320.0,
        "sweet_spot_p": 0.72,
        "of_velo": 78.0,
        "player_state": "TX",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "South",
        "primary_position": "CF"
    }
    result = pipeline.predict(balanced_player)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"] 