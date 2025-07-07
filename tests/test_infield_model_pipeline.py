import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/ml'))
from backend.ml.pipeline.infielder_pipeline import InfielderPredictionPipeline

@pytest.fixture(scope="module")
def pipeline():
    return InfielderPredictionPipeline(models_dir=os.path.join(os.path.dirname(__file__), '../backend/ml/models'))

def test_high_performer(pipeline):
    high_performer = {
        "age": 17.5,
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
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "SS"
    }
    result = pipeline.predict(high_performer)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_average_performer(pipeline):
    average_player = {
        "age": 17.0,
        "height": 70.0,
        "weight": 165.0,
        "hand_speed_max": 20.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 7.2,
        "thirty_time": 3.5,
        "ten_yard_time": 1.8,
        "run_speed_max": 20.0,
        "exit_velo_max": 82.0,
        "exit_velo_avg": 72.0,
        "distance_max": 280.0,
        "sweet_spot_p": 0.65,
        "inf_velo": 72.0,
        "player_state": "TX",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "South",
        "primary_position": "2B"
    }
    result = pipeline.predict(average_player)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_low_performer(pipeline):
    lower_performer = {
        "age": 16.5,
        "height": 68.0,
        "weight": 150.0,
        "hand_speed_max": 18.0,
        "bat_speed_max": 65.0,
        "rot_acc_max": 12.0,
        "sixty_time": 7.8,
        "thirty_time": 3.8,
        "ten_yard_time": 2.0,
        "run_speed_max": 18.0,
        "exit_velo_max": 75.0,
        "exit_velo_avg": 68.0,
        "distance_max": 250.0,
        "sweet_spot_p": 0.55,
        "inf_velo": 68.0,
        "player_state": "FL",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "Southeast",
        "primary_position": "3B"
    }
    result = pipeline.predict(lower_performer)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_minimal_input(pipeline):
    minimal_data = {
        "age": 17.0,
        "inf_velo": 75.0,
        "exit_velo_max": 85.0
    }
    result = pipeline.predict(minimal_data)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_feature_info(pipeline):
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
        "height": 80.0,
        "weight": 250.0,
        "hand_speed_max": 30.0,
        "bat_speed_max": 90.0,
        "rot_acc_max": 25.0,
        "sixty_time": 6.0,
        "thirty_time": 2.8,
        "ten_yard_time": 1.5,
        "run_speed_max": 25.0,
        "exit_velo_max": 100.0,
        "exit_velo_avg": 90.0,
        "distance_max": 400.0,
        "sweet_spot_p": 0.95,
        "inf_velo": 90.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "SS"
    }
    result = pipeline.predict(high_values)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
    
    # Test with very low values
    low_values = {
        "age": 15.0,
        "height": 60.0,
        "weight": 120.0,
        "hand_speed_max": 15.0,
        "bat_speed_max": 60.0,
        "rot_acc_max": 8.0,
        "sixty_time": 8.5,
        "thirty_time": 4.2,
        "ten_yard_time": 2.5,
        "run_speed_max": 15.0,
        "exit_velo_max": 70.0,
        "exit_velo_avg": 65.0,
        "distance_max": 200.0,
        "sweet_spot_p": 0.45,
        "inf_velo": 60.0,
        "player_state": "FL",
        "throwing_hand": "L",
        "hitting_handedness": "L",
        "player_region": "Southeast",
        "primary_position": "2B"
    }
    result = pipeline.predict(low_values)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

def test_different_positions(pipeline):
    """Test predictions for different infield positions"""
    positions = ["SS", "2B", "3B", "1B"]
    
    base_data = {
        "age": 17.0,
        "height": 70.0,
        "weight": 170.0,
        "hand_speed_max": 20.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 7.0,
        "thirty_time": 3.5,
        "ten_yard_time": 1.8,
        "run_speed_max": 20.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 300.0,
        "sweet_spot_p": 0.7,
        "inf_velo": 75.0,
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
        "height": 70.0,
        "weight": 170.0,
        "hand_speed_max": 20.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 7.0,
        "thirty_time": 3.5,
        "ten_yard_time": 1.8,
        "run_speed_max": 20.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 300.0,
        "sweet_spot_p": 0.7,
        "inf_velo": 75.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "primary_position": "SS"
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
        "height": 70.0,
        "weight": 170.0,
        "hand_speed_max": 20.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 7.0,
        "thirty_time": 3.5,
        "ten_yard_time": 1.8,
        "run_speed_max": 20.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 300.0,
        "sweet_spot_p": 0.7,
        "inf_velo": 75.0,
        "player_state": "CA",
        "player_region": "West",
        "primary_position": "SS"
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
        "exit_velo_max": 85.0,
        "inf_velo": 75.0,
        "primary_position": "SS"
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
        "primary_position": "SS"
    }
    result = pipeline.predict(categorical_only)
    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Non D1", "Non P4 D1", "Power 4 D1"]

def test_probability_distribution(pipeline):
    """Test that probabilities sum to approximately 1.0"""
    test_data = {
        "age": 17.0,
        "height": 70.0,
        "weight": 170.0,
        "hand_speed_max": 20.0,
        "bat_speed_max": 70.0,
        "rot_acc_max": 15.0,
        "sixty_time": 7.0,
        "thirty_time": 3.5,
        "ten_yard_time": 1.8,
        "run_speed_max": 20.0,
        "exit_velo_max": 85.0,
        "exit_velo_avg": 75.0,
        "distance_max": 300.0,
        "sweet_spot_p": 0.7,
        "inf_velo": 75.0,
        "player_state": "CA",
        "throwing_hand": "R",
        "hitting_handedness": "R",
        "player_region": "West",
        "primary_position": "SS"
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
    expected_features = ["exit_velo_max", "inf_velo", "primary_position"]
    for feature in expected_features:
        assert feature in required_features 