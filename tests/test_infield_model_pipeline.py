import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/ml'))
from infielder_pipeline import InfielderPredictionPipeline

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