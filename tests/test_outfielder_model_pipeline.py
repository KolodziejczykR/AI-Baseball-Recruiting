import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/ml'))
from backend.ml.pipeline.outfielder_pipeline import OutfielderPredictionPipeline
from backend.utils.player_types import PlayerOutfielder

@pytest.fixture(scope="module")
def pipeline():
    return OutfielderPredictionPipeline()

def test_high_performer(pipeline):
    high_performer = PlayerOutfielder(
        height=73,
        weight=185,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=92.0,
        of_velo=82.0,
        sixty_time=6.6
    )
    result = pipeline.predict(high_performer)
    assert "final_prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["confidence"], str)

def test_average_performer(pipeline):
    average_player = PlayerOutfielder(
        height=71,
        weight=170,
        primary_position="RF",
        hitting_handedness="R",
        throwing_hand="R",
        region="South",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.1
    )
    result = pipeline.predict(average_player)
    assert "final_prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["confidence"], str)

def test_low_performer(pipeline):
    lower_performer = PlayerOutfielder(
        height=69,
        weight=155,
        primary_position="LF",
        hitting_handedness="R",
        throwing_hand="R",
        region="South",
        exit_velo_max=78.0,
        of_velo=70.0,
        sixty_time=7.6
    )
    result = pipeline.predict(lower_performer)
    assert "final_prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    assert all(0.0 <= v <= 1.0 for v in result["probabilities"].values())
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    assert isinstance(result["confidence"], str)

def test_different_positions(pipeline):
    """Test predictions for different outfield positions"""
    positions = ["CF", "LF", "RF", "OF"]
    
    for position in positions:
        player = PlayerOutfielder(
            height=70,
            weight=170,
            primary_position=position,
            hitting_handedness="R",
            throwing_hand="R",
            region="South",
            exit_velo_max=85.0,
            of_velo=75.0,
            sixty_time=7.0
        )
        result = pipeline.predict(player)
        assert "final_prediction" in result
        assert "probabilities" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_different_regions(pipeline):
    """Test predictions for different player regions"""
    regions = ["West", "South", "Northeast"]
    
    for region in regions:
        player = PlayerOutfielder(
            height=70,
            weight=170,
            primary_position="CF",
            hitting_handedness="R",
            throwing_hand="R",
            region=region,
            exit_velo_max=85.0,
            of_velo=75.0,
            sixty_time=7.0
        )
        result = pipeline.predict(player)
        assert "final_prediction" in result
        assert "probabilities" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_handedness_combinations(pipeline):
    """Test different throwing and hitting handedness combinations"""
    handedness_combinations = [
        ("R", "R"), ("R", "S"),
        ("L", "R"), ("L", "S")
    ]
    
    for throw_hand, hit_hand in handedness_combinations:
        player = PlayerOutfielder(
            height=70,
            weight=170,
            primary_position="CF",
            hitting_handedness=hit_hand,
            throwing_hand=throw_hand,
            region="South",
            exit_velo_max=85.0,
            of_velo=75.0,
            sixty_time=7.0
        )
        result = pipeline.predict(player)
        assert "final_prediction" in result
        assert "probabilities" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]


def test_probability_distribution(pipeline):
    """Test that probabilities sum to approximately 1.0"""
    player = PlayerOutfielder(
        height=71,
        weight=175,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    
    result = pipeline.predict(player)
    probabilities = result["probabilities"]
    total_prob = sum(probabilities.values())
    
    # Allow for small floating point errors
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, expected 1.0"

def test_model_info(pipeline):
    """Test the get_model_info method"""
    info = pipeline.get_model_info()
    assert isinstance(info, dict)
    assert "pipeline_type" in info
    assert "stage_1" in info
    assert "stage_2" in info
    assert "required_input" in info
    assert "supported_features" in info

def test_individual_probabilities(pipeline):
    """Test that individual d1_probability and p4_probability are returned"""
    player = PlayerOutfielder(
        height=73,
        weight=185,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=92.0,
        of_velo=82.0,
        sixty_time=6.6
    )
    
    result = pipeline.predict(player)
    assert "d1_probability" in result
    assert isinstance(result["d1_probability"], float)
    assert 0.0 <= result["d1_probability"] <= 1.0
    
    # p4_probability should be present if D1 is predicted, None if Non-D1
    if result["final_prediction"] != "Non-D1":
        assert "p4_probability" in result
        assert isinstance(result["p4_probability"], float)
        assert 0.0 <= result["p4_probability"] <= 1.0
    else:
        assert result["p4_probability"] is None

def test_extreme_values_boundary(pipeline):
    """Test extreme boundary values that could break the model"""
    # Extremely high values
    extreme_high = PlayerOutfielder(
        height=84,  # Very tall
        weight=250,  # Very heavy
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=110.0,  # Unrealistically high
        of_velo=95.0,  # Unrealistically high
        sixty_time=5.8  # Extremely fast
    )
    result = pipeline.predict(extreme_high)
    assert "final_prediction" in result
    assert "probabilities" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]
    
    # Extremely low values
    extreme_low = PlayerOutfielder(
        height=60,  # Very short
        weight=120,  # Very light
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=60.0,  # Very low
        of_velo=50.0,  # Very low
        sixty_time=9.5  # Very slow
    )
    result = pipeline.predict(extreme_low)
    assert "final_prediction" in result
    assert "probabilities" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_invalid_input_handling(pipeline):
    """Test handling of invalid inputs"""
    with pytest.raises(TypeError):
        # Test with None player
        pipeline.predict(None)
    
    # Test with invalid position - should still work but may produce unexpected results
    invalid_player = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="INVALID",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    result = pipeline.predict(invalid_player)
    assert "final_prediction" in result

def test_missing_required_attributes(pipeline):
    """Test behavior when player object is missing required attributes"""
    # Create a minimal player object that might be missing attributes
    class IncompletePlayer:
        def __init__(self):
            self.height = 70
            self.weight = 170
            # Missing other required attributes
    
    with pytest.raises(TypeError):
        pipeline.predict(IncompletePlayer())

def test_unsupported_categorical_values(pipeline):
    """Test handling of unsupported categorical values"""
    # Test with unsupported region - pipeline handles gracefully
    unsupported_region = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="Antarctica",  # Unsupported region
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    result = pipeline.predict(unsupported_region)
    assert "final_prediction" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_negative_values(pipeline):
    """Test handling of negative values"""
    # Pipeline accepts negative values but may produce unexpected results
    negative_player = PlayerOutfielder(
        height=-70,  # Negative height
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    result = pipeline.predict(negative_player)
    assert "final_prediction" in result
    assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_zero_values(pipeline):
    """Test handling of zero values"""
    zero_player = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=0.0,  # Zero exit velocity
        of_velo=0.0,  # Zero outfield velocity
        sixty_time=7.0
    )
    result = pipeline.predict(zero_player)
    assert "final_prediction" in result
    assert "probabilities" in result

def test_nan_infinity_values(pipeline):
    """Test handling of NaN and infinity values"""
    # Pipeline may handle NaN values, let's test if it produces a result
    nan_player = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=float('nan'),  # NaN value
        of_velo=75.0,
        sixty_time=7.0
    )
    # This may raise an exception or return a result - both are acceptable
    try:
        result = pipeline.predict(nan_player)
        assert "final_prediction" in result
    except (ValueError, TypeError, Exception):
        # It's acceptable if the pipeline raises an exception for NaN values
        pass

def test_prediction_consistency(pipeline):
    """Test that identical inputs produce identical outputs"""
    player1 = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    
    player2 = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    
    result1 = pipeline.predict(player1)
    result2 = pipeline.predict(player2)
    
    assert result1["final_prediction"] == result2["final_prediction"]
    assert result1["d1_probability"] == result2["d1_probability"]
    if result1["p4_probability"] is not None:
        assert result1["p4_probability"] == result2["p4_probability"]

def test_all_supported_positions_exist(pipeline):
    """Test that all positions mentioned in comments are actually supported"""
    supported_positions = ["CF", "LF", "RF", "OF"]  # Based on comment in test_different_positions
    
    for position in supported_positions:
        player = PlayerOutfielder(
            height=70,
            weight=170,
            primary_position=position,
            hitting_handedness="R",
            throwing_hand="R",
            region="West",
            exit_velo_max=85.0,
            of_velo=75.0,
            sixty_time=7.0
        )
        result = pipeline.predict(player)
        assert "final_prediction" in result
        assert result["final_prediction"] in ["Non-D1", "Non-P4 D1", "Power 4 D1"]

def test_confidence_levels_validity(pipeline):
    """Test that confidence levels are valid strings"""
    player = PlayerOutfielder(
        height=70,
        weight=170,
        primary_position="CF",
        hitting_handedness="R",
        throwing_hand="R",
        region="West",
        exit_velo_max=85.0,
        of_velo=75.0,
        sixty_time=7.0
    )
    
    result = pipeline.predict(player)
    assert isinstance(result["confidence"], str)
    # Assuming confidence levels should be one of these
    valid_confidence_levels = ["low", "medium", "high", "very_high", "very_low"]
    assert result["confidence"].lower() in valid_confidence_levels 