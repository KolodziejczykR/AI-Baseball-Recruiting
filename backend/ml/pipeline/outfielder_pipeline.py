import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path to enable backend imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.player_types import PlayerOutfielder
from backend.ml.models.models_of.models_d1_or_not_of.version_08072025 import prediction_pipeline as d1_pipeline
from backend.ml.models.models_of.models_p4_or_not_of.version_08072025 import prediction_pipeline as p4_pipeline

class OutfielderPredictionPipeline:
    def __init__(self):
        """
        Initialize the outfielder prediction pipeline using the latest production models.
        """
        print("Successfully loaded outfielder prediction pipeline functions")
    
    def predict(self, player: PlayerOutfielder) -> dict:
        """
        Run the complete two-stage outfielder prediction pipeline.
        
        Args:
            player: PlayerOutfielder object containing player statistics
        
        Returns:
            Dictionary with comprehensive prediction results
        """
        if not isinstance(player, PlayerOutfielder):
            raise TypeError("Input must be a PlayerOutfielder object")
        
        try:
            # Convert player to dictionary format expected by models
            player_data = player.to_dict()
            
            # Stage 1: Predict D1 vs Non-D1
            d1_result = d1_pipeline.predict_outfielder_d1_probability(player_data)
            
            # If predicted as Non-D1, return early
            if d1_result['d1_prediction'] == 0:
                return {
                    'final_prediction': 'Non-D1',
                    'final_category': 0,  # Non-D1
                    'probabilities': {
                        'non_d1': float(1 - d1_result['d1_probability']),
                        'd1_total': float(d1_result['d1_probability']),
                        'non_p4_d1': 0.0,
                        'p4_d1': 0.0
                    },
                    'confidence': d1_result['confidence'],
                    'model_chain': 'D1_only',
                    'd1_details': d1_result,
                    'p4_details': None
                }
            
            # Stage 2: For D1-predicted players, predict P4 vs Non-P4 D1
            p4_result = p4_pipeline.predict_outfielder_p4_probability(player_data)
            
            # Calculate final probabilities
            d1_prob = d1_result['d1_probability']
            p4_conditional_prob = p4_result['p4_probability']
            
            # Final probabilities (conditional on D1 prediction)
            non_d1_prob = 1 - d1_prob
            non_p4_d1_prob = d1_prob * (1 - p4_conditional_prob)
            p4_d1_prob = d1_prob * p4_conditional_prob
            
            # Determine final prediction
            if p4_result['p4_prediction'] == 1:
                final_prediction = 'Power 4 D1'
                final_category = 2  # P4 D1
            else:
                final_prediction = 'Non-P4 D1'
                final_category = 1  # Non-P4 D1
            
            # Overall confidence is minimum of both stages
            overall_confidence = min(d1_result['confidence'], p4_result['confidence'])
            
            return {
                'final_prediction': final_prediction,
                'final_category': final_category,
                'probabilities': {
                    'non_d1': float(non_d1_prob),
                    'd1_total': float(d1_prob),
                    'non_p4_d1': float(non_p4_d1_prob),
                    'p4_d1': float(p4_d1_prob)
                },
                'confidence': overall_confidence,
                'model_chain': 'D1_then_P4',
                'd1_details': d1_result,
                'p4_details': p4_result,
                'player_info': {
                    'player_type': player.get_player_type(),
                    'region': player.region,
                    'elite_candidate_d1': d1_result.get('is_elite_candidate', False),
                    'elite_candidate_p4': p4_result.get('is_elite_candidate', False)
                }
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'final_prediction': None,
                'final_category': None,
                'probabilities': None,
                'confidence': None
            }
    
    def get_model_info(self) -> dict:
        """Get information about the loaded models"""
        return {
            'pipeline_type': 'Two-stage hierarchical ensemble',
            'stage_1': {
                'target': 'D1 vs Non-D1 classification',
                'architecture': 'Elite Detection + XGBoost + LightGBM + DNN + SVM'
            },
            'stage_2': {
                'target': 'P4 vs Non-P4 D1 classification',
                'architecture': 'Elite Detection + XGBoost + LightGBM + MLP + SVM'
            },
            'required_input': 'PlayerOutfielder object',
            'supported_features': [
                'height', 'weight', 'sixty_time', 'exit_velo_max', 
                'of_velo', 'player_region', 'throwing_hand', 'hitting_handedness'
            ]
        }

def predict_outfielder_college_level(player: PlayerOutfielder) -> dict:
    """
    Convenience function for complete outfielder college level prediction.
    
    Args:
        player: PlayerOutfielder object
    
    Returns:
        Complete prediction results
    """
    pipeline = OutfielderPredictionPipeline()
    return pipeline.predict(player)


# Example usage
if __name__ == "__main__":
    # Create example outfielder
    example_player = PlayerOutfielder(
        height=74,
        weight=190,
        primary_position='OF',
        hitting_handedness='Right',
        throwing_hand='Right', 
        region='South',
        exit_velo_max=98.0,
        of_velo=88.0,
        sixty_time=6.5
    )
    
    # Initialize pipeline and make prediction
    pipeline = OutfielderPredictionPipeline()
    result = pipeline.predict(example_player)
    
    print("\n" + "="*50)
    print("OUTFIELDER PREDICTION RESULTS")
    print("="*50)
    print(f"Final Prediction: {result.get('final_prediction', 'Error')}")
    
    if result.get('probabilities'):
        probs = result['probabilities']
        print(f"Non-D1: {probs['non_d1']:.1%}")
        print(f"Non-P4 D1: {probs['non_p4_d1']:.1%}")
        print(f"Power 4 D1: {probs['p4_d1']:.1%}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")

    print('Player Info: \n' + str(example_player))