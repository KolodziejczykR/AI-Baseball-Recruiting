import os
import sys
import warnings
import logging
import time
warnings.filterwarnings('ignore')

# Add project root to Python path to enable backend imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MODEL_DIR_D1 = os.path.join(project_root, 'backend/ml/models/models_inf/models_d1_or_not_inf/version_08072025')
MODEL_DIR_P4 = os.path.join(project_root, 'backend/ml/models/models_inf/models_p4_or_not_inf/version_08072025')

from backend.utils.player_types import PlayerInfielder
from backend.ml.models.models_inf.models_d1_or_not_inf.version_08072025 import prediction_pipeline as d1_pipeline
from backend.ml.models.models_inf.models_p4_or_not_inf.version_08072025 import prediction_pipeline as p4_pipeline

class InfielderPredictionPipeline:
    def __init__(self):
        """
        Initialize the infielder prediction pipeline using the latest production models.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing InfielderPredictionPipeline")
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        
        self.logger.info("InfielderPredictionPipeline initialized successfully")
    
    def predict(self, player: PlayerInfielder) -> dict:
        """
        Run the complete two-stage infielder prediction pipeline.
        
        Model Thresholds:
            D1 Stage: Uses weighted ensemble voting (XGBoost, LightGBM, CatBoost, SVM)
            P4 Stage: ~52.0% probability threshold (optimal_threshold from pickled config)
        
        Args:
            player: PlayerInfielder object containing player statistics
        
        Returns:
            Dictionary with comprehensive prediction results including:
            - d1_probability: Raw D1 probability (0.0-1.0)
            - p4_probability: Raw P4 probability (0.0-1.0) or None if Non-D1
            - final_prediction: 'Non-D1', 'Non-P4 D1', or 'Power 4 D1'
            - probabilities: 3-way breakdown dict
        """
        start_time = time.time()
        self.prediction_count += 1
        
        if not isinstance(player, PlayerInfielder):
            self.logger.error(f"Invalid input type: {type(player)}. Expected PlayerInfielder")
            raise TypeError("Input must be a PlayerInfielder object")
        
        self.logger.info(f"Starting infielder prediction #{self.prediction_count} for position: {player.primary_position}")
        self.logger.debug(f"Player stats: height={player.height}, weight={player.weight}, "
                         f"exit_velo_max={player.exit_velo_max}, inf_velo={player.inf_velo}")
        
        try:
            # Convert player to dictionary format expected by models
            player_data = player.to_dict()
            self.logger.debug(f"Player data converted to dict with {len(player_data)} features")
            
            # Stage 1: Predict D1 vs Non-D1
            self.logger.info("Running Stage 1: D1 vs Non-D1 prediction")
            stage1_start = time.time()
            d1_result = d1_pipeline.predict_infielder_d1_probability(player_data, MODEL_DIR_D1)
            stage1_time = time.time() - stage1_start
            
            self.logger.info(f"Stage 1 completed in {stage1_time:.3f}s - "
                           f"D1 probability: {d1_result['d1_probability']:.3f}, "
                           f"Prediction: {'D1' if d1_result['d1_prediction'] == 1 else 'Non-D1'}")
            
            # If predicted as Non-D1, return early
            if d1_result['d1_prediction'] == 0:
                total_time = time.time() - start_time
                self.total_prediction_time += total_time
                
                self.logger.info(f"Prediction completed: Non-D1 (D1 prob: {d1_result['d1_probability']:.3f}) "
                               f"in {total_time:.3f}s")
                self.logger.info(f"Pipeline stats: {self.prediction_count} predictions, "
                               f"avg time: {self.total_prediction_time/self.prediction_count:.3f}s")
                
                return {
                    'final_prediction': 'Non-D1',
                    'final_category': 0,  # Non-D1
                    'd1_probability': float(d1_result['d1_probability']),
                    'p4_probability': None,
                    'probabilities': {
                        'non_d1': float(1 - d1_result['d1_probability']),
                        'd1_total': float(d1_result['d1_probability']),
                        'non_p4_d1': 0.0,
                        'p4_d1': 0.0
                    },
                    'confidence': d1_result['confidence_level'],
                    'model_chain': 'D1_only',
                    'd1_details': d1_result,
                    'p4_details': None
                }
            
            # Stage 2: For D1-predicted players, predict P4 vs Non-P4 D1
            self.logger.info("Running Stage 2: P4 vs Non-P4 D1 prediction")
            stage2_start = time.time()
            p4_result = p4_pipeline.predict_infielder_p4_probability(player_data, MODEL_DIR_P4)
            stage2_time = time.time() - stage2_start
            
            self.logger.info(f"Stage 2 completed in {stage2_time:.3f}s - "
                           f"P4 probability: {p4_result['p4_probability']:.3f}, "
                           f"Prediction: {'P4' if p4_result['p4_prediction'] == 1 else 'Non-P4'}")
            
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
            overall_confidence = min(d1_result['confidence_level'], p4_result['confidence'])
            
            total_time = time.time() - start_time
            self.total_prediction_time += total_time
            
            self.logger.info(f"Prediction completed: {final_prediction} "
                           f"(D1: {d1_prob:.3f}, P4: {p4_conditional_prob:.3f}) "
                           f"in {total_time:.3f}s")
            self.logger.info(f"Pipeline stats: {self.prediction_count} predictions, "
                           f"avg time: {self.total_prediction_time/self.prediction_count:.3f}s")
            
            return {
                'final_prediction': final_prediction,
                'final_category': final_category,
                'd1_probability': float(d1_prob),
                'p4_probability': float(p4_conditional_prob),
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
            total_time = time.time() - start_time
            self.logger.error(f"Prediction failed after {total_time:.3f}s: {str(e)}", exc_info=True)
            
            return {
                'error': f"Prediction failed: {str(e)}",
                'final_prediction': None,
                'final_category': None,
                'probabilities': None,
                'confidence': None
            }
    
    def get_required_features(self) -> list:
        """
        Get list of required features for prediction
        """
        return [
            'height', 'weight', 'sixty_time', 'exit_velo_max', 
            'inf_velo', 'primary_position', 'player_region', 
            'throwing_hand', 'hitting_handedness'
        ]
    
    def get_feature_info(self) -> dict:
        """
        Get detailed information about features
        """
        return {
            "numerical_features": [
                'height', 'weight', 'sixty_time', 'exit_velo_max', 'inf_velo'
            ],
            "categorical_features": [
                'primary_position', 'player_region', 'throwing_hand', 'hitting_handedness'
            ],
            "descriptions": {
                'height': 'Player height in inches',
                'weight': 'Player weight in pounds',
                'sixty_time': '60-yard dash time in seconds',
                'exit_velo_max': 'Maximum exit velocity in mph',
                'inf_velo': 'Infield velocity in mph',
                'primary_position': 'Primary playing position (SS, 2B, 3B, 1B)',
                'player_region': 'Geographic region (Midwest, West, South, Northeast)',
                'throwing_hand': 'Throwing hand (R, L)',
                'hitting_handedness': 'Hitting handedness (R, L, S)'
            }
        }

    def get_model_info(self) -> dict:
        """
        Get information about the loaded models
        """
        return {
            'pipeline_type': 'Two-stage hierarchical ensemble',
            'stage_1': {
                'target': 'D1 vs Non-D1 classification',
                'architecture': 'Weighted ensemble: XGBoost + LightGBM + CatBoost + SVM'
            },
            'stage_2': {
                'target': 'P4 vs Non-P4 D1 classification',
                'architecture': 'Elite Detection + Hierarchical ensemble: XGBoost + CatBoost + LightGBM + SVM'
            },
            'required_input': 'PlayerInfielder object',
            'supported_features': [
                'height', 'weight', 'sixty_time', 'exit_velo_max', 
                'inf_velo', 'player_region', 'throwing_hand', 'hitting_handedness', 'primary_position'
            ]
        }


# Example usage
if __name__ == "__main__":
    # Create example infielder
    example_player = PlayerInfielder(
        height=75,
        weight=210,
        primary_position='SS',
        hitting_handedness='R',
        throwing_hand='R', 
        region='West',
        exit_velo_max=105.0,
        inf_velo=93.0,
        sixty_time=6.5
    )
    
    # Initialize pipeline and make prediction
    pipeline = InfielderPredictionPipeline()
    result = pipeline.predict(example_player)
    
    print("\n" + "="*50)
    print("INFIELDER PREDICTION RESULTS")
    print("="*50)
    print(f"Final Prediction: {result.get('final_prediction', 'Error')}")
    
    if result.get('probabilities'):
        probs = result['probabilities']
        print(f"Non-D1: {probs['non_d1']:.1%}")
        print(f"D1: {probs['d1_total']:.1%}")
        print(f"Non-P4 D1: {probs['non_p4_d1']:.1%}")
        print(f"Power 4 D1: {probs['p4_d1']:.1%}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")

    print('Player Info: \n' + str(example_player))