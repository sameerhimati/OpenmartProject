import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from data_processor import PPPDataProcessor
from feature_engineering import PPPFeatureEngineer


def resume_from_checkpoint(checkpoint_dir='checkpoints'):
    """
    Resume feature engineering from the latest checkpoint
    """
    import pickle
    import os
    from datetime import datetime
    
    logger.info("Looking for checkpoints...")
    
    # Get all checkpoint files
    if not os.path.exists(checkpoint_dir):
        logger.info("No checkpoints found.")
        return None
        
    checkpoint_files = os.listdir(checkpoint_dir)
    
    # Check for final checkpoint
    final_checkpoints = [f for f in checkpoint_files if 'final' in f]
    if final_checkpoints:
        logger.info("Found completed analysis! Loading final results...")
        latest_final = max(final_checkpoints)
        with open(f"{checkpoint_dir}/{latest_final}", 'rb') as f:
            importance_dict = pickle.load(f)
            
        # Create visualization with fixed style
        logger.info("Creating visualization...")
        plt.style.use('seaborn-v0_8')  # Fixed style name
        plt.figure(figsize=(15, 10))
        
        # Get aggregate importance
        aggregate_importance = {}
        for feature in importance_dict['rf_importance'].keys():
            aggregate_importance[feature] = np.mean([
                scores[feature] for scores in importance_dict.values()
            ])
        
        # Sort and select top features
        sorted_features = sorted(aggregate_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:30]
        
        # Create bar plot
        features, scores = zip(*sorted_features)
        plt.barh(range(len(features)), scores, color='#2ecc71', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Aggregate Importance Score', fontsize=12)
        plt.title('Top Feature Importance', fontsize=14, pad=20)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualization saved to outputs/feature_importance.png")
        
        # Display top features
        logger.info("\nTop 10 Most Important Features (Aggregate):")
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            logger.info(f"{i}. {feature}: {score:.4f}")
        
        return importance_dict
    
    logger.info("No final checkpoint found. Proceeding with feature analysis...")
    return None

def main():
    try:
        logger.info("🚀 Starting feature engineering pipeline...")
        
        # Try to resume from checkpoint
        importance_dict = resume_from_checkpoint()
        
        if importance_dict is not None:
            logger.info("Successfully loaded results from checkpoint!")
            return
        
        # If no checkpoint, proceed with full analysis
        logger.info("Loading processed data from parquet...")
        df = pd.read_parquet("processed_ppp_data.parquet")
        
        # Initialize feature engineer
        feature_engineer = PPPFeatureEngineer()
        
        # Analyze and select features
        selected_df, importance_dict = feature_engineer.analyze_features(df)
        
        # Save results
        logger.info("Saving final results...")
        selected_df.to_parquet("feature_selected_data.parquet",
                             engine='pyarrow',
                             compression='snappy',
                             index=False)
        
        logger.info("\n✨ Feature engineering pipeline complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()