import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import spearmanr
import logging
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import torch  # For GPU acceleration
from sklearn.utils import parallel_backend
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPPFeatureEngineer:
    def __init__(self, target_column: str = 'JobsReported'):
        self.target_column = target_column
        self.selected_features = None
        self.feature_importances = {}
        self.correlation_threshold = 0.95
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Check if GPU is available (Metal on M1/M2/M3 Macs)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if torch.backends.mps.is_available():
            logger.info("🚀 Metal GPU acceleration is available and will be used!")
        else:
            logger.info("Running on CPU")
        
    def _get_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize columns by their basic types.
        """
        type_dict = {
            'numeric': [],
            'datetime': [],
            'categorical': []
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_dict['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_dict['datetime'].append(col)
            else:
                type_dict['categorical'].append(col)
                
        return type_dict
    
    def analyze_features(self, df: pd.DataFrame, n_features: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze and select the most important features using multiple methods.
        """
        logger.info("Starting feature analysis...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Initial feature type analysis
        column_types = self._get_column_types(X)
        logger.info("\nFeature type analysis:")
        for type_name, columns in column_types.items():
            logger.info(f"{type_name}: {len(columns)} columns")
            if len(columns) < 10:  # Only show if few columns
                logger.info(f"Columns: {columns}")
        
        # Convert all features to numeric
        X = self._convert_to_numeric(X)
        
        # Remove constant and quasi-constant features
        X = self._remove_low_variance_features(X)
        
        # Remove highly correlated features
        X = self._remove_correlated_features(X)
        
        # Calculate feature importance using multiple methods
        importance_dict = self._calculate_feature_importance(X, y)
        
        # Select top features
        selected_features = self._select_top_features(importance_dict, n_features)
        
        # Store selected features
        self.selected_features = selected_features
        self.feature_importances = importance_dict
        
        # Create visualization
        self._create_importance_plot(importance_dict, n_features)
        
        logger.info(f"Selected {len(selected_features)} features")
        return df[selected_features + [self.target_column]], importance_dict
    
    def _convert_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all features to numeric format.
        """
        logger.info("Converting features to numeric format...")
        X_numeric = X.copy()
        
        for column in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[column]):
                # Convert datetime to numeric (days since earliest date)
                X_numeric[column] = (X[column] - X[column].min()).dt.total_seconds() / (24 * 60 * 60)
                logger.info(f"Converted datetime column {column} to days elapsed")
            elif X[column].dtype == 'object' or pd.api.types.is_string_dtype(X[column]):
                try:
                    # First try to convert to numeric directly
                    X_numeric[column] = pd.to_numeric(X[column], errors='coerce')
                    if X_numeric[column].isna().sum() > 0:
                        # If we have NAs after conversion, use label encoding
                        self.label_encoders[column] = LabelEncoder()
                        X_numeric[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
                        logger.info(f"Label encoded column {column}")
                except:
                    # If direct conversion fails, use label encoding
                    self.label_encoders[column] = LabelEncoder()
                    X_numeric[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
                    logger.info(f"Label encoded column {column}")
        
        return X_numeric
    
    def _remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove constant and quasi-constant features.
        """
        logger.info("Removing low variance features...")
        
        # Calculate variance for all columns (now all numeric)
        variances = X.var()
        low_variance_features = variances[variances < threshold].index
        
        if len(low_variance_features) > 0:
            logger.info(f"Removing {len(low_variance_features)} low variance features")
            logger.info(f"Removed features: {list(low_variance_features)}")
            X = X.drop(columns=low_variance_features)
            
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features keeping the one with highest correlation with target.
        """
        logger.info("Removing highly correlated features...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        if len(to_drop) > 0:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            logger.info(f"Removed features: {list(to_drop)}")
            X = X.drop(columns=to_drop)
            
        return X
    
    def _select_top_features(self, importance_dict: Dict, n_features: int) -> List[str]:
        """
        Select top features based on aggregate importance across all methods.
        """
        # Normalize scores for each method
        normalized_scores = {}
        for method, scores in importance_dict.items():
            max_score = max(scores.values())
            normalized_scores[method] = {k: v/max_score for k, v in scores.items()}
        
        # Calculate aggregate importance
        aggregate_importance = {}
        for feature in importance_dict['rf_importance'].keys():
            aggregate_importance[feature] = np.mean([
                scores[feature] for scores in normalized_scores.values()
            ])
        
        # Sort features by importance
        sorted_features = sorted(aggregate_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        return [feature for feature, _ in sorted_features[:n_features]]

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate feature importance with progress saving at key stages.
        """
        import pickle
        import os
        from datetime import datetime
        
        # Create checkpoints directory
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Function to save checkpoint
        def save_checkpoint(stage: str, data: dict):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{checkpoint_dir}/feature_importance_{stage}_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved checkpoint for stage '{stage}' to {filename}")
        
        # Function to load latest checkpoint for a stage
        def load_latest_checkpoint(stage: str) -> dict:
            checkpoints = [f for f in os.listdir(checkpoint_dir) 
                        if f.startswith(f"feature_importance_{stage}_")]
            if checkpoints:
                latest = max(checkpoints)
                with open(f"{checkpoint_dir}/{latest}", 'rb') as f:
                    return pickle.load(f)
            return None

        logger.info("Starting feature importance calculations...")
        importance_dict = {}
        
        try:
            # Check for and handle NaN values before scaling
            logger.info("Checking for NaN values...")
            nan_counts = X.isna().sum()
            columns_with_nans = nan_counts[nan_counts > 0]
            if not columns_with_nans.empty:
                logger.info("Found NaN values in the following columns:")
                for col, count in columns_with_nans.items():
                    logger.info(f"  {col}: {count} NaN values")
                
                # Fill NaN values with appropriate strategies
                X_clean = X.copy()
                numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
                
                # For numeric columns, fill with median
                for col in numeric_columns:
                    if col in columns_with_nans:
                        median_val = X_clean[col].median()
                        X_clean[col].fillna(median_val, inplace=True)
                        logger.info(f"Filled NaN values in {col} with median: {median_val}")
                
                # For non-numeric columns, fill with mode
                non_numeric_columns = X_clean.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_columns:
                    if col in columns_with_nans:
                        mode_val = X_clean[col].mode()[0]
                        X_clean[col].fillna(mode_val, inplace=True)
                        logger.info(f"Filled NaN values in {col} with mode: {mode_val}")
            else:
                X_clean = X
                logger.info("No NaN values found in features.")
            
            # Save cleaned data checkpoint
            save_checkpoint('cleaned_data', {'X_clean': X_clean})
            
            # Scale features
            with tqdm(total=1, desc="Scaling features") as pbar:
                X_scaled = self.scaler.fit_transform(X_clean)
                X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
                pbar.update(1)
            
            # Save scaled data checkpoint
            save_checkpoint('scaled_data', {'X_scaled': X_scaled, 'scaler': self.scaler})
            
            # 1. Random Forest importance
            logger.info("Training Random Forest...")
            rf_checkpoint = load_latest_checkpoint('random_forest')
            if rf_checkpoint is None:
                total_trees = 50
                rf = RandomForestRegressor(n_estimators=total_trees, 
                                        random_state=42,
                                        verbose=1,
                                        n_jobs=-1)
                
                with tqdm(total=1, desc="Random Forest Training") as pbar:
                    rf.fit(X_scaled, y)
                    pbar.update(1)
                
                importance_dict['rf_importance'] = dict(zip(X_clean.columns, rf.feature_importances_))
                save_checkpoint('random_forest', {'rf_importance': importance_dict['rf_importance']})
            else:
                logger.info("Loading Random Forest results from checkpoint")
                importance_dict['rf_importance'] = rf_checkpoint['rf_importance']
            
            # 2. F-regression
            logger.info("Calculating F-regression scores...")
            f_reg_checkpoint = load_latest_checkpoint('f_regression')
            if f_reg_checkpoint is None:
                with tqdm(total=1, desc="F-regression") as pbar:
                    f_scores, _ = f_regression(X_scaled, y)
                    importance_dict['f_regression'] = dict(zip(X_clean.columns, f_scores))
                    pbar.update(1)
                save_checkpoint('f_regression', {'f_regression': importance_dict['f_regression']})
            else:
                logger.info("Loading F-regression results from checkpoint")
                importance_dict['f_regression'] = f_reg_checkpoint['f_regression']
            
            # 3. Mutual Information
            logger.info("Calculating Mutual Information scores...")
            mi_checkpoint = load_latest_checkpoint('mutual_info')
            if mi_checkpoint is None:
                with tqdm(total=1, desc="Mutual Information") as pbar:
                    mi_scores = mutual_info_regression(X_scaled, y)
                    importance_dict['mutual_info'] = dict(zip(X_clean.columns, mi_scores))
                    pbar.update(1)
                save_checkpoint('mutual_info', {'mutual_info': importance_dict['mutual_info']})
            else:
                logger.info("Loading Mutual Information results from checkpoint")
                importance_dict['mutual_info'] = mi_checkpoint['mutual_info']
            
            # 4. Spearman correlation
            logger.info("Calculating Spearman correlations...")
            spearman_checkpoint = load_latest_checkpoint('spearman')
            if spearman_checkpoint is None:
                spearman_corr = []
                
                if torch.cuda.is_available():
                    X_torch = torch.from_numpy(X_scaled.values).float().to(self.device)
                    batch_size = self._get_optimal_batch_size(n_features=X_scaled.shape[1], 
                                                            n_samples=y.shape[0])
                    n_features = X_scaled.shape[1]
                    
                    for i in tqdm(range(0, n_features, batch_size), desc="Spearman correlation"):
                        batch_end = min(i + batch_size, n_features)
                        batch_X = X_torch[:, i:batch_end]
                        
                        for j in range(i, batch_end):
                            corr = spearmanr(batch_X[:, j-i].cpu().numpy(), y.values)[0]
                            spearman_corr.append(corr)
                        
                        # Save intermediate checkpoint every batch
                        if (i + batch_size) < n_features:
                            save_checkpoint(f'spearman_intermediate_{i}', 
                                        {'spearman_corr': spearman_corr})
                else:
                    for col in tqdm(X_scaled.columns, desc="Spearman correlation"):
                        corr = spearmanr(X_scaled[col], y)[0]
                        spearman_corr.append(corr)
                
                importance_dict['spearman_corr'] = dict(zip(X_clean.columns, np.abs(spearman_corr)))
                save_checkpoint('spearman', {'spearman_corr': importance_dict['spearman_corr']})
            else:
                logger.info("Loading Spearman correlation results from checkpoint")
                importance_dict['spearman_corr'] = spearman_checkpoint['spearman_corr']
            
            # Save final results
            save_checkpoint('final', importance_dict)
            
            logger.info("✨ Feature importance calculation complete!")
            
            # Log top features from each method
            for method, scores in importance_dict.items():
                top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"\nTop 5 features from {method}:")
                for feat, score in top_features:
                    logger.info(f"  {feat}: {score:.4f}")
            
            return importance_dict
        
        except Exception as e:
            logger.error(f"Error during feature importance calculation: {str(e)}")
            # Save current progress before raising exception
            save_checkpoint('error_state', {
                'importance_dict': importance_dict,
                'error': str(e),
                'stage': 'error'
            })
            raise

    def _create_importance_plot(self, importance_dict: Dict, n_features: int):
        """
        Create and save feature importance visualization with improved styling.
        """
        plt.style.use('seaborn-v0_8')
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
                               reverse=True)[:n_features]
        
        # Create bar plot with better styling
        features, scores = zip(*sorted_features)
        plt.barh(range(len(features)), scores, color='#2ecc71', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Aggregate Importance Score', fontsize=12)
        plt.title('Top Feature Importance', fontsize=14, pad=20)
        
        # Add grid
        plt.grid(True, axis='x', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _get_optimal_batch_size(self, n_features: int, n_samples: int) -> int:
        """
        Determine optimal batch size based on available system memory and dataset size.
        """
        
        # Get system memory info
        system_memory = psutil.virtual_memory().available / (1024 ** 3)  # Available memory in GB
        
        # Calculate approximate memory per feature
        # Assuming float32 data type (4 bytes) and some overhead
        memory_per_feature = (n_samples * 4 * 2) / (1024 ** 3)  # in GB (doubled for safety)
        
        # Calculate how many features we can process at once
        features_per_batch = int((system_memory * 0.3) / memory_per_feature)  # Use 30% of available memory
        
        # Set minimum and maximum batch sizes
        min_batch_size = 100
        max_batch_size = 5000
        
        # Calculate optimal batch size
        optimal_batch_size = max(min_batch_size, min(features_per_batch, max_batch_size))
        
        logger.info(f"System memory: {system_memory:.2f}GB")
        logger.info(f"Memory per feature: {memory_per_feature:.4f}GB")
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size

def main():
    try:
        logger.info("🚀 Starting feature engineering pipeline...")
        logger.info("Loading processed data from parquet...")
        df = pd.read_parquet("../data/processed/processed_ppp_data_half.parquet")
        
        # Initialize feature engineer
        feature_engineer = PPPFeatureEngineer()
        
        # Analyze and select features
        with tqdm(total=1, desc="Feature analysis") as pbar:
            selected_df, importance_dict = feature_engineer.analyze_features(df)
            pbar.update(1)
        
        # Save selected features in parquet format
        logger.info("Saving results...")
        selected_df.to_parquet("../data/processed/feature_selected_data_half.parquet",
                             engine='pyarrow',
                             compression='snappy',
                             index=False)
        
        logger.info("\n🎯 Top selected features:")
        for i, feature in enumerate(feature_engineer.selected_features[:10], 1):
            logger.info(f"{i}. {feature}")
        
        logger.info("\n✨ Feature engineering pipeline complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()