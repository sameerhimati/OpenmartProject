import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from pathlib import Path
import pickle
import json
import sys
from datetime import datetime

# Set up logging to both file and console
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PPPModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def prepare_data(self, data_path: str):
        """
        Prepare data for training based on feature importance results
        """
        logger.info("Loading and preparing data...")
        df = pd.read_parquet(data_path)
        
        # Separate features based on importance
        self.primary_features = [
            'InitialApprovalAmount', 'InitialApprovalAmount_log',
            'RENT_PROCEED', 'HEALTH_CARE_PROCEED', 'UTILITIES_PROCEED'
        ]
        
        self.categorical_features = ['BusinessType', 'source_file']
        
        logger.info("Data loaded successfully")
        logger.info(f"Numeric features: {self.primary_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
        
        return df
    
    def add_feature_interactions(self, X):
        """
        Add interaction terms between important features
        """
        logger.info("Adding feature interactions...")
        
        # Amount-based interactions
        X['amount_rent_ratio'] = X['InitialApprovalAmount'] / (X['RENT_PROCEED'] + 1)
        X['amount_healthcare_ratio'] = X['InitialApprovalAmount'] / (X['HEALTH_CARE_PROCEED'] + 1)
        X['amount_utilities_ratio'] = X['InitialApprovalAmount'] / (X['UTILITIES_PROCEED'] + 1)
        
        # Time-based interactions
        if 'days_since_start' in X.columns:
            X['amount_time_interaction'] = X['InitialApprovalAmount'] * X['days_since_start']
        
        # Log transformations of ratios
        X['amount_rent_ratio_log'] = np.log1p(X['amount_rent_ratio'])
        X['amount_healthcare_ratio_log'] = np.log1p(X['amount_healthcare_ratio'])
        X['amount_utilities_ratio_log'] = np.log1p(X['amount_utilities_ratio'])
        
        return X
    
    def train_models(self, df: pd.DataFrame, use_interactions=True):
        """
        Train multiple models with enhanced features and parameters
        """
        logger.info("Preparing features for training...")
        
        # Convert datetime
        if 'DateApproved' in df.columns:
            df['days_since_start'] = (pd.to_datetime(df['DateApproved']) - 
                                    pd.to_datetime(df['DateApproved']).min()).dt.total_seconds() / (24 * 60 * 60)
        
        # Update feature lists
        self.temporal_features = ['days_since_start']
        numeric_features = [f for f in self.primary_features if f != 'DateApproved'] + self.temporal_features
        
        # Encode categorical features
        X = df[numeric_features + self.categorical_features].copy()
        y = df['JobsReported']
        
        # Add feature interactions
        if use_interactions:
            X = self.add_feature_interactions(X)
            numeric_features = [col for col in X.columns if col not in self.categorical_features]
        
        logger.info("Encoding categorical features...")
        for cat_feat in self.categorical_features:
            logger.info(f"Encoding {cat_feat}")
            label_encoder = LabelEncoder()
            X[cat_feat] = label_encoder.fit_transform(X[cat_feat].astype(str))
            setattr(self, f"{cat_feat}_encoder", label_encoder)
        
        logger.info(f"Training with features: {X.columns.tolist()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale numeric features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # Store scaler for future use
        self.scaler = scaler
        
        # Enhanced model parameters
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            )
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Store predictions for ensemble
            predictions[name] = {
                'train': model.predict(X_train_scaled),
                'test': model.predict(X_test_scaled)
            }
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'metrics': {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, predictions[name]['train'])),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, predictions[name]['test'])),
                    'train_r2': r2_score(y_train, predictions[name]['train']),
                    'test_r2': r2_score(y_test, predictions[name]['test']),
                    'train_mae': mean_absolute_error(y_train, predictions[name]['train']),
                    'test_mae': mean_absolute_error(y_test, predictions[name]['test'])
                }
            }
            
            logger.info(f"{name} Results:")
            for metric, value in results[name]['metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                logger.info(f"\n{name} Feature Importance:")
                importance = pd.DataFrame({
                    'feature': X_train_scaled.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for _, row in importance.iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create and evaluate ensemble
        weights = {'rf': 0.5, 'xgb': 0.3, 'lgb': 0.2}
        ensemble_train = np.zeros_like(y_train, dtype=float)
        ensemble_test = np.zeros_like(y_test, dtype=float)
        
        for name, weight in weights.items():
            ensemble_train += predictions[name]['train'] * weight
            ensemble_test += predictions[name]['test'] * weight
        
        results['ensemble'] = {
            'metrics': {
                'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_test)),
                'train_r2': r2_score(y_train, ensemble_train),
                'test_r2': r2_score(y_test, ensemble_test),
                'train_mae': mean_absolute_error(y_train, ensemble_train),
                'test_mae': mean_absolute_error(y_test, ensemble_test)
            }
        }
        
        logger.info("\nEnsemble Results:")
        for metric, value in results['ensemble']['metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def train_industry_models(self, df):
        """
        Train separate models for different business types
        """
        logger.info("Training industry-specific models...")
        industry_results = {}
        
        for business_type in df['BusinessType'].unique():
            df_industry = df[df['BusinessType'] == business_type]
            if len(df_industry) > 1000:  # Only train if enough data
                logger.info(f"\nTraining models for business type: {business_type}")
                logger.info(f"Number of samples: {len(df_industry)}")
                industry_results[business_type] = self.train_models(df_industry)
        
        return industry_results
    
    def save_models(self, results, output_dir):
        """
        Save trained models and results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, result in results.items():
            if 'model' in result:
                model_path = output_path / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
        
        # Save metrics
        metrics = {name: result['metrics'] for name, result in results.items()}
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save feature encoders and scaler
        encoders = {
            name: getattr(self, f"{name}_encoder") 
            for name in self.categorical_features
        }
        with open(output_path / "encoders.pkl", 'wb') as f:
            pickle.dump(encoders, f)
            
        with open(output_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    def cluster_industries(self, df: pd.DataFrame, n_clusters: int = 5):
        """
        Cluster similar industries based on their loan characteristics
        """
        logger.info(f"Clustering industries into {n_clusters} groups...")
        
        # Aggregate metrics by NAICS code
        industry_metrics = df.groupby('NAICSCode').agg({
            'JobsReported': ['mean', 'std', 'median'],
            'InitialApprovalAmount': ['mean', 'std', 'median'],
            'PAYROLL_PROCEED': ['mean', 'median'],
            'RENT_PROCEED': ['mean', 'median']
        }).fillna(0)
        
        # Flatten column names
        industry_metrics.columns = [f"{col[0]}_{col[1]}" for col in industry_metrics.columns]
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(industry_metrics)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        industry_clusters = kmeans.fit_predict(features_scaled)
        
        # Create mapping (convert numpy types to native Python types)
        cluster_mapping = {
            str(k): int(v) for k, v in 
            dict(zip(industry_metrics.index, industry_clusters)).items()
        }
        
        # Add cluster to original dataframe
        df['IndustryCluster'] = df['NAICSCode'].astype(str).map(cluster_mapping)
        
        return df, cluster_mapping
        
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create more sophisticated features based on analysis
        """
        logger.info("Creating enhanced features...")
        # Size-based features
        df['size_category'] = pd.qcut(df['InitialApprovalAmount'], q=10, labels=False)
        
        # Time-based features
        df['days_since_start'] = (pd.to_datetime(df['DateApproved']) - 
                                pd.to_datetime(df['DateApproved']).min()).dt.days
        df['month'] = pd.to_datetime(df['DateApproved']).dt.month
        df['is_quarter_end'] = pd.to_datetime(df['DateApproved']).dt.month.isin([3, 6, 9, 12]).astype(int)
        
        # Proceed ratios with smoothing
        epsilon = 1e-5
        proceeds = ['PAYROLL_PROCEED', 'RENT_PROCEED', 'UTILITIES_PROCEED', 'HEALTH_CARE_PROCEED']
        
        for proc in proceeds:
            # Smoothed ratio
            df[f'{proc}_ratio_smooth'] = (df[proc] + epsilon) / (df['InitialApprovalAmount'] + epsilon)
            # Log transform
            df[f'{proc}_log'] = np.log1p(df[proc])
        
        # Interaction features
        df['amount_time'] = df['InitialApprovalAmount'] * df['days_since_start']
        df['amount_size_time'] = df['InitialApprovalAmount'] * df['size_category'] * df['days_since_start']
        
        # Geographic features
        df['state_avg_amount'] = df.groupby('BorrowerState')['InitialApprovalAmount'].transform('mean')
        df['state_avg_jobs'] = df.groupby('BorrowerState')['JobsReported'].transform('mean')
        
        # Industry-specific features
        df['industry_avg_amount'] = df.groupby('NAICSCode')['InitialApprovalAmount'].transform('mean')
        df['industry_avg_jobs'] = df.groupby('NAICSCode')['JobsReported'].transform('mean')
        df['amount_to_industry_avg'] = df['InitialApprovalAmount'] / df['industry_avg_amount']
        
        return df

def main():
    try:
        trainer = PPPModelTrainer()
        
        # Prepare data
        logger.info("Loading and preparing data...")
        df = trainer.prepare_data("../data/processed/processed_ppp_data_full.parquet")
        
        # Add enhanced features
        df = trainer.create_enhanced_features(df)
        
        # Cluster industries
        df, cluster_mapping = trainer.cluster_industries(df)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save cluster mapping
        with open('industry_clusters.json', 'w') as f:
            json.dump(cluster_mapping, f, indent=4, default=convert_to_serializable)
        
        # Train general models
        logger.info("\nTraining general models...")
        results = trainer.train_models(df, use_interactions=True)
        
        # Train industry-specific models
        logger.info("\nTraining industry-specific models...")
        industry_results = trainer.train_industry_models(df)
        
        # Save results summary
        summary = {
            'general_results': {
                name: {
                    k: convert_to_serializable(v)
                    for k, v in result['metrics'].items()
                }
                for name, result in results.items()
            },
            'industry_results': {
                str(ind): {
                    name: {
                        k: convert_to_serializable(v)
                        for k, v in result['metrics'].items()
                    }
                    for name, result in ind_results.items()
                }
                for ind, ind_results in industry_results.items()
            }
        }
        
        with open('model_results_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Save all models and results
        trainer.save_models(results, "models/general")
        for industry, ind_results in industry_results.items():
            trainer.save_models(ind_results, f"models/industry_{industry}")
        
        logger.info("\nTraining pipeline completed successfully!")
        logger.info(f"Results saved to model_results_summary.json")
        logger.info(f"Full log saved to {log_filename}")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()