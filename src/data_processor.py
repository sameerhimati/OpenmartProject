import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPPDataProcessor:
    """
    Handles the processing of PPP loan data for jobs prediction.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.categorical_columns = [
            'ProcessingMethod', 'BusinessType', 'RuralUrbanIndicator',
            'HubzoneIndicator', 'LMIIndicator', 'BusinessAgeDescription',
            'Race', 'Ethnicity', 'Gender', 'Veteran'
        ]
        self.numeric_columns = [
            'InitialApprovalAmount', 'CurrentApprovalAmount', 'PAYROLL_PROCEED',
            'UTILITIES_PROCEED', 'MORTGAGE_INTEREST_PROCEED', 'RENT_PROCEED',
            'REFINANCE_EIDL_PROCEED', 'HEALTH_CARE_PROCEED', 'DEBT_INTEREST_PROCEED'
        ]
        self.target_column = 'JobsReported'

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input data structure and types.
        """
        # Check for required columns
        required_columns = [self.target_column] + self.numeric_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate target variable
        if not pd.api.types.is_numeric_dtype(df[self.target_column]):
            raise ValueError(f"Target column {self.target_column} must be numeric")
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline for PPP loan data.
        """
        logger.info("Starting data processing...")
        
        # Validate input data
        self.validate_data(df)
        
        # Make a deep copy to avoid SettingWithCopyWarning
        df_processed = df.copy(deep=True)
        
        # Create derived features first (before encoding)
        df_processed = self._create_derived_features(df_processed)
        
        # Basic cleaning
        df_processed = self._clean_data(df_processed)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Process categorical features
        df_processed = self._process_categorical_features(df_processed)
        
        # Process numerical features
        df_processed = self._process_numerical_features(df_processed)
        
        logger.info("Data processing completed")
        return df_processed
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        """
        logger.info("Cleaning data...")
        
        # Remove rows with null JobsReported (target variable)
        df = df[df[self.target_column].notna()].copy()
        
        # Handle NAICS code properly
        if 'NAICSCode' in df.columns:
            # Convert to string first, then handle missing values
            df.loc[:, 'NAICSCode'] = df['NAICSCode'].astype(str)
            df.loc[df['NAICSCode'].isin(['nan', 'None', '0']), 'NAICSCode'] = '999999'
            
            # Extract sector (first 2 digits)
            df.loc[:, 'NAICSSector'] = df['NAICSCode'].str[:2]
            
            # Handle invalid sectors
            df.loc[~df['NAICSSector'].str.match(r'^\d{2}$'), 'NAICSSector'] = '99'
        
        # Convert date columns to datetime
        date_columns = ['DateApproved', 'LoanStatusDate', 'ForgivenessDate']
        for col in date_columns:
            if col in df.columns:
                df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        logger.info("Handling missing values...")
        
        # Fill missing values in numeric columns with 0
        for col in self.numeric_columns:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill categorical missing values with 'Unknown'
        for col in self.categorical_columns:
            if col in df.columns:
                df.loc[:, col] = df[col].fillna('Unknown')
        
        return df
    
    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features using label encoding.
        """
        logger.info("Processing categorical features...")
        
        for col in self.categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df.loc[:, col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def _process_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process numerical features.
        """
        logger.info("Processing numerical features...")
        
        # Ensure all numeric columns are float
        for col in self.numeric_columns:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Log transform monetary values (adding 1 to handle zeros)
        for col in self.numeric_columns:
            if col in df.columns:
                # Handle negative values if any
                min_val = df[col].min()
                if min_val < 0:
                    df.loc[:, col] = df[col] - min_val + 1
                df.loc[:, f'{col}_log'] = np.log1p(df[col])
        
        # Calculate ratios between different monetary amounts
        df.loc[:, 'PayrollToLoanRatio'] = (df['PAYROLL_PROCEED'] / 
                                          df['InitialApprovalAmount'].replace(0, np.nan)).fillna(0)
        
        if 'ForgivenessAmount' in df.columns:
            df.loc[:, 'ForgivenessRatio'] = (df['ForgivenessAmount'] / 
                                            df['InitialApprovalAmount'].replace(0, np.nan)).fillna(0)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing data.
        """
        logger.info("Creating derived features...")
        
        # Create binary flags before encoding
        df.loc[:, 'IsNonProfit'] = (df['NonProfit'] == 'Y').astype(int)
        df.loc[:, 'IsNewBusiness'] = df['BusinessAgeDescription'].isin(
            ['New Business or 2 years or less', 'Startup, Loan Funds will Open Business']
        ).astype(int)
        
        # Time-based features
        if 'DateApproved' in df.columns:
            date_approved = pd.to_datetime(df['DateApproved'], errors='coerce')
            df.loc[:, 'ApprovalYear'] = date_approved.dt.year
            df.loc[:, 'ApprovalMonth'] = date_approved.dt.month
            df.loc[:, 'ApprovalDayOfWeek'] = date_approved.dt.dayofweek
        
        # Calculate total proceeds
        proceed_columns = [col for col in self.numeric_columns if 'PROCEED' in col]
        for col in proceed_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df.loc[:, 'TotalProceeds'] = df[proceed_columns].sum(axis=1)
        df.loc[:, 'ProceedsToLoanRatio'] = (df['TotalProceeds'] / 
                                           df['InitialApprovalAmount'].replace(0, np.nan)).fillna(0)
        
        return df

def main():
    """
    Main function to run the data processing pipeline.
    """
    try:
        logger.info("Loading data...")
        df = pd.read_csv("../combined_ppp_data20241026213556.csv", low_memory=False)
        
        processor = PPPDataProcessor()
        processed_df = processor.process_data(df)
        
        # Save processed data as parquet
        logger.info("Saving processed data to parquet format...")
        processed_df.to_parquet("../data/processed/processed_ppp_data_half.parquet", 
                              engine='pyarrow',
                              compression='snappy',
                              index=False)
        logger.info("Saved processed data to processed_ppp_data.parquet")
        
        # Print shape and features
        logger.info(f"Processed data shape: {processed_df.shape}")
        logger.info(f"Features created: {list(processed_df.columns)}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()