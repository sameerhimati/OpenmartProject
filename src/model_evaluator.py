import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

class PPPModelEvaluator:
    """
    Evaluates PPP loan model performance and generates business insights
    """
    def __init__(self, results_path: str = "model_results_summary.json"):
        self.results = self._load_results(results_path)
        self.predictions = {}
        self.actual_values = {}
        
    def _load_results(self, path: str) -> Dict:
        """Load model results from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_predictions(self, predictions_path: str):
        """Load model predictions"""
        with open(predictions_path, 'rb') as f:
            self.predictions = pickle.load(f)
            
    def analyze_job_creation_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in job creation
        """
        insights = {}
        
        # Job creation by loan size
        df['loan_size_bucket'] = pd.qcut(df['InitialApprovalAmount'], 
                                       q=10, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                                   'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])
        
        job_metrics = df.groupby('loan_size_bucket').agg({
            'JobsReported': ['mean', 'median', 'std', 'count'],
            'InitialApprovalAmount': 'mean'
        })
        
        job_metrics.columns = ['avg_jobs', 'median_jobs', 'std_jobs', 
                             'loan_count', 'avg_loan_amount']
        job_metrics['jobs_per_dollar'] = job_metrics['avg_jobs'] / job_metrics['avg_loan_amount']
        insights['job_metrics_by_size'] = job_metrics
        
        # Job creation efficiency analysis
        df['jobs_per_dollar'] = df['JobsReported'] / df['InitialApprovalAmount']
        df['jobs_per_dollar_norm'] = stats.zscore(df['jobs_per_dollar'])
        
        # Find most efficient job creators (excluding outliers)
        efficient_creators = df[
            (df['jobs_per_dollar_norm'] > 1) & 
            (df['jobs_per_dollar_norm'] < 3)
        ].sort_values('jobs_per_dollar', ascending=False)
        
        insights['efficient_creators'] = efficient_creators.head(100)
        
        # Industry analysis
        industry_metrics = df.groupby('NAICSCode').agg({
            'JobsReported': ['mean', 'median', 'sum', 'count'],
            'InitialApprovalAmount': 'sum'
        })
        
        industry_metrics.columns = ['avg_jobs', 'median_jobs', 'total_jobs',
                                  'loan_count', 'total_loan_amount']
        industry_metrics['jobs_per_million'] = (
            industry_metrics['total_jobs'] / 
            (industry_metrics['total_loan_amount'] / 1_000_000)
        )
        
        insights['industry_metrics'] = industry_metrics.sort_values('jobs_per_million', 
                                                                  ascending=False)
        
        return insights
    
    def create_visualizations(self, df: pd.DataFrame, insights: Dict):
        """
        Create visualizations for non-technical stakeholders
        """
        # 1. Jobs per Dollar by Loan Size
        plt.figure(figsize=(12, 6))
        sns.barplot(data=insights['job_metrics_by_size'].reset_index(), 
                   x='loan_size_bucket', y='jobs_per_dollar')
        plt.title('Job Creation Efficiency by Loan Size')
        plt.xlabel('Loan Size Decile')
        plt.ylabel('Jobs Created per Dollar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/job_efficiency.png')
        
        # 2. Industry Performance
        top_industries = insights['industry_metrics'].head(15)
        
        plt.figure(figsize=(14, 7))
        sns.barplot(data=top_industries.reset_index(), 
                   x='NAICSCode', y='jobs_per_million')
        plt.title('Top 15 Industries by Jobs Created per Million Dollars')
        plt.xlabel('NAICS Code')
        plt.ylabel('Jobs per Million Dollars')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/top_industries.png')
        
        # 3. Prediction Error Analysis
        if self.predictions:
            errors = np.abs(self.predictions['ensemble'] - df['JobsReported'])
            df['prediction_error'] = errors
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='loan_size_bucket', y='prediction_error')
            plt.title('Model Prediction Error by Loan Size')
            plt.xlabel('Loan Size Decile')
            plt.ylabel('Absolute Prediction Error (Jobs)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/prediction_errors.png')
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate a business-friendly report
        """
        insights = self.analyze_job_creation_patterns(df)
        self.create_visualizations(df, insights)
        
        report = []
        report.append("PPP Loan Job Creation Analysis\n")
        report.append("================================\n\n")
        
        # Overall Statistics
        report.append("Overall Impact:")
        report.append(f"Total Jobs Created: {df['JobsReported'].sum():,.0f}")
        report.append(f"Total Loan Amount: ${df['InitialApprovalAmount'].sum():,.2f}")
        report.append(f"Average Jobs per Loan: {df['JobsReported'].mean():.1f}")
        report.append("")
        
        # Most Efficient Loan Sizes
        job_metrics = insights['job_metrics_by_size']
        most_efficient = job_metrics['jobs_per_dollar'].idxmax()
        report.append("Loan Size Efficiency:")
        report.append(f"Most efficient loan size category: {most_efficient}")
        report.append(f"Average jobs created in this category: {job_metrics.loc[most_efficient, 'avg_jobs']:.1f}")
        report.append("")
        
        # Top Performing Industries
        top_industries = insights['industry_metrics'].head(5)
        report.append("Top Performing Industries (Jobs per Million Dollars):")
        for idx, row in top_industries.iterrows():
            report.append(f"NAICS {idx}: {row['jobs_per_million']:.1f} jobs per million")
        report.append("")
        
        # Model Performance Insights
        if self.predictions:
            mape = mean_absolute_percentage_error(
                df['JobsReported'], self.predictions['ensemble']
            ) * 100
            report.append("Model Performance:")
            report.append(f"Average Percentage Error: {mape:.1f}%")
            report.append(f"Within 20% accuracy: {(np.abs(df['JobsReported'] - self.predictions['ensemble']) / df['JobsReported'] <= 0.2).mean()*100:.1f}%")
        
        return "\n".join(report)

def main():
    # Load the data
    df = pd.read_parquet("../data/processed/processed_ppp_data_full.parquet")
    
    # Initialize evaluator
    evaluator = PPPModelEvaluator()
    
    # Generate and save report
    report = evaluator.generate_report(df)
    
    with open('outputs/business_report.txt', 'w') as f:
        f.write(report)
        
    print("Evaluation complete. Check outputs/ directory for visualizations and report.")

if __name__ == "__main__":
    main()