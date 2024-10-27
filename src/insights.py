import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import logging
from pathlib import Path

class PPPInsightsAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path)
        self.insights = {}
        
    def analyze_feature_importance(self):
        """Analyze which features are most predictive of job creation"""
        # Prepare numeric features
        numeric_features = [
            'InitialApprovalAmount',
            'PAYROLL_PROCEED',
            'UTILITIES_PROCEED',
            'RENT_PROCEED',
            'MORTGAGE_INTEREST_PROCEED',
            'REFINANCE_EIDL_PROCEED',
            'HEALTH_CARE_PROCEED'
        ]
        
        X = self.df[numeric_features].fillna(0)
        y = self.df['JobsReported']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.insights['feature_importance'] = importance
        
        # Analyze correlations
        correlations = self.df[numeric_features + ['JobsReported']].corr()['JobsReported'].sort_values(ascending=False)
        self.insights['correlations'] = correlations
        
        return importance
    
    def analyze_regional_patterns(self):
        """Analyze regional patterns in job creation efficiency"""
        regional_metrics = self.df.groupby('BorrowerState').agg({
            'JobsReported': ['mean', 'median', 'sum', 'count'],
            'InitialApprovalAmount': ['mean', 'sum']
        }).round(2)
        
        # Calculate efficiency metrics
        regional_metrics['jobs_per_million'] = (
            regional_metrics[('JobsReported', 'sum')] / 
            regional_metrics[('InitialApprovalAmount', 'sum')] * 1_000_000
        )
        
        self.insights['regional_metrics'] = regional_metrics
        
        # Analyze urban vs rural patterns
        urban_rural = self.df.groupby('RuralUrbanIndicator').agg({
            'JobsReported': ['mean', 'median'],
            'InitialApprovalAmount': ['mean'],
            'JobsReported': lambda x: len(x[x > 0]) / len(x)  # Success rate
        }).round(2)
        
        self.insights['urban_rural'] = urban_rural
        
        return regional_metrics
    
    def analyze_industry_patterns(self):
        """Analyze industry-specific patterns and insights"""
        industry_metrics = self.df.groupby('NAICSCode').agg({
            'JobsReported': ['mean', 'median', 'sum', 'std'],
            'InitialApprovalAmount': ['mean', 'sum'],
            'BusinessType': 'count'
        }).round(2)
        
        # Calculate efficiency and stability metrics
        industry_metrics['efficiency'] = (
            industry_metrics[('JobsReported', 'sum')] / 
            industry_metrics[('InitialApprovalAmount', 'sum')] * 1_000_000
        )
        
        industry_metrics['consistency'] = (
            industry_metrics[('JobsReported', 'median')] / 
            industry_metrics[('JobsReported', 'mean')]
        )
        
        self.insights['industry_metrics'] = industry_metrics
        
        return industry_metrics
    
    def analyze_loan_size_impact(self):
        """Analyze how loan size impacts job creation"""
        self.df['loan_size_bucket'] = pd.qcut(
            self.df['InitialApprovalAmount'], 
            q=10, 
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
        )
        
        loan_size_metrics = self.df.groupby('loan_size_bucket').agg({
            'JobsReported': ['mean', 'median', 'std'],
            'InitialApprovalAmount': ['mean', 'min', 'max']
        }).round(2)
        
        # Calculate efficiency by loan size
        loan_size_metrics['jobs_per_million'] = (
            loan_size_metrics[('JobsReported', 'mean')] / 
            loan_size_metrics[('InitialApprovalAmount', 'mean')] * 1_000_000
        )
        
        self.insights['loan_size_metrics'] = loan_size_metrics
        
        return loan_size_metrics
    
    def generate_report(self):
        """Generate a comprehensive insights report"""
        # Run all analyses
        self.analyze_feature_importance()
        self.analyze_regional_patterns()
        self.analyze_industry_patterns()
        self.analyze_loan_size_impact()
        
        report = []
        report.append("PPP Loan Program Analysis Insights")
        report.append("================================\n")
        
        # Top Predictors of Job Creation
        report.append("Key Predictors of Job Creation:")
        for idx, row in self.insights['feature_importance'].head().iterrows():
            report.append(f"- {row['feature']}: {row['importance']:.3f} importance score")
        report.append("")
        
        # Loan Size Impact
        report.append("Loan Size Impact Analysis:")
        loan_metrics = self.insights['loan_size_metrics']
        report.append(f"- Smallest loans (Q1) average {loan_metrics[('JobsReported', 'mean')]['Q1']:.1f} jobs")
        report.append(f"- Largest loans (Q10) average {loan_metrics[('JobsReported', 'mean')]['Q10']:.1f} jobs")
        report.append(f"- Most efficient size bracket: {loan_metrics['jobs_per_million'].idxmax()}")
        report.append("")
        
        # Regional Insights
        report.append("Regional Insights:")
        regional = self.insights['regional_metrics']
        top_states = regional['jobs_per_million'].nlargest(5)
        report.append("Top performing states (jobs per million dollars):")
        for state, value in top_states.items():
            report.append(f"- {state}: {value:.1f}")
        report.append("")
        
        # Industry Insights
        report.append("Industry Insights:")
        industry = self.insights['industry_metrics']
        top_industries = industry['efficiency'].nlargest(5)
        report.append("Most efficient industries (jobs per million dollars):")
        for naics, value in top_industries.items():
            report.append(f"- NAICS {naics}: {value:.1f}")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create insightful visualizations"""
        # Create output directory
        Path('outputs').mkdir(exist_ok=True)
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 6))
        importance_data = self.insights['feature_importance']
        sns.barplot(x='importance', y='feature', data=importance_data)
        plt.title('Feature Importance in Predicting Job Creation')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png')
        
        # 2. Regional Performance Map
        # (Would require additional mapping library)
        
        # 3. Loan Size Impact
        plt.figure(figsize=(12, 6))
        loan_metrics = self.insights['loan_size_metrics']
        sns.lineplot(data=loan_metrics['jobs_per_million'])
        plt.title('Job Creation Efficiency by Loan Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/loan_size_impact.png')

def main():
    analyzer = PPPInsightsAnalyzer("../data/processed/processed_ppp_data_full.parquet")
    report = analyzer.generate_report()
    
    # Save report
    with open('outputs/ppp_insights_report.txt', 'w') as f:
        f.write(report)
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print("Analysis complete. Check outputs/ directory for results.")

if __name__ == "__main__":
    main()