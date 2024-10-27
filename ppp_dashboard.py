import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Simplified industry data with just the essentials
industries = {
    'Agriculture': {
        'efficiency': 45.2,
        'risk_factor': 0.8,
    },
    'Manufacturing': {
        'efficiency': 203.5,
        'risk_factor': 0.9,
    },
    'Retail': {
        'efficiency': 112.7,
        'risk_factor': 1.0,
    },
    'Healthcare': {
        'efficiency': 88.7,
        'risk_factor': 0.7,
    },
    'Construction': {
        'efficiency': 52.3,
        'risk_factor': 1.1,
    },
    'Technology': {
        'efficiency': 95.4,
        'risk_factor': 0.85,
    }
}

# Simple regional adjustments
regions = {
    "Northeast": 1.15,
    "Midwest": 1.08,
    "South": 0.95,
    "West": 1.05
}

def predict_jobs(loan_amount: float, industry: str, region: str = None) -> dict:
    """Simple job prediction"""
    # Base calculation
    base_jobs = (loan_amount / 1_000_000) * industries[industry]['efficiency']
    
    # Apply risk factor
    adjusted_jobs = base_jobs * industries[industry]['risk_factor']
    
    # Apply regional adjustment if selected
    if region and region != "No Region Selected":
        adjusted_jobs *= regions[region]
    
    # Add uncertainty range (±15%)
    return {
        'prediction': round(adjusted_jobs),
        'lower': round(adjusted_jobs * 0.85),
        'upper': round(adjusted_jobs * 1.15)
    }

def main():
    st.set_page_config(page_title="PPP Loan Job Predictor", layout="wide")
    
    st.title("PPP Loan Jobs Predictor")
    st.write("Predict jobs created based on loan amount and industry")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Information")
        
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=0,
            max_value=10000000,
            step=1000,
            format="%d"
        )
        
        industry = st.selectbox(
            "Select Industry",
            options=list(industries.keys())
        )
    
    with col2:
        st.subheader("Regional Information (Optional)")
        region = st.selectbox(
            "Select Region",
            options=["No Region Selected"] + list(regions.keys())
        )
    
    # Make prediction when button is clicked
    if st.button("Predict Jobs"):
        if loan_amount > 0 and industry:
            prediction = predict_jobs(loan_amount, industry, region)
            
            # Display results
            st.header("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Jobs", prediction['prediction'])
            with col2:
                st.metric("Lower Estimate", prediction['lower'])
            with col3:
                st.metric("Upper Estimate", prediction['upper'])
            
            # Add explanation
            st.info(f"""
                Based on:
                - Industry efficiency: {industries[industry]['efficiency']} jobs per million
                - Industry risk factor: {industries[industry]['risk_factor']}
                - Regional adjustment: {regions[region] if region != "No Region Selected" else "None"}
            """)
    
    # Show industry comparison
    st.header("Industry Comparison")
    
    industry_data = pd.DataFrame([
        {
            'Industry': name,
            'Jobs per Million': data['efficiency'],
            'Risk Factor': data['risk_factor']
        }
        for name, data in industries.items()
    ])
    
    fig = px.bar(
        industry_data,
        x='Industry',
        y='Jobs per Million',
        color='Risk Factor',
        title='Jobs Created per Million Dollars by Industry'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation of methodology
    with st.expander("How predictions are calculated"):
        st.markdown("""
            The job prediction is calculated using:
            1. Base calculation: Loan Amount × Industry Efficiency
            2. Risk adjustment: Base × Industry Risk Factor
            3. Regional adjustment (if selected): Risk-adjusted × Regional Factor
            4. Uncertainty range: ±15% of final prediction
            
            **Note:** These predictions are estimates based on historical PPP data.
        """)

if __name__ == "__main__":
    main()