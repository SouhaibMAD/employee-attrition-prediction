"""
Streamlit Interface for Employee Attrition Prediction

This app allows users to input employee information and get predictions
about whether they are likely to leave the company.

Author: Souhaib MADHOUR
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HR Analytics - Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_pipeline():
    """Load the best model, preprocessing pipeline, and feature names."""
    try:
        # Try loading from models directory (if running from root)
        model_path = "models/best_model.pkl"
        pipeline_path = "models/preprocessing_pipeline.pkl"
        feature_names_path = "models/feature_names.pkl"
        model_info_path = "models/best_model_info.json"
        
        # If not found, try from src/models
        if not os.path.exists(model_path):
            model_path = "src/models/best_model.pkl"
            pipeline_path = "src/models/preprocessing_pipeline.pkl"
            feature_names_path = "src/models/feature_names.pkl"
            model_info_path = "src/models/best_model_info.json"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(pipeline_path)
        feature_names = joblib.load(feature_names_path)
        
        # Load model info if available
        model_info = {}
        if os.path.exists(model_info_path):
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        
        return model, preprocessor, feature_names, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def create_input_form():
    """Create input form for employee features."""
    
    st.sidebar.header("📊 Employee Information")
    
    # Numerical features
    st.sidebar.subheader("Personal & Demographics")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    
    st.sidebar.subheader("Job Information")
    department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    job_level = st.sidebar.slider("Job Level", min_value=1, max_value=5, value=2, step=1)
    business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    
    st.sidebar.subheader("Education")
    education = st.sidebar.slider("Education Level", min_value=1, max_value=5, value=3, 
                                  help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
    education_field = st.sidebar.selectbox("Education Field", [
        "Life Sciences", "Other", "Medical", "Marketing", 
        "Technical Degree", "Human Resources"
    ])
    
    st.sidebar.subheader("Compensation & Benefits")
    monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=5000, step=100)
    daily_rate = st.sidebar.number_input("Daily Rate ($)", min_value=0, max_value=3000, value=800, step=10)
    hourly_rate = st.sidebar.number_input("Hourly Rate ($)", min_value=0, max_value=200, value=65, step=1)
    monthly_rate = st.sidebar.number_input("Monthly Rate ($)", min_value=0, max_value=30000, value=15000, step=100)
    percent_salary_hike = st.sidebar.number_input("Percent Salary Hike (%)", min_value=0, max_value=50, value=11, step=1)
    stock_option_level = st.sidebar.slider("Stock Option Level", min_value=0, max_value=3, value=0, step=1)
    
    st.sidebar.subheader("Work Experience")
    total_working_years = st.sidebar.number_input("Total Working Years", min_value=0, max_value=50, value=10, step=1)
    num_companies_worked = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=20, value=1, step=1)
    years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=50, value=5, step=1)
    years_in_current_role = st.sidebar.number_input("Years in Current Role", min_value=0, max_value=30, value=3, step=1)
    years_since_last_promotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1, step=1)
    years_with_curr_manager = st.sidebar.number_input("Years with Current Manager", min_value=0, max_value=20, value=2, step=1)
    training_times_last_year = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, value=2, step=1)
    
    st.sidebar.subheader("Location")
    distance_from_home = st.sidebar.number_input("Distance from Home (miles)", min_value=0, max_value=50, value=5, step=1)
    
    st.sidebar.subheader("Satisfaction & Engagement")
    environment_satisfaction = st.sidebar.slider("Environment Satisfaction", min_value=1, max_value=4, value=3,
                                                 help="1=Low, 2=Medium, 3=High, 4=Very High")
    job_satisfaction = st.sidebar.slider("Job Satisfaction", min_value=1, max_value=4, value=3,
                                        help="1=Low, 2=Medium, 3=High, 4=Very High")
    relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction", min_value=1, max_value=4, value=3,
                                                 help="1=Low, 2=Medium, 3=High, 4=Very High")
    work_life_balance = st.sidebar.slider("Work Life Balance", min_value=1, max_value=4, value=3,
                                         help="1=Bad, 2=Good, 3=Better, 4=Best")
    job_involvement = st.sidebar.slider("Job Involvement", min_value=1, max_value=4, value=3,
                                       help="1=Low, 2=Medium, 3=High, 4=Very High")
    performance_rating = st.sidebar.slider("Performance Rating", min_value=1, max_value=4, value=3,
                                          help="1=Low, 2=Good, 3=Excellent, 4=Outstanding")
    
    # Create DataFrame with all features
    data = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager
    }
    
    return pd.DataFrame([data])

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">👥 Employee Attrition Prediction System</div>', unsafe_allow_html=True)
    
    # Load model and pipeline
    with st.spinner("Loading model and preprocessing pipeline..."):
        model, preprocessor, feature_names, model_info = load_model_and_pipeline()
    
    # Display model information
    if model_info:
        model_name = model_info.get('model_name', 'Best Model')
        st.info(f"**Model:** {model_name} | **Training Date:** {model_info.get('training_date', 'N/A')}")
    
    # Create input form
    input_df = create_input_form()
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Preprocess input
            with st.spinner("Processing input data..."):
                X_processed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            prediction_proba = model.predict_proba(X_processed)[0]
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Result")
                if prediction == 1:
                    st.markdown('<p class="high-risk">⚠️ HIGH RISK: Employee is likely to leave</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="low-risk">✅ LOW RISK: Employee is likely to stay</p>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("📈 Probability Scores")
                prob_leave = prediction_proba[1] * 100
                prob_stay = prediction_proba[0] * 100
                
                st.metric("Probability of Leaving", f"{prob_leave:.2f}%")
                st.metric("Probability of Staying", f"{prob_stay:.2f}%")
                
                # Progress bars
                st.progress(prob_leave / 100, text=f"Risk Level: {prob_leave:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("🔍 Top Contributing Factors")
                
                # Get feature importances
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(feature_importance_df.set_index('Feature'))
                
                # Show top factors
                st.write("**Top 5 Most Important Features:**")
                for idx, row in feature_importance_df.head(5).iterrows():
                    st.write(f"- **{row['Feature']}**: {row['Importance']:.4f}")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                **This employee shows a high risk of attrition. Consider:**
                - Reviewing compensation and benefits
                - Discussing career development opportunities
                - Addressing work-life balance concerns
                - Improving job satisfaction through engagement initiatives
                - Providing additional training and growth opportunities
                """)
            else:
                st.success("""
                **This employee shows low risk of attrition. To maintain retention:**
                - Continue providing growth opportunities
                - Maintain competitive compensation
                - Keep engagement levels high
                - Regular check-ins and feedback
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    
    # Information section
    with st.expander("ℹ️ About This Model"):
        st.write("""
        This prediction model uses machine learning to assess the likelihood of employee attrition.
        
        **Model Performance:**
        - The model was trained on historical employee data
        - It considers multiple factors including job satisfaction, compensation, work-life balance, and career progression
        - Predictions are based on patterns learned from past employee data
        
        **How to Use:**
        1. Fill in all the employee information in the sidebar
        2. Click the "Predict Attrition Risk" button
        3. Review the prediction and probability scores
        4. Use the recommendations to take appropriate action
        
        **Note:** This is a predictive tool and should be used as a guide. Individual circumstances may vary.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>HR Analytics - Employee Attrition Prediction System</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

