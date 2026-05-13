#!/usr/bin/env python3
"""
Credit Scoring & Creditworthiness Prediction System
Streamlit Interactive Dashboard

Run: streamlit run app.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.model_io import load_all
from src.models.predictor import CreditPredictor
from src.utils.logger import get_logger
import shap
import matplotlib.pyplot as plt

log = get_logger("streamlit_app")

# PAGE CONFIG & STYLING
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main Header Styling */
    .main-header {
        color: #1a3a52;
        text-align: center;
        margin-bottom: 30px;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* Metric Cards with Enhanced Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Risk Level Cards */
    .risk-low { 
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(46, 204, 113, 0.2);
    }
    .risk-medium { 
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.2);
    }
    .risk-high { 
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(231, 76, 60, 0.2);
    }
    
    /* Input Section Styling */
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    /* Improved Button Styling */
    .stButton > button {
        font-size: 16px;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .badge-approved {
        background-color: #d4edda;
        color: #155724;
    }
    
    .badge-rejected {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Results Container */
    .results-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Info Box Styling */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    /* Success Box Styling */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    /* Error Box Styling */
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# LOAD CONFIG & MODELS
@st.cache_resource
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models_and_data():
    try:
        cfg = load_config()
        models, scaler, feature_names = load_all(cfg["outputs"]["models_dir"])
        
        processed_data = pd.read_csv(cfg["data"]["processed_path"])
        metrics_csv = pd.read_csv(cfg["outputs"]["metrics_csv"])
        
        return models, scaler, feature_names, cfg, processed_data, metrics_csv
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Run `python main.py` first to train models")
        st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Home", "Predict Risk", "Model Comparison", 
     "Feature Analysis", "SHAP Explainability", "Dataset Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Predicts loan default risk using ML models trained on real "
    "financial data to assess creditworthiness of applicants."
)

# Load data at startup
models, scaler, feature_names, cfg, processed_data, metrics_csv = load_models_and_data()

# PAGE 1: HOME
if page == "Home":
    st.markdown("<h1 class='main-header'>Credit Scoring & Creditworthiness Prediction System</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(processed_data):,}",
            delta="Training Data"
        )
    
    with col2:
        default_rate = (processed_data["loan_status"].sum() / len(processed_data) * 100)
        st.metric(
            label="Default Rate",
            value=f"{default_rate:.1f}%",
            delta=f"{processed_data['loan_status'].sum():,} defaults"
        )
    
    with col3:
        best_model = metrics_csv.loc[metrics_csv["ROC-AUC"].idxmax(), "Model"]
        best_auc = metrics_csv["ROC-AUC"].max()
        st.metric(
            label="Best Model",
            value=f"{best_model}",
            delta=f"ROC-AUC: {best_auc:.4f}"
        )
    
    st.markdown("---")
    
    st.subheader("System Overview")
    st.write("""
    This system uses machine learning to predict credit risk and assess the creditworthiness of loan applicants.
    
    **Key Capabilities:**
    - Real-time Risk Prediction — Classify applicants as low/medium/high risk
    - Model Comparison — Compare performance of 4 different classifiers
    - Feature Analysis — Understand which factors drive credit decisions
    - SHAP Explainability — Transparent AI for model interpretability
    - Performance Metrics — Detailed evaluation using industry-standard metrics
    
    **Use Cases:**
    - Approve/Reject loan applications
    - Set appropriate interest rates based on risk
    - Identify high-risk applicants for manual review
    - Ensure fair and transparent lending decisions
    """)
    
    st.markdown("---")
    st.subheader("How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **1. Data Preprocessing**
        - Handle missing values
        - Remove outliers
        - Encode categorical variables
        - Balance classes with SMOTE
        
        **2. Feature Engineering**
        - Debt-to-Income Ratio
        - Loan-to-Income Ratio
        - Income per Credit Year
        - High-Risk Grade Flag
        - Interest Rate × Loan Amount
        - Employment Stability Score
        """)
    
    with col2:
        st.write("""
        **3. Model Training**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - XGBoost
        
        **4. Evaluation**
        - Accuracy, Precision, Recall
        - F1-Score, ROC-AUC, PR-AUC
        - Cross-Validation (5-fold)
        - Confusion Matrix Analysis
        """)

# PAGE 2: PREDICT RISK
elif page == "Predict Risk":
    st.markdown("<h1 class='main-header'>Credit Risk Prediction</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Model selection with info
    best_model_name = metrics_csv.loc[metrics_csv["ROC-AUC"].idxmax(), "Model"]
    col_model, col_info = st.columns([3, 1])
    
    with col_model:
        selected_model = st.selectbox(
            "Select Prediction Model:",
            list(models.keys()),
            index=list(models.keys()).index(best_model_name),
            help="Choose which ML model to use for prediction"
        )
    
    with col_info:
        best_auc = metrics_csv[metrics_csv["Model"] == selected_model]["ROC-AUC"].values[0]
        st.metric("Model ROC-AUC", f"{best_auc:.4f}")
    
    st.markdown("---")
    
    # Personal Information Section
    with st.expander("Personal Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=35, 
                                        help="Applicant's age in years")
        
        with col2:
            person_income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, 
                                           value=85000, step=1000, help="Total annual income")
        
        with col3:
            person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, 
                                               max_value=60.0, value=8.0, step=0.5, help="Years in current job")
    
    # Loan Details Section
    with st.expander("Loan Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=500000, 
                                       value=10000, step=500, help="Amount requested")
        
        with col2:
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, 
                                           value=7.5, step=0.1, help="Proposed interest rate")
        
        with col3:
            loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, 
                                                 value=0.12, step=0.01, help="Loan/Income ratio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_intent = st.selectbox("Loan Intent", 
                                      ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                                      help="Purpose of the loan")
        
        with col2:
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"],
                                     help="Lender-assigned credit grade (A=best, G=worst)")
        
        with col3:
            person_home_ownership = st.selectbox("Home Ownership", 
                                               ["OWN", "RENT", "MORTGAGE", "OTHER"],
                                               help="Current housing status")
    
    # Credit History Section
    with st.expander("Credit History & Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cb_person_default_on_file = st.selectbox("Previous Default on File", ["N", "Y"],
                                                    help="Has applicant defaulted before?")
        
        with col2:
            cb_person_cred_hist_length = st.number_input("Credit History (years)", min_value=0, 
                                                        max_value=50, value=10, help="Years of credit history")
        
        with col3:
            risk_threshold = st.slider("Risk Threshold", min_value=0.0, max_value=1.0, value=0.5, 
                                      step=0.05, help="Probability threshold for default classification")
    
    st.markdown("---")
    
    # Prediction Button
    predict_col, spacer_col = st.columns([1, 2])
    with predict_col:
        predict_button = st.button("Predict Credit Risk", key="predict_btn", use_container_width=True, 
                                   help="Click to analyze the applicant's creditworthiness")
    
    if predict_button:
        
        applicant_data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }
        
        try:
            use_scaling = (selected_model == "Logistic Regression")
            predictor = CreditPredictor(
                model=models[selected_model],
                scaler=scaler,
                feature_names=feature_names,
                use_scaling=use_scaling
            )
            
            result = predictor.predict(applicant_data, threshold=risk_threshold)
            
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Prediction Results</h2>", unsafe_allow_html=True)
            
            # Main Results - Risk Status
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if result["label"] == 1:
                    st.markdown(f"<div class='risk-high'><h3>HIGH RISK</h3></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-low'><h3>LOW RISK</h3></div>", unsafe_allow_html=True)
            
            with col2:
                prob = result['probability']
                st.metric("Default Probability", f"{prob:.1%}", 
                         delta=f"Threshold: {risk_threshold:.1%}")
            
            with col3:
                st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
            
            with col4:
                st.metric("Risk Band", result['risk_band'].split()[0])
            
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["Decision", "Details", "Risk Factors"])
            
            with tab1:
                recommendation = result['recommendation']
                if "Approve" in recommendation:
                    st.markdown(f"<div class='success-box'><h4>Recommendation: APPROVE</h4><p>{recommendation}</p></div>",
                              unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='error-box'><h4>Recommendation: REJECT</h4><p>{recommendation}</p></div>",
                              unsafe_allow_html=True)
            
            with tab2:
                st.write("Prediction Details")
                details_df = pd.DataFrame({
                    "Metric": ["Default Probability", "Risk Score", "Risk Band", "Classification", "Model Used"],
                    "Value": [
                        f"{result['probability']:.4f}",
                        f"{result['risk_score']:.1f}",
                        result['risk_band'],
                        "Default" if result["label"] == 1 else "Good Standing",
                        selected_model
                    ]
                })
                st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            with tab3:
                st.write("Top Risk Contributing Factors")
                factors = result.get('key_risk_factors', [])
                if factors:
                    for i, factor in enumerate(factors[:7], 1):
                        st.write(f"{i}. {factor}")
                else:
                    st.info("No major risk factors detected")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            log.error(f"Prediction error: {e}")

# PAGE 3: MODEL COMPARISON
elif page == "Model Comparison":
    st.markdown("<h1 class='main-header'>Model Performance Comparison</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Performance Metrics")
    
    # Create a clean display table
    display_df = metrics_csv.copy()
    display_df = display_df[[col for col in display_df.columns if not col.startswith("_")]]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC-AUC Comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_csv["Model"],
            y=metrics_csv["ROC-AUC"],
            marker=dict(color=metrics_csv["ROC-AUC"], colorscale="Viridis"),
            text=[f"{x:.4f}" for x in metrics_csv["ROC-AUC"]],
            textposition="outside"
        ))
        fig.update_layout(
            title="ROC-AUC Score Comparison",
            xaxis_title="Model",
            yaxis_title="ROC-AUC",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1-Score Comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_csv["Model"],
            y=metrics_csv["F1-Score"],
            marker=dict(color=metrics_csv["F1-Score"], colorscale="Blues"),
            text=[f"{x:.4f}" for x in metrics_csv["F1-Score"]],
            textposition="outside"
        ))
        fig.update_layout(
            title="F1-Score Comparison",
            xaxis_title="Model",
            yaxis_title="F1-Score",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision vs Recall
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_csv["Recall"],
            y=metrics_csv["Precision"],
            mode="markers+text",
            text=metrics_csv["Model"],
            textposition="top center",
            marker=dict(size=12, color=metrics_csv["ROC-AUC"], colorscale="Plasma")
        ))
        fig.update_layout(
            title="Precision vs Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Accuracy Comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_csv["Model"],
            y=metrics_csv["Accuracy"],
            marker=dict(color=metrics_csv["Accuracy"], colorscale="Greens"),
            text=[f"{x:.4f}" for x in metrics_csv["Accuracy"]],
            textposition="outside"
        ))
        fig.update_layout(
            title="Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Metrics Explanation")
    
    metrics_info = {
        "Accuracy": "Percentage of all predictions that are correct. Can be misleading on imbalanced data.",
        "Precision": "Of applicants flagged as default, what % actually default. (Minimize false alarms)",
        "Recall": "Of actual defaults, what % did the model catch. (Minimize missed defaults)",
        "F1-Score": "Harmonic mean of Precision and Recall. Best single metric for imbalanced classification.",
        "ROC-AUC": "Area under ROC curve (0.5=random, 1.0=perfect). Industry standard is ≥ 0.75.",
        "PR-AUC": "Area under Precision-Recall curve. More informative on heavily imbalanced data."
    }
    
    for metric, explanation in metrics_info.items():
        st.write(f"**{metric}**: {explanation}")

# PAGE 4: FEATURE ANALYSIS
elif page == "Feature Analysis":
    st.markdown("<h1 class='main-header'>Feature Importance Analysis</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Select Model for Feature Analysis")
    
    analysis_model = st.selectbox("Choose Model:", list(models.keys()), key="feature_analysis_model")
    
    model = models[analysis_model]
    
    # Check if model has feature_importances_
    if hasattr(model, "feature_importances_"):
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True).tail(20)
        
        # Visualization
        fig = go.Figure(go.Bar(
            x=feature_importance_df["Importance"],
            y=feature_importance_df["Feature"],
            orientation="h",
            marker=dict(color=feature_importance_df["Importance"], colorscale="Viridis")
        ))
        fig.update_layout(
            title=f"Top 20 Important Features — {analysis_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Feature Statistics")
        
        stats_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("This model doesn't have built-in feature importance. "
                "Check the SHAP Explainability page for feature contributions.")

# PAGE 5: SHAP EXPLAINABILITY
elif page == "SHAP Explainability":
    st.markdown("<h1 class='main-header'>SHAP Model Explainability</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("SHAP (SHapley Additive exPlanations)")
    st.write("""
    SHAP assigns each feature a contribution score for every prediction, making AI decisions transparent and explainable.
    """)
    
    st.markdown("---")
    
    shap_dir = cfg["outputs"]["shap_dir"]
    
    if os.path.exists(shap_dir) and os.listdir(shap_dir):
        
        # Get available SHAP plots
        shap_files = sorted([f for f in os.listdir(shap_dir) if f.endswith(".png")])
        
        if shap_files:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Plots")
                summary_files = [f for f in shap_files if "summary" in f]
                if summary_files:
                    for f in summary_files:
                        model_name = f.replace("shap_summary_", "").replace(".png", "").replace("_", " ").title()
                        st.write(f"**{model_name}**")
                        img = Image.open(os.path.join(shap_dir, f))
                        st.image(img, use_column_width=True)
            
            with col2:
                st.subheader("Bar Plots")
                bar_files = [f for f in shap_files if "bar" in f]
                if bar_files:
                    for f in bar_files:
                        model_name = f.replace("shap_bar_", "").replace(".png", "").replace("_", " ").title()
                        st.write(f"**{model_name}**")
                        img = Image.open(os.path.join(shap_dir, f))
                        st.image(img, use_column_width=True)
        
        else:
            st.warning("No SHAP plots found. Run `python main.py` to generate them.")
    
    else:
        st.warning("SHAP directory not found. Run `python main.py` to generate explainability plots.")

# PAGE 6: DATASET INFO
elif page == "Dataset Info":
    st.markdown("<h1 class='main-header'>Dataset Information</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(processed_data):,}")
    
    with col2:
        st.metric("Total Features", len(feature_names))
    
    with col3:
        st.metric("Good Loans", f"{(processed_data['loan_status'] == 0).sum():,}")
    
    with col4:
        st.metric("Default Loans", f"{(processed_data['loan_status'] == 1).sum():,}")
    
    st.markdown("---")
    
    st.subheader("Target Variable Distribution")
    
    fig = go.Figure()
    target_dist = processed_data['loan_status'].value_counts()
    fig.add_trace(go.Pie(
        labels=["Good Standing (0)", "Default (1)"],
        values=target_dist.values,
        marker=dict(colors=["#2ecc71", "#e74c3c"]),
        textinfo="label+percent+value"
    ))
    fig.update_layout(title="Loan Status Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Numeric Features Statistics")
    
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'loan_status' in numeric_cols:
        numeric_cols.remove('loan_status')
    
    stats_df = processed_data[numeric_cols].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Feature Engineering Summary")
    
    engineered_features = {
        "debt_to_income_ratio": "Monthly loan payment ÷ monthly income. Core affordability metric.",
        "loan_to_income_ratio": "Total loan ÷ annual income. Borrowing burden indicator.",
        "income_per_cred_year": "Income ÷ credit history years. Financial maturity proxy.",
        "high_risk_grade_flag": "1 if grade ∈ {E,F,G}. Sub-prime indicator.",
        "int_rate_x_loan_amnt": "Interest rate × loan amount. Total interest burden.",
        "employment_stability_score": "Sigmoid-scaled employment length. Job stability (0–1)."
    }
    
    for feature, description in engineered_features.items():
        st.write(f"**{feature}**  \n{description}")
    
    st.markdown("---")
    
    st.subheader("Data Sample")
    
    n_samples = st.slider("Number of samples to display:", min_value=5, max_value=50, value=10)
    st.dataframe(processed_data.head(n_samples), use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 30px;'>
    <p><strong>Credit Scoring System</strong> | Streamlit, scikit-learn, XGBoost</p>
    <p>SHAP for model explainability | Fair & Transparent Lending</p>
</div>
""", unsafe_allow_html=True)
