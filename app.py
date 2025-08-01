import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Page Config
st.set_page_config(page_title='Telco Churn Dashboard', layout='wide', page_icon='📈')

# Helper functions
def load_models():
    """Load trained models and preprocessing objects"""
    models = {}
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        file_name = f"{model_name}_model.pkl"
        if os.path.exists(file_name):
            models[model_name] = joblib.load(file_name)

    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
    label_encoders = joblib.load('label_encoders.pkl') if os.path.exists('label_encoders.pkl') else None
    feature_names = joblib.load('feature_names.pkl') if os.path.exists('feature_names.pkl') else None

    return models, scaler, label_encoders, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    return df

# Load data and models
df = load_data()
models, scaler, label_encoders, feature_names = load_models()

st.title('📈 Telco Customer Churn Dashboard')
st.markdown('Interactive dashboard for exploring churn patterns, model performance, and individual customer risk.')

# KPI Metrics
churn_rate = (df['Churn'] == 'Yes').mean()
total_customers = len(df)
total_churned = (df['Churn'] == 'Yes').sum()

col1, col2, col3 = st.columns(3)
col1.metric('Total Customers', f"{total_customers}")
col2.metric('Churned Customers', f"{total_churned}")
col3.metric('Churn Rate', f"{churn_rate:.2%}")

st.markdown('---')

## Sidebar filters
st.sidebar.header('Filter Customers')
contract_filter = st.sidebar.multiselect('Contract Type', options=df['Contract'].unique(), default=list(df['Contract'].unique()))
service_filter = st.sidebar.multiselect('Internet Service', options=df['InternetService'].unique(), default=list(df['InternetService'].unique()))

filtered_df = df[(df['Contract'].isin(contract_filter)) & (df['InternetService'].isin(service_filter))]

st.subheader('Customer Segmentation')
seg_col1, seg_col2 = st.columns(2)

with seg_col1:
    fig_contract = px.histogram(filtered_df, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
    st.plotly_chart(fig_contract, use_container_width=True)

with seg_col2:
    fig_tenure = px.histogram(filtered_df, x='tenure', color='Churn', nbins=20, title='Churn by Tenure')
    st.plotly_chart(fig_tenure, use_container_width=True)

st.subheader('Monthly Charges vs. Tenure')
fig_scatter = px.scatter(filtered_df, x='tenure', y='MonthlyCharges', color='Churn', title='Charges vs Tenure')
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown('---')

st.subheader('Model Predictions')
if not models:
    st.warning('No trained models found. Please run model_training.py first.')
else:
    model_choice = st.selectbox('Select Model', options=list(models.keys()))
    model = models[model_choice]

    st.markdown('### Predict Churn Probability')
    customer_id = st.selectbox('Select CustomerID', options=df['customerID'])
    customer_row = df[df['customerID'] == customer_id]
    st.write(customer_row)

    # Preprocess single row
    input_df = customer_row.drop('Churn', axis=1).copy()

    # Encode categorical columns
    for col in input_df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Align features
    X_input = input_df[feature_names]

    # Scale if logistic regression
    if model_choice == 'logistic_regression' and scaler is not None:
        X_input_scaled = scaler.transform(X_input)
        pred_proba = model.predict_proba(X_input_scaled)[0][1]
    else:
        pred_proba = model.predict_proba(X_input)[0][1]

    risk_level = 'High' if pred_proba > 0.7 else 'Medium' if pred_proba > 0.3 else 'Low'

    st.metric('Churn Probability', f"{pred_proba:.2%}")
    st.metric('Risk Level', risk_level)

    if st.checkbox('Show Feature Importance (Tree-based Models only)'):
        if model_choice in ['random_forest', 'xgboost']:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Top 10 Feature Importances')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info('Feature importance is not available for the selected model.')
