import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Goldman Sachs Stock Prediction",
    layout="wide" 
)

@st.cache_resource
def load_models():
    models = joblib.load('gs_models.joblib')
    scalers = joblib.load('gs_scalers.joblib')
    feature_cols = joblib.load('feature_columns.joblib')
    predictions = joblib.load('gs_predictions.joblib')
    results = joblib.load('gs_results.joblib')
    return models, scalers, feature_cols, predictions, results

@st.cache_data
def load_data():
    test_df = pd.read_csv('gs_test_data.csv')
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    return test_df

models, scalers, feature_cols, predictions, results = load_models()
test_df = load_data()

st.title("Goldman Sachs Stock Price Prediction Dashboard")
st.markdown("### Multi-Horizon Machine Learning Forecasting System")

st.sidebar.header("Configuration")

horizon = st.sidebar.selectbox(
    "Select Prediction Horizon",
    options=['1-day', '5-day', '20-day'],
    index=0
)

horizon_key = horizon.split('-')[0] + 'd'

model_name = st.sidebar.selectbox(
    "Select Model",
    options=['Linear Regression', 'Random Forest', 'XGBoost', 'Ensemble'],
    index=0
)

min_date = test_df['Date'].min().date()
max_date = test_df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

st.header(f"{horizon} Prediction Performance - {model_name}")

metrics = results[horizon_key][model_name]
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("RMSE", f"${metrics['RMSE']:.2f}")
with col2:
    st.metric("MAE", f"${metrics['MAE']:.2f}")
with col3:
    st.metric("R² Score", f"{metrics['R²']:.4f}")
with col4:
    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
with col5:
    st.metric("Direction Acc.", f"{metrics['Directional Accuracy']:.2f}%")

if metrics['R²'] < 0:
    st.warning(f"**Warning:** {model_name} has negative R² ({metrics['R²']:.4f}), meaning it performs worse than predicting the average price. This model overfitted to training data.")

st.header("Actual vs Predicted Prices")

#-------------------------------------------
actual = predictions[horizon_key]['actual']
pred = predictions[horizon_key][model_name]

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (test_df['Date'].dt.date >= start_date) & (test_df['Date'].dt.date <= end_date)
    filtered_dates = test_df[mask]['Date'].values[:len(actual)]
    
    date_mask = (test_df['Date'].dt.date[:len(actual)] >= start_date) & (test_df['Date'].dt.date[:len(actual)] <= end_date)
    filtered_actual = actual[date_mask]
    filtered_pred = pred[date_mask]
else:
    # If no date range selected, show all data
    filtered_dates = test_df['Date'].values[:len(actual)]
    filtered_actual = actual
    filtered_pred = pred

#-------------------------------------------
# Chart
fig = go.Figure()

# Actual price line
fig.add_trace(go.Scatter(
    x=filtered_dates,
    y=filtered_actual,
    mode='lines',
    name='Actual Price',
    line=dict(color='black', width=2)
))

# Predicted price line 
fig.add_trace(go.Scatter(
    x=filtered_dates,
    y=filtered_pred,
    mode='lines',
    name=f'{model_name} Prediction',
    line=dict(color='blue', width=2, dash='dash'),
    opacity=0.7
))

fig.update_layout(
    title=f'{horizon} Ahead Predictions - {model_name}',
    xaxis_title='Date',
    yaxis_title='Stock Price ($)',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

#-------------------------------------------
st.header("Model Comparison Across All Horizons")
comparison_data = []
for h in ['1d', '5d', '20d']:
    for m in ['Linear Regression', 'Random Forest', 'XGBoost', 'Ensemble']:
        metrics = results[h][m]
        comparison_data.append({
            'Horizon': h.replace('d', '-day'),
            'Model': m,
            'RMSE ($)': f"${metrics['RMSE']:.2f}",
            'MAE ($)': f"${metrics['MAE']:.2f}",
            'R²': f"{metrics['R²']:.4f}",
            'MAPE (%)': f"{metrics['MAPE']:.2f}%",
            'Direction Acc. (%)': f"{metrics['Directional Accuracy']:.2f}%"
        })

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(comparison_df, use_container_width=True, height=600)

#-------------------------------------------
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("What Worked")
    st.markdown("""
    - **Linear Regression** performed exceptionally well across all horizons
    - 1-day predictions: RMSE of $12.95 (only 1.6% error)
    - R² scores: 0.99 (1-day), 0.97 (5-day), 0.88 (20-day)
    - Model captures trend very well
    """)

with col2:
    st.subheader("What Didn't Work")
    st.markdown("""
    - **Random Forest & XGBoost** severely overfitted
    - Negative R² scores = worse than predicting average
    - RMSE $230 - 280 (vs Linear's $13-50)
    - Likely need hyperparameter tuning or feature engineering
    """)

#-------------------------------------------
st.header("Download Results")

col1, col2 = st.columns(2)

with col1:
    download_df = pd.DataFrame({
        'Date': filtered_dates,
        'Actual': filtered_actual,
        f'{model_name}_Prediction': filtered_pred,
        'Error': filtered_actual - filtered_pred
    })
    
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name=f"gs_predictions_{horizon}_{model_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )

with col2:
    csv_comparison = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download Model Comparison CSV",
        data=csv_comparison,
        file_name="gs_model_comparison.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("""
**Disclaimer:** This is an educational machine learning project. Predictions are based on historical patterns 
and do not constitute financial advice. Past performance does not guarantee future results.
""")