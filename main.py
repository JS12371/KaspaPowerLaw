import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Constants for the regression
FAIR_SLOPE = 4.231517555680599
FAIR_INTERCEPT = -30.619674611147577
STD_DEV = 0.40220538718586324
GENESIS_DATE = datetime(2021, 11, 7)
END_DATE = datetime(2024, 7, 28)

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to plot results with heatmap and dates on x-axis
def plot_results(data):
    data['Days Since Genesis'] = (data.index - GENESIS_DATE).days
    dates = data.index
    prices = data['Close'].values
    days = data['Days Since Genesis'].values

    # Generate future dates up to 2035
    future_dates = pd.date_range(start=dates[-1], end='2035-12-31')
    future_days = (future_dates - GENESIS_DATE).days

    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_dates = dates.append(future_dates)

    # Calculate the regression lines
    regression_line = np.exp(FAIR_INTERCEPT) * all_days ** FAIR_SLOPE
    support_line = np.exp(FAIR_INTERCEPT - STD_DEV) * all_days ** FAIR_SLOPE
    resistance_line = np.exp(FAIR_INTERCEPT + STD_DEV) * all_days ** FAIR_SLOPE

    # Calculate percent deviations and percentiles
    percent_deviation = (prices - np.exp(FAIR_INTERCEPT) * days ** FAIR_SLOPE) / (np.exp(FAIR_INTERCEPT) * days ** FAIR_SLOPE) * 100
    percentiles = pd.qcut(percent_deviation, 100, labels=False)
    percentileNow = percentiles[-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sc = ax.scatter(dates, prices, c=percentiles, cmap='turbo', label='Price', alpha=0.75)
    plt.colorbar(sc, label='Percentile')
    ax.semilogy(all_dates, regression_line, label='Fair Line', color='white', linestyle='--')
    ax.semilogy(all_dates, support_line, label='Support Line', color='green', linestyle='--')
    ax.semilogy(all_dates, resistance_line, label='Resistance Line', color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.set_title('Price Data with Power Law Regression (Extended to 2035)', color='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.setp(ax.get_xticklabels(), color='white')
    plt.setp(ax.get_yticklabels(), color='white')
    st.pyplot(fig)

    return percentileNow

# Function to plot log-log graph with regression and bands
def plot_log_log(data):
    data['Days Since Genesis'] = (data.index - GENESIS_DATE).days
    days = data['Days Since Genesis'].values
    prices = data['Close'].values

    # Generate future days up to 2035
    future_days = (pd.date_range(start=data.index[-1], end='2035-12-31') - GENESIS_DATE).days

    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_prices = np.concatenate([prices, [np.nan] * len(future_days)])

    # Calculate the regression lines
    regression_line = np.exp(FAIR_INTERCEPT) * all_days ** FAIR_SLOPE
    support_line = np.exp(FAIR_INTERCEPT - STD_DEV) * all_days ** FAIR_SLOPE
    resistance_line = np.exp(FAIR_INTERCEPT + STD_DEV) * all_days ** FAIR_SLOPE

    # Calculate percent deviations and percentiles
    percent_deviation = (prices - np.exp(FAIR_INTERCEPT) * days ** FAIR_SLOPE) / (np.exp(FAIR_INTERCEPT) * days ** FAIR_SLOPE) * 100
    percentiles = pd.qcut(percent_deviation, 100, labels=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sc = ax.scatter(days, prices, c=percentiles, cmap='turbo', label='Price', alpha=0.75)
    plt.colorbar(sc, label='Percentile')
    ax.loglog(all_days, regression_line, label='Fair Line', color='white', linestyle='--')
    ax.loglog(all_days, support_line, label='Support Line', color='green', linestyle='--')
    ax.loglog(all_days, resistance_line, label='Resistance Line', color='red', linestyle='--')

    # Add vertical lines and labels for each new year starting from 2023
    for year in range(2023, 2036):
        year_days = (datetime(year, 1, 1) - GENESIS_DATE).days
        ax.axvline(year_days, color='mediumseagreen', linestyle=':')
        ax.text(year_days, ax.get_ylim()[0], str(year), color='mediumseagreen', verticalalignment='bottom', horizontalalignment='right', rotation=90)

    ax.set_xlabel('Days Since Genesis', color='white')
    ax.set_ylabel('Close Price', color='white')
    ax.legend()
    ax.set_title('Log-Log Plot of Days Since Genesis vs. Close Price with Regression Lines (Extended to 2035)', color='white')
    ax.grid(True, which="both", ls="--", color='gray')
    plt.setp(ax.get_xticklabels(), color='white')
    plt.setp(ax.get_yticklabels(), color='white')
    st.pyplot(fig)

# Function to calculate predicted values for a given date
def calculate_predicted_values(target_date):
    days_since_genesis = (pd.to_datetime(target_date) - GENESIS_DATE).days
    fair_value = np.exp(FAIR_INTERCEPT) * days_since_genesis ** FAIR_SLOPE
    support_value = np.exp(FAIR_INTERCEPT - STD_DEV) * days_since_genesis ** FAIR_SLOPE
    resistance_value = np.exp(FAIR_INTERCEPT + STD_DEV) * days_since_genesis ** FAIR_SLOPE
    return fair_value, support_value, resistance_value

# Streamlit App
st.set_page_config(page_title="KASPA POWER LAW", layout="wide", initial_sidebar_state="expanded")
st.title("KASPA POWER LAW")

# Sidebar settings
st.sidebar.header("Settings")
ticker = 'KAS-USD'
start_date = datetime(2022, 6, 1)
end_date = END_DATE

# Fetch data
data = fetch_data(ticker, start_date, end_date)

if not data.empty:
    st.write(f"### Log-Log Regression Parameters")

    # Log-transform the data
    data['Days Since Genesis'] = (data.index - GENESIS_DATE).days
    log_days = np.log(data['Days Since Genesis'].values).reshape(-1, 1)
    log_prices = np.log(data['Close'].values)

    # Perform linear regression on the log-transformed data
    model = LinearRegression()
    model.fit(log_days, log_prices)
    log_predicted_prices = model.predict(log_days)

    # Calculate R² value
    r_squared = r2_score(log_prices, log_predicted_prices)
    st.write(f"**R² Value**: {r_squared:.4f}")
    st.write("**Regression Done on**: 07/28/2024")


    # Plot results with support and resistance lines
    st.subheader("Price Data with Power Law Regression")
    percentileNow = plot_results(data)

    # Plot log-log graph with regression and bands
    st.subheader("Log-Log Plot with Regression Lines")
    plot_log_log(data)

    st.write("### Other Data")
    st.write(f"**Current Residual Risk Value**: {percentileNow:.0f}%")

    

    # Input date for prediction
    st.sidebar.header("Prediction Settings")
    target_date = st.sidebar.date_input("Select a date for prediction", value=datetime.now())
    
    if st.sidebar.button("Calculate Predicted Values"):
        fair_value, support_value, resistance_value = calculate_predicted_values(target_date)
        st.write(f"### Predicted values for {target_date}:")
        st.write(f"**Fair Value**: {fair_value:.2f}")
        st.write(f"**Support Value**: {support_value:.2f}")
        st.write(f"**Resistance Value**: {resistance_value:.2f}")
else:
    st.write("No data found for the given ticker and date range.")
