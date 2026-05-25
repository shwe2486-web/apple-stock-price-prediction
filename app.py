import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# TITLE
# =========================================

st.title("📈 Stock Price Forecasting Dashboard")

st.markdown("""
This application predicts future stock prices using a trained model.
""")

# =========================================
# LOAD DATA
# =========================================

st.header("📂 Load Dataset")

df = pd.read_csv("Processed_Apple_Stock_Data.csv")

# =========================================
# DATA PREPARATION
# =========================================

st.header("🛠 Data Preparation")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

st.subheader("Dataset Preview")
st.write(df.tail())

# =========================================
# DOWNLOAD DATA BUTTON
# =========================================

csv = df.to_csv().encode('utf-8')

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name='stock_data.csv',
    mime='text/csv'
)

# =========================================
# ACTUAL PRICE CHART
# =========================================

st.header("📊 Actual Closing Price Chart")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df['Close'], label='Actual Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# =========================================
# LOAD MODEL
# =========================================

st.header("🧠 Load Trained Model")

model = pickle.load(open("arima_model.pkl", "rb"))

st.success("Model Loaded Successfully ✅")

# =========================================
# FUTURE PREDICTION
# =========================================

st.header("🔮 Future Prediction")

steps = st.slider(
    "Select number of days to forecast",
    min_value=1,
    max_value=30,
    value=7
)

if st.button("Predict Future Prices"):

    # Forecast
    forecast = model.forecast(steps=steps)

    # Future dates
    future_dates = pd.date_range(
        start=df.index[-1],
        periods=steps+1,
        freq='D'
    )[1:]

    # Future dataframe
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': forecast
    })

    # =========================================
    # FUTURE DATAFRAME
    # =========================================

    st.subheader("📋 Forecast Data")

    st.write(forecast_df)

    # =========================================
    # BUY / SELL SIGNAL
    # =========================================

    st.header("📈 Buy / Sell Signal")

    last_actual_price = df['Close'].iloc[-1]
    future_price = forecast.iloc[-1]

    if future_price > last_actual_price:
        st.success("🟢 BUY SIGNAL")
    else:
        st.error("🔴 SELL SIGNAL")

    # =========================================
    # FORECAST CHART
    # =========================================

    st.header("📉 Forecast Chart")

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(
        future_dates,
        forecast,
        label='Forecast',
        color='orange'
    )

    ax2.set_xlabel("Future Date")
    ax2.set_ylabel("Predicted Price")

    ax2.legend()

    st.pyplot(fig2)

    # =========================================
    # COMPARISON CHART
    # =========================================

    st.header("📊 Actual vs Forecast Comparison")

    fig3, ax3 = plt.subplots(figsize=(12,5))

    ax3.plot(
        df.index,
        df['Close'],
        label='Actual Price'
    )

    ax3.plot(
        future_dates,
        forecast,
        label='Forecast Price',
        color='red'
    )

    ax3.set_xlabel("Date")
    ax3.set_ylabel("Stock Price")

    ax3.legend()

    st.pyplot(fig3)

    # =========================================
    # DOWNLOAD FORECAST BUTTON
    # =========================================

    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download Forecast",
        data=forecast_csv,
        file_name='forecast.csv',
        mime='text/csv'
    )
