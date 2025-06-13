# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from pmdarima import auto_arima
from prophet import Prophet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Stock Price Forecasting App")

# Sidebar - Input
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-10"))

# Fetch Data
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

df = get_data(ticker, start_date, end_date)
train_size = int(len(df) * 0.8)

st.subheader("Raw and Processed Data")
st.write(df.tail())

# Tabs for models
tabs = st.tabs(["📊 EDA", "🌀 SARIMA", "📆 Prophet", "🔮 LSTM", "📊 Model Comparison"])

# =============================
# 📊 Tab 1: EDA
# =============================
with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].plot(df['Close'])
    axs[0].set_title('Close Price')
    axs[1].plot(df['Daily_Return'])
    axs[1].set_title('Daily Returns')
    axs[2].plot(df['Volume'])
    axs[2].set_title('Volume')
    st.pyplot(fig)

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition")
    result = seasonal_decompose(df['Close'], model='multiplicative', period=252)
    fig = result.plot()
    st.pyplot(fig)

# =============================
# 🌀 Tab 2: SARIMA
# =============================
with tabs[1]:
    st.subheader("SARIMA Forecasting")

    train = df['Close'][:train_size]
    test = df['Close'][train_size:]

    with st.spinner("Fitting SARIMA model..."):
        arima_model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
        order = arima_model.order
        seasonal_order = arima_model.seasonal_order
        model_sarima = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        results_sarima = model_sarima.fit(disp=False)
        forecast = results_sarima.get_forecast(steps=len(test))
        pred_mean = forecast.predicted_mean
        ci = forecast.conf_int()

    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mae = mean_absolute_error(test, pred_mean)

    st.write(f"**RMSE:** {rmse:.2f} | **MAE:** {mae:.2f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Test')
    ax.plot(test.index, pred_mean, label='SARIMA Forecast', color='red')
    ax.fill_between(test.index, ci.iloc[:, 0], ci.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# =============================
# 📆 Tab 3: Prophet
# =============================
with tabs[2]:
    st.subheader("Prophet Forecasting")

    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    prophet_train = prophet_df.iloc[:train_size]
    prophet_test = prophet_df.iloc[train_size:]

    with st.spinner("Training Prophet..."):
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model_prophet.add_country_holidays(country_name='US')
        model_prophet.fit(prophet_train)
        future = model_prophet.make_future_dataframe(periods=len(prophet_test))
        forecast = model_prophet.predict(future)
        prophet_forecast = forecast.iloc[train_size:][['ds', 'yhat']]

    prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_forecast['yhat']))
    prophet_mae = mean_absolute_error(prophet_test['y'], prophet_forecast['yhat'])

    st.write(f"**RMSE:** {prophet_rmse:.2f} | **MAE:** {prophet_mae:.2f}")

    fig1 = model_prophet.plot(forecast)
    st.pyplot(fig1)

    fig2 = model_prophet.plot_components(forecast)
    st.pyplot(fig2)

# =============================
# 🔮 Tab 4: LSTM
# =============================
with tabs[3]:
    st.subheader("LSTM Forecasting")

    SEQ_LENGTH = 60
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    with st.spinner("Training LSTM..."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    test_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_pred_inv = scaler.inverse_transform(test_pred)

    lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    lstm_mae = mean_absolute_error(y_test_inv, test_pred_inv)

    st.write(f"**RMSE:** {lstm_rmse:.2f} | **MAE:** {lstm_mae:.2f}")

    test_dates = df.index[train_size + SEQ_LENGTH:]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, y_test_inv, label='Test Actual')
    ax.plot(test_dates, test_pred_inv, label='LSTM Forecast', color='red')
    ax.legend()
    st.pyplot(fig)

# =============================
# 📊 Tab 5: Comparison
# =============================
with tabs[4]:
    st.subheader("Model Comparison")

    results = pd.DataFrame({
        'Model': ['SARIMA', 'Prophet', 'LSTM'],
        'RMSE': [rmse, prophet_rmse, lstm_rmse],
        'MAE': [mae, prophet_mae, lstm_mae]
    })

    st.dataframe(results.set_index('Model'))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='Model', y='RMSE', data=results, ax=axs[0])
    axs[0].set_title("RMSE Comparison")

    sns.barplot(x='Model', y='MAE', data=results, ax=axs[1])
    axs[1].set_title("MAE Comparison")
    st.pyplot(fig)
