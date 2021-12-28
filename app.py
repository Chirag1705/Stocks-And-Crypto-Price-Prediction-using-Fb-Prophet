import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
import yfinance as yf
import datetime as dt
import pandas_datareader.data as collector
import matplotlib.pyplot as plt

#Stock Prediction ->

st.title("Stock Prediction")

#getting the stock data using yahoo finance
start_period = dt.datetime(2012, 1, 1)
end_period = dt.datetime(2021,1,1)
df = collector.DataReader("FB", 'yahoo', start_period, end_period)  # Collects data
#prices in USD

# years = st.slider("Prediction of Years:", 1, 10)
# period = years*365

st.subheader('Raw Data')
df

df.reset_index(inplace=True)
data_compacted=df[["Date","Adj Close"]]
data_compacted=data_compacted.rename(columns={"Date": "ds", "Adj Close": "y"})

# PLot
st.subheader('Raw Data Plot')

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

st.subheader("Data Sample After Formatting")
st.write(data_compacted.tail())

mod = Prophet(daily_seasonality = True)
mod.fit(data_compacted)

future = mod.make_future_dataframe(periods=3650)

forecast = mod.predict(future)
st.subheader("Stock Values Forecast")
forecast

# st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.subheader("Stocks Forecasted Data Plot")
fig1 = plot_plotly(mod, forecast)
st.plotly_chart(fig1)

st.subheader("Plots of Components of Forecasted Stocks Values")
fig2 = plot_components_plotly(mod, forecast)
st.plotly_chart(fig2)

# df.reset_index(inplace=True)
# df.set_index("Date", inplace=True)

def plot_adjclose():
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x = df['Date'], y = df['Adj Close']))
    fig3.layout.update(title_text = "Adj Close", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

plot_adjclose()

# Crypto Prediction ->

st.title("Crypto(Bitcoin) Prediction")

#getting crypto prices
df1 = yf.download('BTC-USD', start = '2016-01-01')

st.subheader('Raw Data')
df1

df1.reset_index(inplace=True)
data1=df1[["Date","Adj Close"]]
data1=data1.rename(columns={"Date": "ds", "Adj Close": "y"})

# PLot
st.subheader('Raw Data Plot')

def plot_raw_data():
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x = df1['Date'], y = df1['Open'], name = 'stock_open'))
    fig4.add_trace(go.Scatter(x = df1['Date'], y = df1['Close'], name = 'stock_close'))
    fig4.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig4)

plot_raw_data()

#Forecasting

m1 = Prophet(daily_seasonality=True)
m1.fit(data1)

future1 = m1.make_future_dataframe(periods=3650)
forecast1 = m1.predict(future1)

st.subheader('Crypto Values Forecast')
forecast1

st.subheader("Crypto Forecasted Data Plot")
fig5 = plot_plotly(mod, forecast)
st.plotly_chart(fig5)

st.subheader("Plots of Components of Forecasted Crypto Values")
fig6 = plot_components_plotly(mod, forecast)
st.plotly_chart(fig6)

# df.reset_index(inplace=True)
# df.set_index("Date", inplace=True)

def plot_adjclose():
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x = df1['Date'], y = df1['Adj Close']))
    fig7.layout.update(title_text = "Adj Close", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig7)

plot_adjclose()
