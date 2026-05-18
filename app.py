import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Market Analysis Dashboard")

ticker = st.sidebar.text_input("Enter NSE Stock Ticker", value="RELIANCE.NS")
period = st.sidebar.selectbox("Select Period", ["3mo", "6mo", "1y", "2y"], index=2)

@st.cache_data(ttl=300)
def get_data(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    return df

df = get_data(ticker, period)

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close']
)])
fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='blue')))
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
col3.metric("52W High", f"₹{df['High'].max():.2f}")
col4.metric("52W Low", f"₹{df['Low'].min():.2f}")