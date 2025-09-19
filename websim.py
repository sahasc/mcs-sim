# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")
st.title("Interactive Monte Carlo Stock Price Simulator")

# ----------------- User Inputs -----------------
tickers = st.text_input("Enter stock tickers (comma-separated, e.g., AAPL,RBLX):", "AAPL").upper().replace(" ", "").split(",")
simulation_days = st.slider("Days to simulate:", 30, 730, 252)
num_simulations = st.slider("Number of simulations:", 100, 2000, 1000)
top_paths_to_plot = st.slider("Highlighted paths:", 5, 50, 20)
training_days = st.slider("Days for historical data:", 30, 500, 252)

# ----------------- Function: Monte Carlo Simulation -----------------
def monte_carlo_sim(ticker):
    today = datetime.date.today()
    data = yf.download(ticker, start="2022-01-01", end=str(today), auto_adjust=True)
    if data.empty:
        st.warning(f"No data found for {ticker}")
        return None

    close_prices = data["Close"]
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    recent_returns = log_returns[-training_days:]

    annual_drift = float(recent_returns.mean() * 252)
    annual_volatility = float(recent_returns.std() * np.sqrt(252))
    S0 = float(close_prices.iloc[-1])

    dt = 1/252
    all_paths = np.zeros((simulation_days, num_simulations))

    for i in range(num_simulations):
        prices = [S0]
        for t in range(1, simulation_days):
            Z = np.random.normal()
            S_next = prices[-1] * np.exp((annual_drift - 0.5 * annual_volatility**2) * dt + annual_volatility * np.sqrt(dt) * Z)
            prices.append(S_next)
        all_paths[:, i] = prices

    # Likelihood-based top paths
    final_prices = all_paths[-1, :]
    mean_final = np.mean(final_prices)
    distance_from_mean = np.abs(final_prices - mean_final)
    most_likely_idx = np.argsort(distance_from_mean)[:top_paths_to_plot]
    highlight_paths = all_paths[:, most_likely_idx]
    highlight_distances = distance_from_mean[most_likely_idx]
    normalized = (highlight_distances - highlight_distances.min()) / (highlight_distances.max() - highlight_distances.min() + 1e-8)
    colors = [(val, 1-val, 0) for val in normalized]  # red → green

    mean_path = np.mean(all_paths, axis=1)
    std_path = np.std(all_paths, axis=1)

    return S0, mean_path, std_path, highlight_paths, colors, all_paths

# ----------------- Plotting -----------------
for ticker in tickers:
    result = monte_carlo_sim(ticker)
    if result is None:
        continue

    S0, mean_path, std_path, highlight_paths, colors, all_paths = result
    st.subheader(f"Ticker: {ticker} | Current Price: {S0:.2f}")

    fig = go.Figure()

    # Plot top paths with gradient
    for i in range(top_paths_to_plot):
        r, g, b = colors[i]
        fig.add_trace(go.Scatter(
            y=highlight_paths[:, i],
            mode='lines',
            line=dict(color=f'rgb({int(255*r)},{int(255*g)},{int(255*b)})', width=2),
            name=f'Likely Path {i+1}',
            hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
        ))

    # Mean path
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Expected Mean Path',
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # ±1 Std Dev shaded band
    fig.add_trace(go.Scatter(
        y=mean_path + std_path,
        mode='lines',
        line=dict(color='rgba(128,128,128,0.2)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        y=mean_path - std_path,
        mode='lines',
        fill='tonexty',
        line=dict(color='rgba(128,128,128,0.2)'),
        name='±1 Std Dev',
        hoverinfo='skip'
    ))

    # Current price
    fig.add_trace(go.Scatter(
        y=[S0]*simulation_days,
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Current Price',
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Monte Carlo Simulation for {ticker}",
        xaxis_title="Days",
        yaxis_title="Price",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Option to download CSV
    all_paths_df = pd.DataFrame(all_paths)
    csv = all_paths_df.to_csv(index=False)
    st.download_button(
        label="Download Simulated Paths as CSV",
        data=csv,
        file_name=f"{ticker}_monte_carlo.csv",
        mime="text/csv"
    )
