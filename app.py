import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import black_scholes_model as bsm # Import model logic

# --- Page Setup ---
st.set_page_config(
    page_title="Option Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Title ---
st.title("📈 Option Analytics Dashboard")
st.markdown("Interactive BSM pricing, Greeks, and payoff visualization.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust parameters for real-time calculation.")

# Use session state to persist inputs across tabs
if 'S' not in st.session_state:
    st.session_state.S = 100.0
if 'K' not in st.session_state:
    st.session_state.K = 105.0
if 'days_to_expiry' not in st.session_state:
    st.session_state.days_to_expiry = 365
if 'r' not in st.session_state:
    st.session_state.r = 5.0
if 'sigma' not in st.session_state:
    st.session_state.sigma = 20.0

# Sidebar input fields
S = st.sidebar.number_input("Current Stock Price (S)", min_value=1.0, value=st.session_state.S, step=1.0, key='S')
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=st.session_state.K, step=1.0, key='K')

days_to_expiry = st.sidebar.slider(
    "Days to Expiry", 
    min_value=1, 
    max_value=1095, # Max 3 years
    value=st.session_state.days_to_expiry, 
    step=1, 
    key='days_to_expiry',
    help="Converted to years (T = days/365) for the BSM model."
)

r_percent = st.sidebar.slider("Risk-Free Interest Rate (r %)", 0.0, 20.0, st.session_state.r, 0.1, key='r')
sigma_percent = st.sidebar.slider("Volatility (σ %)", 1.0, 100.0, st.session_state.sigma, 0.5, key='sigma')

# --- Model Input Conversions ---
r = r_percent / 100.0
sigma = sigma_percent / 100.0
T = days_to_expiry / 365.0  # Convert days to annualized T for BSM formula

# --- Calculations ---
try:
    # Calculate prices
    call_price = bsm.black_scholes_price(S, K, T, r, sigma, "call")
    put_price = bsm.black_scholes_price(S, K, T, r, sigma, "put")

    # Calculate Greeks
    call_delta = bsm.delta(S, K, T, r, sigma, "call")
    put_delta = bsm.delta(S, K, T, r, sigma, "put")
    gamma = bsm.gamma(S, K, T, r, sigma)
    vega = bsm.vega(S, K, T, r, sigma)
    call_theta = bsm.theta(S, K, T, r, sigma, "call")
    put_theta = bsm.theta(S, K, T, r, sigma, "put")
    call_rho = bsm.rho(S, K, T, r, sigma, "call")
    put_rho = bsm.rho(S, K, T, r, sigma, "put")

except Exception as e:
    st.error(f"Error in BSM calculation: {e}")
    st.stop()


# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["Option Pricer & Greeks", "Price & Payoff Analysis"])

with tab1:
    st.header("Calculated Prices & Greeks")
    st.markdown(f"For S=**${S:,.2f}**, K=**${K:,.2f}**, and **{days_to_expiry} days** to expiry.")
    
    st.subheader("Option Prices")
    col1, col2 = st.columns(2)
    col1.metric("Call Option Price", f"${call_price:,.4f}")
    col2.metric("Put Option Price", f"${put_price:,.4f}")

    st.subheader("The Greeks (Option Sensitivities)")
    
    # Greeks display grid
    gcol1, gcol2, gcol3 = st.columns(3)
    
    with gcol1:
        st.metric("Delta (Call)", f"{call_delta:,.4f}")
        st.metric("Delta (Put)", f"{put_delta:,.4f}")
    with gcol2:
        st.metric("Gamma", f"{gamma:,.4f}")
        st.metric("Vega (per 1% vol)", f"{vega/100.0:,.4f}")
    with gcol3:
        st.metric("Theta (Call, per day)", f"{call_theta:,.4f}")
        st.metric("Theta (Put, per day)", f"{put_theta:,.4f}")

    st.markdown("---")
    gcol4, gcol5 = st.columns(2)
    with gcol4:
        st.metric("Rho (Call, per 1% rate)", f"{call_rho:,.4f}")
    with gcol5:
        st.metric("Rho (Put, per 1% rate)", f"{put_rho:,.4f}")
    
    # Explanations in an expander
    with st.expander("Greeks Definitions"):
        st.markdown("""
        - **Delta:** Change in option price per $1 change in stock price.
        - **Gamma:** Change in Delta per $1 change in stock price.
        - **Vega:** Change in option price per 1% change in volatility.
        - **Theta:** Change in option price per 1-day decrease in time (time decay).
        - **Rho:** Change in option price per 1% change in the risk-free rate.
        """)

with tab2:
    st.header("Interactive Price & Payoff Analysis")
    st.markdown("Compares the option's current (theoretical) value vs. its intrinsic value (P/L) at expiry.")

    option_to_analyze = st.radio("Analyze Option Type", ["Call", "Put"], horizontal=True)

    # --- Graphing Logic ---
    
    # 1. Define x-axis (stock price range)
    S_range = np.linspace(K * 0.7, K * 1.3, 100) 

    # 2. Calculate y-axis values
    if option_to_analyze == "Call":
        current_price = call_price
        # Calculate BSM value across stock price range
        price_curve = [bsm.black_scholes_price(s_val, K, T, r, sigma, "call") for s_val in S_range]
        # Calculate P/L at expiry
        payoff_at_expiry = np.maximum(S_range - K, 0) - current_price
        
    else: # Put
        current_price = put_price
        # Calculate BSM value across stock price range
        price_curve = [bsm.black_scholes_price(s_val, K, T, r, sigma, "put") for s_val in S_range]
        # Calculate P/L at expiry
        payoff_at_expiry = np.maximum(K - S_range, 0) - current_price

    # 3. Create Plotly Figure
    fig = go.Figure()

    # Add Payoff at Expiry trace
    fig.add_trace(go.Scatter(
        x=S_range,
        y=payoff_at_expiry,
        mode='lines',
        name='Profit/Loss at Expiry',
        line=dict(color='red', dash='dash', width=3)
    ))
    
    # Add Current BSM Value trace
    fig.add_trace(go.Scatter(
        x=S_range,
        y=price_curve,
        mode='lines',
        name='Current Option Value',
        line=dict(color='blue', width=3)
    ))
    
    # 4. Style the graph
    fig.update_layout(
        title=f"<b>{option_to_analyze} Option: Value vs. Expiry P/L</b>",
        xaxis_title="Stock Price",
        yaxis_title="Option Value / P&L",
        legend_title="Curves",
        hovermode="x unified",
        xaxis=dict(tickformat='$,.2f'), # Format x-axis as currency
        yaxis=dict(tickformat='$,.2f')  # Format y-axis as currency
    )
    
    # Add key reference lines
    fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="gray", 
                   annotation_text="Strike Price (K)", annotation_position="top right")
    fig.add_vline(x=S, line_width=1, line_dash="dot", line_color="green", 
                   annotation_text="Current Price (S)", annotation_position="top left")
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="black")
    
    # Calculate and show breakeven
    if option_to_analyze == 'Call':
        breakeven = K + current_price
    else:
        breakeven = K - current_price
    
    fig.add_vline(x=breakeven, line_width=1, line_dash="dot", line_color="orange", 
                   annotation_text="Breakeven", annotation_position="bottom right")

    # Render the plot
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    - **Red Dashed Line:** Profit/Loss if you buy the option today for **${current_price:,.4f}** and hold until expiry.
    - **Blue Solid Line:** The option's theoretical (BSM) value *today* at various stock prices.
    - The difference between the blue line (current value) and the red line (intrinsic value) is the **Extrinsic Value** (Time Value + Volatility Value).
    """)