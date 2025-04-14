import streamlit as st
import numpy as np
import pandas as pd
import json
import yfinance as yf
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt



# Load proposed portfolios
@st.cache_data
def load_best_portfolios():
    portfolios = {}
    for key in ['10_Years', '5_Years', '3_Years', '2_Years']:
        with open(f"path/best_portfolios_{key}.json", 'r') as f:
            portfolios[key] = json.load(f)
    return portfolios


# Helper functions
def log_returns(prices):
    return np.log(prices / prices.shift(1))

# Simulate Returns using Normal Distribution
def Monte_Carlo_Returns(initial_value: int, weights: np.ndarray, mean_returns_daily: pd.Series, cov_matrix_daily: np.ndarray, time_horizon: int, n_sims=10000):

    """
    Runs a Monte Carlo simulation for portfolio growth.
    """

    # time_horizon = 3,4,5, ... (number of years)
    time_frame = 252*time_horizon 
    
    simulated_portfolios = np.zeros((time_frame, n_sims))

    for i in range(n_sims):
        simulated_returns = np.random.multivariate_normal(mean_returns_daily, cov_matrix_daily, time_frame)
        weighted_returns = simulated_returns @ weights
        cumulative_returns = np.exp(np.cumsum(weighted_returns)) # exponent of cumulative sum of our returns (since we took log returns)
        simulated_portfolios[:, i] = initial_value * cumulative_returns

    return simulated_portfolios


# FOR LATER USE

# def Monte_Carlo_Returns_w_Crash(initial_value: int, weights: np.ndarray, mean_returns_daily: pd.Series, cov_matrix_daily: np.ndarray, 
#                    crash_yrs: int, pct_crash: float, volatility_level: str, time_horizon: int, n_sims=1000):
    
#     """
#     Runs a Monte Carlo simulation for portfolio growth - simulating an x% market crash with user inputed volatility
#     """

#     mean_returns_crash = mean_returns_daily*252 # yearly avg
#     mean_returns_crash_daily = (mean_returns_crash - mean_returns_crash*pct_crash)/252 # crash & conver to daily 

#     # Optionally adjust volatility
#     multiplier = {'high': 2.5, 'medium': 1.75, 'normal': 1.0}
#     crash_cov_matrix_daily = cov_matrix_daily * multiplier[volatility_level] # higher volatility

#     # time_horizon = 3,4,5, ... (number of years)
#     time_frame = 252*time_horizon 

#     # ---- Run Simulations ----
#     simulated_portfolios = np.zeros((time_frame, n_sims))


#     for i in range(n_sims):
#         crash_simulated_returns = np.vstack([
#             np.random.multivariate_normal(mean_returns_crash_daily, crash_cov_matrix_daily, size=252*crash_yrs), # x yeas of x% crash and volatility
#             np.random.multivariate_normal(mean_returns_daily, cov_matrix_daily * multiplier['medium'], size=252), # following a crash we see expected behavior with medium vol
#             np.random.multivariate_normal(mean_returns_daily, cov_matrix_daily, size=252) # normal/expected behavior with normal vol
#         ])
#         weighted_returns = crash_simulated_returns @ weights
#         cumulative_returns = np.exp(np.cumsum(weighted_returns)) # exponent of cumulative sum of our returns (since we took log returns)
#         simulated_portfolios[:, i] = initial_value * cumulative_returns

#         return simulated_portfolios

# Main app
st.title("📈 Portfolio Monte Carlo Simulator")

# User options
all_tickers = ['PLTR', 'COIN', 'AAPL', 'NVDA', 'TSLA', 'MCD', 'KO', 'TSM', 'GE', 'VOO', 'SCHD', 'VTI',
               'BTC-USD', 'XRP-USD', 'META', 'AMZN', 'GLD', 'SLV', 'MSFT', 'AMD', 'TQQQ']


best_portfolios = load_best_portfolios()
portfolio_option = st.selectbox("Choose portfolio type:", ["Custom", "10_Years", "5_Years", "3_Years", "2_Years"])

if portfolio_option == "Custom":
    selected_tickers = st.multiselect("Pick your tickers:", all_tickers)
    weights = []
    if selected_tickers:
        for t in selected_tickers:
            w = st.number_input(f"Weight for {t}", min_value=0.0, max_value=1.0, value=1.0 / len(selected_tickers))
            weights.append(w)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # Normalize
else:
    selected_tickers = best_portfolios[portfolio_option]['tickers']
    weights = best_portfolios[portfolio_option]['weights']
    sharpe = best_portfolios[portfolio_option]['sharpe']
    rets = best_portfolios[portfolio_option]['return']
    vols = best_portfolios[portfolio_option]['volatility']
    df_weights = pd.DataFrame( {
                'Asset' : list(selected_tickers),
                'Weight': weights
    }
    )


# Download prices
if selected_tickers:
    prices = yf.download(selected_tickers, start='2014-12-31', end='2025-01-01', progress=False)['Close']

    sub_close = prices[list(selected_tickers)].dropna().copy() # pass tickers in best time_window tickers
    returns = sub_close.apply(lambda x: log_returns(x)).dropna() # get log returns

    

    if portfolio_option == '10_Years':
        returns = returns[returns.index.year >= 2015]
    elif portfolio_option == '5_Years':
        returns = returns[returns.index.year >= 2020]
    elif portfolio_option == '3_Years':
        returns = returns[returns.index.year >= 2022]
    elif portfolio_option == '2_Years':
        returns = returns[returns.index.year >= 2023]


    st.markdown(f"- **Portfolio Timeline**: {returns.index.year[0]} - {returns.index.year[-1]}")
    st.markdown(f"- **Portfolio Sharpe:** {sharpe:,.2f}")
    st.markdown(f"- **Portfolio Return:** {rets*100:,.2f}%")
    st.markdown(f"- **Portfolio Volatility:** {vols*100:,.2f}%")


    df_weights["Weight (%)"] = df_weights["Weight"] * 100
    st.markdown("### 🧾 Portfolio Breakdown")
    st.dataframe(df_weights[df_weights['Weight (%)']> 0.01][["Asset", "Weight (%)"]].style.format({"Weight (%)": "{:.2f}%"}), use_container_width=True)

    mean_returns = returns.mean() * 252 # yearly 
    cov_matrix = LedoitWolf().fit(returns).covariance_ * 252 # yearly

    mean_returns_daily = returns.mean()                  # Daily mean returns
    cov_matrix_daily = LedoitWolf().fit(returns).covariance_  # Daily covariance


    st.subheader("Monte Carlo Settings")
    years = st.selectbox("Projection duration (years):", [1, 3, 10])
    # time_horizon = years * 252
    # num_simulations = st.slider("Number of simulations", 100, 5000, 10000)
    starting_amount = st.slider("💰 Starting Investment Amount", min_value=1000, max_value=100000, value=10000, step=1000)
    st.markdown(f"**Selected Investment:** ${starting_amount:,.0f}")

    if st.button("🚀 Run Sim"):
        sim_results = Monte_Carlo_Returns(initial_value=starting_amount, weights=weights, mean_returns_daily=mean_returns_daily, cov_matrix_daily=cov_matrix_daily,
                                        time_horizon=years)
        
        # ---- Plot Results in Streamlit ----
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all simulations
        ax.plot(sim_results, color='gray', alpha=0.1)

        # Plot mean line
        ax.plot(sim_results.mean(axis=1), color='red', label='Expected Portfolio Value')

        # Titles and labels
        ax.set_title(f"Monte Carlo Simulation of Portfolio Value ({portfolio_option} Portfolio) after {years} years")
        ax.set_xlabel("Days")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True)

        # Show plot in Streamlit
        st.pyplot(fig)

        # ---- Summary Stats ----
        final_values = sim_results[-1, :]

        st.subheader("Portfolio Summary Statistics")
        st.markdown(f"- **Expected Final Value:** ${final_values.mean():,.2f}")
        st.markdown(f"- **5% Value-at-Risk (VaR):** ${np.percentile(final_values, 5):,.2f}")
        st.markdown(f"- **95% Value-at-Gain:** ${np.percentile(final_values, 95):,.2f}")

    
else:
    st.info("Please select tickers to begin.")
    










