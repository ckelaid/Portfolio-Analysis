# course_enrollments

# This scripts allows one to pass a series of Tickers (assets)
# and build a portfolio that return the max Sharpe ratio based on all tickers and all ticker combinations 
# to find the best portfolio (in terms of Max Sharpe) given the assets

# The Max Sharpe is present on varying time-horizons 
#   2 years - 2023-2024
#   3 years - 2022-2023-2024
#   5 years - 2020-2021-2022-2023-2024
#   10 years - 2015 - 2024

# NOTE: Use custom list of Tickers (will change based on user prompt) => can be LLM based


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import itertools
from itertools import combinations


# we set any amount of tickers in here
starting_tickers = ['PLTR', 'COIN', 'AAPL', 'NVDA', 'TSLA', 'MCD', 'KO', 'TSM',
                    'GE', 'VOO', 'SCHD', 'VTI', 'BTC-USD', 'XRP-USD', 'VOO', 'SCHD', 'VTI',
                    'META', 'AMZN', 'GLD', 'MSFT', 'AMD', 'TQQQ']

# Download full price history
prices = yf.download(starting_tickers, start='2014-12-31', end='2025-01-01', progress=False)['Close']
prices_2020 = prices[prices.index.year >= 2020]
prices_2022 = prices[prices.index.year >= 2022]
prices_2023 = prices[prices.index.year >= 2023]

# Create filtered datasets per time window
close_dict = {
    '10 Years': prices.loc[:, prices.iloc[0, :].isna() == False].copy(), # tickers we keep for each year
    '5 Years': prices_2020.loc[:, prices_2020.iloc[1, :].isna() == False].copy(),  # tickers we keep for each year
    '3 Years': prices_2022.loc[:, prices_2022.iloc[2, :].isna() == False].copy(),  # tickers we keep for each year
    '2 Years': prices_2023.loc[:, prices_2023.iloc[2, :].isna() == False].copy()  # tickers we keep for each year
}

# Helper functions
def log_returns(prices):
    return np.log(prices / prices.shift(1))

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def constraint(weights):
    return np.sum(weights) - 1



# Store best portfolio per time window
best_portfolios = {}

# Loop through each time window
for key, close in close_dict.items():
    print(f"\n⏳ Processing time window: {key}")
    tickers_clean = list(close.columns) # tickers available in time_frame

    best_result = None # reset best result for every time_frame

    # Loop through combinations from size 2 to N
    for k in range(2, len(tickers_clean) + 1):
        for tickers in combinations(tickers_clean, k):
            try:
                sub_close = close[list(tickers)].dropna()
                if sub_close.shape[1] != len(tickers):  # Drop if any NaNs
                    continue

                returns = sub_close.apply(lambda x: log_returns(x)).dropna() # get log returns
                mean_returns = returns.mean() * 252 # mean returns per trading year
                cov_matrix = LedoitWolf().fit(returns).covariance_ * 252 # covariance matrix per trading year

                bounds = [(0, 1)] * len(tickers)
                initial_guess = np.array([1 / len(tickers)] * len(tickers)) # initial portfolio weights
                constraints = {'type': 'eq', 'fun': constraint}
                # minimize negative Sharpe (maximize Sharpe)
                opt = minimize(negative_sharpe, initial_guess, args=(mean_returns, cov_matrix),
                               method='SLSQP', bounds=bounds, constraints=constraints)

                if not opt.success:
                    continue
                # Get Portfolio returns and volatility
                ret, vol = portfolio_performance(opt.x, mean_returns, cov_matrix)
                sharpe = -opt.fun # get Sharpe

                result = {
                    'tickers': tickers,
                    'weights': opt.x,
                    'sharpe': sharpe,
                    'return': ret,
                    'volatility': vol
                }

                if best_result is None or sharpe > best_result['sharpe']:
                    best_result = result

            except Exception as e:
                continue  # Gracefully skip bad combos

    best_portfolios[key] = best_result

# SAVE
import json
filename = "best_portfolios.json"

with open(filename, 'w') as file:
    json.dump(best_portfolios, file, indent=4)

# 📊 Display results
for key, result in best_portfolios.items():
    print(f"\n📅 Best Portfolio for {key}:")
    for t, w in zip(result['tickers'], result['weights']):
        if w > 0.001:
            print(f"  {t}: {w*100:.2f}%")
    print(f"  Sharpe: {result['sharpe']:.2f}")
    print(f"  Return: {result['return']*100:.2f}%")
    print(f"  Volatility: {result['volatility']:.2f}")
    print('---------------------------------------------------------------------------------------\n')

