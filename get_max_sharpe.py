# course_enrollments

# This scripts allows one to pass a series of Tickers (assets)
# and build a portfolio that return the max Sharpe ratio 

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


# Step 1: Load historical prices
tickers = ['PLTR', 'COIN', 'AAPL', 'NVDA', 'TSLA', 'MCD', 'KO', 'TSM', 'GE'] # 2.8 Sharpe in close_2yrs => 78% returns
# tickers = ['VOO', 'SCHD', 'VTI', 'BTC-USD', 'XRP-USD']
# tickers = ['BTC-USD', 'XRP-USD']
# tickers = ['VOO', 'SCHD', 'VTI']

#tickers = ['PLTR', 'COIN', 'AAPL', 'NVDA', 'TSLA', 'MCD', 'KO', 'TSM', 'GE', 'VOO', 'SCHD', 'VTI', 'BTC-USD', 'XRP-USD']

prices = yf.download(tickers, start='2014-12-31', end='2025-01-01')#['Adj Close'] # pre market crash (from tarrifs)
close = prices['Close']

# Weed out the NaNs (tickers with no action before a certain date time)
close_10yrs = close.loc[:, close.isna().sum() == 0].copy()

close_5yrs = close[close.index.year>=2020].loc[:, close[close.index.year>=2020].isna().sum() == 0].copy()

close_3yrs = close[close.index.year>=2022].loc[:, close[close.index.year>=2022].isna().sum() == 0].copy()

close_2yrs = close[close.index.year>=2023].loc[:, close[close.index.year>=2023].isna().sum() == 0].copy()


def log_returns(prices):
        return np.log(prices / prices.shift(1))


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) # prtfolio returns
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) # portfolio std
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std # negative Sharpe

def constraint(weights):
    return np.sum(weights) - 1
      

# print('\nHERE')

close_dict = {
    '10 Years' : close_10yrs,
    '5 Years' : close_5yrs,
    '3 Years' : close_3yrs,
    '2 Years' : close_2yrs
}


# Get log returns
for key in close_dict.keys():

    # print('\nHERE1')
    close = close_dict[key].copy()
    returns = close.apply(lambda x: log_returns(x)).dropna() 
    mean_returns = returns.mean() * 252 # 252 trading days
    # cov_matrix = returns.cov() * 252

    # Use Ledoit-Wolf shrinkage for a more robust covariance estimate
    lw = LedoitWolf()
    cov_matrix_shrink = lw.fit(returns).covariance_ * 252

    bounds = [(0, 1)]* len(close.columns)
    initial_guess = np.array([1/len(close.columns)] * len(close.columns)) #  initial portfolio weights
    constraints = {'type': 'eq', 'fun': constraint}
    # print('\nHERE2')
    # Solve for max Sharpe using shrinkage covariance
    opt = minimize(negative_sharpe, initial_guess, args=(mean_returns, cov_matrix_shrink), 
                method='SLSQP', bounds=bounds, constraints=constraints)
    
    optim_ret, optim_vol = portfolio_performance(opt.x, mean_returns, cov_matrix_shrink)

    # Print optimal weights and maximum Sharpe ratio
    print(f"Optimal Portfolio to Maximize Sharpe over the last {key}:\n")
    for i, ticker in enumerate(close.columns):
        if np.round(opt.x[i]*100,2) != 0.00:
            print(f"            {ticker}: {np.round(opt.x[i]*100,2):.2f}%")
    
    print('\nMaximum Sharpe ratio:', -opt.fun)
    print(f"Optimal Returns: {np.round(optim_ret*100,2)}% with volatility of {optim_vol}")
    print('---------------------------------------------------------------------------------------\n')
        


















