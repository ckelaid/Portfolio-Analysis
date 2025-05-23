# Portfolio-Analysis


Exploring Portfolios based on stock tickers to generate Max Sharpe and associated return - on historical 10, 5, 3, and 2 year time-horizons
  - **get_max_sharpe_allPortfolios.py**: Checks all possible combinations of assets, of lengths 2 to total assets present, to find optimal portfolio for each time-horizon that maximizes the Sharpe Ratio.
  - **app.py**: Streamlit app that displays Monte Carlo simulated returns (using normal distribution) for a selected Portfolio of assets (gathered from **get_max_sharpe_allPortfolios.py**) in the next 1, 3, or 10 years based on a user input.
    - The app is available at: https://portfolio-returns-mc.streamlit.app/ - please allow 10-15 seconds for the simulation to run (10,000 iterations)
    - **NOTE**: The simulated returns assume the market will continue behaving as it has in the time_horizon of the selected Portfolio (i.e., the return distribution for the *10_Year* Optimal Portfolio is generated using the mean daily returns and volatility of the assets during the 10-year time frame - in this case 2015-2024.)
    - **NEXT STEPS**: Looking to include scenario-base modeling, such as market crashes and high volatility periods - stay tuned...
      
      

