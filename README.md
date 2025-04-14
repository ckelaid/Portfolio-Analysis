# Portfolio-Analysis


Exploring Portfolios based on stock tickers to generate Max Sharpe and associated return - on historical 10, 5, 3, and 2 year time-horizons
  - **get_max_sharpe_allPortfolios.py**: Checks all possible combinations of assets, of lengths 2 to total assets present, to find optimal portfolio for each time-horizon that maximizes the Sharpe Ratio.
  - **app.py**: Streamlit app that displays Monte Carlo simulated returns for a selected Portfolio of assets (gathered from **get_max_sharpe_allPortfolios.py**) in the next 1, 3, or 10 years based on a user input.
    - The app is available at: https://portfolio-returns-mc.streamlit.app/
      

