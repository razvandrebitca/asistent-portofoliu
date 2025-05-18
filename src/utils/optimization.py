import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    # Download data - using tickers parameter for new API
    data = yf.download(tickers=selected_assets, start=start_date, end=end_date, auto_adjust=False)
    
    # Handle data structure based on number of assets
    if len(selected_assets) == 1:
        my_portfolio = pd.DataFrame({selected_assets[0]: data['Adj Close']})
    else:
        my_portfolio = data['Adj Close']
    
    # Handle missing values
    my_portfolio = my_portfolio.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate returns
    my_portfolio_returns = my_portfolio.pct_change().dropna()
    
    # Calculate expected returns and covariance
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)
    
    # Ensure matrix is symmetric (fix for the initial error)
    S = (S + S.T) / 2
    
    # Create efficient frontier
    ef = EfficientFrontier(mu, S)
    
    # Optimize based on risk tolerance
    if risk_tolerance == "Low":
        weights = ef.min_volatility()
    elif risk_tolerance == "Medium":
        weights = ef.efficient_return(target_return=0.1)
    elif risk_tolerance == "High":
        weights = ef.max_sharpe()
    
    cleaned_weights = ef.clean_weights()
    
    # Get latest prices and allocate
    latest_prices = get_latest_prices(my_portfolio)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()
    
    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover