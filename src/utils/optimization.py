import yfinance as yf
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    my_portfolio = pd.DataFrame()
    for asset in selected_assets:
        my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date,auto_adjust=False)['Adj Close']

    my_portfolio_returns = my_portfolio.pct_change().dropna()
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)

    ef = EfficientFrontier(mu, S)

    # Adjust optimization strategy based on risk tolerance
    if risk_tolerance == "Low":
        weights = ef.min_volatility()  # Optimizes for minimum volatility
    elif risk_tolerance == "Medium":
        weights = ef.efficient_return(target_return=0.1)  # Target moderate return
    elif risk_tolerance == "High":
        weights = ef.max_sharpe()  # Optimizes for maximum Sharpe ratio

    cleaned_weights = ef.clean_weights()

    latest_prices = get_latest_prices(my_portfolio)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()

    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover