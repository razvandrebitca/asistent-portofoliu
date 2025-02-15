import pandas as pd
from yahooquery import Ticker
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def get_stock_data(symbols, start_date, end_date):
    """Fetches adjusted closing price using yahooquery."""
    ticker = Ticker(symbols)
    data = ticker.history(period="5y", interval="1d")  # Fetch daily data for the last 5 years
    
    if data.empty:
        print(f" No data retrieved for {symbols}. Check the symbols.")
        return None

    # Convert start_date and end_date to pandas datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Reset index for filtering
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])  # Ensure proper datetime format

    # Filter data within the given date range
    data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    # Ensure we return only adjusted close prices
    adj_close_data = data.pivot(index="date", columns="symbol", values="adjclose")

    return adj_close_data

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    my_portfolio = get_stock_data(selected_assets, start_date, end_date)

    if my_portfolio is None or my_portfolio.empty:
        raise ValueError("No valid asset data retrieved. Check symbols.")

    my_portfolio_returns = my_portfolio.pct_change().dropna()
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)

    ef = EfficientFrontier(mu, S)

    # Adjust optimization strategy based on risk tolerance
    if risk_tolerance == "Low":
        ef.min_volatility()
    elif risk_tolerance == "Medium":
        ef.efficient_return(target_return=0.1)
    elif risk_tolerance == "High":
        ef.max_sharpe()
    else:
        raise ValueError("Invalid risk tolerance level. Choose from: 'Low', 'Medium', 'High'.")

    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(my_portfolio)

    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()

    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover
