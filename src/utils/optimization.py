import pandas as pd
import yfinance as yf
import time
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def fetch_stock_data(selected_assets, start_date, end_date, max_retries=3):
    """ Fetch stock data using yfinance with retries for cloud deployment. """
    my_portfolio = pd.DataFrame()
    
    for asset in selected_assets:
        for attempt in range(max_retries):
            try:
                # Download stock data with auto_adjust=True (default)
                df = yf.download(asset, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                # Check if 'Adj Close' is present, otherwise fall back to 'Close'
                if 'Adj Close' in df.columns:
                    my_portfolio[asset] = df["Adj Close"]
                elif 'Close' in df.columns:
                    my_portfolio[asset] = df["Close"]
                else:
                    print(f"⚠️ No valid price data for {asset}, attempt {attempt + 1}/{max_retries}")
                    time.sleep(2)  # Wait before retrying
                    continue  # Try next attempt for the same ticker
                
                # Successfully fetched the data
                break  # Exit retry loop if data is fetched successfully

            except Exception as e:
                print(f"❌ Error fetching {asset}: {e}")
                time.sleep(3)  # Small delay before retrying

    if my_portfolio.empty:
        raise ValueError("❌ No valid stock data retrieved. Check ticker symbols or API limits.")

    return my_portfolio

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    """ Optimizes portfolio based on selected assets and risk tolerance. """
    my_portfolio = fetch_stock_data(selected_assets, start_date, end_date)
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

    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(my_portfolio)
    
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()

    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover


