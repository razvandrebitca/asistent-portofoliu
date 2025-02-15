import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Alpha Vantage API Key
API_KEY = "YECJNAEG3LQJQB5U"

# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize="full")

    # Rename columns to match Yahoo Finance
    data = data.rename(columns={
        "5. adjusted close": "Adj Close",
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume"
    })

    # Convert index to datetime and filter by date range
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data.loc[start_date:end_date]

# Portfolio Optimization Function
def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    my_portfolio = pd.DataFrame()

    for asset in selected_assets:
        try:
            stock_data = fetch_stock_data(asset, start_date, end_date)
            my_portfolio[asset] = stock_data["Adj Close"]
            time.sleep(12)  # Avoid hitting API rate limits
        except Exception as e:
            print(f"Error fetching data for {asset}: {e}")

    my_portfolio_returns = my_portfolio.pct_change().dropna()
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)

    ef = EfficientFrontier(mu, S)

    # Adjust optimization based on risk tolerance
    if risk_tolerance == "Low":
        weights = ef.min_volatility()  # Optimize for low volatility
    elif risk_tolerance == "Medium":
        weights = ef.efficient_return(target_return=0.1)  # Moderate return target
    elif risk_tolerance == "High":
        weights = ef.max_sharpe()  # Optimize for max Sharpe ratio

    cleaned_weights = ef.clean_weights()

    latest_prices = get_latest_prices(my_portfolio)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()

    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover
