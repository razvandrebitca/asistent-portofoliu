import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

API_KEY = "YECJNAEG3LQJQB5U"

def fetch_stock_data(symbol, start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize="full")
    except Exception as e:
        print(f"⚠️ Error fetching data for {symbol}: {e}")
        return None

    # Rename columns to match Yahoo Finance format
    data = data.rename(columns={
        "5. adjusted close": "Adj Close"
    })

    # Convert index to datetime and filter by date range
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    filtered_data = data.loc[start_date:end_date]
    
    if filtered_data.empty:
        print(f"⚠️ No data available for {symbol} in the given date range.")
        return None

    return filtered_data

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    my_portfolio = pd.DataFrame()

    for asset in selected_assets:
        stock_data = fetch_stock_data(asset, start_date, end_date)
        if stock_data is not None:
            my_portfolio[asset] = stock_data["Adj Close"]
        time.sleep(12)  # Avoid hitting API rate limits

    # Check if portfolio has valid data
    if my_portfolio.empty:
        raise ValueError("No valid stock data retrieved. Check API rate limits or ticker symbols.")

    print("📊 Portfolio Data Preview:\n", my_portfolio.head())

    my_portfolio_returns = my_portfolio.pct_change().dropna()

    # Ensure the portfolio has data for risk calculations
    if my_portfolio_returns.empty:
        raise ValueError("Portfolio returns are empty. No valid price movements detected.")

    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)

    if mu.empty or S.empty:
        raise ValueError("Covariance matrix or expected returns vector is empty.")

    ef = EfficientFrontier(mu, S)

    if risk_tolerance == "Low":
        weights = ef.min_volatility()
    elif risk_tolerance == "Medium":
        weights = ef.efficient_return(target_return=0.1)
    elif risk_tolerance == "High":
        weights = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()

    latest_prices = get_latest_prices(my_portfolio)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()

    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover
