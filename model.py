import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    # Download historical data
    my_portfolio = pd.DataFrame()
    for asset in selected_assets:
        my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date,auto_adjust=False)['Adj Close']
    
    # Calculate daily returns
    my_portfolio_returns = my_portfolio.pct_change().dropna()

    # Machine Learning: Predict future returns using rolling linear regression
    window_size = 30  # Use past 30 days for prediction
    predicted_returns = []

    for asset in selected_assets:
        asset_returns = my_portfolio_returns[asset].dropna()
        if len(asset_returns) > window_size:
            X = np.array(range(window_size)).reshape(-1, 1)  # Days as features
            y = asset_returns[-window_size:]  # Use last 'window_size' returns
            
            model = LinearRegression()
            model.fit(X, y)
            predicted_return = model.predict(np.array([[window_size]]))[0]  # Predict next day return
        else:
            predicted_return = asset_returns.mean()  # Fallback to historical mean if not enough data
        
        predicted_returns.append(predicted_return)

    # Convert predicted returns to pandas Series
    predicted_returns = pd.Series(predicted_returns, index=selected_assets)

    # Compute exponentially weighted covariance matrix for better risk estimation
    cov_matrix = my_portfolio_returns.ewm(span=60).cov().dropna().iloc[-len(selected_assets):, -len(selected_assets):] * 252  # Annualized covariance

    num_assets = len(selected_assets)
    risk_free_rate = 0.02  # Assume a 2% annual risk-free rate

    # Portfolio return function
    def calculate_portfolio_return(weights):
        return np.sum(weights * predicted_returns)

    # Portfolio volatility function
    def calculate_portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Sharpe ratio function (higher is better)
    def calculate_sharpe_ratio(weights):
        portfolio_return = calculate_portfolio_return(weights)
        portfolio_volatility = calculate_portfolio_volatility(weights)
        return (portfolio_return - risk_free_rate) / (portfolio_volatility + 1e-8)  # Avoid div-by-zero

    # Minimize volatility (for low risk)
    def min_volatility(weights):
        return calculate_portfolio_volatility(weights)

    # Maximize Sharpe Ratio (for high risk)
    def negative_sharpe_ratio(weights):
        return -calculate_sharpe_ratio(weights)

    # Constraint: Weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

    # Initial equal weights
    initial_guess = np.array([1. / num_assets] * num_assets)

    # Choose optimization strategy based on risk tolerance
    if risk_tolerance == "Low":
        result = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_tolerance == "Medium":
        target_return = 0.1  # Example: Target 10% annual return
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: calculate_portfolio_return(weights) - target_return})
        result = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    else:  # High risk
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x

    # Ensure portfolio allocation uses **integer shares**
    latest_prices = my_portfolio.iloc[-1]
    allocation = {selected_assets[i]: int(np.floor((optimal_weights[i] * portfolio_amount) / latest_prices[i])) for i in range(len(selected_assets))}
    allocated_value = sum(allocation[asset] * latest_prices[asset] for asset in selected_assets)
    leftover = portfolio_amount - allocated_value

    return my_portfolio, my_portfolio_returns, optimal_weights, latest_prices, allocation, leftover
