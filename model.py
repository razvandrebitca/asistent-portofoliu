import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    # Download historical data
    my_portfolio = pd.DataFrame()
    for asset in selected_assets:
        my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date)['Adj Close']
    
    # Calculate returns
    my_portfolio_returns = my_portfolio.pct_change().dropna()

    # Use machine learning (Linear Regression) to predict future returns
    # Create lagged data for the model (shift by 1 for previous day's return as feature)
    X = my_portfolio_returns.shift(1).dropna()  # Use previous day's returns as a feature
    y = my_portfolio_returns.iloc[1:]  # Predict today's return based on previous day's return

    # Initialize linear regression model
    model = LinearRegression()

    # Fit model for each asset
    predicted_returns = []
    for asset in selected_assets:
        model.fit(X[asset].values.reshape(-1, 1), y[asset].values)
        predicted_returns.append(model.predict(X[asset].values.reshape(-1, 1))[-1])  # Use last prediction

    # Convert to a pandas Series
    predicted_returns = pd.Series(predicted_returns, index=selected_assets)

    # Use the predicted returns instead of historical mean returns
    mean_returns = predicted_returns  # These are now the predicted returns

    # Compute the covariance matrix (still based on historical returns)
    cov_matrix = my_portfolio_returns.cov() * 252  # Annualized covariance matrix
    num_assets = len(selected_assets)
    risk_free_rate = 0.02  # Assume a 2% annual risk-free rate

    def calculate_portfolio_return(weights):
        return np.sum(weights * mean_returns)

    def calculate_portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def calculate_sharpe_ratio(weights):
        portfolio_return = calculate_portfolio_return(weights)
        portfolio_volatility = calculate_portfolio_volatility(weights)
        return (portfolio_return - risk_free_rate) / portfolio_volatility

    def min_volatility(weights):
        return calculate_portfolio_volatility(weights)

    def negative_sharpe_ratio(weights):
        return -calculate_sharpe_ratio(weights)

    def target_return(weights, target):
        return calculate_portfolio_return(weights) - target

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = [1. / num_assets] * num_assets

    if risk_tolerance == "Low":
        result = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_tolerance == "Medium":
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: target_return(weights, 0.1)})
        result = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    else:  # High risk or default
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    latest_prices = my_portfolio.iloc[-1]
    allocation = {selected_assets[i]: int(np.floor((optimal_weights[i] * portfolio_amount) / latest_prices[i])) for i in range(len(selected_assets))}
    allocation_dict = (optimal_weights * portfolio_amount / latest_prices).astype(int) 
    leftover = portfolio_amount - np.sum(allocation_dict * latest_prices)

    return my_portfolio, my_portfolio_returns, optimal_weights, latest_prices, allocation, leftover
