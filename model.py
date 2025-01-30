import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Calculate portfolio return
def calculate_portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)

# Calculate portfolio volatility
def calculate_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Calculate Sharpe ratio
def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = calculate_portfolio_return(weights, mean_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    return (portfolio_return - risk_free_rate) / portfolio_volatility

# Optimize portfolio for maximum Sharpe ratio
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(lambda weights: -calculate_sharpe_ratio(weights, *args), 
                      num_assets * [1. / num_assets,], 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    return result.x

# Fetch data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']  # We are interested in adjusted close prices

# Portfolio performance: return and volatility
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# EfficientFrontier Class
class EfficientFrontier:
    def __init__(self, mean_returns, cov_matrix, risk_free_rate=0.0):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(mean_returns)
    
    # Portfolio performance
    def portfolio_performance(self, weights):
        return portfolio_performance(weights, self.mean_returns, self.cov_matrix)
    
    # Optimization to maximize Sharpe Ratio
    def maximize_sharpe_ratio(self):
        def negative_sharpe_ratio(weights):
            portfolio_return, portfolio_volatility = self.portfolio_performance(weights)
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Constraints (sum of weights must be 1, weights between 0 and 1)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))  # Assuming no shorting allowed
        
        # Initial guess: equal distribution of weights
        initial_guess = [1. / self.num_assets] * self.num_assets
        
        # Minimize the negative Sharpe ratio
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x  # Optimal weights that maximize Sharpe Ratio
    
    # Calculate efficient frontier
    def efficient_frontier(self, returns_range):
        volatilities = []
        for target_return in returns_range:
            def min_volatility(weights):
                portfolio_return, portfolio_volatility = self.portfolio_performance(weights)
                return portfolio_volatility
            
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                           {'type': 'eq', 'fun': lambda weights: self.portfolio_performance(weights)[0] - target_return})
            bounds = tuple((0, 1) for _ in range(self.num_assets))
            initial_guess = [1. / self.num_assets] * self.num_assets
            
            result = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            volatilities.append(result.fun)
        return volatilities

# Main script
if __name__ == "__main__":
    # Define tickers and date range
    tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']  # Example: Apple, Google, Microsoft, Amazon
    start_date = '2020-01-01'
    end_date = '2024-12-10'

    # Fetch adjusted close prices from Yahoo Finance
    data = fetch_data(tickers, start_date, end_date)

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate annualized mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualize mean returns (assuming 252 trading days in a year)
    cov_matrix = returns.cov() * 252      # Annualize the covariance matrix

    # Create an instance of EfficientFrontier
    ef = EfficientFrontier(mean_returns, cov_matrix)

    # Generate a range of target returns (from 5% to 25% annualized)
    target_returns = np.linspace(0.05, 0.25, 50)

    # Calculate the corresponding portfolio volatilities for the efficient frontier
    volatilities = ef.efficient_frontier(target_returns)

    # Plot the efficient frontier
    plt.plot(volatilities, target_returns, label='Efficient Frontier', color='blue')
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Risk)')
    plt.ylabel('Portfolio Return')
    plt.show()

    # Calculate and display the portfolio with the maximum Sharpe ratio
    optimal_weights = ef.maximize_sharpe_ratio()
    optimal_return, optimal_volatility = ef.portfolio_performance(optimal_weights)

    print(f"Optimal Weights for Maximum Sharpe Ratio: {optimal_weights}")
    print(f"Expected Portfolio Return: {optimal_return}")
    print(f"Expected Portfolio Volatility: {optimal_volatility}")
