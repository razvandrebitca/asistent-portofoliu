import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from login import login, register, logout, is_logged_in, initialize_session_state

st.set_page_config(page_title="Portfolio Analysis Assistant", layout="wide")

# Function to retrieve S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    table = data[0]
    tickers = table['Symbol'].tolist()
    company_names = table['Security'].tolist()
    ticker_company_dict = dict(zip(tickers, company_names))
    return ticker_company_dict

# Function to retrieve cryptocurrency price data
def get_crypto_prices(crypto_symbols, start_date, end_date):
    crypto_data = {}
    for symbol in crypto_symbols:
        crypto_data[symbol] = yf.download(symbol + '-USD', start=start_date, end=end_date)['Adj Close']
    return pd.DataFrame(crypto_data)

# Function to retrieve S&P 500 price data
def get_sp500_prices(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_prices = sp500_data['Adj Close']
    return sp500_prices

# Function to optimize portfolio using Markowitz efficient frontier
def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount):
    my_portfolio = pd.DataFrame()
    for asset in selected_assets:
        my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date)['Adj Close']
    
    my_portfolio_returns = my_portfolio.pct_change().dropna()
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # Optimizes for maximum Sharpe ratio
    cleaned_weights = ef.clean_weights()
    
    latest_prices = get_latest_prices(my_portfolio)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()
    
    return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover

# Function to calculate performance metrics
def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.01):
    portfolio_returns = pd.Series(portfolio_returns)  # Convert to Pandas Series
    sharpe_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std() * np.sqrt(252)
    sortino_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Beta and Treynor Ratio
    beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
    treynor_ratio = (portfolio_returns.mean() - risk_free_rate) / beta
    
    return sharpe_ratio, sortino_ratio, max_drawdown, beta, treynor_ratio

# Main function for the Streamlit app
def main():
    names = ["John Doe", "Peter Miller"]
    usernames = ["john", "peter"]
    passwords = ["abc123", "test1234"]
    
    file_path = Path(__file__).parent / "hashed_pw.pk1"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)
        
    credentials = {
        "usernames": {
            usernames[0]: {
                "name": names[0],
                "password": hashed_passwords[0]
            },
            usernames[1]: {
                "name": names[1],
                "password": hashed_passwords[1]
            }
        }
    }
    
    authenticator = stauth.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=30)
    name, authentification_status, username = authenticator.login('main', fields={'Form name': 'Login'})
    
    if authentification_status == False:
        st.error("Invalid username or password")
    
    if authentification_status:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Welcome, {name}!")      
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

        st.title("Portfolio Analysis Assistant")
        st.markdown("---")
        st.subheader("Description")
        st.info(
            "This app uses Markowitz's algorithm to optimize portfolios containing S&P 500 stocks and cryptocurrencies, "
            "using historical trading data."
        )
        st.markdown("---")

        input_col = st.sidebar
        input_col.header("Select Time Period for Historical Data")

        start_date = input_col.date_input("Start Date:", dt.datetime(2020, 1, 1))
        end_date = input_col.date_input("End Date:", dt.datetime.now())

        ticker_company_dict = get_sp500_tickers()

        input_col.header("Portfolio")

        selected_tickers = input_col.multiselect(
            "Select S&P 500 stocks for your portfolio:", 
            list(ticker_company_dict.keys()), 
            format_func=lambda ticker: f"{ticker}: {ticker_company_dict[ticker]}"
        )

        # Input for custom ticker symbols
        custom_tickers = input_col.text_input("Enter custom stock tickers separated by commas (e.g., AAPL, MSFT, TSLA)")

        if custom_tickers:
            custom_tickers_list = [ticker.strip() for ticker in custom_tickers.split(',')]
            selected_tickers.extend(custom_tickers_list)

        # Input for cryptocurrencies
        crypto_symbols = input_col.text_input("Enter cryptocurrency symbols separated by commas (e.g., BTC, ETH, ADA)")

        if crypto_symbols:
            crypto_symbols_list = [symbol.strip() for symbol in crypto_symbols.split(',')]
            selected_tickers.extend([symbol + '-USD' for symbol in crypto_symbols_list])

        portfolio_amount = input_col.number_input(
            "Enter portfolio amount:", 
            min_value=1000.0, 
            step=1000.0, 
            value=1000.0, 
            format="%.2f"
        )

        risk_free_rate = input_col.number_input(
            "Enter risk-free rate (in %):", 
            min_value=0.0, 
            max_value=10.0, 
            step=0.1, 
            value=1.0
        ) / 100

        if input_col.button("Optimize Portfolio"):
            if len(selected_tickers) < 2:
                st.warning("At least 2 assets must be selected.")
            else:
                my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount)

                df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                df_allocation['Price per Share'] = '$' + latest_price.round(2).astype(str)
                df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)

                col1, col2 = st.columns([2, 2.5])

                with col1:
                    st.write("Allocated Funds:")
                    st.dataframe(df_allocation)
                    st.write("Remaining Funds: ${:.2f}".format(leftover))

                with col2:
                    st.write("Optimized Portfolio Composition:")
                    
                    colors = sns.color_palette('Set3', len(df_allocation))
                    explode = [0.05 if shares == max(df_allocation['Shares']) else 0 for shares in df_allocation['Shares']]

                    plt.figure(figsize=(8, 8))
                    plt.pie(df_allocation['Shares'], labels=df_allocation.index, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
                    plt.axis('equal')
                    st.pyplot(plt)
                
                sp500_prices = get_sp500_prices(start_date, end_date)
                sp500_returns = sp500_prices.pct_change().dropna()
                portfolio_returns = my_portfolio_returns.mean(axis=1)

                sharpe_ratio, sortino_ratio, max_drawdown, beta, treynor_ratio = calculate_performance_metrics(portfolio_returns, sp500_returns, risk_free_rate)

                st.write("Portfolio Performance Metrics:")
                st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
                st.write(f"Max Drawdown: {max_drawdown:.2%}")
                st.write(f"Beta: {beta:.2f}")
                st.write(f"Treynor Ratio: {treynor_ratio:.2f}")

                plt.figure(figsize=(10, 6))
                cumulative_returns = (1 + portfolio_returns).cumprod()
                sp500_cumulative_returns = (1 + sp500_returns).cumprod()
                plt.plot(cumulative_returns, label="Portfolio")
                plt.plot(sp500_cumulative_returns, label="S&P 500")
                plt.legend(loc="best")
                plt.title("Cumulative Returns")
                st.pyplot(plt)

if __name__ == "__main__":
    main()
