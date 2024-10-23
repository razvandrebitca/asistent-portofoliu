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
language = st.sidebar.radio("Language / Limba:", ("English", "Română"))
translations = {
    "English": {
        "title": "Portfolio Analysis Assistant",
        "description_text": "This app uses Markowitz's algorithm to optimize portfolios containing S&P 500 stocks and cryptocurrencies, using historical trading data.",
        "start_date": "Start Date:",
        "end_date": "End Date:",
        "select_assets": "Select S&P 500 stocks for your portfolio:",
        "custom_tickers": "Enter custom stock tickers separated by commas (e.g., AAPL, MSFT, TSLA)",
        "crypto_symbols": "Enter cryptocurrency symbols separated by commas (e.g., BTC, ETH, ADA)",
        "portfolio_amount": "Enter portfolio amount:",
        "risk_free_rate": "Enter risk-free rate (in %):",
        "optimize_button": "Optimize Portfolio",
        "warning_assets": "At least 2 assets must be selected.",
        "allocated_funds": "Allocated Funds:",
        "remaining_funds": "Remaining Funds:",
        "optimized_portfolio": "Optimized Portfolio Composition:",
        "cumulative_return": "S&P 500 vs. Optimized Portfolio",
        "correlation_heatmap": "Correlation Heatmap of Selected Assets:",
        "portfolio_vs_sp500": "Portfolio vs. S&P 500:",
        "sharpe_ratio": "Sharpe Ratio:",
        "sortino_ratio": "Sortino Ratio:",
        "max_drawdown": "Max Drawdown:",
        "portfolio_beta": "Portfolio Beta:",
        "treynor_ratio": "Treynor Ratio:",
        "welcome":"Welcome",
        "time":"Select Time Period for Historical Data",
        "portfolio":"Portfolio",
        "description":"Description",
        "theme":"Theme"
    },
    "Română": {
        "title": "Asistent de Analiză a Portofoliului",
        "description_text": "Această aplicație folosește algoritmul lui Markowitz pentru a optimiza portofoliile care conțin acțiuni S&P 500 și criptomonede, folosind date istorice de tranzacționare.",
        "start_date": "Data de început:",
        "end_date": "Data de sfârșit:",
        "select_assets": "Selectează acțiunile S&P 500 pentru portofoliul tău:",
        "custom_tickers": "Introdu simbolurile acțiunilor personalizate separate prin virgule (de ex., AAPL, MSFT, TSLA)",
        "crypto_symbols": "Introdu simbolurile criptomonedelor separate prin virgule (de ex., BTC, ETH, ADA)",
        "portfolio_amount": "Introdu suma portofoliului:",
        "risk_free_rate": "Introdu rata fără risc (în %):",
        "optimize_button": "Optimizează Portofoliul",
        "warning_assets": "Trebuie selectate cel puțin 2 active.",
        "allocated_funds": "Fonduri Alocate:",
        "remaining_funds": "Fonduri rămase:",
        "optimized_portfolio": "Compoziția Portofoliului Optimizat:",
        "cumulative_return": "S&P 500 vs. Portofoliu Optimizat",
        "correlation_heatmap": "Harta de Corelație a Activelor Selectate:",
        "portfolio_vs_sp500": "Portofoliul vs. S&P 500:",
        "sharpe_ratio": "Raport Sharpe:",
        "sortino_ratio": "Raport Sortino:",
        "max_drawdown": "Maximă Scădere:",
        "portfolio_beta": "Beta Portofoliu:",
        "treynor_ratio": "Raport Treynor:",
        "welcome":"Bun venit",
        "time":"Selectați intervalul de timp pentru datele istorice",
        "portfolio":"Portofoliu",
        "description":"Descriere",
        "theme":"Temă"
    }
}
def apply_theme(theme):
   if theme == "Dark":
        dark_css = """
        <style>
            body, .stApp {
                background-color: #303030 !important;
                color: white !important;
            }
            .stButton>button, .stTextInput>div>div>input {
                background-color: #565656 !important;
                color: white !important;
            }
            .stDataFrame {
                background-color: #3e3e3e !important;
                color: white !important;
            }
            .stSidebar, .sidebar-content {
                background-color: #333333 !important;
                color: white !important;
            }
            .stTable {
                background-color: #3e3e3e !important;
                color: white !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: white !important;
            }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
   else:
        light_css = """
        <style>
            body, .stApp {
                background-color: white !important;
                color: black !important;
            }
            .stButton>button, .stTextInput>div>div>input {
                background-color: #f0f0f0 !important;
                color: black !important;
            }
            .stDataFrame {
                background-color: white !important;
                color: black !important;
            }
            .stSidebar, .sidebar-content {
                background-color: #f8f9fa !important;
                color: black !important;
            }
            .stTable {
                background-color: white !important;
                color: black !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: black !important;
            }
            .st-c6 {
            color: rgb(51, 184, 255);
            }
          .st-emotion-cache-6qob1r {
           position: relative;
            height: 100%;
            width: 100%;
           overflow: overlay;
           background-color:white!important;
           }
            p{
                color:black;
            }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

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
    t = translations[language]
    theme_choice = st.sidebar.radio(t["theme"], ("Light", "Dark"))
    apply_theme(theme_choice)
    
    
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
        st.sidebar.title(t['welcome']+f", {name}!")      
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

        st.title(t["title"])
        st.markdown("---")
        st.subheader(t["description"])
        st.info(t['description_text'])
        st.markdown("---")

        input_col = st.sidebar
        input_col.header(t['time'])

        start_date = input_col.date_input(t['start_date'], dt.datetime(2020, 1, 1))
        end_date = input_col.date_input(t['end_date'], dt.datetime.now())

        ticker_company_dict = get_sp500_tickers()

        input_col.header(t['portfolio'])

        selected_tickers = input_col.multiselect(
            t['select_assets'], 
            list(ticker_company_dict.keys()), 
            format_func=lambda ticker: f"{ticker}: {ticker_company_dict[ticker]}"
        )

        # Input for custom ticker symbols
        custom_tickers = input_col.text_input(t['custom_tickers'])

        if custom_tickers:
            custom_tickers_list = [ticker.strip() for ticker in custom_tickers.split(',')]
            selected_tickers.extend(custom_tickers_list)

        # Input for cryptocurrencies
        crypto_symbols = input_col.text_input(t['crypto_symbols'])

        if crypto_symbols:
            crypto_symbols_list = [symbol.strip() for symbol in crypto_symbols.split(',')]
            selected_tickers.extend([symbol + '-USD' for symbol in crypto_symbols_list])

        portfolio_amount = input_col.number_input(
            t['portfolio_amount'], 
            min_value=1000.0, 
            step=1000.0, 
            value=1000.0, 
            format="%.2f"
        )

        risk_free_rate = input_col.number_input(
            t['risk_free_rate'], 
            min_value=0.0, 
            max_value=10.0, 
            step=0.1, 
            value=1.0
        ) / 100

        if input_col.button(t['optimize_button']):
            if len(selected_tickers) < 2:
                st.warning(t['warning_assets'])
            else:
                my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount)

                df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                df_allocation['Price per Share'] = '$' + latest_price.round(2).astype(str)
                df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)

                col1, col2 = st.columns([2, 2.5])

                with col1:
                    st.write(t['allocated_funds'])
                    st.dataframe(df_allocation)
                    st.write(t['remaining_funds']+" ${:.2f}".format(leftover))

                with col2:
                    st.write(t['optimized_portfolio'])
                    
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
                st.write(t['sharpe_ratio']+f" {sharpe_ratio:.2f}")
                st.write(t['sortino_ratio']+f" {sortino_ratio:.2f}")
                st.write(t['max_drawdown']+f" {max_drawdown:.2%}")
                st.write(t['portfolio_beta']+f": {beta:.2f}")
                st.write(t['treynor_ratio']+f" {treynor_ratio:.2f}")

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
