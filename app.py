import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from translations import translations
import io

st.set_page_config(page_title="Portfolio Analysis Assistant", layout="wide")
language = st.sidebar.radio("Language / Limba:", ("English", "Română"))
t = translations[language]
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

# # Function to optimize portfolio using Markowitz efficient frontier
# def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount):
#     my_portfolio = pd.DataFrame()
#     for asset in selected_assets:
#         my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date)['Adj Close']
    
#     my_portfolio_returns = my_portfolio.pct_change().dropna()
#     mu = expected_returns.mean_historical_return(my_portfolio)
#     S = risk_models.sample_cov(my_portfolio)
    
#     ef = EfficientFrontier(mu, S)
#     weights = ef.max_sharpe()  # Optimizes for maximum Sharpe ratio
#     cleaned_weights = ef.clean_weights()
    
#     latest_prices = get_latest_prices(my_portfolio)
#     da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
#     allocation, leftover = da.lp_portfolio()
    
#     return my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover
# Adjusted optimize_portfolio function with risk tolerance
def optimize_portfolio(selected_assets, start_date, end_date, portfolio_amount, risk_tolerance):
    my_portfolio = pd.DataFrame()
    for asset in selected_assets:
        my_portfolio[asset] = yf.download(asset, start=start_date, end=end_date)['Adj Close']

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
    else:
        weights = ef.max_sharpe()  # Default to max Sharpe ratio

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



def plot_asset_prices(portfolio_data):
    st.subheader(t['price_movement'])
    st.line_chart(portfolio_data)

def plot_risk_return(my_portfolio_returns):
    st.subheader(t['risk_analysis'])
    asset_means = my_portfolio_returns.mean() * 252
    asset_vols = my_portfolio_returns.std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.scatter(asset_vols, asset_means, color="blue", alpha=0.7)
    for i, asset in enumerate(asset_means.index):
        plt.text(asset_vols.iloc[i], asset_means.iloc[i], asset, fontsize=9)
    plt.xlabel(t['annual_risk'])
    plt.ylabel(t['annual_return'])
    plt.title(t['risk_return'])
    st.pyplot(plt)

def plot_correlation_heatmap(my_portfolio_returns):
    st.subheader(t['heatmap'])
    corr_matrix = my_portfolio_returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(t['correlation_matrix'])
    st.pyplot(plt)

def backtest_portfolio(portfolio_returns, benchmark_returns):
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_portfolio, label="Portfolio", color="green")
    plt.plot(cumulative_benchmark, label="Benchmark (S&P 500)", color="orange")
    plt.legend()
    plt.title(t['cumulative_return'])
    st.pyplot(plt)

# Generate Full PDF Report
def generate_full_pdf_report(selected_assets, start_date, end_date, portfolio_amount, risk_free_rate, risk_tolerance, file_name="Portfolio_Report.pdf"):
    # Optimize Portfolio
    my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover = optimize_portfolio(
        selected_assets, start_date, end_date, portfolio_amount, risk_tolerance
    )
    
    # Retrieve Benchmark Data (S&P 500)
    sp500_prices = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    sp500_returns = sp500_prices.pct_change().dropna()
    portfolio_returns = my_portfolio_returns.mean(axis=1)
    
    # Calculate Performance Metrics
    sharpe_ratio, sortino_ratio, max_drawdown, beta, treynor_ratio = calculate_performance_metrics(
        portfolio_returns, sp500_returns, risk_free_rate
    )
    
    # Prepare Data for PDF
    metrics = {
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Sortino Ratio": f"{sortino_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Beta": f"{beta:.2f}",
        "Treynor Ratio": f"{treynor_ratio:.2f}"
    }
    
    # Generate PDF
    pdf_buffer = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.cell(200, 10, txt=t['report'], ln=True, align='C')
    pdf.ln(10)
    
    # Allocation
    pdf.cell(200, 10, txt="Optimized Allocation", ln=True, align='L')
    pdf.ln(5)
    for asset, shares in allocation.items():
        pdf.cell(200, 10, txt=f"{asset}: {shares} shares", ln=True, align='L')
    pdf.ln(10)
    
    # Metrics
    pdf.cell(200, 10, txt="Performance Metrics", ln=True, align='L')
    pdf.ln(5)
    for metric, value in metrics.items():
        pdf.cell(200, 10, txt=f"{metric}: {value}", ln=True, align='L')
    pdf.ln(10)
    
    # Remaining Funds
    pdf.cell(200, 10, txt=f"Remaining Funds: ${leftover:.2f}", ln=True, align='L')
    
    # Save PDF to Buffer
    pdf.output(dest="S").encode("latin1")
    pdf_buffer.write(pdf.output(dest="S").encode("latin1"))
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Main function for the Streamlit app
def main():
   
    theme_choice = st.sidebar.radio(t["theme"], ("Dark", "Light"))
    apply_theme(theme_choice)

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


    # File uploader for tickers
    uploaded_file = input_col.file_uploader(t['upload'], type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith("csv"):
                file_tickers = pd.read_csv(uploaded_file, header=None).iloc[:, 0].tolist()
            else:
                file_tickers = uploaded_file.read().decode("utf-8").splitlines()
            selected_tickers.extend([ticker.strip() for ticker in file_tickers if ticker.strip()])
            st.sidebar.success(t['success'])
        except Exception as e:
            st.sidebar.error(f"{t['error']}: {e}")

    portfolio_amount = input_col.number_input(
        t['portfolio_amount'], 
        min_value=1000.0, 
        step=1000.0, 
        value=1000.0, 
        format="%.2f"
    )

    col_rfr, col_tooltip = st.sidebar.columns([8, 1])
    with col_rfr:
        risk_free_rate = st.number_input(
            t['risk_free_rate'], 
            min_value=0.0, 
            max_value=10.0, 
            step=0.1, 
            value=1.0
        ) / 100
    with col_tooltip:
        st.sidebar.markdown(
            "", 
            help=t['tooltip_risk']
        )

        risk_tolerance = input_col.radio(
        t["risk_tolerance"], 
        ("Low", "Medium", "High"), 
        help="Help"
        )


    if input_col.button(t['optimize_button']):
            if len(selected_tickers) < 2:
                st.warning(t['warning_assets'])
            else:
                my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount, risk_tolerance)

                df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                df_allocation['Price per Share'] = '$' + latest_price.round(2).astype(str)
                df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)
                col1, col2 = st.columns([2, 2.5])

                with col1:
                    st.subheader(t['allocated_funds'])
                    st.dataframe(df_allocation)
                    st.subheader(t['remaining_funds']+" ${:.2f}".format(leftover))

                with col2:
                    st.subheader(t['optimized_portfolio'])
                    
                    colors = sns.color_palette('Set3', len(df_allocation))
                    explode = [0.05 if shares == max(df_allocation['Shares']) else 0 for shares in df_allocation['Shares']]

                    plt.figure(figsize=(8, 8))
                    plt.pie(df_allocation['Shares'], labels=df_allocation.index, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
                    plt.axis('equal')
                    st.pyplot(plt)
                
                sp500_prices = get_sp500_prices(start_date, end_date)
                sp500_returns = sp500_prices.pct_change().dropna()
                portfolio_returns = my_portfolio_returns.mean(axis=1)
                backtest_portfolio(my_portfolio_returns.mean(axis=1), sp500_returns)
                sharpe_ratio, sortino_ratio, max_drawdown, beta, treynor_ratio = calculate_performance_metrics(portfolio_returns, sp500_returns, risk_free_rate)

                st.subheader(t['portfolio_metrics'])
                
                st.markdown(
                 f"""
                  <div style="margin-top:20px; font-size: 16px;">
                 {t['sharpe_ratio']} {sharpe_ratio:.2f}
                 <span style="color: gray; cursor: pointer; border: 1px solid #ddd; padding: 5px; border-radius: 5px;"
                      title="{t['tooltip_sharpe']}">
                      ℹ
                 </span>
                 </div>
                 """,
                  unsafe_allow_html=True
                 )

                st.markdown(
                f"""
             <div style="margin-top:20px; font-size: 16px;">
                {t['sortino_ratio']} {sortino_ratio:.2f}
                <span style="color: gray; cursor: pointer; border: 1px solid #ddd; padding: 5px; border-radius: 5px;"
                      title="{t['tooltip_sortino']}">
                      ℹ
                </span>
                </div>
                """,
              unsafe_allow_html=True
              )
                st.markdown(
               f"""
               <div style="margin-top:20px; font-size: 16px;">
                {t['max_drawdown']} {max_drawdown:.2%}
                <span style="color: gray; cursor: pointer; border: 1px solid #ddd; padding: 5px; border-radius: 5px;"
                      title="{t['tooltip_max']}">
                      ℹ
                </span>
               </div>
              """,
               unsafe_allow_html=True
              )

                st.markdown(
              f"""
             <div style="margin-top:20px;font-size: 16px;">
                {t['portfolio_beta']} {beta:.2f}
                <span style="color: gray; cursor: pointer; border: 1px solid #ddd; padding: 5px; border-radius: 5px;"
                      title="{t['tooltip_beta']}">
                      ℹ
                </span>
               </div>
              """,
               unsafe_allow_html=True
               )

                st.markdown(
             f"""
             <div style="margin-top:20px; font-size: 16px;">
                {t['treynor_ratio']} {treynor_ratio:.2f}
                <span style="color: gray; cursor: pointer; border: 1px solid #ddd; padding: 5px; border-radius: 5px;"
                      title="{t['tooltip_treynor']}">
                      ℹ
                </span>
              </div>
              """,
              unsafe_allow_html=True
             )
                pdf_buffer = generate_full_pdf_report(selected_tickers, start_date, end_date, portfolio_amount, risk_free_rate,risk_tolerance)
                st.download_button(
                label="Download Portfolio Report",
                data=pdf_buffer,
                file_name="Portfolio_Report.pdf",
                mime="application/pdf"
                ) 
                plot_asset_prices(my_portfolio)
                plot_risk_return(my_portfolio_returns)
                plot_correlation_heatmap(my_portfolio_returns) 
                


if __name__ == "__main__":
    main()