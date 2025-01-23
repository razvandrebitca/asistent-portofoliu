import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from translations import translations
import yfinance as yf
from fpdf import FPDF
import datetime as dt

# # Generate Full PDF Report
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

# Function to retrieve S&P 500 price data
def get_sp500_prices(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_prices = sp500_data['Adj Close']
    return sp500_prices

def backtest_portfolio(portfolio_returns, benchmark_returns):
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_portfolio, label="Portfolio", color="green")
    plt.plot(cumulative_benchmark, label="Benchmark (S&P 500)", color="orange")
    plt.legend()
    plt.title("S&P 500 vs. Optimized Portfolio")
    st.pyplot(plt)

# Function to calculate performance metrics
def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate):
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
    st.subheader("Historical Price Movement")
    st.line_chart(portfolio_data)

def plot_risk_return(my_portfolio_returns):
    st.subheader("Risk vs Return Analysis")
    asset_means = my_portfolio_returns.mean() * 252
    asset_vols = my_portfolio_returns.std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.scatter(asset_vols, asset_means, color="blue", alpha=0.7)
    for i, asset in enumerate(asset_means.index):
        plt.text(asset_vols.iloc[i], asset_means.iloc[i], asset, fontsize=9)
    plt.xlabel("Annual Risk (Volatility)")
    plt.ylabel("Annual Return")
    plt.title("Risk vs Return of Selected Assets")
    st.pyplot(plt)

def plot_correlation_heatmap(my_portfolio_returns):
    st.subheader("Heatmap")
    corr_matrix = my_portfolio_returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Selected Assets")
    st.pyplot(plt)

def display_portfolio(my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover,risk_free_rate,start_date,end_date, language):
    t = translations[language]
    df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
    df_allocation['Price per Share'] = '$' + latest_price.round(2).astype(str)
    df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str) 
    # Display metrics
    col1, col2, col3 = st.columns(3)
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
    # pdf_buffer = generate_full_pdf_report(selected_tickers, start_date, end_date, portfolio_amount, risk_free_rate,risk_tolerance)
    # st.download_button(
    # label="Download Report",
    # data=pdf_buffer,
    # file_name="Portfolio_Report.pdf",
    # mime="application/pdf"
    # ) 
    plot_asset_prices(my_portfolio)
    plot_risk_return(my_portfolio_returns)
    plot_correlation_heatmap(my_portfolio_returns) 
