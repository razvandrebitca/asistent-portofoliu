import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from translations import translations
import yfinance as yf
from fpdf import FPDF
import datetime as dt
from src.utils.optimization import optimize_portfolio
import io
import base64
# Generate Full PDF Report
def generate_full_pdf_report(selected_assets, start_date, end_date, portfolio_amount, risk_free_rate, risk_tolerance, language, file_name="Portfolio_Report.pdf"):
    t = translations[language]  # Assuming translations are defined elsewhere in your code
    
    # Optimize Portfolio
    my_portfolio, my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover = optimize_portfolio(
        selected_assets, start_date, end_date, portfolio_amount, risk_tolerance
    )
    
    # Retrieve Benchmark Data (S&P 500)
    sp500_prices = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    sp500_returns = sp500_prices.pct_change().dropna()
    portfolio_returns = my_portfolio_returns.mean(axis=1)
    
    # Calculate Performance Metrics
    sharpe_ratio, sortino_ratio, max_drawdown, beta = calculate_performance_metrics(
        portfolio_returns, sp500_returns, risk_free_rate, language
    )
    
    # Prepare Data for PDF
    metrics = {
        f"{t['sharpe_ratio']}": f"{sharpe_ratio:.2f}",
        f"{t['sortino_ratio']}": f"{sortino_ratio:.2f}",
        f"{t['max_drawdown']}": f"{max_drawdown:.2%}",
        f"{t['portfolio_beta']}": f"{beta:.2f}",
    }
    
    # Generate PDF
    pdf_buffer = io.BytesIO()
    pdf = FPDF()

    # Load DejaVu font (make sure you have DejaVuSans.ttf and DejaVuSans-Bold.ttf in the correct folder)
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)

    
    pdf.set_font("DejaVu", size=12)  # Use DejaVu font, which supports Romanian characters
    
    pdf.add_page()
    
    # Title
    pdf.cell(200, 10, txt=f"{t['report']}", ln=True, align='C')
    pdf.ln(10)
    
    # Allocation
    pdf.cell(200, 10, txt=f"{t['allocated_funds']}", ln=True, align='L')
    pdf.ln(5)
    for asset, shares in allocation.items():
        pdf.cell(200, 10, txt=f"{asset}: {shares} shares", ln=True, align='L')
    pdf.ln(10)
    
    # Metrics
    pdf.cell(200, 10, txt=f"{t['portfolio_metrics']}", ln=True, align='L')
    pdf.ln(5)
    for metric, value in metrics.items():
        pdf.cell(200, 10, txt=f"{metric}: {value}", ln=True, align='L')
    pdf.ln(10)
    
    # Remaining Funds
    pdf.cell(200, 10, txt=f"{t['remaining_funds']} ${leftover:.2f}", ln=True, align='L')
    
    # Save PDF to Buffer
    pdf_output = pdf.output(dest="S")  # This returns the PDF as a string
    pdf_buffer.write(pdf_output.encode('latin1'))  # Encode the string as bytes before writing to the buffer
    pdf_buffer.seek(0)
    
    return pdf_buffer


def monte_carlo_simulation(my_portfolio_returns, num_simulations, num_days,language):
    """
    Monte Carlo simulation to estimate future portfolio return distributions.
    
    Parameters:
    - my_portfolio_returns (pd.Series): Historical portfolio returns.
    - num_simulations (int): Number of simulations to run.
    - num_days (int): Number of days to project forward (default: 252 trading days in a year).
    
    Returns:
    - Plots a histogram of simulated future portfolio returns.
    """
    t = translations[language]
    # Calculate mean and standard deviation of daily returns
    mean_return = my_portfolio_returns.mean()
    std_dev = my_portfolio_returns.std()
    
    # Generate random daily returns using a normal distribution
    simulated_returns = np.random.normal(mean_return, std_dev, (num_simulations, num_days))
    
    # Compute cumulative returns over the simulated period
    cumulative_returns = (1 + simulated_returns).cumprod(axis=1)
    
    # Final portfolio values at the end of simulation period
    final_returns = cumulative_returns[:, -1] - 1  # Extract the final portfolio return
    
    # Plot histogram of simulated portfolio returns
    st.subheader(t['monte_carlo_title'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(final_returns, bins=50, kde=True, color="blue", alpha=0.7)
    ax.axvline(final_returns.mean(), color="red", linestyle="dashed", linewidth=2, label="Mean Expected Return")
    ax.axvline(np.percentile(final_returns, 5), color="orange", linestyle="dashed", linewidth=2, label="5th Percentile (Worst Case)")
    ax.axvline(np.percentile(final_returns, 95), color="green", linestyle="dashed", linewidth=2, label="95th Percentile (Best Case)")
    
    plt.xlabel(t['sim_return'])
    plt.ylabel("Frequency")
    plt.title(t['monte_carlo_title'])
    plt.legend()
    
    st.pyplot(fig)


# Function to retrieve S&P 500 price data
def get_sp500_prices(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_prices = sp500_data['Adj Close']
    return sp500_prices

def backtest_portfolio(portfolio_returns, benchmark_returns,language):
    t = translations[language]
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_portfolio, label="Portfolio", color="green")
    plt.plot(cumulative_benchmark, label="Benchmark (S&P 500)", color="orange")
    plt.legend()
    plt.title(t['portfolio_vs_sp500'])
    st.pyplot(plt)

# Function to calculate performance metrics
def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate,language):
    t = translations[language]
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError(t['empty_error'])

    # Ensure returns align on the same dates
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]

    # Convert annual risk-free rate to daily (assuming 252 trading days)
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

    # Excess return over risk-free rate
    excess_returns = portfolio_returns - daily_risk_free_rate

    # Sharpe Ratio (Risk-Adjusted Return)
    sharpe_ratio = excess_returns.mean() * np.sqrt(252) / (excess_returns.std() + 1e-8)  # Avoid div-by-zero

    # Sortino Ratio (Downside Risk-Adjusted Return)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() + 1e-8  # Avoid div-by-zero
    sortino_ratio = excess_returns.mean() * np.sqrt(252) / downside_std

    # Maximum Drawdown (Worst Peak-to-Trough Loss)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Beta Calculation (Covariance of portfolio & market / Variance of market)
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    if cov_matrix.shape == (2, 2) and np.var(benchmark_returns) > 0:
        beta = cov_matrix[0, 1] / np.var(benchmark_returns)
    else:
        beta = np.nan  # Handle cases where beta cannot be computed
    
    return sharpe_ratio, sortino_ratio, max_drawdown, beta


def plot_asset_prices(portfolio_data,language):
    t = translations[language]
    st.subheader(t['price_movement'])
    st.line_chart(portfolio_data)

def plot_risk_return(my_portfolio_returns,language):
    t = translations[language]
    st.subheader(t['risk_return'])
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

def plot_correlation_heatmap(my_portfolio_returns,language):
    t = translations[language]
    st.subheader(t['heatmap'])
    corr_matrix = my_portfolio_returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(t['correlation_heatmap'])
    st.pyplot(plt)

# Function to create a base64 download link for the PDF
def get_download_link(pdf_data):
    # Convert the BytesIO object to bytes and encode it to base64
    pdf_bytes = pdf_data.getvalue()
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="Portfolio_Report.pdf">📥</a>'

def display_portfolio(my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover, risk_free_rate, selected_assets, portfolio_amount, risk_tolerance, start_date, end_date, language):
    t = translations[language]
    # Generate the PDF only if not already done
    if "pdf_data" not in st.session_state:
        st.session_state["pdf_data"] = generate_full_pdf_report(
            selected_assets, start_date, end_date, portfolio_amount, risk_free_rate, risk_tolerance, language
        )

    # Convert allocation dictionary to a DataFrame
    df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
    
    # Add Price per Share and Cost columns
    df_allocation['Price per Share'] = '$' + latest_price.round(2).astype(str)
    df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)
    
    # Handle NaN values in 'Shares' column (replace with 0 and cast to int)
    df_allocation['Shares'] = df_allocation['Shares'].fillna(0).astype(int)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)

    # Portfolio allocation with collapsible section
    with col1:
        with st.expander(t['allocated_funds']):
            st.dataframe(df_allocation)
            st.write(f"{t['remaining_funds']} ${leftover:.2f}")

    # Portfolio metrics with collapsible section
    with col2:
        with st.expander(t['portfolio_metrics']):
            sharpe_ratio, sortino_ratio, max_drawdown, beta = calculate_performance_metrics(my_portfolio_returns.mean(axis=1), get_sp500_prices(start_date, end_date).pct_change().dropna(), risk_free_rate,language)
            st.markdown(
                f"{t['sharpe_ratio']}: {sharpe_ratio:.2f}",
                help=f"{t['tooltip_sharpe']}"
            )

            st.markdown(
                f"{t['sortino_ratio']}: {sortino_ratio:.2f}",
                help=f"{t['tooltip_sortino']}"
            )

            st.markdown(
                f"{t['max_drawdown']}: {max_drawdown:.2%}",
                help=f"{t['tooltip_max']}"
            )

            st.markdown(
                f"{t['portfolio_beta']}: {beta:.2f}",
                help=f"{t['tooltip_beta']}"
            )

    # Pie chart with collapsible section
    with col2:
        with st.expander(t['optimized_portfolio']):
            # Generate color palette for pie chart
            colors = sns.color_palette('Set3', len(df_allocation))

            # Create an explode array to emphasize the largest slice (optional)
            explode = [0.05 if shares == max(df_allocation['Shares']) else 0 for shares in df_allocation['Shares']]

            # Create the pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(df_allocation['Shares'], labels=df_allocation.index, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(plt)
    
    # Generate full PDF report with a collapsible section
    with st.expander(t['download_report']):
        st.markdown(get_download_link(st.session_state["pdf_data"]), unsafe_allow_html=True) 
    
    # Plot other asset-related charts in collapsible sections
    with st.expander(t['price_movement']):
        plot_asset_prices(my_portfolio,language)

    with st.expander(t['risk_return']):
        plot_risk_return(my_portfolio_returns,language)

    with st.expander(t['correlation_matrix']):
        plot_correlation_heatmap(my_portfolio_returns,language)

    with st.expander(t['monte_carlo_title']):
     monte_carlo_simulation(my_portfolio_returns.mean(axis=1), num_simulations=1000, num_days=252,language=language)
