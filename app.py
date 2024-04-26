import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Funcție pentru a prelua toate companiile din S&P 500 de pe wikipedia
def get_sp500_tickers():
  
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    table = data[0]
    tickers = table['Symbol'].tolist()
    company_names = table['Security'].tolist()
    ticker_company_dict = dict(zip(tickers, company_names))
    return ticker_company_dict

# Obținere istoric prețuri pt. S&P 500
def get_sp500_prices(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_prices = sp500_data['Adj Close']
    return sp500_prices

# Funcție pentru optimizarea portofoliului folosind frontiera Markowitz
def optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount):
    n = len(selected_tickers)
    
   
    my_portfolio = pd.DataFrame()
    for ticker in selected_tickers:
        my_portfolio[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    

    
    my_portfolio_returns = my_portfolio.pct_change().dropna()

    
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)
    
   
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    latest_prices = get_latest_prices(my_portfolio)
    
   
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()
    
    return my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover

def main():
  
    st.set_page_config(page_title="Asistent Virtual pentru Analiza Portofoliului")

 
    st.title("Asistent Virtual pentru Analiza Portofoliului")
    
    st.markdown("---")
    st.subheader("Descriere")
    st.info(
            "Această aplicație folosește algoritmul lui Markowitz pentru a optimiza portofolii ce conțin acțiuni S&P 500, folosind date istorice de tranzacționare a indexului S&P 500.\n"
        )
    st.markdown("---")

    input_col = st.sidebar
    input_col.header("Selectare interval de timp pentru instoricul de tranzacționare")

    
    start_date = input_col.date_input("Dată început:", dt.datetime(2020, 1, 1))
    end_date = input_col.date_input("Dată sfârșit:", dt.datetime.now())

 
    ticker_company_dict = get_sp500_tickers()

    input_col.header("Portofoliu")

    selected_tickers = input_col.multiselect("Selectați tipurile de acțiuni din portofoliul dvs.:", list(ticker_company_dict.keys()), format_func=lambda ticker: f"{ticker}: {ticker_company_dict[ticker]}")

  
    portfolio_amount = input_col.number_input("Introduceți suma investită în portfoliu:", min_value=1000.0, step=1000.0, value=1000.0, format="%.2f")

   
    if input_col.button("Optimizare"):
        if len(selected_tickers) < 2:
            st.warning("Trebuie selectate cel puțin 2 tipuri de active.")
        else:
            my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount)
            
     
            df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
            df_allocation['Preț acțiune'] = '$' + latest_price.round(2).astype(str)
            df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)
    
           
            col1, col2 = st.columns([2, 2.5])

           
            with col1:
                st.write("Fonduri alocate:")
                st.dataframe(df_allocation)
                st.write("Fonduri rămase: ${:.2f}".format(leftover))

           
            with col2:
                st.write("Compoziție portofoliu optimizat:")
                
                colors = sns.color_palette('Set3', len(df_allocation))

               
                explode = [0.05 if shares == max(df_allocation['Shares']) else 0 for shares in df_allocation['Shares']]

               
                plt.figure(figsize=(8,8))
                plt.pie(df_allocation['Shares'], labels=df_allocation.index, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
                plt.axis('equal')

                st.pyplot(plt)
            
            
            sp500_prices = get_sp500_prices(start_date, end_date)

            
            sp500_returns = sp500_prices.pct_change().dropna()

            
            my_portfolio_returns_array = my_portfolio_returns.values
            cleaned_weights_array = np.array(list(cleaned_weights.values()))

            
            portfolio_returns = np.dot(my_portfolio_returns_array, cleaned_weights_array)

           
            sp500_expected_returns = sp500_returns.mean() * 252 
            sp500_volatility = sp500_returns.std() * np.sqrt(252)
            portfolio_expected_returns = portfolio_returns.mean() * 252
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

          
            combined_returns = pd.DataFrame({'S&P 500': sp500_returns, 'Portfoliu dvs.': portfolio_returns}, index=my_portfolio_returns.index)

        
            plt.figure(figsize=(12, 6))
            plt.plot(combined_returns.index, 100 * (combined_returns + 1).cumprod(), lw=2)
            plt.legend(combined_returns.columns)
            plt.xlabel('Dată')
            plt.ylabel('Rentabilitatea cumulativă (%)')
            plt.title('S&P 500 vs. Portofoliu optimizat')
            plt.grid(True)
            plt.tight_layout()     
            st.pyplot(plt)
            df_info = pd.DataFrame({
                'S&P 500': ['{:.2f}%'.format(100 * sp500_expected_returns), '{:.2f}%'.format(100 * sp500_volatility)],
                'Portfoliu': ['{:.2f}%'.format(100 * portfolio_expected_returns), '{:.2f}%'.format(100 * portfolio_volatility)]
            }, index=['Profit așteptat', 'Volatilitate'])

            st.dataframe(df_info)


if __name__ == "__main__":
    main()