import streamlit as st
import datetime as dt
from src.utils.data import get_sp500_tickers

def create_sidebar(t):
    with st.sidebar:
        st.header(t["settings_header"])


        theme = st.radio(t["theme"], ("Dark", "Light"))
        # Date range selection
        st.subheader(t["date_range_header"])
        start_date = st.date_input(t["start_date"], dt.datetime(2020, 1, 1))
        end_date = st.date_input(t["end_date"], dt.datetime.now())
        # Ticker selection
        tickers = get_sp500_tickers()
        selected_tickers = st.multiselect(
        t['select_assets'], 
        list(tickers.keys()), 
        format_func=lambda ticker: f"{ticker}: {tickers[ticker]}"
          )
        # Input for custom ticker symbols
        custom_tickers = st.text_input(t['custom_tickers'])

        if custom_tickers:
            custom_tickers_list = [ticker.strip() for ticker in custom_tickers.split(',')]
            selected_tickers.extend(custom_tickers_list)

     # Input for cryptocurrencies
        crypto_symbols = st.text_input(t['crypto_symbols'])

        if crypto_symbols:
            crypto_symbols_list = [symbol.strip() for symbol in crypto_symbols.split(',')]
            selected_tickers.extend([symbol + '-USD' for symbol in crypto_symbols_list])
            
        # File uploader for tickers
        uploaded_file = st.file_uploader(t['upload'], type=["csv", "txt"])
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
        portfolio_amount = st.number_input(t["portfolio_amount"], min_value=1000.0)
       # risk_free_rate = st.number_input(t['risk_free_rate'],  min_value=0.0,  max_value=10.0,  step=0.1,  value=1.0) / 100
        risk_tolerance = st.radio(t["risk_tolerance"], ("Low", "Medium", "High"))
        
        
    return {
        "theme": theme,
        "start_date": start_date,
        "end_date": end_date,
        "portfolio_amount": portfolio_amount,
        "risk_tolerance": risk_tolerance,
        "risk_free_rate": 1, # risk_free_rate,
        "selected_tickers": selected_tickers
    }