import streamlit as st
from src.styles.theme import apply_theme
from src.components.sidebar import create_sidebar
from src.components.portfolio import display_portfolio
from src.utils.optimization import optimize_portfolio
from translations import translations

def main():
    st.set_page_config(page_title="Portfolio Analysis Assistant", layout="wide")
    
    # Language selection
    language = st.sidebar.radio("Language / Limba:", ("English", "Română"))
    t = translations[language]
    
    # Create sidebar and get settings
    settings = create_sidebar(t)
    
    # Apply theme
    st.markdown(apply_theme(settings["theme"]), unsafe_allow_html=True)
    
    # Main content
    st.title(t["title"])
    st.markdown("---")
    
    # Portfolio optimization
    if st.sidebar.button(t["optimize_button"]):
        with st.spinner(t["optimizing"]):
         my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(
                settings["selected_tickers"],
                settings["start_date"],
                settings["end_date"],
                settings["portfolio_amount"],
                settings["risk_tolerance"]
            )
         display_portfolio(my_portfolio, my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover,settings["risk_free_rate"],settings["selected_tickers"],settings["portfolio_amount"],settings["risk_tolerance"],settings["start_date"],settings["end_date"],language)

if __name__ == "__main__":
    main()