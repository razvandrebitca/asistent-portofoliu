import pandas as pd

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    table = data[0]
    tickers = table['Symbol'].tolist()
    company_names = table['Security'].tolist()
    ticker_company_dict = dict(zip(tickers, company_names))
    return ticker_company_dict