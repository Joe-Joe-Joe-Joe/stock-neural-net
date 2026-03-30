import yfinance as yf
import datetime

company_tickers = {
    "Tesla": 'TSLA',
    "McDonald's": 'MCD',
    "Meta": 'META'
}

start_date = datetime.datetime(2025,1,1)
end_date = datetime.datetime(2025,12,31)

for name in company_tickers.keys():
    stock_data = yf.download(company_tickers[name], start_date, end_date)
    print(stock_data.head())
    with open(f"stock_data/{name}.json", "w") as f:
        stock_data_json = stock_data.to_json(date_format="iso")
        f.writelines(stock_data_json)