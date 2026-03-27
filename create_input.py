import pandas as pd
import json
import os
import datetime as dt
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_SENTIMENT_DIR = os.path.join(SCRIPT_DIR, "scraped_news_sentiment\\")
STOCK_DIR = os.path.join(SCRIPT_DIR, "stock_data\\")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "stacked_data\\")
COMPANIES = ["Tesla", "McDonald's", "Meta"]
WINDOW_SIZE = 4

target_company = 2
start_date = dt.datetime(2025, 1, 4)
end_date = dt.datetime(2025, 1, 31)

# class stock_data:
#         def __init__(self, date: str, close: float = 0, high: float = 0, low: float = 0, volume: int = 0):
#             self.date = date
#             self.close = close
#             self.high = high
#             self.low = low
#             self.volume = volume

# class date_data:
#     def __init__(self, date: dt.datetime, combined_sentiment: float = 5.0, stock_price: stock_data = None):
#         self.date = date
#         self.combined_sentiment = combined_sentiment
#         self.stock_data = stock_price

def str_to_date(str:str):
    return dt.datetime.strptime(str[0:str.find("T")], "%Y-%m-%d")

date_data = {"date": dt.datetime(1999,1,1),
             "combined_sentiment": 5.0,
             "open": 0.0,
             "close": 0.0,
             "high": 0.0,  
             "low": 0.0,
             "volume": 0.0}

date_data_nostockdata = {"date": dt.datetime(1999,1,1),
                     "combined_sentiment": 5.0}


# initialize date_data_list with all date_data between start_date and end_date
date_data_list = []
for day in range((end_date - start_date).days + 1):
    date = (start_date + dt.timedelta(days=day))
    tmp = date_data.copy()
    tmp["date"] = date
    date_data_list.append(tmp)


# fill array with sentiment data
for file in os.listdir(NEWS_SENTIMENT_DIR):
    if COMPANIES[target_company] not in file:
        continue

    with open(f"{NEWS_SENTIMENT_DIR}/{file}", encoding='utf-8') as f:
        sentiment_data = json.load(f)
    
    dates_of_multiple_articles = {}
    for date, articles in sentiment_data.items():
        for article in articles:
            publish_date_str = article.get("publish_date")
            if publish_date_str == None:
                publish_date = dt.datetime.strptime(date, "%Y_%m_%d")
            else:
                publish_date = str_to_date(article.get("publish_date"))
            assert publish_date <= end_date and publish_date >= start_date, "Article publish date is out of range"

            index = (publish_date - start_date).days
            # if multiple articles on same date, take combined sentiment
            cur_sentiment = date_data_list[index]["combined_sentiment"]
            if cur_sentiment == 5.0:
                date_data_list[index]["combined_sentiment"] = article.get("combined_sentiment")
            else:
                date_data_list[index]["combined_sentiment"] = (cur_sentiment) + article.get("combined_sentiment")
                if dates_of_multiple_articles.get(publish_date) == None:
                    dates_of_multiple_articles[publish_date] = 2
                else:
                    dates_of_multiple_articles[publish_date] += 1

    for date, num_of_articles in dates_of_multiple_articles.items():
        index = (date - start_date).days
        date_data_list[index]["combined_sentiment"] /= num_of_articles


# fill array with stock data
for file in os.listdir(STOCK_DIR):
    if COMPANIES[target_company] not in file:
        continue

    with open(f"{STOCK_DIR}/{file}", encoding='utf-8') as f:
        stock_data = json.load(f)
        for stock_infoandticker, stock_data_date in stock_data.items():
            stock_infotype = re.sub("[()']","",stock_infoandticker).split(',')[0].lower()
            for date, price in stock_data_date.items():
                date = str_to_date(date)
                index = (date - start_date).days
                date_data_list[index][stock_infotype] = price

# create blocks
            

df = pd.DataFrame(date_data_list)
with open(f"{COMPANIES[target_company]}_data.json", "w", encoding="utf-8") as f:
    df.to_json(f, indent=4, index=False, date_format="iso")