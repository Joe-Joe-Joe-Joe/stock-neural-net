import pandas as pd
import json
import os
import datetime as dt
import re
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_SENTIMENT_DIR = os.path.join(SCRIPT_DIR, "scraped_news_sentiment\\")
STOCK_DIR = os.path.join(SCRIPT_DIR, "stock_data\\")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "stacked_data\\")
COMPANY_NAME = ["Tesla", "McDonald's", "Meta"]
# NAICS code from https://www.naics.com/search/ and inputting company's name
COMPANY_TYPE = [[33,6,1,1,0], [72,2,5,1,3], [51,9,2,9,0]]
# location of company's corporate HQ
COMPANY_STATE = ["TX", "IL", "CA"]
# company networth, according to https://companiesmarketcap.com
COMPANY_MARKETCAP = [1396000000000, 220450000000, 1385000000000]
WINDOW_SIZE = 4
EMPTY_DATE = dt.datetime(1900,1,1)

target_company = 2
start_date = dt.datetime(2025, 1, 4)
end_date = dt.datetime(2025, 1, 31)

# uncomment once working with data that is aligned with sunday/saturday frame
# assert start_date.weekday() == 6, "Start date is not a Sunday!"
# assert start_date.weekday() == 5, "End date is not a Saturday!"

def str_to_date(str:str):
    return dt.datetime.strptime(str[0:str.find("T")], "%Y-%m-%d")

date_data = {"date": EMPTY_DATE,
             "combined_sentiment": 5.0,
             "open": 0.0,
             "close": 0.0,
             "high": 0.0,  
             "low": 0.0,
             "volume": 0.0}
week = {"week_id": -1,
        "week_start_date": EMPTY_DATE,
        "week_end_date": EMPTY_DATE,
        "days": []
        }
block = {"block_id": -1,
         "block_start_date": EMPTY_DATE,
         "block_end_date": EMPTY_DATE,
         "weeks": []}
input_data = {"company_name": COMPANY_NAME[target_company],
              "company_type": COMPANY_TYPE[target_company],
              "company_state": COMPANY_STATE[target_company],
              "company_marketcap":COMPANY_MARKETCAP[target_company],
              "blocks": []}

# initialize date_data_list with all date_data between start_date and end_date
date_data_list = []
for day in range((end_date - start_date).days + 1):
    date = (start_date + dt.timedelta(days=day))
    tmp = date_data.copy()
    tmp["date"] = date
    date_data_list.append(tmp)

assert len(date_data_list)%7 == 0 and len(date_data_list)//7 >= WINDOW_SIZE, "\"date_data_list\" is wrong size!"

# fill array with sentiment data
for file in os.listdir(NEWS_SENTIMENT_DIR):
    if COMPANY_NAME[target_company] not in file:
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
    if COMPANY_NAME[target_company] not in file:
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
for block_id in range(len(date_data_list)//7 - WINDOW_SIZE + 1):
    cur_block = deepcopy(block)
    cur_block["block_id"] = block_id
    # TO DO: add start and end dates to block
    for week_id in range(WINDOW_SIZE):
        cur_week = deepcopy(week)
        cur_week["week_id"] = week_id
        for day_id in range(7):
            index = day_id + week_id*7 + block_id*WINDOW_SIZE
            cur_week["days"].append(date_data_list[index])
        cur_week["week_start_date"] = cur_week["days"][0]["date"]
        cur_week["week_end_date"] = cur_week["days"][-1]["date"]
        cur_block["weeks"].append(cur_week)
    cur_block["block_start_date"] = cur_block["weeks"][0]["days"][0]["date"]
    cur_block["block_end_date"] = cur_block["weeks"][-1]["days"][-1]["date"]
    input_data["blocks"].append(cur_block)

with open(f"{COMPANY_NAME[target_company]}_data.json", "w", encoding="utf-8") as f:
    json.dump(input_data, f, indent=4, default=str)