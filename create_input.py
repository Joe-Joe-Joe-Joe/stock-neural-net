import pandas as pd
import json
import os
import datetime as dt
import re
from copy import deepcopy
import tensorflow as tf
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_SENTIMENT_DIR = os.path.join(SCRIPT_DIR, "scraped_news_sentiment\\")
STOCK_DIR = os.path.join(SCRIPT_DIR, "stock_data\\")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "stacked_data\\")
COMPANY_NAME = ["Tesla", "McDonald's", "Meta"]
# NAICS code from https://www.naics.com/search/
COMPANY_TYPE = [[33,6,1,1,0], [72,2,5,1,3], [51,9,2,9,0]]
COMPANY_TYPE_LEN = len(COMPANY_TYPE[0])
# location of company's corporate HQ
COMPANY_STATE = ["TX", "IL", "CA"]
COMPANY_STATE_LEN = len(COMPANY_STATE)
# company networth, according to https://companiesmarketcap.com as of 2026-03-26
COMPANY_MARKETCAP = [1396000000000, 220450000000, 1385000000000]
# number of days in a batch
WINDOW_SIZE = 7
# features are per day
NUM_COMPANY_FEATURES = COMPANY_TYPE_LEN + COMPANY_STATE_LEN + 1        # each number in NAICS code counts as feature, each state counts as feature (used in one-hot vector)
NUM_STOCK_FEATURES = 5
NUM_SENTIMENT_FEATURES = 1
NUM_FEATURES = NUM_COMPANY_FEATURES + NUM_STOCK_FEATURES + NUM_SENTIMENT_FEATURES
EMPTY_DATE = dt.datetime(1900,1,1)

target_company = 2
start_date = dt.datetime(2025, 1, 4)        # start date is inclusive
end_date = dt.datetime(2025, 1, 31)         # end date is inclusive

def str_to_date(str:str):
    return dt.datetime.strptime(str[0:str.find("T")], "%Y-%m-%d")

# takes in json file and outputs 3D tensor with dimensions [(start_date-end_date).days-WINDOW_SIZE, WINDOW_SIZE, # of features (company info, stock data, sentiment values)]
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
def unpack_to_tensor(data:dict):
    '''
    Returns a TensorFlow Tensor with data formatted from the provided JSON file.\n
    Ensure JSON is the same one created by create_input.py.\n

    Tensor Dimensions: [Number of Batches, Window Size, Number of Features]\n
    Note: The first two indices of the Tensor object are sorted chronologically (i.e. [0,0,x] returns earliest data available, [-1, -1, x] returns latest data available).\n

    Tensor Feature Indices:\n
    0-4: NAICS Company Code\n
    5-7: Company State (1-Hot Vector)\n
    8: Company Market Cap\n
    9: Open Price\n
    10: Close Price\n
    11: High Price\n
    12: Low Price\n
    13: Volume\n
    14: Combined Sentiment Score\n
    '''
    assert data.keys() == input_data.keys(), "Key mismatch in JSON header data! Ensure correct file loaded"
    assert data["blocks"][0].keys() == block.keys(), "Key mismatch in JSON block data!"
    assert data["blocks"][0]["days"][0].keys() == date_data.keys(), "Key mismatch in JSON day data!"

    input_tensor = np.empty([len(data["blocks"]), len(data["blocks"][0]["days"]), NUM_FEATURES])

    company_data = []
    for i in range(COMPANY_TYPE_LEN): company_data.append(data["company_type"][i])
    for i in range(COMPANY_STATE_LEN): company_data.append((data["company_state"] == COMPANY_STATE[i]))
    company_data.append(data["company_marketcap"])
    
    for block_id in range(input_tensor.shape[0]):
        for day_id in range(input_tensor.shape[1]):
            curr_day_stockdata = [data["blocks"][block_id]["days"][day_id]["open"],
                                  data["blocks"][block_id]["days"][day_id]["close"],
                                  data["blocks"][block_id]["days"][day_id]["high"],
                                  data["blocks"][block_id]["days"][day_id]["low"],
                                  data["blocks"][block_id]["days"][day_id]["volume"]]
            for i in range(len(company_data)): input_tensor[block_id, day_id, i] = company_data[i]
            for i in range(len(curr_day_stockdata)): input_tensor[block_id, day_id, i+len(company_data)] = curr_day_stockdata[i]
            input_tensor[block_id, day_id, NUM_FEATURES-1] = data["blocks"][block_id]["days"][day_id]["combined_sentiment"]
 
    return tf.convert_to_tensor(input_tensor)

date_data = {"day_id": -1,
             "date": EMPTY_DATE,
             "open": 0.0,
             "close": 0.0,
             "high": 0.0,  
             "low": 0.0,
             "volume": 0.0,
             "combined_sentiment": 5.0}
block = {"block_id": -1,
         "block_start_date": EMPTY_DATE,
         "block_end_date": EMPTY_DATE,
         "days": []}
input_data = {"company_name": COMPANY_NAME[target_company],
              "company_type": COMPANY_TYPE[target_company],
              "company_state": COMPANY_STATE[target_company],
              "company_marketcap":COMPANY_MARKETCAP[target_company],
              "blocks": []}

# initialize date_data_list with all date_data between start_date and end_date
date_data_list = []
for day in range((end_date - start_date).days + 2):
    date = (start_date + dt.timedelta(days=day))
    tmp = date_data.copy()
    tmp["date"] = date
    date_data_list.append(tmp)

assert len(date_data_list) >= WINDOW_SIZE, "\"date_data_list\" is too small for WINDOW_SIZE! Either increase date range or reduce \"WINDOW_SIZE\"."

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
            cur_sentiment = date_data_list[index]["combined_sentiment"]
            # if multiple articles on same date, take combined sentiment
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

# TO DO: develop method to handle missing stock values (interpolate between dates, add mask to allow network to identify interpolated data)
# create blocks
for block_id in range(len(date_data_list)-WINDOW_SIZE):
    cur_block = deepcopy(block)
    cur_block["block_id"] = block_id
    # TO DO: add start and end dates to block
    for day_id in range(WINDOW_SIZE):
        index = day_id + block_id
        cur_block["days"].append(deepcopy(date_data_list[index]))
        cur_block["days"][day_id]["day_id"] = day_id
    cur_block["block_start_date"] = cur_block["days"][0]["date"]
    cur_block["block_end_date"] = cur_block["days"][-1]["date"]
    input_data["blocks"].append(cur_block)

with open(f"{COMPANY_NAME[target_company]}_data.json", "w", encoding="utf-8") as f:
    json.dump(input_data, f, indent=4, default=str)