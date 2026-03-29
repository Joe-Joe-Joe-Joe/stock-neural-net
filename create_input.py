import pandas as pd
import json
import os
import datetime as dt
import re
from copy import deepcopy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_SENTIMENT_DIR = os.path.join(SCRIPT_DIR, "scraped_news_sentiment\\")
STOCK_DIR = os.path.join(SCRIPT_DIR, "stock_data\\")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "stacked_data\\")
COMPANY_NAME = ["Tesla", "McDonald's", "Meta"]      # update create_date_data_list docstring if this list changes
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
NUM_FLAGS = 2
NUM_FEATURES = NUM_COMPANY_FEATURES + NUM_STOCK_FEATURES + NUM_SENTIMENT_FEATURES + NUM_FLAGS
EMPTY_DATE = dt.datetime(1900,1,1)
EMPTY_SENTIMENT = -1.0

# ensure flags at end of date_data, visualize() relies on flags being at end of dict
date_data = {"day_id": -1,
             "date": EMPTY_DATE,
             "open": 0.0,
             "close": 0.0,
             "high": 0.0,  
             "low": 0.0,
             "volume": 0.0,
             "combined_sentiment": EMPTY_SENTIMENT,
             "nostockdata_flag": 0.0,
             "nosentimentdata_flag": 0.0}
block = {"block_id": -1,
         "block_start_date": EMPTY_DATE,
         "block_end_date": EMPTY_DATE,
         "days": []}
input_data = {"company_name": "",
              "company_type": [],
              "company_state": "",
              "company_marketcap": 0,
              "blocks": []}

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
    15: No Stock Data Available Flag
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
            curr_day_data = [data["blocks"][block_id]["days"][day_id]["open"],
                             data["blocks"][block_id]["days"][day_id]["close"],
                             data["blocks"][block_id]["days"][day_id]["high"],
                             data["blocks"][block_id]["days"][day_id]["low"],
                             data["blocks"][block_id]["days"][day_id]["volume"],
                             data["blocks"][block_id]["days"][day_id]["combined_sentiment"],
                             data["blocks"][block_id]["days"][day_id]["nostockdata_flag"],
                             data["blocks"][block_id]["days"][day_id]["nosentimentdata_flag"]]
            for i in range(len(company_data)): input_tensor[block_id, day_id, i] = company_data[i]
            for i in range(len(curr_day_data)): input_tensor[block_id, day_id, i+len(company_data)] = curr_day_data[i]
 
    return tf.convert_to_tensor(input_tensor)

def visualize_data(date_data_list:list):
    '''
    Takes in date_data_list and creates an array of I/O plots across all features.
    '''
    num_rows = NUM_FEATURES-NUM_FLAGS
    num_cols = NUM_FEATURES-NUM_FLAGS

    fig, axs = plt.subplots(3,3, figsize=(10, 10))

    X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

    for i in range(num_rows):
        for j in range(num_cols):
            pc = axs[i,j].scatter(X,Y,c=Z, cmap='plasma')

    plt.show()
    


def create_date_data_list(start_date:dt.datetime, end_date:dt.datetime, target_company:int):
    f'''
    Creates a date_data_list that includes all data for the target company between the start and end date provided (inclusive).\n
    For target company, use index of company in following list: ["Tesla", "McDonald's", "Meta"]
    '''
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
                if not (publish_date <= end_date and publish_date >= start_date):
                    print(f"Article publish date is out of range ({publish_date})")
                    continue

                index = (publish_date - start_date).days
                cur_sentiment = date_data_list[index]["combined_sentiment"]
                # if multiple articles on same date, take combined sentiment
                if cur_sentiment == EMPTY_SENTIMENT:
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

    for day in date_data_list:
        if day["close"] == 0:
            day["nostockdata_flag"] = 1.0
        if day["combined_sentiment"] == EMPTY_SENTIMENT:
            day["nosentimentdata_flag"] = 1.0
    return date_data_list

def pack(date_data_list:list, target_company:int):
    company_data = deepcopy(input_data)
    company_data["company_name"] = COMPANY_NAME[target_company]
    company_data["company_type"] = COMPANY_TYPE[target_company]
    company_data["company_state"] = COMPANY_STATE[target_company]
    company_data["company_marketcap"] = COMPANY_MARKETCAP[target_company]

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
        company_data["blocks"].append(cur_block)

    with open(f"{COMPANY_NAME[target_company]}_data.json", "w", encoding="utf-8") as f:
        json.dump(company_data, f, indent=4, default=str)

if __name__ == "__main__":
    target_company = 2
    start_date = dt.datetime(2025, 1, 4)        # start date is inclusive
    end_date = dt.datetime(2025, 1, 31)         # end date is inclusive

    try:
        with open(f"{COMPANY_NAME[target_company]}_data.json",'r') as f:
            json_data = json.load(f)
            input_tensor = unpack_to_tensor(json_data)
    except FileNotFoundError:
        date_data_list = create_date_data_list(start_date, end_date, target_company)
        pack(date_data_list, target_company)
        with open(f"{COMPANY_NAME[target_company]}_data.json",'r') as f:
            json_data = json.load(f)
            input_tensor = unpack_to_tensor(json_data)

    print(input_tensor[0,0,NUM_FEATURES-2])
    print(input_tensor[0,0,NUM_FEATURES-1])