import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
import random
import os
import yfinance as yf
from datetime import datetime, timedelta
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

company_tickers = {
    "Tesla": 'TSLA',
    "McDonald's": 'MCD',
    "Meta": 'META'
}

start_date=datetime(2026,1,4)
end_date=datetime(2026,1,24)

def save_stock_data():
    for name in company_tickers.keys():
        stock_data = yf.download(company_tickers[name], start_date, end_date)
        with open(f"stock_data/{name}.json", "w") as f:
            f.writelines(stock_data.to_json())

if __name__ == "__main__":
    save_stock_data()