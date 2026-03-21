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

MODEL_FILE = "model.keras"
company_tickers = {
    "Tesla": 'TSLA',
    "McDonald's": 'MCD',
    "Meta": 'META'
}

target_company = "Meta"

start_date=datetime(2025,1,8)
end_date=datetime(2026,1,7)
#end_date=datetime(2025,1,15)

def save_stock_data():
    for name in company_tickers.keys():
        stock_data = yf.download(company_tickers[name], start_date, end_date)
        with open(f"stock_data/{name}.json", "w") as f:
            f.writelines(stock_data.to_json())

def load_stock_data(company):
    with open(f"stock_data/{company}.json", "r") as f:
        data = pd.read_json(f)
    return data

def import_news_json():
    pass
# for each json file in test stock (Meta), turn it into a dataframe row, sort by date
# for each week, select dataframe rows from that date in order
# assign each sentiment to the corresponding column, up to 5
#import the 7 day stock data for the period prior to start
# on each loop, import the current week of stock prices as the training output
#ith row of x is input values, ith row of Y is output stock values
# reassign current week of stock prices as old prices and make them inputs

#functions: get 7 days of stock price starting on (date)

# want final data to have 7 input values for historical stock price, 5 for news sentiment scores, 7 outputs for next 7 days

def generate_xy(stock):
    stock_data = load_stock_data(target_company)

# ----------------------------
# Deterministic reproducibility
# ----------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------
# Model definition
# ----------------------------
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(7)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ----------------------------
# Training
# ----------------------------
def train_model(seed=42):
    set_seed(seed)

    model = build_model()
    #preprocesses

    # Example synthetic dataset
    X = np.random.rand(1000, 4)
    Y = np.random.rand(1000, 7)

    model.fit(X, Y, epochs=20, batch_size=32)

    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


# ----------------------------
# Load model
# ----------------------------
def load_model():
    return keras.models.load_model(MODEL_FILE)


# ----------------------------
# Inference
# ----------------------------
def predict(model, inputs):
    vec = np.array([inputs])
    out = model.predict(vec, verbose=0)
    return out[0]


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command")

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--seed", type=int, default=42)

    pred_cmd = sub.add_parser("predict")
    pred_cmd.add_argument("x1", type=float)
    pred_cmd.add_argument("x2", type=float)
    pred_cmd.add_argument("x3", type=float)
    pred_cmd.add_argument("x4", type=float)

    args = parser.parse_args()

    if args.command == "train":
        train_model(seed=args.seed)

    elif args.command == "predict":
        model = load_model()
        inputs = [args.x1, args.x2, args.x3, args.x4]
        result = predict(model, inputs)

        print("Input:", inputs)
        print("Output:", result)

    else:
        parser.print_help()


if __name__ == "__main__":
    pass
   # main()
   #load_stock_data()
