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

MODEL_FILE = "model.keras"
company_tickers = {
    "Tesla": 'TSLA',
    "McDonald's": 'MCD',
    "Meta": 'META'
}

target_company = "Meta"

start_date=datetime(2025,1,8)
#end_date=datetime(2026,1,7)
end_date=datetime(2025,12,30)

def save_stock_data():
    for name in company_tickers.keys():
        stock_data = yf.download(company_tickers[name], start_date, end_date)
        with open(f"stock_data/{name}.json", "w") as f:
            f.writelines(stock_data.to_json())

def load_stock_data(company):
    with open(f"stock_data/{company}.json", "r") as f:
        data = pd.read_json(f)
    return data

def import_news_json(company):
    with open("sentiment_condensed.json", encoding="utf-8") as f:
        data = pd.read_json(f)
        filtered = data[data["company"] == company]
    return filtered

def generate_xy(stock):
    X = pd.DataFrame()
    Y = pd.DataFrame()
    # determines number of sentiments to pull in with stock data
    articles = 5
    stock_data = load_stock_data(stock)
    colname = f"('Close', '{company_tickers[stock]}')"
    news_sentiment = import_news_json(stock)
    news_sentiment["date"] = pd.to_datetime(news_sentiment["date"], format="%Y%m%d")
    week_starting = start_date
    while week_starting < (end_date-timedelta(days=7)):
        stock_prices = stock_data.loc[week_starting:week_starting + timedelta(days=7)]
        sentiments = news_sentiment[(news_sentiment["date"] >= week_starting) & (news_sentiment["date"] < week_starting + timedelta(days=7))]

        # select 5 days of stock prices for features
        this_week_stocks = stock_prices.iloc[:5][colname].values  # shape (5,)

        # select up to 5 sentiment values within the week
        sentiments_window = sentiments["combined_sentiment"].values[:articles]  # shape (<=5,)

        # if fewer than 5 sentiments, pad with zeros
        if len(sentiments_window) < articles:
            sentiments_window = np.pad(sentiments_window, (0, articles - len(sentiments_window)))

        # combine features: 5 stock prices + 5 sentiments -> shape (10,)
        features = np.concatenate([this_week_stocks, sentiments_window])

        # append to X and Y
        X = pd.concat([X, pd.DataFrame([features])], ignore_index=True)


        stock_prices_nxt_week = stock_data.loc[week_starting+timedelta(days=7):week_starting + timedelta(days=14)]
        next_week =  stock_prices_nxt_week.iloc[:5][colname].values  # shape (5,)
        Y = pd.concat([Y, pd.DataFrame([next_week])], ignore_index=True)

        # move to next week
        week_starting += timedelta(days=7)
    return X, Y

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
# Model input is [stock price_1 ... stock price_5, news_sentiment_1, .... news_sentiment_5]
# output is [stock price_6 ... stock price_10]
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(5)
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

def train_model(seed=42, test_size=0.2):
    set_seed(seed)

    model = build_model()

    X, Y = generate_xy(target_company)

    # Split into train and evaluation sets
    X_train, X_eval, Y_train, Y_eval = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=True
    )

    # Train on training set, validate on evaluation set
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_eval, Y_eval),
        epochs=20,
        batch_size=32
    )

    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # Show validation results
    print("Validation results per epoch:")
    for epoch, val_loss in enumerate(history.history['val_loss'], start=1):
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

    # Optionally, plot training vs validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss for {target_company}')
    plt.legend()
    plt.show()


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
    main()