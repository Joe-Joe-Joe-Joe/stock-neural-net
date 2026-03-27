import argparse
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
STOCK_DIR = BASE_DIR / "stock_data"
MODEL_FILE = BASE_DIR / "model.keras"
X_SCALER_FILE = BASE_DIR / "x_scaler.pkl"
Y_SCALER_FILE = BASE_DIR / "y_scaler.pkl"
SENTIMENT_FILE = BASE_DIR / "sentiment_condensed.json"

# ------------------------------------------------------------
# GLOBAL VARIABLE: choose the stock here
# Options: "Tesla", "Meta", "McDonald's"
# ------------------------------------------------------------
target_company = "McDonald's"

company_tickers = {
    "Tesla": "TSLA",
    "McDonald's": "MCD",
    "Meta": "META",
}

start_date = datetime(2025, 1, 8)
end_date = datetime(2025, 12, 30)

WINDOW_DAYS = 5
TARGET_DAYS = 5
ARTICLES = 5

# ------------------------------------------------------------
# New 3-week January 2026 prediction data files
# ------------------------------------------------------------
PREDICTION_STOCK_FILES = {
    "Tesla": BASE_DIR / "SentimentData_Jan4_to_Jan24" / "Tesla_3weeks.json",
    "Meta": BASE_DIR / "SentimentData_Jan4_to_Jan24" / "Meta_3weeks.json",
    "McDonald's": BASE_DIR / "SentimentData_Jan4_to_Jan24" / "McDonald's_3weeks.json",
}

PREDICTION_NEWS_FILES = {
    "Tesla": [
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Tesla_2026_01_04.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Tesla_2026_01_11.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Tesla_2026_01_18.json",
    ],
    "Meta": [
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Meta_2026_01_04.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Meta_2026_01_11.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_Meta_2026_01_18.json",
    ],
    "McDonald's": [
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_McDonald's_2026_01_04.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_McDonald's_2026_01_11.json",
        BASE_DIR / "SentimentData_Jan4_to_Jan24" / "news_McDonald's_2026_01_18.json",
    ],
}


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_stock_dir():
    STOCK_DIR.mkdir(parents=True, exist_ok=True)


def stock_file(company):
    return STOCK_DIR / f"{company}.json"


def save_stock_data():
    ensure_stock_dir()
    for company, ticker in company_tickers.items():
        print(f"Downloading {company} ({ticker})...")
        stock_data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )

        if stock_data.empty:
            raise ValueError(f"No stock data downloaded for {company}")

        stock_data.to_json(stock_file(company))
        print(f"Saved to {stock_file(company)}")


def load_stock_data(company):
    f = stock_file(company)
    if not f.exists():
        print(f"{f} missing, downloading stock data...")
        save_stock_data()

    df = pd.read_json(f)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def import_news_json(company):
    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(f"Missing {SENTIMENT_FILE}")

    data = pd.read_json(SENTIMENT_FILE)

    filtered = data[data["company"] == company].copy()
    filtered["date"] = pd.to_datetime(filtered["date"], format="%Y%m%d", errors="coerce")
    filtered = filtered.dropna(subset=["date"]).sort_values("date")
    return filtered


def generate_xy(company):
    stock_data = load_stock_data(company)
    news_sentiment = import_news_json(company)

    ticker = company_tickers[company]
    colname = f"('Close', '{ticker}')"

    if colname not in stock_data.columns:
        raise KeyError(
            f"Could not find column {colname} in stock data.\n"
            f"Available columns: {list(stock_data.columns)}"
        )

    stock_prices = stock_data[colname].dropna()

    X_rows = []
    Y_rows = []

    week_starting = start_date

    while week_starting < (end_date - timedelta(days=14)):
        this_week_end = week_starting + timedelta(days=7)
        next_week_end = week_starting + timedelta(days=14)

        this_week_stocks = stock_prices.loc[week_starting:this_week_end].iloc[:WINDOW_DAYS].values
        next_week_stocks = stock_prices.loc[this_week_end:next_week_end].iloc[:TARGET_DAYS].values

        sentiments = news_sentiment[
            (news_sentiment["date"] >= week_starting) &
            (news_sentiment["date"] < this_week_end)
        ]["combined_sentiment"].values[:ARTICLES]

        if len(sentiments) < ARTICLES:
            sentiments = np.pad(sentiments, (0, ARTICLES - len(sentiments)))

        if len(this_week_stocks) < WINDOW_DAYS or len(next_week_stocks) < TARGET_DAYS:
            week_starting += timedelta(days=7)
            continue

        X_rows.append(np.concatenate([this_week_stocks, sentiments]))
        Y_rows.append(next_week_stocks)

        week_starting += timedelta(days=7)

    if len(X_rows) == 0:
        raise ValueError("No training samples were created.")

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    return X, Y


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(5)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_model(seed=42, epochs=100):
    set_seed(seed)

    X, Y = generate_xy(target_company)

    X_train, X_eval, Y_train, Y_eval = train_test_split(
        X, Y, test_size=0.2, random_state=seed, shuffle=True
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_eval_scaled = x_scaler.transform(X_eval)

    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_eval_scaled = y_scaler.transform(Y_eval)

    model = build_model()

    history = model.fit(
        X_train_scaled,
        Y_train_scaled,
        validation_data=(X_eval_scaled, Y_eval_scaled),
        epochs=epochs,
        batch_size=8,
        verbose=1
    )

    model.save(MODEL_FILE)
    joblib.dump(x_scaler, X_SCALER_FILE)
    joblib.dump(y_scaler, Y_SCALER_FILE)

    print(f"Saved model to {MODEL_FILE}")

    eval_loss, eval_mae = model.evaluate(X_eval_scaled, Y_eval_scaled, verbose=0)
    print(f"Eval loss: {eval_loss:.6f}")
    print(f"Eval MAE : {eval_mae:.6f}")

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Loss")
    plt.title(f"Training vs Validation Loss for {target_company}")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_trained_objects():
    model = keras.models.load_model(MODEL_FILE)
    x_scaler = joblib.load(X_SCALER_FILE)
    y_scaler = joblib.load(Y_SCALER_FILE)
    return model, x_scaler, y_scaler


# ============================================================
# NEW HELPERS FOR JAN 2026 3-WEEK PREDICTION DATA
# ============================================================
def load_prediction_stock_data(company):
    file_path = PREDICTION_STOCK_FILES[company]

    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction stock file: {file_path}")

    df = pd.read_json(file_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_prediction_news_week(company, week_start_str):
    file_path = BASE_DIR / "SentimentData_Jan4_to_Jan24" / f"news_{company}_{week_start_str}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction news file: {file_path}")

    import json

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Expected structure:
    # {
    #   "2026_01_11": [ {...}, {...}, ... ]
    # }
    if len(raw) != 1:
        raise ValueError(f"Unexpected news JSON structure in {file_path}")

    only_key = next(iter(raw.keys()))
    items = raw[only_key]

    if not isinstance(items, list):
        raise ValueError(f"Unexpected news JSON item format in {file_path}")

    df = pd.DataFrame(items)

    if df.empty:
        return df

    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    return df

def get_week_sentiments(company, week_start_str):
    df = load_prediction_news_week(company, week_start_str)

    if df.empty or "combined_sentiment" not in df.columns:
        sentiments = np.zeros(ARTICLES, dtype=np.float32)
    else:
        sentiments = df["combined_sentiment"].dropna().values[:ARTICLES]
        if len(sentiments) < ARTICLES:
            sentiments = np.pad(sentiments, (0, ARTICLES - len(sentiments)))

    return sentiments.astype(np.float32)

def get_backtest_input_and_target(company):
    stock_data = load_prediction_stock_data(company)
    ticker = company_tickers[company]
    colname = f"('Close', '{ticker}')"

    if colname not in stock_data.columns:
        raise KeyError(
            f"Could not find column {colname} in 3-week stock data.\n"
            f"Available columns: {list(stock_data.columns)}"
        )

    stock_prices = stock_data[colname].dropna().sort_index()

    # Only use the first 2 calendar weeks
    # Week 1: Jan 5–Jan 9
    # Week 2: Jan 12–Jan 16
    week1 = stock_prices.loc["2026-01-05":"2026-01-09"]
    week2 = stock_prices.loc["2026-01-12":"2026-01-16"]

    if len(week1) == 0:
        raise ValueError("Week 1 has no stock data.")
    if len(week2) < 2:
        raise ValueError("Week 2 must have at least 2 trading days.")

    # Model expects 5 stock inputs
    week1_closes = week1.values.astype(np.float32)
    if len(week1_closes) < 5:
        week1_closes = np.pad(week1_closes, (0, 5 - len(week1_closes)), mode="edge")
    else:
        week1_closes = week1_closes[:5]

    # Use week 1 sentiments to predict week 2
    week1_sentiments = get_week_sentiments(company, "2026_01_04")

    model_input = np.concatenate([week1_closes, week1_sentiments]).astype(np.float32)

    actual_week2 = week2.values.astype(np.float32)
    actual_first_day_week2 = actual_week2[0]
    actual_remaining_week2 = actual_week2[1:]

    return {
        "all_prices": pd.concat([week1, week2]),
        "week1": week1,
        "week2": week2,
        "model_input": model_input,
        "actual_week2": actual_week2,
        "actual_first_day_week2": actual_first_day_week2,
        "actual_remaining_week2": actual_remaining_week2,
    }


def get_last_week_sentiments(company):
    # Use week starting Jan 18, 2026 as the last week for prediction input
    df = load_prediction_news_week(company, "2026_01_18")

    if df.empty or "combined_sentiment" not in df.columns:
        sentiments = np.zeros(ARTICLES, dtype=np.float32)
    else:
        sentiments = df["combined_sentiment"].dropna().values[:ARTICLES]
        if len(sentiments) < ARTICLES:
            sentiments = np.pad(sentiments, (0, ARTICLES - len(sentiments)))

    return sentiments.astype(np.float32)


def get_prediction_input_from_new_data(company):
    stock_data = load_prediction_stock_data(company)
    ticker = company_tickers[company]
    colname = f"('Close', '{ticker}')"

    if colname not in stock_data.columns:
        raise KeyError(
            f"Could not find column {colname} in 3-week stock data.\n"
            f"Available columns: {list(stock_data.columns)}"
        )

    stock_prices = stock_data[colname].dropna()

    if len(stock_prices) < 5:
        raise ValueError("Not enough stock data to extract last week's 5 closing prices.")

    last_week_closes = stock_prices.iloc[-5:].values.astype(np.float32)
    last_week_sentiments = get_last_week_sentiments(company)

    model_input = np.concatenate([last_week_closes, last_week_sentiments]).astype(np.float32)
    return model_input, stock_prices


def predict(company=target_company, show_plot=True):
    model, x_scaler, y_scaler = load_trained_objects()

    data = get_backtest_input_and_target(company)
    inputs = data["model_input"]
    all_prices = data["all_prices"]
    week1 = data["week1"]
    week2 = data["week2"]
    actual_week2 = data["actual_week2"]
    actual_first_day_week2 = data["actual_first_day_week2"]
    actual_remaining_week2 = data["actual_remaining_week2"]

    arr = np.array([inputs], dtype=np.float32)
    arr_scaled = x_scaler.transform(arr)
    pred_scaled = model.predict(arr_scaled, verbose=0)
    pred = y_scaler.inverse_transform(pred_scaled)[0]

    # Use actual day 1 of week 2, then predicted days 2-5
    predicted_remaining_week2 = pred[1:1 + len(actual_remaining_week2)]

    # Combined overlay series: actual first day + predicted remaining days
    combined_week2_overlay = np.concatenate((
        np.array([actual_first_day_week2], dtype=np.float32),
        predicted_remaining_week2
    ))

    print(f"\nCompany: {company}")
    print("Week 1 stock closes used as input :", np.round(inputs[:5], 2))
    print("Week 1 sentiments used as input   :", np.round(inputs[5:], 3))
    print("Full predicted week 2 closes      :", np.round(pred, 2))
    print("Actual week 2 closes              :", np.round(actual_week2, 2))
    print("Using actual first day of week 2  :", np.round(actual_first_day_week2, 2))
    print("Predicted remaining week 2 closes :", np.round(predicted_remaining_week2, 2))

    mae_remaining = np.mean(np.abs(predicted_remaining_week2 - actual_remaining_week2))
    print(f"Week 2 remaining-days MAE         : {mae_remaining:.4f}")

    if show_plot:
        plt.figure(figsize=(11, 5))

        # Plot actual prices for weeks 1-2
        plt.plot(
            all_prices.index,
            all_prices.values,
            marker="o",
            label="Actual Prices (Weeks 1–2)"
        )

        # Highlight week 1 input
        plt.plot(
            week1.index,
            week1.values,
            marker="o",
            linewidth=3,
            label="Actual Week 1 (Input)"
        )

        # Actual week 2
        plt.plot(
            week2.index,
            actual_week2,
            marker="o",
            linewidth=3,
            label="Actual Week 2"
        )

        # Overlay: actual day 1 + predicted remaining days
        plt.plot(
            week2.index[:len(combined_week2_overlay)],
            combined_week2_overlay,
            marker="x",
            linestyle="--",
            linewidth=2,
            label="Week 2 Overlay (Actual Day 1 + Predicted Days 2–5)"
        )

        # Optional: show only the predicted part more explicitly
        plt.plot(
            week2.index[1:1 + len(predicted_remaining_week2)],
            predicted_remaining_week2,
            marker="x",
            linestyle=":",
            linewidth=2,
            label="Predicted Week 2 Days 2–5"
        )

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{company}: Actual Weeks 1–2 with Week 2 Day 1 Fixed")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return pred

def main():
    parser = argparse.ArgumentParser(description="Stock sentiment neural network")
    sub = parser.add_subparsers(dest="command")

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--epochs", type=int, default=100)

    sub.add_parser("download")
    sub.add_parser("predict")

    args = parser.parse_args()

    if args.command == "download":
        save_stock_data()
    elif args.command == "train":
        train_model(seed=args.seed, epochs=args.epochs)
    elif args.command == "predict":
        result = predict(company=target_company, show_plot=True)
        print("Predicted week 2 closes:", np.round(result, 2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()