import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from create_input import create_date_data_list
from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_TEST_RATIO = 0.8
WINDOW_SIZE = 14
COMPANY_NAME = ["Tesla", "McDonald's", "Meta"]

LOAD_MODEL = True
MODEL_WEIGHT_FILE = "model.weights.h5"

target_company = 2 # ["Tesla", "McDonald's", "Meta"] 
start_date = datetime(2025, 1, 1)         # start date is inclusive
end_date = datetime(2025, 12, 31)         # end date is inclusive
# visualize = False

# try:
#     with open(f"{COMPANY_NAME[target_company]}_data.json",'r') as f:
#         json_data = json.load(f)
#         input_tensor, mask = unpack_to_tensor(json_data)
# except FileNotFoundError:
#     date_data_list = create_date_data_list(start_date, end_date, target_company)
#     pack(date_data_list, target_company)
#     with open(f"{COMPANY_NAME[target_company]}_data.json",'r') as f:
#         json_data = json.load(f)
#         input_tensor, mask = unpack_to_tensor(json_data)

# if visualize:
#     try:
#         date_data_list
#     except NameError:
#         date_data_list = create_date_data_list(start_date, end_date, target_company)
#     visualize_data(date_data_list, company_name=COMPANY_NAME[target_company], outlook=0)
#     visualize_data(date_data_list, company_name=COMPANY_NAME[target_company], outlook=7)

# WORKING WITH DATE_DATA_LIST FOR NOW

date_data_list = create_date_data_list(start_date, end_date, target_company)
# split data into training and verification set

num_days_missing_data = 0
# prune data missing sentiment or stock data
for i, day_data in enumerate(list(date_data_list)):
    if day_data["nostockdata_flag"] == 1.0 or day_data["nosentimentdata_flag"] == 1.0:
        date_data_list.remove(date_data_list[i-num_days_missing_data])
        num_days_missing_data += 1

# turn date_data_list from list of dicts to list of arrays
date_data_2darray = []
date_stockdata_2darray = []
date_sentiment_array = []
for day_dict in date_data_list:
    day_array = []
    day_stockarray = []
    for feature in day_dict.keys():
        day_array.append(day_dict[feature])
        if not feature in ("day_id", "date", "nostockdata_flag", "nosentimentdata_flag", "combined_sentiment"):
            day_stockarray.append(day_dict[feature])

    date_data_2darray.append(day_array)
    date_stockdata_2darray.append(day_stockarray)
    date_sentiment_array.append(day_dict["combined_sentiment"])
    

date_data_2darray = np.array(date_data_2darray)
date_stockdata_2darray = np.array(date_stockdata_2darray)
date_sentiment_array = np.array(date_sentiment_array)

scaler_stocks = StandardScaler()
date_data_scaled = scaler_stocks.fit_transform(date_stockdata_2darray)
date_data_scaled = np.c_[date_data_scaled, date_sentiment_array]
split_index = int(np.ceil(date_data_scaled.shape[0]*TRAIN_TEST_RATIO))

training_data, testing_data = date_data_scaled[:split_index,:], date_data_scaled[split_index-WINDOW_SIZE:,:]

X_train, Y_train = [], []
for i in range(WINDOW_SIZE, len(training_data)):
    X_train.append(training_data[i-WINDOW_SIZE:i,:])
    Y_train.append(training_data[i,1])

X_test, Y_test = [], []
for i in range(WINDOW_SIZE, len(testing_data)):
    X_test.append(testing_data[i-WINDOW_SIZE:i,:])
    Y_test.append(testing_data[i,1])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_train.shape)
print(X_train[0,:,:])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(7,6)))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1))

#model.summary()
model.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError()])

if LOAD_MODEL:
    print("Loading model weights...")
    model.load_weights(f"{COMPANY_NAME[target_company]}_{MODEL_WEIGHT_FILE}")
else:
    print("Training new model weights")
    training = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
    model.save_weights(f"{COMPANY_NAME[target_company]}_{MODEL_WEIGHT_FILE}", overwrite=False)


predictions = model.predict(X_test)
invert_scale = StandardScaler()
invert_scale.mean_ = scaler_stocks.mean_[1]
invert_scale.scale_ = scaler_stocks.scale_[1]
predictions = invert_scale.inverse_transform(predictions)

print(type(predictions))

train = date_data_2darray[:split_index,:]
test = date_data_2darray[split_index:,:]
test = test.copy()
test = np.c_[test, predictions]

plt.figure(figsize=(12,8))
plt.plot(train[:,1], train[:,4], label="Train (Actual)", color='blue')
plt.plot(test[:,1], test[:,4], label="Test (Actual)", color='red')
plt.plot(test[:,1], test[:,-1], label="Predictions", color='orange')
plt.title(f"Stock Predictions: {COMPANY_NAME[target_company]} from {start_date.date()} to {end_date.date()}")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# determine correct output for 5 business day projection
# linearly interprets between dates for missing stock data

# create model

# compile, train, then save model

# use verifcation set to check model performance