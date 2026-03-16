
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the model
model = keras.Sequential([
    layers.Input(shape=(4,)),        # 4 input values
    layers.Dense(16, activation='relu'),
    layers.Dense(7)                  # 7 outputs
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse',                      # typical for regression outputs
    metrics=['mae']
)

model.summary()

# Example training data
import numpy as np
X = np.random.rand(1000, 4)          # 1000 samples, 4 inputs
Y = np.random.rand(1000, 7)          # 7 outputs

# Train
model.fit(X, Y, epochs=20, batch_size=32)

# Example inference
test_input = np.array([[0.1, 0.2, 0.3, 0.4]])
prediction = model.predict(test_input)

print(prediction)