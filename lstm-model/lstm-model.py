#!/usr/bin/env python3
#
# Price prediction for stocks using LSTM model
# Main steps:
# 1) Get stock data
# 2) Preprocess data for LSTM model
# 3) Separate the data into training and testing sets
# 4) Build the LSTM model
# 5) Apply the model to make predictions
import yfinance as yf
import pandas as pd

# Example: Get historical stock data for a company (e.g., Apple)
ticker = 'AAPL'  # Example ticker
stock_data = yf.download(ticker, start="2010-01-01", end="2024-11-07")

# Display the first few rows of the data
print(stock_data.head())

# Use the 'Close' price column for predictions
data = stock_data[['Close']]

# Normalize the data (optional but recommended for LSTM models)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the sequence length (how many past days to look at for prediction)
sequence_length = 60

def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])  # X: past 'sequence_length' days
        y.append(data[i, 0])  # y: next day's closing price
    return np.array(X), np.array(y)

# Create dataset for training
import numpy as np
X, y = create_dataset(scaled_data, sequence_length)

# Reshape X to be suitable for LSTM input (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(X.shape, y.shape)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # To prevent overfitting

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))  # Predicting the next closing price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predicting the stock price
y_pred = model.predict(X_test)

# Invert scaling to get actual stock prices
y_pred = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the results
import matplotlib.pyplot as plt

# Get the dates for the test data
dates = stock_data.index[-len(y_test_rescaled):]  # Use the last 'n' dates corresponding to y_test

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(dates, y_test_rescaled, color='blue', label='Actual Stock Price')
plt.plot(dates, y_pred, color='red', label='Predicted Stock Price')

# Set labels and title
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)  # Rotate the date labels for better visibility

# Add legend
plt.legend()

# Show the plot
plt.savefig('lstm-model.png')
plt.tight_layout()  # Adjust layout to prevent label cut-off

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Compute MAE, MSE, RMSE, R-squared
mae = mean_absolute_error(y_test_rescaled, y_pred)
mse = mean_squared_error(y_test_rescaled, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared: {r2}')