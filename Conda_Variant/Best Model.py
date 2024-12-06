from datetime import date, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from matplotlib.dates import DateFormatter
from pandas.tseries.offsets import DateOffset


# Define constants
TODAY = date.today().strftime("%Y-%m-%d")
START = (pd.Timestamp(TODAY) - pd.DateOffset(months=30)).strftime("%Y-%m-%d")

# Define a function to load the dataset
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Keep the Date column
    return data

# Load Apple stock data
ticker = "AAPL"
data = load_data(ticker)

# Split data into train and test sets
validation_period = data['Date'].max() - DateOffset(months=6)
test = data[data['Date'] >= validation_period]
train = data[data['Date'] < validation_period]

# Plot 1: Training and Testing Data Split
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Close'], label='Training Data', color='blue')
plt.plot(test['Date'], test['Close'], label='Validation Data', color='orange')
date_form = DateFormatter("%b %Y")
plt.gca().xaxis.set_major_formatter(date_form)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title(f'Training and Validation Data Split for {ticker} Stocks', size=15)
plt.legend()
plt.grid(True)
plt.show()

# Apply the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_close = train['Close'].values.reshape(-1, 1)
test_close = test['Close'].values.reshape(-1, 1)
data_training_array = scaler.fit_transform(train_close)

# Prepare the training dataset for LSTM
def prepare_lstm_data(data_scaled, window_size=30):
    x, y = [], []
    for i in range(window_size, data_scaled.shape[0]):
        x.append(data_scaled[i - window_size:i])
        y.append(data_scaled[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = prepare_lstm_data(data_training_array)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
# Combine the last 30 days of the training set with the test set for validation
past_30_days = train['Close'].tail(30).values.reshape(-1, 1)
final_test_data = np.concatenate([past_30_days, test_close], axis=0)

# Scale the combined data
input_data = scaler.transform(final_test_data)

# Prepare x_test and y_test with the combined data
x_test, y_test = prepare_lstm_data(input_data)

# Define the LSTM model
model = Sequential()

# First LSTM layer
model.add(
    LSTM(
        units=80,
        activation='tanh',
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)
model.add(Dropout(0.1))

# Second LSTM layer
model.add(
    LSTM(
        units=70,
        return_sequences=False
    )
)
model.add(Dropout(0.2))

# Dense layers
model.add(Dense(1))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.00733257094226877),
    loss='mean_squared_error'
)

# Train the model with explicit validation data
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=30,
    validation_data=(x_test, y_test)
)

# Predict on the validation (test) set
y_pred = model.predict(x_test)

# Rescale predictions and true values back to original values
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae_percentage = (mae / np.mean(y_test)) * 100
r2 = r2_score(y_test, y_pred)



# Plot the results for the validation set
plt.figure(figsize=(12, 6))
plt.plot(test['Date'], y_test, 'b', label="Original Price")
plt.plot(test['Date'], y_pred[-len(test):], 'r--', label="Predicted Price",)
date_form = DateFormatter("%b %Y")
plt.gca().xaxis.set_major_formatter(date_form)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Price Prediction vs Actual (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()


# Create subplots: Training/Validation Loss and R² scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns
fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
# Plot 1: Training and Validation Loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: R² with y=x line
ax2.scatter(y_test, y_pred, alpha=0.6, color='red', label='Predicted vs Actual')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='y=x (Perfect Prediction)')
ax2.set_xlabel('Actual Prices')
ax2.set_ylabel('Predicted Prices')
ax2.set_title(f'Predicted vs Actual Prices (R²={r2:.2f})')
ax2.legend()
ax2.grid(True)
# Print metrics
print("\nMetrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Error (MAE): {mse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mae_percentage:.2f}%")
print(f"R² Score: {r2:.2f}")


# Print training and validation date ranges
print("\nTraining Data Date Range:")
print(f"Start Date: {train['Date'].min()} | End Date: {train['Date'].max()}")
print("\nValidation (Test) Data Date Range:")
print(f"Start Date: {test['Date'].min()} | End Date: {test['Date'].max()}")

def predict_future(model, last_180_days, scaler, days=5):
    """
    Predict the next 'days' number of days using the trained model.
    Excludes weekends (Saturday and Sunday).
    """
    future_predictions = []
    input_sequence = last_180_days.reshape(1, 30, 1)  # Reshape for the model input
    future_dates = []
    current_date = data['Date'].max()  # Start from the last date in the dataset

    while len(future_predictions) < days:
        # Skip weekends
        current_date += timedelta(days=1)
        if current_date.weekday() in [5, 6]:  # Saturday and Sunday
            continue

        # Predict the next day
        next_day_pred = model.predict(input_sequence)[0, 0]
        future_predictions.append(next_day_pred)
        future_dates.append(current_date)

        # Update the input sequence by appending the predicted value and sliding the window
        next_day_scaled = np.append(input_sequence[0, 1:], [[next_day_pred]], axis=0)
        input_sequence = next_day_scaled.reshape(1, 30, 1)

    # Convert predictions back to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).reshape(-1)
    return future_dates, future_predictions


look_back = 30
last_n_days_scaled = scaler.transform(data['Close'].tail(look_back).values.reshape(-1, 1))
future_dates, future_7_days = predict_future(model, last_n_days_scaled, scaler,days= 5)
plt.figure(figsize=(12, 6))
last_month_data = data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=1)]
plt.plot(last_month_data['Date'], last_month_data['Close'], label="Historical Prices (Last Month)", color='blue')
plt.plot(future_dates, future_7_days, label="Future Predictions (7 Days)", color='red', marker='o')
all_dates = pd.to_datetime(list(last_month_data['Date']) + future_dates)
mondays = [date for date in all_dates if date.weekday() == 0]  # Filter only Mondays
plt.xticks(ticks=mondays, labels=[d.strftime("%b %d") for d in mondays])

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('5-Day Future Stock Price Prediction (Weekdays Only)')
plt.legend()
plt.grid(True)
plt.show()

# Print the future predicted prices
print("7-Day Predictions (Weekdays Only):")
for date, price in zip(future_dates, future_7_days):
    print(f"Predicted Price on {date.strftime('%Y-%m-%d')}: ${price:.2f}")

