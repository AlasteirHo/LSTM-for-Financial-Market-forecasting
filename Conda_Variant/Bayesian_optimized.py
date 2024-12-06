from datetime import date, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.dates import DateFormatter
from keras_tuner import BayesianOptimization
from pandas.tseries.offsets import DateOffset


# Define constants
TODAY = date.today().strftime("%Y-%m-%d")
START = (pd.Timestamp(TODAY) - pd.DateOffset(months=36)).strftime("%Y-%m-%d")


# Function to load the dataset
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# Load Apple stock data
ticker = "AAPL"
data = load_data(ticker)

# Split data into train and test sets
validation_period = data['Date'].max() - DateOffset(months=6)
test = data[data['Date'] >= validation_period]
train = data[data['Date'] < validation_period]

# Apply MinMaxScaler
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

# Combine the last 30 days of the training set with the test set for validation
past_n_days = train['Close'].tail(30).values.reshape(-1, 1)
final_test_data = np.concatenate([past_n_days, test_close], axis=0)

# Scale the combined data
input_data = scaler.transform(final_test_data)

# Prepare x_test and y_test
x_test, y_test = prepare_lstm_data(input_data)


# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()

    # First LSTM layer
    model.add(
        LSTM(
            units=hp.Int('units_1', min_value=20, max_value=100, step=10),
            activation='tanh',
            return_sequences=True,
            input_shape=(x_train.shape[1], 1)
        )
    )
    model.add(Dropout(hp.Float('dropout_1', min_value=0, max_value=0.5, step=0.1)))

    # Second LSTM layer
    model.add(
        LSTM(
            units=hp.Int('units_2', min_value=50, max_value=100, step=10),
            return_sequences=False
        )
    )
    model.add(Dropout(hp.Float('dropout_2', min_value=0, max_value=0.5, step=0.1)))
    # Dense layers
    model.add(Dense(units=hp.Int('dense_units', min_value=0, max_value=50, step=10)))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(
        optimizer=Nadam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error'
    )
    return model


# Set up the Keras Tuner with Bayesian Optimization
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='Models',
    project_name='AAPL_1'
)

# Run the tuner to search for the best hyperparameters
tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    batch_size=32,
    # callbacks=[early_stopping]
)

# Predict the test data
y_pred = model.predict(x_test)

# Rescale back to original values
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mae_percentage = (mse / np.mean(y_test)) * 100
r2 = r2_score(y_test, y_pred)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(test['Date'], y_test, 'b', label="Original Price")
plt.plot(test['Date'], y_pred[-len(test):], 'r', label="Predicted Price")
date_form = DateFormatter("%b %Y")
plt.gca().xaxis.set_major_formatter(date_form)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title(f'Price Prediction vs Actual (Last 6 Months) for {ticker}')
plt.legend()
plt.grid(True)
plt.show()

# Training and Validation Split Plot
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Close'], label='Training Data', color='blue')
plt.plot(test['Date'], test['Close'], label='Validation Data', color='orange')
plt.gca().xaxis.set_major_formatter(date_form)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title(f'Training and Validation Data Split for {ticker} stocks')
plt.legend()
plt.grid(True)
plt.show()

# Training/Validation Loss and R² scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Training and Validation Loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: R² Scatter Plot
ax2.scatter(y_test, y_pred, alpha=0.6, color='red', label='Predicted vs Actual')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='y=x (Perfect Prediction)')
ax2.set_xlabel('Actual Prices')
ax2.set_ylabel('Predicted Prices')
ax2.set_title(f'Predicted vs Actual Prices (R²={r2:.2f})')
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Print metrics and hyperparameters
print("MAPE: {:.2f}%".format(mae_percentage))
print("MAE: {:.2f}%".format(mae))
print("R² score on test set:", r2)
print("MSE: ", mse)

print("\nBest Hyperparameters Found:")
print(f"First LSTM Units: {best_hps.get('units_1')}")
print(f"First Dropout Rate: {best_hps.get('dropout_1')}")
print(f"Second LSTM Units: {best_hps.get('units_2')}")
print(f"Second Dropout Rate: {best_hps.get('dropout_2')}")
print(f"Dense Units: {best_hps.get('dense_units')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}\n")

def predict_future(model, last_n_days, scaler, days=5):
    """
    Predict the next 'days' number of days using the trained model.
    Excludes weekends (Saturday and Sunday).
    """
    future_predictions = []
    input_sequence = last_n_days.reshape(1, 30, 1)  # Reshape for the model input
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