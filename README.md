# LSTM for Stock Price Prediction

This repository contains the implementation of a Long Short-Term Memory (LSTM) neural network for predicting the stock prices of Apple Inc. (AAPL). The project demonstrates the effectiveness of LSTM in modeling temporal dependencies in financial time-series data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Stock price prediction is critical for developing investment strategies and managing risks. Traditional regression models fall short in capturing the non-linear and sequential dependencies inherent in stock price data. This project employs LSTM neural networks to overcome these limitations and achieve robust prediction accuracy.

Key results:
- **R² Score**: 0.92
- **Mean Squared Error (MSE)**: 9.89
- **Mean Absolute Percentage Error (MAPE)**: 1.02%

## Features

- Predicts stock prices using historical closing prices.
- Fine-tuned hyperparameters using Bayesian optimization.
- Forecasts up to 5 trading days into the future.
- Provides insights into the benefits and limitations of LSTM models in financial forecasting.

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- pandas
- Matplotlib
- yfinance

## Dataset

The dataset comprises 4 years of daily closing prices for Apple Inc. (AAPL), obtained using the `yfinance` library. The data was preprocessed by:
- Normalizing values using `MinMaxScaler`.
- Splitting into training and validation sets (80:20 ratio).
- Using a sliding-window approach for sequence preparation.

## Model Architecture

The model consists of:
- Two LSTM layers with dropout for regularization.
- Dense layers for output predictions.
- Optimized using the Adam optimizer with Mean Squared Error (MSE) as the loss function.

Hyperparameters were fine-tuned using Bayesian optimization, resulting in improved model performance.

## Performance

The model achieved high accuracy and effectively captured trends:
- **Validation R² Score**: 0.92
- **Validation MAPE**: 1.02%
- Predicted vs. actual values closely aligned, showcasing minimal overfitting.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LSTM-for-stock-prediction.git
   cd LSTM-for-stock-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset using the `yfinance` library.
2. Preprocess the data using the provided scripts.
3. Train the LSTM model:
   ```bash
   python train_model.py
   ```
4. Evaluate the model and generate predictions:
   ```bash
   python evaluate_model.py
   ```

## Future Work

- Explore hybrid CNN-LSTM architectures for improved performance.
- Integrate external features like trading volume, sentiment analysis, and economic indicators.
- Incorporate attention mechanisms for better adaptability to market volatility.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
