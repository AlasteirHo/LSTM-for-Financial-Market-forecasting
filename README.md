# LSTM for Stock Price Prediction

This repository contains the implementation of a Long Short-Term Memory (LSTM) neural network for predicting the stock prices of Apple Inc. (AAPL). The project demonstrates the effectiveness of LSTM in modeling temporal dependencies in financial time-series data.
**DISCLAIMER: The predictions made by this algorithm should not be taken as financial advice. I am not responsible for any losses caused as a result of using this model.**
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
- **Compute time on NVIDIA RTX 3070ti:**
   a) Bayesian_Optimization = 5 Minutes 21 seconds
   b) Best_model = 20 seconds
## Installation (Use Google Colab if you are unable to run on a dedicated GPU)

1. Clone the repository:
   ```bash
   git clone https://github.com/AlasteirHo/LSTM-for-Financial-Market-forecasting.git
   cd LSTM-for-Financial-Market-forecasting
   ```
2. Install dependencies:
   ```bash
   pip install pandas matplotlib tensorflow yfinance numpy scikit-learn keras-tuner
   ```
3. Optional (Run the code on a conda enviornment utilizing CUDA accleration, NVIDIA GPUs only)*
   
```bash
conda create -n lstm_env python=3.10
conda activate lstm_env
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
pip install pandas matplotlib yfinance numpy scikit-learn keras-tuner
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
*Ensure you have 8GB of VRAM available and Anaconda/Miniconda installed
## Usage

1. Download the dataset using the `yfinance` library.
2. Preprocess the data using the provided scripts.
3. Train the LSTM model.
4. Evaluate the model and generate predictions.
5. Predict 5 trading days into the future

## Future Work

- Explore hybrid CNN-LSTM architectures for improved performance.
- Integrate external features like trading volume, sentiment analysis, and economic indicators.
- Incorporate attention mechanisms for better adaptability to market volatility.


