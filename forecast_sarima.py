import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


def download_prices(ticker, start, end):
    # Monthly average closing prices
    price = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if price.empty or "Close" not in price.columns:
        raise ValueError(f"Failed to retrieve data for {ticker}")
    return price["Close"].resample("M").mean()


def evaluate(actual, predicted):
    return {
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted)
    }


def run_sarima(series, steps=12):
    # Train SARIMA and forecast
    series = series.dropna()
    train = series[:-steps]
    test = series[-steps:]

    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=steps)

    return preds, test, evaluate(test, preds), fit


def plot(series, forecast, title):
    # Plot SARIMA forecast
    plt.figure(figsize=(14, 6))
    plt.plot(series, label="Historical", linewidth=2)
    plt.plot(forecast, label="Forecast", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    symbol = "AAPL"
    start = "2015-01-01"
    end = "2024-01-01"
    steps = 12

    series = download_prices(symbol, start, end)
    forecast, test, metrics, _ = run_sarima(series, steps)
    forecast.index = test.index

    plot(series, forecast, f"{symbol} – SARIMA Forecast")

    print("\nEvaluation:")
    for key, val in metrics.items():
        print(f"{key}: {val:.2f}")

    print("\nForecast:")
    print(forecast)
