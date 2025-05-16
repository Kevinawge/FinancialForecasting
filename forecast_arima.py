import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_arima_forecast(data, steps=30, show_plot=True):
    # Fit and forecast using auto_arima
    series = data[["Close"]].copy().dropna()
    series = series.asfreq("D").fillna(method="ffill")

    train = series.iloc[:-steps]
    test = series.iloc[-steps:]

    model = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
    preds = model.predict(n_periods=steps)
    forecast = pd.DataFrame({"Forecast": preds}, index=test.index)

    rmse = np.sqrt(mean_squared_error(test["Close"], preds))
    mae = mean_absolute_error(test["Close"], preds)

    if show_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(series.index, series["Close"], label="Actual", linewidth=2)
        plt.plot(forecast.index, forecast["Forecast"], linestyle="--", label="Forecast")
        plt.title("ARIMA Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return forecast, test, {"RMSE": rmse, "MAE": mae}


if __name__ == "__main__":
    symbol = "AAPL"
    print(f"\nFetching: {symbol}")

    try:
        price = yf.download(symbol, start="2020-01-01", end="2023-01-01", auto_adjust=True, repair=True)
        if price.empty:
            raise ValueError("Primary source failed.")
    except Exception as err:
        print(f"Download error: {err}")
        try:
            price = yf.Ticker(symbol).history(start="2020-01-01", end="2023-01-01", auto_adjust=True)
        except Exception as err2:
            print(f"Backup failed: {err2}")
            exit("Download failed. Exiting.")

    if price.empty:
        exit("No data returned. Exiting.")

    print("Download complete.")

    forecast, test, metrics = run_arima_forecast(price, steps=30, show_plot=True)

    print("\nEvaluation:")
    for metric, val in metrics.items():
        print(f"{metric}: {val:.2f}")

    print("\nLatest Forecast:")
    print(forecast.tail())
