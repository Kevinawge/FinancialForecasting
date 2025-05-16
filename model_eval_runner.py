import yfinance as yf
import pandas as pd
from stats_tests import adf_test, normality_tests


def analyze(ticker, start="2018-01-01", end="2023-01-01"):
    print(f"\n=== {ticker} Summary ===")

    data = yf.download(ticker, start=start, end=end, auto_adjust=True)[["Close"]]
    data = data.dropna().asfreq("D").fillna(method="ffill")
    data["Return"] = data["Close"].pct_change()

    mean = data["Return"].mean()
    std = data["Return"].std()
    skew = data["Return"].skew()
    kurt = data["Return"].kurt()

    stationarity = adf_test(data["Close"])
    normality = normality_tests(data["Return"].dropna())

    print(f"Mean Return:         {mean:.5f}")
    print(f"Std Deviation:       {std:.5f}")
    print(f"Skewness:            {skew:.4f}")
    print(f"Kurtosis:            {kurt:.4f}")
    print(f"ADF p-value:         {stationarity['p-value']:.4f}")
    print(f"Jarque-Bera p-value: {normality['JB p-value']:.4f}")
    print(f"Shapiro p-value:     {normality['Shapiro p-value']:.4f}")

    return {
        "Ticker": ticker,
        "Mean Return": mean,
        "Std Dev": std,
        "Skew": skew,
        "Kurtosis": kurt,
        "ADF p": stationarity["p-value"],
        "JB p": normality["JB p-value"],
        "Shapiro p": normality["Shapiro p-value"]
    }


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results = []

    for symbol in tickers:
        try:
            summary = analyze(symbol)
            results.append(summary)
        except Exception as err:
            print(f"Error for {symbol}: {err}")

    final = pd.DataFrame(results)
    print("\n=== Comparison Summary ===\n")
    print(final.round(4))
