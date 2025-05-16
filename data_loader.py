import os
import time
import pandas as pd
import yfinance as yf


def load_prices(tickers, start="2018-01-01", end="2023-12-31",
                auto_adjust=True, use_cache=False, cache_file="stock_data.csv",
                merge=True):
    # Load historical stock prices with optional caching
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cache: {cache_file}")
        table = pd.read_csv(cache_file, parse_dates=["Date"])
        table.set_index("Date", inplace=True)
        return table if merge else split_by_ticker(table)

    records = []

    for symbol in tickers:
        print(f"Fetching: {symbol}")
        for _ in range(3):
            try:
                temp = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=auto_adjust)
                if temp.empty:
                    print(f"Empty result for: {symbol}")
                    break
                temp["Ticker"] = symbol
                temp.index.name = "Date"
                records.append(temp)
                break
            except Exception as err:
                print(f"Retrying {symbol} due to: {err}")
                time.sleep(2)
        else:
            print(f"Failed to fetch: {symbol}")

    if not records:
        return pd.DataFrame() if merge else {}

    combined = pd.concat(records).sort_index()

    if use_cache:
        combined.reset_index().to_csv(cache_file, index=False)
        print(f"Saved to cache: {cache_file}")

    return combined if merge else split_by_ticker(combined)


def split_by_ticker(data):
    # Split combined data into dict by ticker
    return {name: frame.drop(columns="Ticker") for name, frame in data.groupby("Ticker")}


# Run from command line
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    data = load_prices(tickers, use_cache=False, merge=True)
    print(data.head())