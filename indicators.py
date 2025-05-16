import pandas as pd


def add_moving_averages(data, windows=[20, 50, 100, 200]):
    # Add simple and exponential moving averages
    for w in windows:
        data[f"SMA_{w}"] = data["Close"].rolling(w).mean()
        data[f"EMA_{w}"] = data["Close"].ewm(span=w, adjust=False).mean()
    return data


def add_rsi(data, window=14):
    # Add Relative Strength Index
    change = data["Close"].diff()
    gain = change.where(change > 0, 0).rolling(window).mean()
    loss = -change.where(change < 0, 0).rolling(window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def add_bollinger_bands(data, window=20):
    # Add Bollinger Bands
    mean = data["Close"].rolling(window).mean()
    std = data["Close"].rolling(window).std()
    data["Bollinger_Upper"] = mean + (2 * std)
    data["Bollinger_Lower"] = mean - (2 * std)
    return data


def add_macd(data):
    # Add MACD and signal line
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data


def apply_indicators(data, indicators=["sma", "rsi", "macd", "boll"]):
    # Apply selected technical indicators
    if "sma" in indicators:
        data = add_moving_averages(data)
    if "rsi" in indicators:
        data = add_rsi(data)
    if "macd" in indicators:
        data = add_macd(data)
    if "boll" in indicators:
        data = add_bollinger_bands(data)
    return data


# Run as script
if __name__ == "__main__":
    import yfinance as yf

    price = yf.download("AAPL", start="2022-01-01", end="2022-12-31", auto_adjust=True)
    price = apply_indicators(price)

    cols = ["Close", "SMA_20", "EMA_20", "RSI", "MACD", "Signal_Line", "Bollinger_Upper", "Bollinger_Lower"]
    print(price[cols].tail())
