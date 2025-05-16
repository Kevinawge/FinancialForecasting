import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indicators.indicators import apply_indicators


def plot_price_ma(data, label="AAPL", short=20, long=50, save_path=None):
    # Price chart with SMA crossover lines
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close", linewidth=2)
    plt.plot(data[f"SMA_{short}"], label=f"SMA {short}", linestyle="--")
    plt.plot(data[f"SMA_{long}"], label=f"SMA {long}", linestyle="--")

    crossover = data[f"SMA_{short}"] > data[f"SMA_{long}"]
    changes = crossover != crossover.shift()

    for date in data.index[changes]:
        color = "green" if crossover.loc[date] else "red"
        label_type = "Bullish" if crossover.loc[date] else "Bearish"
        plt.axvline(x=date, color=color, linestyle="--", alpha=0.5, label=f"{label_type} Crossover")

    plt.title(f"{label} - Price with SMA Crossovers")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def plot_rsi(data, label="AAPL", save_path=None):
    # RSI chart with buy/sell zones
    plt.figure(figsize=(10, 3))
    plt.plot(data["RSI"], label="RSI", color="purple")
    plt.axhline(70, linestyle="--", color="red", label="Overbought")
    plt.axhline(30, linestyle="--", color="green", label="Oversold")

    buys = data[data["RSI"] < 30]
    sells = data[data["RSI"] > 70]

    plt.scatter(buys.index, buys["RSI"], marker="^", color="green", label="Buy")
    plt.scatter(sells.index, sells["RSI"], marker="v", color="red", label="Sell")

    plt.title(f"{label} - RSI (Buy/Sell Zones)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == "__main__":
    import yfinance as yf

    os.makedirs("plots", exist_ok=True)
    price = yf.download("AAPL", start="2022-01-01", end="2022-12-31", auto_adjust=True)
    price = apply_indicators(price, indicators=["sma", "rsi"])

    plot_price_ma(price, label="AAPL", short=20, long=50, save_path="plots/AAPL_price_sma.png")
    plot_rsi(price, label="AAPL", save_path="plots/AAPL_rsi.png")
