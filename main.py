import os
import subprocess


def run_script(path):
    if os.path.exists(path):
        subprocess.run(["python", path])
    else:
        print(f"Script not found: {path}")


def load_data():
    run_script("data_loader/data_loader.py")


def visualize():
    run_script("visualizer/visualizer.py")


def forecast_arima():
    run_script("forecast_arima/forecast_arima.py")


def forecast_sarima():
    run_script("forecast_sarima/forecast_sarima.py")


def evaluate_metrics():
    run_script("model_evaluation/model_eval_runner.py")


def main():
    menu = """
=========== Stock Analysis CLI ===========
1. Load and preprocess stock data
2. Visualize technical indicators
3. Forecast with ARIMA
4. Forecast with SARIMA
5. Evaluate metrics across companies
0. Exit
==========================================
"""
    options = {
        "1": load_data,
        "2": visualize,
        "3": forecast_arima,
        "4": forecast_sarima,
        "5": evaluate_metrics
    }

    while True:
        print(menu)
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting.")
            break
        elif choice in options:
            options[choice]()
        else:
            print("Invalid selection. Please choose 0–5.")


if __name__ == "__main__":
    main()