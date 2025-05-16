import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera, shapiro


def adf_test(series):
    # Augmented Dickey-Fuller test for stationarity
    stat, p, *_ = adfuller(series.dropna())
    return {"ADF Statistic": stat, "p-value": p}


def normality_tests(series):
    # Jarque-Bera and Shapiro-Wilk tests for normality
    jb_stat, jb_p = jarque_bera(series)
    shapiro_stat, shapiro_p = shapiro(series)
    return {
        "Jarque-Bera": jb_stat,
        "JB p-value": jb_p,
        "Shapiro-Wilk": shapiro_stat,
        "Shapiro p-value": shapiro_p
    }