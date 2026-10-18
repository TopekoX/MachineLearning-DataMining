"""Fungsi bantu untuk bab Forecasting: memuat data CO2 bulanan dan menghitung metrik.

Data: konsentrasi CO2 atmosfer mingguan di Mauna Loa (1958-2001), dibundel dengan
statsmodels (domain publik). Kutipan: Keeling, C.D. dan T.P. Whorf (2004).
"""
import numpy as np
import pandas as pd
from statsmodels.datasets import co2

SEASON = 12  # periode musiman data bulanan


def load_co2_monthly():
    """Rata-rata bulanan CO2 (ppm), indeks bulanan 'MS'.

    Ada 5 bulan tanpa data (semuanya sebelum 1965). Karena berada jauh di dalam
    data latih, bulan itu diisi dengan interpolasi berbasis waktu.
    """
    weekly = co2.load_pandas().data["co2"]
    monthly = weekly.resample("MS").mean()
    return monthly.interpolate(method="time").rename("co2")


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)


def mase(y_true, y_pred, y_train, m=SEASON):
    """MASE: MAE dibagi MAE seasonal-naive pada DATA LATIH (1.0 = sama dengan seasonal naive)."""
    y_train = np.asarray(y_train)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return mae(y_true, y_pred) / float(scale)


def seasonal_naive(train, horizon, m=SEASON):
    """Prediksi = nilai pada musim yang sama tahun lalu (diulang bila horizon > m)."""
    last_season = np.asarray(train)[-m:]
    return np.resize(last_season, horizon)
