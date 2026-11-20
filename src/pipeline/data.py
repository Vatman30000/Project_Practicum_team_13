from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import FeatureConfig

BASE_NUMERIC = [
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point_temp",
    "solar_radiation",
    "rainfall",
    "snowfall",
]

CAT_COLUMNS = ["seasons", "holiday", "functioning_day"]

_TIME_FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "hour_dist_to_8",
    "hour_dist_to_18",
    "hour_dist_to_peak",
]


def norm_col(name: str) -> str:
    """Приводит название колонки из CSV к единому нижнему регистру."""
    name = str(name).strip().lower()
    name = name.replace("°c", "").replace("°", "")
    name = name.replace("%", "")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


_RENAME_MAP = {
    "wind_speed_m_s": "wind_speed",
    "visibility_10m": "visibility",
    "dew_point_temperature": "dew_point_temp",
    "solar_radiation_mj_m2": "solar_radiation",
    "rainfallmm": "rainfall",
    "snowfall_cm": "snowfall",
}


def load_dataset(data_path: str | None = None, dataset: str = "joebeachcapital/seoul-bike-sharing/versions/1") -> pd.DataFrame:
    """Загружает датасет (локальный CSV или через kagglehub) и нормализует.

    Возвращает DataFrame с нормализованными колонками, колонкой `date`
    типа datetime и признаком `timestamp` для сортировки. Целевая колонка
    — `rented_bike_count`.
    """
    if data_path:
        csv_path = data_path
    else:
        import os

        import kagglehub

        csv_path = os.path.join(kagglehub.dataset_download(dataset), "SeoulBikeData.csv")

    df = pd.read_csv(csv_path, encoding="unicode_escape")

    df.columns = [norm_col(c) for c in df.columns]
    df = df.rename(columns=_RENAME_MAP)

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("В данных есть повторяющиеся комбинации дата + час")
    if len(df) != 8760:
        raise ValueError(f"Ожидалось 8760 строк, получено {len(df)}")

    return df


def add_time_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Добавляет временные признаки согласно FeatureConfig."""
    out = df.copy()

    if cfg.day_of_week:
        out["day_of_week"] = out["date"].dt.day_of_week
    if cfg.month:
        out["month"] = out["date"].dt.month
    if cfg.is_weekend:
        out["is_weekend"] = (out["date"].dt.day_of_week >= 5).astype(int)
    if cfg.hour_cyclic:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    if cfg.hour_peak_distance:
        for peak in (8, 18):
            distance = (out["hour"] - peak).abs()
            out[f"hour_dist_to_{peak}"] = np.minimum(distance, 24 - distance)
        out["hour_dist_to_peak"] = out[["hour_dist_to_8", "hour_dist_to_18"]].min(axis=1)

    return out


def feature_columns(cfg: FeatureConfig) -> list[str]:
    """Порядок признаков, подаваемых в модель (числовые + категориальные)."""
    num = list(BASE_NUMERIC)
    if cfg.day_of_week:
        num.append("day_of_week")
    if cfg.month:
        num.append("month")
    if cfg.is_weekend:
        num.append("is_weekend")
    if cfg.hour_cyclic:
        num += ["hour_sin", "hour_cos"]
    if cfg.hour_peak_distance:
        num += ["hour_dist_to_8", "hour_dist_to_18", "hour_dist_to_peak"]
    return num + CAT_COLUMNS


__all__ = [
    "BASE_NUMERIC",
    "CAT_COLUMNS",
    "norm_col",
    "load_dataset",
    "add_time_features",
    "feature_columns",
]