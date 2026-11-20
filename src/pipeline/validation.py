from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .config import SplitConfig

_TARGET_ORIGINAL = "rented_bike_count"


def chronological_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
    target: str = _TARGET_ORIGINAL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит отсортированный по времени датафрейм на train/val/test.

    Порядок индексов хронологический (без shuffle). Test замораживается и
    не используется при подборе. Если `keep_non_functioning` выключен,
    нерабочие часы отфильтровываются из train и val (test остаётся целым).
    """
    n = len(df)
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)
    if not (0 < train_end < val_end < n):
        raise ValueError("Некорректные доли train/val: проверьте train_frac и val_frac")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if not cfg.keep_non_functioning:
        train_df = train_df[train_df["functioning_day"] == "Yes"].copy()
        val_df = val_df[val_df["functioning_day"] == "Yes"].copy()

    return train_df, val_df, test_df


def make_time_series_cv(cfg: SplitConfig) -> TimeSeriesSplit:
    """TimeSeriesSplit для кросс-валидации внутри train."""
    return TimeSeriesSplit(n_splits=cfg.cv_splits, gap=cfg.cv_gap)


def cross_validate(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    fit_params: dict | None = None,
) -> np.ndarray:
    """RMSE по каждому временному фолду (положительные значения).

    Использует ручной цикл с deepcopy вместо `cross_val_score`: это
    позволяет передавать `fit_params` (например, `cat_features` для
    CatBoost) и не зависит от версии sklearn.
    """
    fit_params = fit_params or {}
    scores = []
    for train_idx, val_idx in cv.split(X_train):
        fold_model = deepcopy(pipeline)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_params)
        pred = fold_model.predict(X_train.iloc[val_idx])
        scores.append(float(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred))))
    return np.asarray(scores)


def regression_metrics(
    y_true,
    y_pred,
    clip_negative: bool = True,
) -> dict[str, float]:
    """Основная (RMSE) и вспомогательные (MAE, R²) метрики."""
    y_pred = np.asarray(y_pred)
    if clip_negative:
        y_pred = np.maximum(y_pred, 0)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


__all__ = ["chronological_split", "make_time_series_cv", "cross_validate", "regression_metrics"]