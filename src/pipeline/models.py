from __future__ import annotations

from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor

FAMILY_BY_MODEL: dict[str, str] = {
    "linear": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "elasticnet": "linear",
    "random_forest": "tree",
    "xgboost": "xgb",
    "lightgbm": "lgb",
    "catboost": "catboost",
}

_DEFAULTS: dict[str, dict] = {
    "linear": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 1.0},
    "elasticnet": {"alpha": 1.0, "l1_ratio": 0.5},
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbosity": -1,
    },
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "allow_writing_files": False,
        "verbose": 0,
    },
}


def _require(lib: str) -> None:
    """Проверяет наличие библиотеки и даёт понятную ошибку при её отсутствии."""
    try:
        __import__(lib)
    except ImportError as exc:
        raise ImportError(
            f"Для модели нужна библиотека '{lib}', которая не установлена. "
            f"Добавьте её в зависимости и выполните установку (например, uv add {lib})."
        ) from exc


def _make_estimator(
    model_name: str,
    params: dict,
    random_state: int,
):
    """Создаёт оценщик (estimator) по имени модели."""
    merged = {**_DEFAULTS.get(model_name, {}), **params}

    if model_name == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    if model_name in ("ridge", "lasso", "elasticnet"):
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        cls = {"ridge": Ridge, "lasso": Lasso, "elasticnet": ElasticNet}[model_name]
        return cls(**merged, random_state=random_state)

    if model_name == "random_forest":
        _require("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**merged, random_state=random_state, n_jobs=-1)

    if model_name == "xgboost":
        _require("xgboost")
        from xgboost import XGBRegressor

        return XGBRegressor(**merged, random_state=random_state)

    if model_name == "lightgbm":
        _require("lightgbm")
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**merged, random_state=random_state)

    if model_name == "catboost":
        _require("catboost")
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**merged, random_seed=random_state)

    raise ValueError(f"Неизвестная модель: {model_name!r}")


def build_pipeline(
    model_name: str,
    params: dict | None = None,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Собирает полный пайплайн (предобработка + модель)."""
    if model_name not in FAMILY_BY_MODEL:
        available = ", ".join(sorted(FAMILY_BY_MODEL))
        raise ValueError(f"Неизвестная модель {model_name!r}. Доступны: {available}")

    family = FAMILY_BY_MODEL[model_name]
    num_cols = num_cols or []
    cat_cols = cat_cols or []

    preprocessor = build_preprocessor(family, num_cols, cat_cols)
    estimator = _make_estimator(model_name, params or {}, random_state)

    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])


__all__ = ["FAMILY_BY_MODEL", "build_pipeline"]