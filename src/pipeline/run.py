from __future__ import annotations

import argparse

from .config import PipelineConfig, config_from_file
from .data import CAT_COLUMNS, add_time_features, feature_columns, load_dataset
from .models import FAMILY_BY_MODEL, build_pipeline
from .tuning import tune
from .validation import (
    chronological_split,
    cross_validate,
    make_time_series_cv,
    regression_metrics,
)


def _default_param_distributions(model_name: str) -> dict:
    """Небольшая сетка по умолчанию, чтобы `--tune` работал сразу."""
    if model_name == "catboost":
        return {
            "model__iterations": [300, 500, 800],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__depth": [4, 6, 8],
        }
    if model_name in ("xgboost", "lightgbm"):
        return {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6, 8],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [80, 120, 180],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 3, 6],
        }
    if model_name in ("ridge", "lasso", "elasticnet"):
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0]}
    return {}


def run(cfg: PipelineConfig) -> dict:
    """Выполняет полный прогон пайплайна по конфигу и возвращает результаты."""
    print(f"=== Загрузка данных: {cfg.dataset or cfg.data_path} ===")
    df = load_dataset(cfg.data_path, cfg.dataset)
    df = add_time_features(df, cfg.features)

    cols = feature_columns(cfg.features)
    num_cols = [c for c in cols if c not in CAT_COLUMNS]
    cat_cols = CAT_COLUMNS
    fit_params = {"model__cat_features": cat_cols} if cfg.model.name == "catboost" else {}

    train_df, val_df, test_df = chronological_split(df, cfg.split, cfg.target)
    print(
        f"Сплит (train/val/test): {len(train_df)} / {len(val_df)} / {len(test_df)}"
    )

    X_train, y_train = train_df[cols], train_df[cfg.target]
    X_val, y_val = val_df[cols], val_df[cfg.target]

    pipeline = build_pipeline(
        cfg.model.name,
        cfg.model.params,
        num_cols=num_cols,
        cat_cols=cat_cols,
        random_state=cfg.random_state,
    )

    cv = make_time_series_cv(cfg.split)
    rmse_folds = cross_validate(pipeline, X_train, y_train, cv, fit_params=fit_params)
    print(
        f"CV RMSE (train): {rmse_folds.mean():.2f} ± {rmse_folds.std():.2f}"
    )

    if cfg.tuning.method != "none":
        if not cfg.tuning.param_distributions:
            cfg.tuning.param_distributions = _default_param_distributions(
                cfg.model.name
            )
        print(f"=== Подбор гиперпараметров: {cfg.tuning.method} ===")
        search = tune(
            pipeline, X_train, y_train, cfg.tuning, cv, cfg.random_state,
            fit_params=fit_params,
        )
        pipeline = search.best_estimator_
        print(f"Лучшие параметры: {search.best_params_}")
        print(f"Лучший CV RMSE: {-search.best_score_:.2f}")

    pipeline.fit(X_train, y_train, **fit_params)

    y_val_pred = pipeline.predict(X_val)
    val_metrics = regression_metrics(y_val, y_val_pred, cfg.clip_negative)

    y_train_pred = pipeline.predict(X_train)
    train_metrics = regression_metrics(
        y_train, y_train_pred, cfg.clip_negative
    )

    results: dict = {
        "model": cfg.model.name,
        "cv_rmse_mean": float(rmse_folds.mean()),
        "cv_rmse_std": float(rmse_folds.std()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    if cfg.evaluate_test:
        X_test, y_test = test_df[cols], test_df[cfg.target]
        y_test_pred = pipeline.predict(X_test)
        results["test_metrics"] = regression_metrics(
            y_test, y_test_pred, cfg.clip_negative
        )

    _print_results(results)
    return results


def _print_results(results: dict) -> None:
    metrics_names = ["RMSE", "MAE", "R2"]
    rows = {
        "train": results["train_metrics"],
        "val": results["val_metrics"],
    }
    if "test_metrics" in results:
        rows["test"] = results["test_metrics"]

    print("\n=== Метрики ===")
    header = f"{'split':<8}" + "".join(f"{m:>12}" for m in metrics_names)
    print(header)
    for split, m in rows.items():
        line = f"{split:<8}" + "".join(f"{m[name]:>12.3f}" for name in metrics_names)
        print(line)
    print(f"CV RMSE: {results['cv_rmse_mean']:.2f} ± {results['cv_rmse_std']:.2f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Пайплайн прогноза спроса на велосипеды")
    parser.add_argument("--config", help="Путь к JSON-конфигу")
    parser.add_argument("--model", choices=sorted(FAMILY_BY_MODEL), help="Имя модели")
    parser.add_argument("--data-path", help="Локальный CSV вместо kagglehub")
    parser.add_argument("--tune", action="store_true", help="Включить подбор гиперпараметров")
    parser.add_argument("--method", choices=["random", "grid"], help="Метод тюнинга")
    parser.add_argument("--evaluate-test", action="store_true", help="Оценить на test")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.config:
        cfg = config_from_file(args.config)
    else:
        cfg = PipelineConfig()

    if args.model:
        cfg.model.name = args.model
    if args.data_path:
        cfg.data_path = args.data_path
    if args.tune:
        cfg.tuning.method = "random"
    if args.method:
        cfg.tuning.method = args.method
    if args.evaluate_test:
        cfg.evaluate_test = True

    run(cfg)


if __name__ == "__main__":
    main()