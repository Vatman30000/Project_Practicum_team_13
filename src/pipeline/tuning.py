from __future__ import annotations

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .config import TuningConfig


def tune(
    pipeline,
    X_train,
    y_train,
    cfg: TuningConfig,
    cv,
    random_state: int = 42,
    fit_params: dict | None = None,
):
    """Подбирает гиперпараметры. Возвращает обученный search-объект.

    У search доступны `.best_params_`, `.best_score_`, `.best_estimator_`,
    `.cv_results_`. Для `method="none"` возвращает None (без тюнинга).
    `fit_params` (например, `cat_features` для CatBoost) передаются в fit
    каждой оцениваемой конфигурации.
    """
    if cfg.method == "none":
        return None
    if not cfg.param_distributions:
        raise ValueError("Для тюнинга укажите TuningConfig.param_distributions")

    common = dict(
        estimator=pipeline,
        scoring=cfg.scoring,
        cv=cv,
        refit=True,
        return_train_score=True,
        n_jobs=2,
    )

    if cfg.method == "random":
        search = RandomizedSearchCV(
            param_distributions=cfg.param_distributions,
            n_iter=cfg.n_iter,
            random_state=random_state,
            **common,
        )
    elif cfg.method == "grid":
        search = GridSearchCV(param_grid=cfg.param_distributions, **common)
    else:
        raise ValueError(
            f"Неизвестный метод тюнинга {cfg.method!r}: допустимо 'none', 'random', 'grid'"
        )

    search.fit(X_train, y_train, **(fit_params or {}))
    return search


__all__ = ["tune"]