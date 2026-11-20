from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORY_LEVELS

_OHE_FAMILIES = {"linear", "tree", "xgb", "lgb"}

_SCALED_FAMILIES = {"linear"}


def build_preprocessor(
    family: str,
    num_cols: list[str],
    cat_cols: list[str],
) -> object:
    """Возвращает трансформер признаков для заданного семейства модели."""
    if family == "catboost":
        return FunctionTransformer(func=lambda x: x, validate=False)

    if family not in _OHE_FAMILIES:
        raise ValueError(f"Неизвестное семейство предобработки: {family!r}")

    categories = [CATEGORY_LEVELS[col] for col in cat_cols]

    num_transform: object
    if family in _SCALED_FAMILIES:
        num_transform = StandardScaler()
    else:
        num_transform = "passthrough"

    cat_kwargs = dict(categories=categories, handle_unknown="ignore", sparse_output=False)
    if family == "linear":
        cat_kwargs["drop"] = "first"

    return ColumnTransformer(
        transformers=[
            ("num", num_transform, num_cols),
            ("cat", OneHotEncoder(**cat_kwargs), cat_cols),
        ],
        remainder="drop",
    )


__all__ = ["build_preprocessor"]