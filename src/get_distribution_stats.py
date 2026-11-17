import numpy as np
import pandas as pd


def analyze_distribution(
    series,
    percentiles_list=None,
    clip_quantiles=None,
    apply_log=False,
    print_deciles=False,
):
    """
    Выводит статистику числового ряда.

    Порядок обработки:
    1. Подсчёт исходных пропусков и бесконечностей.
    2. Замена бесконечностей на NaN.
    3. Опциональное клипирование.
    4. Опциональное log1p.
    5. Расчёт статистик по конечным непустым значениям.

    Доли нулей и выходов за IQR-пороги считаются относительно
    количества значений, участвующих в анализе.

    Возвращает словарь со статистиками и преобразованным рядом.
    """
    if percentiles_list is None:
        percentiles_list = [
            0.0,
            0.001,
            0.01,
            0.05,
            0.25,
            0.5,
            0.75,
            0.95,
            0.99,
            0.999,
            1.0,
        ]

    if any(not 0 <= p <= 1 for p in percentiles_list):
        raise ValueError("Процентили должны находиться в диапазоне [0, 1]")

    s = pd.to_numeric(pd.Series(series).copy(), errors="raise")

    n_total = len(s)
    n_missing = int(s.isna().sum())
    n_infinite = int(s.isin([np.inf, -np.inf]).sum())

    na_pct = n_missing / n_total * 100 if n_total else np.nan
    inf_pct = n_infinite / n_total * 100 if n_total else np.nan

    s = s.replace([np.inf, -np.inf], np.nan)

    if s.notna().sum() == 0:
        raise ValueError("Нет конечных непустых значений для анализа")

    clip_bounds = None

    if clip_quantiles is not None:
        if len(clip_quantiles) != 2:
            raise ValueError("clip_quantiles должен содержать две границы")

        low, high = clip_quantiles

        if not 0 <= low <= high <= 1:
            raise ValueError(
                "Требуется 0 <= clip_quantiles[0] <= clip_quantiles[1] <= 1"
            )

        q_low, q_high = s.quantile([low, high])
        clip_bounds = (q_low, q_high)
        s = s.clip(lower=q_low, upper=q_high)

    if apply_log:
        if (s.dropna() <= -1).any():
            raise ValueError("Для log1p все непустые значения должны быть > -1")
        s = np.log1p(s)

    valid = s.dropna()
    n = len(valid)

    q1, median, q3 = valid.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    n_below = int(valid.lt(lower_fence).sum())
    n_above = int(valid.gt(upper_fence).sum())
    n_outliers = n_below + n_above
    outlier_pct = n_outliers / n * 100

    n_zeros = int(valid.eq(0).sum())
    zero_pct = n_zeros / n * 100

    mean = valid.mean()
    std = valid.std()
    min_value = valid.min()
    max_value = valid.max()
    skew = valid.skew()
    kurt = valid.kurt()

    percentiles = valid.quantile(percentiles_list)

    deciles = valid.quantile(np.arange(1, 10) / 10) if print_deciles else None

    print(f"Всего: {n_total:,} | В анализе: {n:,}")
    print(
        f"Пропуски: {n_missing:,} ({na_pct:.2f}%) | "
        f"Бесконечности: {n_infinite:,} ({inf_pct:.2f}%)"
    )

    if clip_bounds is not None:
        print(
            f"Clip по квантилям {clip_quantiles}: "
            f"[{clip_bounds[0]:.4f}, {clip_bounds[1]:.4f}]"
        )

    if apply_log:
        print("Преобразование: log1p")

    print(f"Среднее: {mean:.4f} | Медиана: {median:.4f} | Std: {std:.4f}")
    print(f"Min / Max: {min_value:.4f} / {max_value:.4f}")
    print(f"Q1: {q1:.4f} | Q3: {q3:.4f} | IQR: {iqr:.4f}")
    print(f"IQR-пороги: [{lower_fence:.4f}, {upper_fence:.4f}]")
    print(
        f"За IQR-порогами: {n_outliers:,} ({outlier_pct:.2f}%) "
        f"| Ниже: {n_below:,} | Выше: {n_above:,}"
    )
    print(f"Нули: {n_zeros:,} ({zero_pct:.2f}%)")
    print(f"Асимметрия: {skew:.4f} | Эксцесс: {kurt:.4f}")

    if iqr == 0:
        print(
            "Внимание: IQR = 0. Пороги совпадают; "
            "выход за них не означает ошибку данных."
        )

    if deciles is not None:
        print("\nДецили:")
        print(deciles.round(4))

    print("\nПроцентили:")
    print(percentiles.round(4))

    return {
        "n_total": n_total,
        "n": n,
        "n_missing": n_missing,
        "na_pct": na_pct,
        "n_infinite": n_infinite,
        "inf_pct": inf_pct,
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_below": n_below,
        "n_above": n_above,
        "n_outliers": n_outliers,
        "outlier_pct": outlier_pct,
        "n_zeros": n_zeros,
        "zero_pct": zero_pct,
        "skew": skew,
        "kurt": kurt,
        "percentiles": percentiles,
        "deciles": deciles,
        "clip_quantiles": clip_quantiles,
        "clip_bounds": clip_bounds,
        "apply_log": apply_log,
        "transformed_series": s,
    }
