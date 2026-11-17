import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_distribution_pair(
    series,
    feature_name,
    clip_quantiles=(0.0, 1.0),
    is_days=False,
    bins=(40, 150),
    kde=False,
    ylabel="Количество наблюдений",
    count_threshold=None,
    custom_titles=None,
    custom_xlabels=None,
    figsize=(14, 5),
    show=True,
):
    """
    Рисует распределение признака и его log1p-преобразование.

    На каждой панели накладываются две гистограммы.

    Параметры
    ----------
    series : pd.Series
        Числовые данные.
    feature_name : str
        Название признака.
    clip_quantiles : tuple или None
        Квантильные границы клипирования.
        None — без клипирования.
    is_days : bool
        Преобразовать отрицательное количество дней в годы: -x / 365.25.
    bins : tuple[int, int]
        Количество бинов двух наложенных гистограмм.
        Одинаковая пара используется на обеих панелях.
    kde : bool
        Добавить одну KDE-кривую на каждую панель.
    ylabel : str
        Подпись оси Y.
    count_threshold : float или None
        Горизонтальная линия на заданном количестве наблюдений.
    custom_titles : tuple[str, str] или None
        Заголовки левой и правой панелей.
    custom_xlabels : tuple[str, str] или None
        Подписи оси X левой и правой панелей.
    figsize : tuple
        Размер фигуры.
    show : bool
        Вызвать plt.show().

    Возвращает
    ----------
    fig, axes
        Фигура и массив осей.
    """
    if len(bins) != 2 or any(
        not isinstance(b, (int, np.integer)) or b < 1 for b in bins
    ):
        raise ValueError("bins должен содержать два положительных целых числа")

    data = (
        pd.to_numeric(pd.Series(series), errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if data.empty:
        raise ValueError("Нет данных после удаления пропусков и бесконечностей")

    clip_applied = False

    if clip_quantiles is not None:
        if len(clip_quantiles) != 2:
            raise ValueError("clip_quantiles должен содержать две границы")

        low, high = clip_quantiles

        if not 0 <= low <= high <= 1:
            raise ValueError(
                "Границы должны удовлетворять условию 0 <= low <= high <= 1"
            )

        q_low, q_high = data.quantile([low, high])
        data = data.clip(lower=q_low, upper=q_high)
        clip_applied = (low, high) != (0.0, 1.0)

    if is_days:
        data = -data / 365.25

    if (data <= -1).any():
        raise ValueError("Для log1p все значения после преобразований должны быть > -1")

    data_log = np.log1p(data)

    suffix = " после clip" if clip_applied else ""

    if custom_titles is None:
        titles = (
            f"Распределение {feature_name}{suffix}",
            f"Распределение log1p({feature_name}){suffix}",
        )
    else:
        if len(custom_titles) != 2:
            raise ValueError("custom_titles должен содержать два заголовка")
        titles = custom_titles

    if custom_xlabels is None:
        xlabels = (
            feature_name,
            f"log1p({feature_name})",
        )
    else:
        if len(custom_xlabels) != 2:
            raise ValueError("custom_xlabels должен содержать две подписи")
        xlabels = custom_xlabels

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, values, title, xlabel in zip(
        axes,
        (data, data_log),
        titles,
        xlabels,
    ):
        # KDE добавляем только к первой гистограмме,
        # чтобы не рисовать одну и ту же кривую дважды.
        sns.histplot(
            x=values,
            bins=bins[0],
            stat="count",
            kde=kde and values.nunique() > 1,
            color="gray",
            alpha=0.4,
            edgecolor=None,
            label=f"bins={bins[0]}",
            ax=ax,
        )

        sns.histplot(
            x=values,
            bins=bins[1],
            stat="count",
            color="tab:blue",
            alpha=0.45,
            edgecolor=None,
            label=f"bins={bins[1]}",
            ax=ax,
        )

        if count_threshold is not None:
            ax.axhline(
                count_threshold,
                color="red",
                linestyle="--",
                label=f"Порог: {count_threshold}",
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
