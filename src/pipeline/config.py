from __future__ import annotations

from dataclasses import dataclass, field, asdict

CATEGORY_LEVELS: dict[str, list[str]] = {
    "seasons": ["Autumn", "Spring", "Summer", "Winter"],
    "holiday": ["Holiday", "No Holiday"],
    "functioning_day": ["No", "Yes"],
}


@dataclass
class SplitConfig:
    """Хронологическое разбиение и схема кросс-валидации."""

    train_frac: float = 0.70
    val_frac: float = 0.15
    keep_non_functioning: bool = True
    cv_splits: int = 4
    cv_gap: int = 24


@dataclass
class FeatureConfig:
    """Какие временные признаки добавлять к базовым погодным."""

    day_of_week: bool = True
    month: bool = True
    is_weekend: bool = True
    hour_cyclic: bool = True
    hour_peak_distance: bool = False


@dataclass
class ModelConfig:
    """Имя модели и её гиперпараметры (переопределяют значения по умолчанию)."""

    name: str = "catboost"
    params: dict = field(default_factory=dict)


@dataclass
class TuningConfig:
    """Подбор гиперпараметров. method: "none" | "random" | "grid"."""

    method: str = "none"
    param_distributions: dict = field(default_factory=dict)
    n_iter: int = 10
    scoring: str = "neg_root_mean_squared_error"


@dataclass
class PipelineConfig:
    """Полная спецификация прогона пайплайна."""

    dataset: str = "joebeachcapital/seoul-bike-sharing/versions/1"
    data_path: str | None = None
    target: str = "rented_bike_count"
    random_state: int = 42
    clip_negative: bool = True
    evaluate_test: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Строит конфиг из вложенного словаря (из JSON-файла)."""
        data = dict(data)
        split = data.pop("split", None)
        features = data.pop("features", None)
        model = data.pop("model", None)
        tuning = data.pop("tuning", None)

        cfg = cls(**data)
        if split:
            cfg.split = SplitConfig(**split)
        if features:
            cfg.features = FeatureConfig(**features)
        if model:
            cfg.model = ModelConfig(**model)
        if tuning:
            cfg.tuning = TuningConfig(**tuning)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_file(path: str) -> PipelineConfig:
    """Загружает конфиг из JSON-файла."""
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return PipelineConfig.from_dict(data)


__all__ = [
    "CATEGORY_LEVELS",
    "SplitConfig",
    "FeatureConfig",
    "ModelConfig",
    "TuningConfig",
    "PipelineConfig",
    "config_from_file",
]