"""Boosting constructors; sampling is configured in the shared pipeline."""
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


def make_model(name, params, seed=42):
    if name == "adaboost":
        return AdaBoostClassifier(**{"random_state": seed, **params})
    if name == "gradient_boosting":
        return GradientBoostingClassifier(**{"random_state": seed, **params})
    raise ValueError(f"Unknown boosting model: {name}")
