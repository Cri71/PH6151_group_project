"""Unfitted baseline estimators; evaluation lives in evaluation.py."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def make_model(name, params, seed=42):
    if name == "dummy":
        return DummyClassifier(**{"strategy": "most_frequent", "random_state": seed, **params})
    if name == "logistic_regression":
        return LogisticRegression(**{"max_iter": 2000, "random_state": seed, **params})
    if name == "linear_svm":
        return LinearSVC(**{"max_iter": 5000, "random_state": seed, **params})
    if name == "knn":
        return KNeighborsClassifier(**params)
    raise ValueError(f"Unknown baseline: {name}")
