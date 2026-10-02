import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold


@pytest.fixture
def bank_sample():
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n), "job": rng.choice(["admin.", "services"], n),
        "marital": rng.choice(["single", "married"], n),
        "education": rng.choice(["primary", "secondary"], n), "default": rng.choice(["yes", "no"], n),
        "balance": rng.normal(500, 1000, n), "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n), "contact": rng.choice(["cellular", "telephone"], n), "day": rng.integers(1, 29, n),
        "month": rng.choice(["may", "jun"], n), "duration": rng.integers(0, 1000, n), "campaign": rng.integers(1, 5, n),
        "pdays": rng.choice([-1, 10, 50], n), "previous": rng.integers(0, 4, n), "poutcome": rng.choice(["unknown", "failure"], n),
    })
    y = pd.Series([0] * 70 + [1] * 30)
    folds = np.zeros(n, dtype=int)
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        folds[valid] = fold
    return X, y, folds
