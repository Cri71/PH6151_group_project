"""Shared five-fold evaluation of complete, unfitted pipelines."""
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef


def evaluate_cv(estimator, X, y, folds):
    folds = np.asarray(folds)
    if len(X) != len(y) or len(folds) != len(y) or set(folds) != set(range(1, 6)):
        raise ValueError("X, y and exactly five folds must describe the same rows")
    predictions = np.empty(len(y), dtype=int)
    records = []
    for fold in range(1, 6):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        if y.iloc[train].nunique() != 2 or y.iloc[valid].nunique() != 2:
            raise ValueError("Each training and validation fold must contain both classes")
        fitted = clone(estimator)
        started = perf_counter()
        fitted.fit(X.iloc[train], y.iloc[train])
        fit_time = perf_counter() - started
        predicted = fitted.predict(X.iloc[valid])
        predictions[valid] = predicted
        records.append({
            "fold": fold, "n_train": len(train), "n_valid": len(valid),
            "mcc": float(matthews_corrcoef(y.iloc[valid], predicted)),
            "fit_time_seconds": fit_time,
        })
    oof = pd.DataFrame({"row_id": np.arange(len(y)), "fold": folds,
                        "y_true": y.to_numpy(), "y_pred": predictions})
    return pd.DataFrame(records), oof
