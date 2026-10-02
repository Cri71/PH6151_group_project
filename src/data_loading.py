"""Load the immutable bank dataset and its common validation folds."""
import hashlib

import numpy as np
import pandas as pd

COLUMNS = ["age", "job", "marital", "education", "default", "balance",
           "housing", "loan", "contact", "day", "month", "duration",
           "campaign", "pdays", "previous", "poutcome", "y"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path):
    frame = pd.read_csv(path, sep=";")
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("Unexpected bank dataset schema or empty dataset")
    if not frame["y"].isin(["yes", "no"]).all():
        raise ValueError("Target must contain only yes/no")
    return frame.drop(columns="y"), frame["y"].map({"no": 0, "yes": 1})


def load_folds(path, row_count):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "fold"]:
        raise ValueError("Expected row_id,fold columns")
    if len(frame) != row_count or not np.array_equal(frame["row_id"], np.arange(row_count)):
        raise ValueError("Fold rows must match the original dataset order exactly")
    folds = frame["fold"].to_numpy()
    if set(folds) != set(range(1, 6)):
        raise ValueError("Exactly five nonempty validation folds are required")
    return folds
