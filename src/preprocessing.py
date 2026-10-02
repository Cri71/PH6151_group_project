"""One training-fold pipeline for every classification workstream."""
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BankFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, include_duration=False):
        self.include_duration = include_duration

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        if not self.include_duration:
            frame = frame.drop(columns="duration")
        frame["previously_contacted"] = (frame["pdays"] >= 0).astype(int)
        frame["pdays"] = frame["pdays"].replace(-1, 0)
        return frame


def build_pipeline(estimator, options, seed=42):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])
    sampler = options.get("sampler", "none")
    if sampler not in {"none", "random"}:
        raise ValueError("Supported samplers: none, random")
    k = options.get("select_k", "all")
    if k != "all" and (type(k) is not int or k < 1):
        raise ValueError("select_k must be a positive integer or all")
    return Pipeline([
        ("features", BankFeatures(options.get("include_duration", False))),
        ("preprocess", preprocess),
        ("sampler", RandomOverSampler(random_state=seed) if sampler == "random" else "passthrough"),
        ("select", SelectKBest(f_classif, k=k) if k != "all" else "passthrough"),
        ("model", estimator),
    ])
