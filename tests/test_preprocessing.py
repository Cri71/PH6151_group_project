import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest

from src.baseline_models import make_model
from src.evaluation import evaluate_cv
from src.preprocessing import BankFeatures, build_pipeline


def test_features_preserve_input_and_exclude_duration(bank_sample):
    X, _, _ = bank_sample
    original = X.copy(deep=True)
    transformed = BankFeatures().fit_transform(X)
    assert X.equals(original)
    assert "duration" not in transformed
    assert (transformed.loc[X.pdays == -1, "previously_contacted"] == 0).all()
    assert (transformed.loc[X.pdays == -1, "pdays"] == 0).all()
    assert "duration" in BankFeatures(include_duration=True).fit_transform(X)


def test_scaling_and_categories_only_learn_training_rows(bank_sample):
    X, y, _ = bank_sample
    train, valid = X.iloc[:80].copy(), X.iloc[80:].copy()
    valid.loc[:, "age"] = 10000
    valid.loc[:, "job"] = "validation_only"
    pipeline = build_pipeline(make_model("dummy", {}), {})
    pipeline.fit(train, y.iloc[:80])
    transformer = pipeline.named_steps["preprocess"]
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], train.age.mean())
    encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
    assert all("validation_only" not in categories for categories in encoder.categories_)
    assert len(pipeline.predict(valid)) == len(valid)


def test_sampling_and_selection_fit_only_training_folds(bank_sample, monkeypatch):
    X, y, folds = bank_sample
    sampled_sizes, selected_sizes = [], []
    original_sample = RandomOverSampler.fit_resample
    original_select = SelectKBest.fit

    def sample(self, X_train, y_train, **kwargs):
        sampled_sizes.append(len(y_train))
        return original_sample(self, X_train, y_train, **kwargs)

    def select(self, X_train, y_train=None, **kwargs):
        selected_sizes.append(len(y_train))
        return original_select(self, X_train, y_train, **kwargs)

    monkeypatch.setattr(RandomOverSampler, "fit_resample", sample)
    monkeypatch.setattr(SelectKBest, "fit", select)
    pipeline = build_pipeline(make_model("logistic_regression", {}), {"sampler": "random", "select_k": 5})
    metrics, predictions = evaluate_cv(pipeline, X, y, folds)
    assert sampled_sizes == [80] * 5
    assert selected_sizes == [112] * 5  # Training folds have 56 negative/24 positive rows.
    assert metrics.n_valid.tolist() == [20] * 5
    assert len(predictions) == len(y)
    assert pipeline.named_steps["model"].__dict__.get("classes_") is None
