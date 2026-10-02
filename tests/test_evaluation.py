import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef

from experiments.aggregate import aggregate
from experiments.run import build_model
from src.data_loading import load_data, load_folds, sha256
from src.evaluation import evaluate_cv

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_manifest_match_protocol():
    protocol = json.loads((ROOT / "experiments/protocol.json").read_text())
    path = ROOT / "data/raw/bank-full.csv"
    assert sha256(path) == protocol["data_sha256"]
    assert sha256(ROOT / "data/folds.csv") == protocol["folds_sha256"]
    X, y = load_data(path)
    folds = load_folds(ROOT / "data/folds.csv", len(y))
    assert len(X) == 45211
    assert int(y.sum()) == 5289
    assert set(folds) == set(range(1, 6))
    for fold in range(1, 6):
        assert abs(y[folds == fold].mean() - y.mean()) < 0.001


@pytest.mark.parametrize("config_path", sorted((ROOT / "experiments/configs").glob("*.json")), ids=lambda p:p.stem)
def test_every_example_uses_five_fold_mcc(config_path, bank_sample):
    X, y, folds = bank_sample
    config = json.loads(config_path.read_text())
    estimator = build_model(config, seed=42, include_duration=False)
    metrics, oof = evaluate_cv(estimator, X, y, folds)
    assert metrics.fold.tolist() == [1, 2, 3, 4, 5]
    assert oof.row_id.tolist() == list(range(len(y)))
    assert metrics.n_valid.sum() == len(y)
    for record in metrics.itertuples():
        held_out = oof[oof.fold == record.fold]
        assert record.mcc == matthews_corrcoef(held_out.y_true, held_out.y_pred)
    if config["model"] == "dummy":
        assert (metrics.mcc == 0).all()


def test_stacking_preprocessing_is_inside_inner_folds(bank_sample, monkeypatch):
    from src.preprocessing import BankFeatures
    X, y, folds = bank_sample
    sizes = []
    original = BankFeatures.fit

    def fit(self, X_train, y_train=None):
        sizes.append(len(X_train))
        return original(self, X_train, y_train)

    monkeypatch.setattr(BankFeatures, "fit", fit)
    config = {"model": "stacking", "params": {}, "preprocessing": {}}
    evaluate_cv(build_model(config, 42, False), X, y, folds)
    # Per outer fold: two full fits and five inner fits per base learner.
    assert sizes.count(80) == 10
    assert sizes.count(64) == 50
    assert len(sizes) == 60


def test_invalid_manifest_and_folds_rejected(tmp_path, bank_sample):
    X, y, folds = bank_sample
    with pytest.raises(ValueError, match="five"):
        evaluate_cv(build_model({"model": "dummy"}, 42, False), X, y, np.ones(len(y)))
    path = tmp_path / "folds.csv"
    pd.DataFrame({"row_id": [0, 0], "fold": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="order"):
        load_folds(path, 2)


def write_run(root, run_id, dirty=False):
    folder = root / run_id
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"experiment_id": run_id, "model": "dummy"}))
    (folder / "protocol.json").write_text(json.dumps({"include_duration": False}))
    metadata = {"dirty_tree": dirty, "code_commit": "synthetic-test-commit", "data_sha256": "data",
                "folds_sha256": "folds", "protocol_sha256": "protocol", "seed": 42,
                "versions": {"test": "1"}, "python_version": "test"}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame({"fold": range(1, 6), "mcc": [0.1, 0.2, 0.3, 0.4, 0.5],
                  "fit_time_seconds": [1.] * 5}).to_csv(folder / "fold_metrics.csv", index=False)


def test_aggregation_and_rejection(tmp_path):
    write_run(tmp_path, "clean")
    table = aggregate(tmp_path, ["clean"])
    assert table.mcc_mean.iloc[0] == pytest.approx(0.3)
    assert table.mcc_std.iloc[0] == pytest.approx(np.std([.1, .2, .3, .4, .5], ddof=1))
    write_run(tmp_path, "dirty", dirty=True)
    with pytest.raises(ValueError, match="clean working tree"):
        aggregate(tmp_path, ["dirty"])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(tmp_path, ["clean", "clean"])
    folder = tmp_path / "clean"
    metrics = pd.read_csv(folder / "fold_metrics.csv").iloc[:4]
    metrics.to_csv(folder / "fold_metrics.csv", index=False)
    with pytest.raises(ValueError, match="five"):
        aggregate(tmp_path, ["clean"])
