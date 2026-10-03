"""Run from repository root: python -m experiments.run --config PATH."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

from src import baseline_models, boosting, ensemble, tree_models
from src.data_loading import load_data, load_folds, sha256
from src.evaluation import evaluate_cv
from src.preprocessing import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def build_model(config, seed, include_duration):
    name = config["model"]
    params = config.get("params", {})
    if "random_state" in params:
        raise ValueError("Set the seed in protocol.json, not model parameters")
    options = {"include_duration": include_duration, **config.get("preprocessing", {})}
    if options["include_duration"] != include_duration:
        raise ValueError("Feature availability must match protocol.json")
    if name in {"voting", "stacking"}:
        if set(params) & {"estimators", "cv", "final_estimator"}:
            raise ValueError("Configure ensemble members/inner CV in src/ensemble.py")
        return ensemble.make_model(name, params, options, seed)
    if name in {"dummy", "logistic_regression", "linear_svm", "knn"}:
        estimator = baseline_models.make_model(name, params, seed)
    elif name in {"decision_tree", "random_forest"}:
        estimator = tree_models.make_model(name, params, seed)
    elif name in {"adaboost", "gradient_boosting"}:
        estimator = boosting.make_model(name, params, seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return build_pipeline(estimator, options, seed)


def source_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        changes = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], cwd=ROOT, text=True)
        return commit, bool(changes.strip())
    except subprocess.CalledProcessError:
        return None, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs")
    args = parser.parse_args()
    protocol_path = args.protocol
    protocol = read_json(protocol_path)
    config = read_json(args.config)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", config["experiment_id"]):
        raise ValueError("experiment_id must use lowercase letters, digits, underscores or hyphens")
    if protocol["n_splits"] != 5 or protocol["metric"] != "mcc":
        raise ValueError("This project requires five folds and MCC")
    data_path = ROOT / "data/raw/bank-full.csv"
    folds_path = ROOT / "data/folds.csv"
    if sha256(data_path) != protocol["data_sha256"] or sha256(folds_path) != protocol["folds_sha256"]:
        raise ValueError("Dataset/folds changed; review and update the protocol before running")
    X, y = load_data(data_path)
    folds = load_folds(folds_path, len(y))
    estimator = build_model(config, protocol["seed"], protocol["include_duration"])
    commit, dirty = source_state()
    metrics, predictions = evaluate_cv(estimator, X, y, folds)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{config['experiment_id']}__{stamp}__{commit[:7] if commit else 'uncommitted'}"
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "scipy", "matplotlib"]}
    metadata = {"run_id": run_id, "code_commit": commit, "dirty_tree": dirty,
                "seed": protocol["seed"], "data_sha256": sha256(data_path),
                "folds_sha256": sha256(folds_path), "protocol_sha256": sha256(protocol_path),
                "versions": versions, "python_version": __import__("platform").python_version()}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    print(f"{run_id}: MCC {metrics.mcc.mean():.4f} +/- {metrics.mcc.std(ddof=1):.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
