"""Build one comparison table from explicitly accepted experiment runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["experiment_id", "model", "include_duration", "sampler", "select_k",
           *[f"mcc_fold_{i}" for i in range(1, 6)], "mcc_mean", "mcc_std",
           "fit_time_mean", "run_id", "code_commit", "data_sha256", "folds_sha256"]


def aggregate(run_root, run_ids):
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run IDs in final_runs.json")
    rows, experiments, signature = [], set(), None
    for run_id in run_ids:
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Run IDs must be directory names")
        folder = run_root / run_id
        config = json.loads((folder / "config.json").read_text())
        metadata = json.loads((folder / "metadata.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        if metadata["dirty_tree"] or not metadata["code_commit"]:
            raise ValueError("Final runs must come from committed source with a clean working tree")
        if config["experiment_id"] in experiments:
            raise ValueError("Accept only one run per experiment ID")
        experiments.add(config["experiment_id"])
        current = (metadata["data_sha256"], metadata["folds_sha256"],
                   metadata["protocol_sha256"], metadata["seed"],
                   json.dumps(metadata["versions"], sort_keys=True), metadata["python_version"])
        if signature is not None and current != signature:
            raise ValueError("Accepted runs must share the same protocol, data, folds and environment")
        signature = current
        metrics = pd.read_csv(folder / "fold_metrics.csv").sort_values("fold")
        if metrics["fold"].tolist() != list(range(1, 6)):
            raise ValueError("Expected exactly five fold metrics")
        if not np.isfinite(metrics[["mcc", "fit_time_seconds"]].to_numpy()).all():
            raise ValueError("Metrics must be finite")
        if not metrics.mcc.between(-1, 1).all() or (metrics.fit_time_seconds < 0).any():
            raise ValueError("Invalid MCC or timing")
        options = config.get("preprocessing", {})
        row = {"experiment_id": config["experiment_id"],
               "model": config["model"], "include_duration": protocol["include_duration"],
               "sampler": options.get("sampler", "none"), "select_k": options.get("select_k", "all"),
               "mcc_mean": metrics.mcc.mean(), "mcc_std": metrics.mcc.std(ddof=1),
               "fit_time_mean": metrics.fit_time_seconds.mean(), "run_id": run_id,
               "code_commit": metadata["code_commit"], "data_sha256": metadata["data_sha256"],
               "folds_sha256": metadata["folds_sha256"]}
        row.update({f"mcc_fold_{int(r.fold)}": r.mcc for r in metrics.itertuples()})
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("mcc_mean", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/final_runs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/model_comparison.csv")
    args = parser.parse_args()
    table = aggregate(args.runs, json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} experiments to {args.output}")


if __name__ == "__main__":
    main()
