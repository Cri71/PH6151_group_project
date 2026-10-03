#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

"${PYTHON_BIN:-python}" - <<'PYTHON'
import json
from pathlib import Path
import sys

from experiments.aggregate import aggregate

run_root = Path("results/runs")
selected = {}
# Timestamped names sort newest first within each experiment.
for folder in sorted(run_root.iterdir(), reverse=True):
    if not folder.is_dir():
        continue
    try:
        config = json.loads((folder / "config.json").read_text())
        experiment_id = config["experiment_id"]
        if experiment_id in selected:
            continue
        aggregate(run_root, [folder.name])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Skipping {folder.name}: {error}", file=sys.stderr)
        continue
    selected[experiment_id] = folder.name

run_ids = [selected[name] for name in sorted(selected)]
if not run_ids:
    sys.exit("No eligible runs found; final_runs.json was left unchanged.")
# Check compatibility across experiments before replacing the manifest.
try:
    aggregate(run_root, run_ids)
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"Selected runs cannot be compared: {error}. final_runs.json was left unchanged.")

manifest = Path("experiments/final_runs.json")
manifest.write_text(json.dumps(run_ids, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Wrote {len(run_ids)} accepted runs to {manifest}")
for run_id in run_ids:
    print(f"  {run_id}")
PYTHON
