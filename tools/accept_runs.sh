#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

"${PYTHON_BIN:-python}" - "$@" <<'PYTHON'
import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.aggregate import aggregate

parser = argparse.ArgumentParser(description="Select newest eligible runs for one protocol.")
parser.add_argument("--protocol", type=Path, default=Path("experiments/protocol.json"))
parser.add_argument("--runs", type=Path, default=Path("results/runs"))
parser.add_argument("--manifest", type=Path, default=Path("experiments/final_runs.json"))
args = parser.parse_args()
protocol_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
run_root = args.runs
selected = {}
# Timestamped names sort newest first within each experiment.
for folder in sorted(run_root.iterdir(), reverse=True):
    if not folder.is_dir():
        continue
    try:
        config = json.loads((folder / "config.json").read_text())
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["protocol_sha256"] != protocol_sha256:
            continue
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
    sys.exit("No eligible runs found for the selected protocol; manifest was left unchanged.")
# Check compatibility across experiments before replacing the manifest.
try:
    aggregate(run_root, run_ids)
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"Selected runs cannot be compared: {error}. manifest was left unchanged.")

manifest = args.manifest
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(run_ids, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Wrote {len(run_ids)} accepted runs to {manifest}")
for run_id in run_ids:
    print(f"  {run_id}")
PYTHON
