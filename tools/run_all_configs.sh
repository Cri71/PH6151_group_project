#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

shopt -s nullglob
configs=(experiments/configs/*.json)
if (( ${#configs[@]} == 0 )); then
    printf 'No experiment configurations found.\n' >&2
    exit 1
fi

printf 'Running %s configurations sequentially.\n' "${#configs[@]}"
for config in "${configs[@]}"; do
    printf 'Running %s\n' "$config"
    "${PYTHON_BIN:-python}" -m experiments.run --config "$config"
done
printf 'All configurations completed. Run tools/accept_runs.sh next.\n'
