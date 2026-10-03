# NTU PH6151 — Bank Term Deposit Classification

A six-person data-mining project predicting whether a bank client subscribes
to a term deposit (`yes`/`no`). This repository contains code, data analysis,
experiment configurations, figures and result tables. The written report and
presentation are managed separately outside this repository.

## Repository structure and purposes

```text
PH6151_group_project/
├── data/                 # Original dataset, documentation and fixed fold assignments
│   ├── raw/
│   │   ├── bank-full.csv
│   │   └── bank-names.txt
│   ├── folds.csv
│   └── README.md
├── notebooks/            # EDA and exploratory analysis; import shared source code
│   └── 01_m1_eda.ipynb
├── src/                  # Reusable analysis code, kept as simple Python modules
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── baseline_models.py
│   ├── tree_models.py
│   ├── boosting.py
│   └── ensemble.py
├── experiments/          # Shared protocol, model configs, runner and aggregation
│   ├── protocol.json
│   ├── configs/
│   ├── run.py
│   ├── aggregate.py
│   └── final_runs.json
├── results/
│   ├── runs/             # Local per-run records; selectively commit accepted records
│   ├── figures/          # Curated EDA/model figures
│   └── tables/           # Generated model_comparison.csv and supporting tables
├── tools/                # Bash helpers for batch runs and accepted-run selection
│   ├── accept_runs.sh
│   └── run_all_configs.sh
├── tests/                # Leakage, folds, scoring, model and aggregation checks
├── CONTRIBUTIONS.md      # Contribution records for coding and non-coding work
├── AGENTS.md             # Instructions for coding assistants
├── requirements.txt      # Pinned dependencies
├── .gitattributes        # Source line endings; preserve raw dataset bytes
└── .gitignore            # Environments, caches, local runs and excluded document types
```

| Source module | Purpose |
| --- | --- |
| `data_loading.py` | Load and validate the CSV, map the target, load fixed folds |
| `preprocessing.py` | Shared features, imputation, scaling, encoding, sampling and selection |
| `evaluation.py` | Fit fresh pipelines across five folds; compute MCC and predictions |
| `baseline_models.py` | Dummy, logistic regression, linear SVM and KNN constructors |
| `tree_models.py` | Decision Tree and Random Forest constructors |
| `boosting.py` | AdaBoost and Gradient Boosting constructors |
| `ensemble.py` | Voting and stacking using shared base pipelines |

These are shared workstreams, not assignments to individual members. Not every
team member needs to write code or own a model. Record actual contributions in
`CONTRIBUTIONS.md`, including dataset research, experiment design, interpretation,
review and work on the separately managed report/presentation. Experiment
configurations contain technical settings only. The starter code and example
configurations are not completed student investigations.

## Setup and commands

The pinned environment was verified on **Python 3.14.4**. Use Python 3.14 for
matching results; other Python versions have not been verified. From the
repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Use the same environment for notebooks. VS Code or Jupyter can open the EDA
notebook; install `ipykernel` in this environment if your notebook frontend
requires it. Notebook UI tooling is separate from the pinned analysis stack.

Run an initial majority-class check and a baseline:

```bash
python -m experiments.run --config experiments/configs/baseline_dummy.json
python -m experiments.run --config experiments/configs/baseline_logistic_regression.json
```

Run any other JSON configuration in `experiments/configs/` using the same
command. Run directories are timestamped, so experiments do not overwrite
one another. `--output /path/to/runs` can direct local runs elsewhere.
Random Forest, boosting, KNN and especially stacking may take longer on the
full dataset. Fit time is recorded for each outer fold.

## Batch execution and accepted-run selection

Use Bash (Linux/macOS, WSL, or Git Bash) with your analysis environment activated.
Both helpers locate the repository root; relative arguments are root-relative.
They use `python` by default, or the executable set in `PYTHON_BIN`.

Default workflow:

```bash
bash tools/run_all_configs.sh
bash tools/accept_runs.sh
python -m experiments.aggregate
```

The batch helper runs the 12 configs sequentially, stops on failure, and forwards
optional `--protocol` and `--output` arguments to every training command.
The acceptance helper supports `--protocol`, `--runs`, and `--manifest`. It
selects the newest eligible run per experiment whose recorded protocol checksum
matches the selected file, then checks that the selected runs are comparable.
If none qualify or the set is incompatible, it preserves the existing manifest.

To run and compare all 12 models with duration included:

```bash
bash tools/run_all_configs.sh \
  --protocol experiments/with_duration/protocol.json \
  --output results/runs/with_duration
bash tools/accept_runs.sh \
  --protocol experiments/with_duration/protocol.json \
  --runs results/runs/with_duration \
  --manifest experiments/with_duration/final_runs.json
python -m experiments.aggregate \
  --runs results/runs/with_duration \
  --manifest experiments/with_duration/final_runs.json \
  --output results/tables/model_comparison_with_duration.csv
```

Individual `experiments.run` commands accept the same `--protocol`/`--output`
options; the example config is `experiments/with_duration/logistic_regression.json`.
Duration is available after the call ends, so interpret that scenario accordingly.
For accepted results, commit pending changes before training: the current
aggregator rejects `dirty_tree: true`. Review selected IDs; an older eligible
run may be chosen when a newer attempt is rejected.

## Shared methodology

- Use the immutable `data/raw/bank-full.csv`; never use `bank.csv` as a test
  set because it overlaps the full dataset.
- Map `yes=1`, `no=0`. MCC is the primary evaluation metric.
- Every reported model uses the same five stratified folds in `data/folds.csv`.
  The runner checks the data/fold hashes in `experiments/protocol.json`.
- The primary feature scenario excludes call `duration`. A with-duration
  experiment requires a reviewed protocol change and a separate comparison.
- Shared features retain literal `unknown` categories and represent the
  `pdays=-1` sentinel with a previous-contact indicator.
- Imputation, numeric scaling, one-hot encoding, oversampling and learned
  feature selection are fitted only on the current training fold. Validation
  rows keep the original class balance and are never resampled.
- All initial models use the same preprocessing; trees also receive scaled
  numeric features for consistency. Sampling and selection are explicit
  experiment variations, recorded in each configuration.
- `sampler` supports `none` (default) or `random`. `select_k` supports `all`
  (default) or a positive feature count. Random oversampling runs after
  encoding and before optional selection inside the imbalanced-learn pipeline.
- Voting fits fresh base pipelines on each outer training fold. Stacking puts
  complete base pipelines inside its five-fold inner CV, including sampling
  and selection, so inner validation data do not fit those transformations.
- Initial configurations use fixed parameters and estimator-default prediction
  rules. Automated tuning is not implemented. Add training-only nested CV
  before introducing hyperparameter/threshold search; never select settings
  using an outer validation fold.
- Report all five fold MCC values, their mean and sample standard deviation.
  Fold standard deviation is not a confidence interval.

Minimal interfaces:

```python
X, y = load_data(path)
pipeline = build_pipeline(estimator, options, seed=42)
fold_metrics, oof_predictions = evaluate_cv(pipeline, X, y, folds)
```

Model constructors return unfitted estimators. `experiments/run.py` is the
single orchestration entry point; reusable logic lives in `src/`. Import this
code in notebooks rather than copying preprocessing or evaluation loops.

## Experiments and comparison table

A configuration specifies an `experiment_id`, `model`, model `params`
and shared `preprocessing` options. For example:

```json
{
  "experiment_id": "logistic_random_select20",
  "model": "logistic_regression",
  "params": {"class_weight": null},
  "preprocessing": {"sampler": "random", "select_k": 20}
}
```

The seed and feature-availability setting belong to the shared protocol.
Each run produces:

```text
results/runs/<experiment_id>__<UTC_timestamp>__<commit>/
├── config.json
├── protocol.json
├── metadata.json
├── fold_metrics.csv
└── oof_predictions.csv
```

Metadata captures the source commit, dirty-tree state, Python/dependency
versions, seed, and data/fold/protocol hashes. Explore locally, then commit
source/config changes and rerun from a clean working tree before accepting
final results. Runs made from uncommitted source are explicitly marked and
rejected by the final aggregator.

Use `bash tools/accept_runs.sh` to select eligible runs automatically, or add
the chosen directory names to `experiments/final_runs.json` manually (a JSON list),
then run:

```bash
python -m experiments.aggregate
```

The aggregator requires five complete MCC records, one accepted run per
experiment ID, committed clean source, and matching protocol/data/folds and
environment. It creates `results/tables/model_comparison.csv`, sorted by mean
MCC, with model, sampling/selection options, each fold score, mean,
standard deviation, timing and provenance. Different feature scenarios or
protocols need separate comparisons. The initial table contains headers only;
no benchmark results are claimed by the scaffold.

## Git workflow and what to commit

Develop on short-lived branches and merge through a peer-reviewed PR:

| Workstream | Branch pattern |
| --- | --- |
| EDA | `feat/eda/<topic>` |
| Preprocessing/features | `feat/preprocessing/<topic>` |
| Baselines | `feat/baselines/<topic>` |
| Trees/forests | `feat/trees/<topic>` |
| Boosting/imbalance | `feat/boosting/<topic>` |
| Ensemble/CV/integration | `feat/integration/<topic>` |

Coordinate edits to shared preprocessing/evaluation code, notebooks and
experiment configurations with the people working on them. Keep `main` runnable.
Enable branch protection on GitHub if your account/repository supports it.

Commit raw data/documentation, the fold manifest, source, tests, configurations,
curated notebooks/figures/tables and contribution records. Exclude environments,
caches, notebook checkpoints, fitted models and exploratory run outputs.
Run folders are ignored by default: use `git add -f` on only the accepted small
`config.json`, `protocol.json`, `metadata.json` and `fold_metrics.csv` records.
Keep bulky out-of-fold predictions local unless the team explicitly needs them.
Do not add report, LaTeX, presentation or PPT files.

Naming examples:

- Notebook: `03_baselines.ipynb`.
- Python module: `tree_models.py`.
- Configuration/experiment ID: `random_forest__balanced`.
- Figure: `random_forest__no_duration__confusion_matrix.png`.
- Table: `model_comparison.csv`.

## Dataset provenance and submission

See `data/README.md` and `data/raw/bank-names.txt` for provenance and citation.
The upstream clone was converted to normal project files; its Git history and
unused reference files are preserved in a backup outside this repository.

For the coding submission, choose accepted results, verify tests and documented
commands from a fresh clone/environment, then archive tracked project files.
Exclude `.git`, environments, caches and exploratory outputs. Reports and
presentation artifacts are submitted from their separate locations.
