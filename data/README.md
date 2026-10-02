# Dataset and validation folds

- Source clone: https://github.com/z-o-e/bank_data_analysis
- Source revision: `f05896b87019e7001e6c78948e59d072dde150b8`.
- File: `raw/bank-full.csv`, semicolon-delimited; original bytes preserved.
- SHA-256: `ada45d7c04ac65e4bf1a0a27996e73510848c28095dd40df9f5d4c153af67931`.
- 45,211 rows, 16 original predictors and binary target `y`.
- Targets: 39,922 `no` and 5,289 `yes` (11.698% positive).
- The loader maps `no=0`, `yes=1`; row IDs are zero-based original row positions.
- `raw/bank-names.txt` is the original data dictionary and citation request.

Citation: S. Moro, R. Laureano and P. Cortez (2011), *Using Data Mining for
Bank Direct Marketing: An Application of the CRISP-DM Methodology*,
Proceedings of ESM'2011, pp. 117–121. Retain the supplied citation and check
redistribution terms before distributing the dataset beyond course use.

## Interpretation

Literal `unknown` categories are retained as explicit levels. `pdays=-1`
means no previous contact: shared feature engineering adds a
`previously_contacted` indicator and substitutes zero for that sentinel.
The primary protocol excludes `duration`, which is unavailable before call
completion. Other contact/campaign features mean this is not automatically
a prediction made before the first call; define the prediction time when
interpreting your analysis.

The original documentation describes date-ordered observations. Shuffled
stratified CV measures performance on mixed historical records, not future
campaign forecasting. There is no client identifier for group-aware splits.
No empty cells or exact duplicate rows were found during initialization.

## Shared folds

`folds.csv` assigns every original row to one validation fold (1–5), generated
with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
Fold SHA-256: `fa8b296de52731a39a7a256b27359c6d87a636ba6b2bbfc2ef4f8c291b006555`.
The runner verifies both checksums against `experiments/protocol.json` before
fitting. Do not reorder the raw data or edit this manifest independently.

The supplied `bank.csv` is a subset of the full dataset: all 4,521 rows occur
in `bank-full.csv`. It must not be used as an independent test set. That file
and the original R script are preserved in the migration backup outside this
repository. No upstream Git metadata was imported.
