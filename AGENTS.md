# PH6151 coding and data-analysis repository

- Keep this repository focused on code, data analysis, configurations and results.
  Written reports, LaTeX and presentation files are managed outside it.
- Read README.md and relevant modules before editing. Change only requested files.
- Preserve data/raw bytes; use the existing checksum-verified five-fold manifest.
- All reported models use src/evaluation.py and shared preprocessing pipelines.
- Fit preprocessing, feature selection and sampling only inside training folds.
  Stacking requires full base pipelines inside its inner cross-validation.
- Do not tune hyperparameters or thresholds on outer validation folds. Add nested
  training-fold tuning before introducing automated search.
- Do not duplicate reusable code in notebooks. Use imports from src.
- Do not claim student experiments or contributions are complete from scaffold code.
- Run `python -m pytest -q` after changes to preprocessing or evaluation.
- Use feature branches and PR review; do not commit or push without authorization.

## Project learnings

- Keep report/presentation files outside this coding-only repository.
- Keep model configurations technical; record coding and non-coding contributions
  separately without assigning model ownership to individual members.
- For scripted migration guards, inspect `git status --porcelain=v1`; its nested
  modification notation differs from `git status --short`.
