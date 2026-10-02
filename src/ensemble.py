"""Voting and stacking with fold-local preprocessing in each base learner."""
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.baseline_models import make_model as make_baseline
from src.preprocessing import build_pipeline
from src.tree_models import make_model as make_tree


def make_model(name, params, options, seed=42):
    # Full pipelines inside the ensemble also protect stacking's inner folds.
    members = [
        ("logistic", build_pipeline(make_baseline("logistic_regression", {}, seed), options, seed)),
        ("forest", build_pipeline(make_tree("random_forest", {}, seed), options, seed)),
    ]
    if name == "voting":
        return VotingClassifier(**{"estimators": members, "voting": "soft", "n_jobs": 1, **params})
    if name == "stacking":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return StackingClassifier(**{
            "estimators": members,
            "final_estimator": LogisticRegression(max_iter=2000, random_state=seed),
            "cv": cv, "n_jobs": 1, **params,
        })
    raise ValueError(f"Unknown ensemble: {name}")
