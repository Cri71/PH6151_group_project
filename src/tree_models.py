"""Decision Tree and Random Forest estimator constructors."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def make_model(name, params, seed=42):
    if name == "decision_tree":
        return DecisionTreeClassifier(**{"random_state": seed, **params})
    if name == "random_forest":
        return RandomForestClassifier(**{"n_estimators": 100, "n_jobs": 1, "random_state": seed, **params})
    raise ValueError(f"Unknown tree model: {name}")
