"""The three model families this project compares, each as one scikit-learn pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FAMILIES = ("logreg", "rf", "hgb")


def _preprocess(scale_numbers):
    numeric = StandardScaler() if scale_numbers else "passthrough"
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_pipeline(family, params=None):
    """An unfitted pipeline for one family, with `params` passed to its estimator.

    Random seeds are fixed here, so the same family and params always fit the
    same model.
    """
    params = dict(params or {})
    if family == "logreg":
        model = LogisticRegression(max_iter=2000, **params)
        scale = True
    elif family == "rf":
        model = RandomForestClassifier(random_state=42, **params)
        scale = False
    elif family == "hgb":
        model = HistGradientBoostingClassifier(random_state=42, **params)
        scale = False
    else:
        raise ValueError("unknown family %r, expected one of %s" % (family, ", ".join(FAMILIES)))
    return Pipeline([("preprocess", _preprocess(scale)), ("model", model)])
