"""The two scores every run in this project is judged by."""
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_roc_auc(pipeline, X_train, y_train):
    """Mean ROC AUC over the fixed five folds of the training split.

    Fits copies of `pipeline`. The pipeline you pass in is left unfitted.
    """
    scores = cross_val_score(clone(pipeline), X_train, y_train, cv=CV, scoring="roc_auc")
    return float(scores.mean())


def holdout_roc_auc(fitted_pipeline, X_test, y_test):
    """ROC AUC of an already fitted pipeline on the holdout."""
    return float(roc_auc_score(y_test, fitted_pipeline.predict_proba(X_test)[:, 1]))
