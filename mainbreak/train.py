"""Train the burst model and write it, with its metadata, into artifacts/.

    python3 -m mainbreak.train

Fits the chosen configuration on the training split, picks a decision
threshold on cross validated predictions, scores the holdout, and writes two
files: artifacts/model.joblib and artifacts/model_meta.json. Running it again
overwrites both.
"""
import datetime as dt
import json
import subprocess
import sys

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

from .data import ROOT, data_sha256, load_split
from .evaluate import CV, holdout_roc_auc
from .pipeline import build_pipeline

ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
META_PATH = ARTIFACTS / "model_meta.json"

FAMILY = "hgb"
PARAMS = {
    "learning_rate": 0.04,
    "max_iter": 300,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
}

# The threshold is the highest cut on cross validated predictions that still
# flags three bursts in every four.
RECALL_TARGET = 0.75


def git_provenance():
    """The commit HEAD points at, and whether the working tree differs from it."""
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20
        )

    try:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    if head.returncode != 0 or status.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def pick_threshold(y_true, proba):
    """Highest cut on the 0.01 grid whose recall reaches RECALL_TARGET."""
    for cut in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if recall_score(y_true, proba >= cut) >= RECALL_TARGET:
            return float(cut)
    return 0.01


def main():
    X_train, X_test, y_train, y_test = load_split()

    oof = cross_val_predict(
        build_pipeline(FAMILY, PARAMS), X_train, y_train, cv=CV, method="predict_proba"
    )[:, 1]
    threshold = pick_threshold(y_train, oof)

    pipeline = build_pipeline(FAMILY, PARAMS).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    flagged = proba >= threshold

    trained_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    meta = {
        "model_version": "%s-%s-1" % (trained_at.strftime("%Y%m%d"), FAMILY),
        "trained_at": trained_at.isoformat(),
        "family": FAMILY,
        "params": PARAMS,
        "decision_threshold": threshold,
        "metrics": {
            "test_roc_auc": round(holdout_roc_auc(pipeline, X_test, y_test), 4),
            "test_average_precision": round(float(average_precision_score(y_test, proba)), 4),
            "test_recall_at_threshold": round(float(recall_score(y_test, flagged)), 4),
            "test_precision_at_threshold": round(float(precision_score(y_test, flagged)), 4),
        },
        "sklearn_version": sklearn.__version__,
        "data_sha256": data_sha256(),
        "rows": {"train": int(len(X_train)), "test": int(len(X_test))},
        "git": git_provenance(),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")

    m = meta["metrics"]
    print("family         : %s" % FAMILY)
    print("train / test   : %d / %d" % (len(X_train), len(X_test)))
    print("threshold      : %.2f" % threshold)
    print("test ROC AUC   : %.4f" % m["test_roc_auc"])
    print("recall at cut  : %.4f" % m["test_recall_at_threshold"])
    print("precision      : %.4f" % m["test_precision_at_threshold"])
    commit = meta["git"]["commit"]
    print("built from     : %s%s" % (commit[:8] if commit else "no commit",
                                      " (uncommitted changes)" if meta["git"]["dirty"] else ""))
    print("wrote %s and %s" % (MODEL_PATH.relative_to(ROOT), META_PATH.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
