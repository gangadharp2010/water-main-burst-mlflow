
import json
import mlflow

from mainbreak.track import configure_tracking
from mainbreak.data import ROOT, load_split
from mainbreak.pipeline import build_pipeline
from mainbreak.evaluate import cv_roc_auc, holdout_roc_auc


def main():
    configure_tracking()

    X_train, X_test, y_train, y_test = load_split()
    configs = json.loads(
        (ROOT / "configs" / "sweep.json").read_text()
    )

    for config in configs:
        name = config["name"]
        family = config["family"]
        params = config["params"]

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("stage", "sweep")
            mlflow.log_param("family", family)
            mlflow.log_params(params)

            pipeline = build_pipeline(family, params)
            cv_score = cv_roc_auc(pipeline, X_train, y_train)
            mlflow.log_metric("cv_roc_auc", cv_score)

            pipeline.fit(X_train, y_train)
            test_score = holdout_roc_auc(
                pipeline, X_test, y_test
            )
            mlflow.log_metric("test_roc_auc", test_score)

            if family == "hgb":
                scores = pipeline.named_steps["model"].validation_score_
                for step, score in enumerate(scores):
                    mlflow.log_metric(
                        "validation_roc_auc",
                        float(score),
                        step=step,
                    )

            print(f"{name:<20} cv {cv_score:.4f}")

    print(f"logged {len(configs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

