
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import skops.io as sio
from mlflow.models import infer_signature

from mainbreak import train
from mainbreak.data import load_data

ROOT = Path(train.ROOT).resolve()
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_URI = (ROOT / "mlruns").as_uri()


def configure_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name("main-break")
    if experiment is None:
        mlflow.create_experiment(
            "main-break",
            artifact_location=ARTIFACT_URI,
        )

    mlflow.set_experiment("main-break")


def main():
    configure_tracking()

    with mlflow.start_run(run_name="mainbreak-train") as run:
        train.main()

        meta = json.loads(train.META_PATH.read_text())
        model = joblib.load(train.MODEL_PATH)

        mlflow.log_param("family", meta["family"])
        for key, value in meta["params"].items():
            mlflow.log_param(key, value)
        mlflow.log_param(
            "decision_threshold", meta["decision_threshold"]
        )

        for key, value in meta["metrics"].items():
            mlflow.log_metric(key, value)

        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_version", meta["model_version"])
        mlflow.set_tag("git_dirty", str(meta["git"]["dirty"]))
        mlflow.set_tag("data_sha256", meta["data_sha256"])

        mlflow.log_artifact(str(train.META_PATH))

        X, _ = load_data()
        example = X.head(5)
        signature = infer_signature(
            example,
            model.predict_proba(example),
        )

        # Serialize the fitted model to skops bytes for type inspection.
        model_bytes = sio.dumps(model)

        trusted_types = sio.get_untrusted_types(
             data=model_bytes
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=example,
            signature=signature,
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=trusted_types,
        )

        run_id = run.info.run_id

    print(f"tracked as run : {run_id} in experiment 'main-break'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
