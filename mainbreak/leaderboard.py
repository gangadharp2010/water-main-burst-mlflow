
import math

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def _finished_runs(tracking_uri, stage):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("main-break")

    if experiment is None:
        return []

    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and tags.stage = '{stage}'"
        ),
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=50000,
    )


def sweep_winner(tracking_uri: str) -> str | None:
    runs = _finished_runs(tracking_uri, "sweep")
    eligible = [
        r for r in runs
        if "cv_roc_auc" in r.data.metrics
        and math.isfinite(r.data.metrics["cv_roc_auc"])
    ]

    if not eligible:
        return None

    winner = max(
        eligible,
        key=lambda r: r.data.metrics["cv_roc_auc"],
    )
    return winner.info.run_id


def latest_model_uri(tracking_uri: str) -> str | None:
    runs = _finished_runs(tracking_uri, "training")
    if not runs:
        return None

    latest = max(
        runs,
        key=lambda r: r.info.start_time or 0,
    )
    return f"runs:/{latest.info.run_id}/model"


def runs_deeper_than(tracking_uri: str, depth: int) -> list[str]:
    runs = _finished_runs(tracking_uri, "sweep")
    eligible = []

    for run in runs:
        value = run.data.params.get("max_depth")
        if value is None:
            continue

        try:
            numeric_depth = float(value)
        except (ValueError, TypeError):
            continue

        if math.isfinite(numeric_depth) and numeric_depth > depth:
            eligible.append((numeric_depth, run.info.run_id))

    eligible.sort(key=lambda item: item[0], reverse=True)
    return [run_id for _, run_id in eligible]

