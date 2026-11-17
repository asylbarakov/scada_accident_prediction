"""Read-only CARE H7 dashboard data and point-in-time inference verification."""

from dataclasses import dataclass
import json
from pathlib import Path
from pickle import UnpicklingError

import joblib
import numpy as np
import pandas as pd

MODELS = {"logistic": "Logistic Regression", "lightgbm": "LightGBM"}
KEYS = ["turbine_id", "source_event", "timestamp"]
PREDICTION_COLUMNS = KEYS + ["score", "alert"]


class ArtifactError(ValueError):
    """A required artifact is missing, malformed, or inconsistent with the run."""


def read_artifact(path, reader, *args, **kwargs):
    try:
        return reader(path, *args, **kwargs)
    except (OSError, ValueError, KeyError, ImportError, EOFError, UnpicklingError) as exc:
        raise ArtifactError(f"{path} — {exc}") from exc


def read_json(path):
    return read_artifact(path, lambda file: json.loads(file.read_text(encoding="utf-8")))


def timestamps(frame, path):
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise ArtifactError(f"{path} — обнаружены некорректные даты.")
    return frame


@dataclass(frozen=True)
class Event:
    id: int
    turbine: str
    fold: str
    failure_at: pd.Timestamp | None
    description: str | None
    related_anchor: int | None


@dataclass
class Experiment:
    run: Path
    sensors: dict
    events: dict[int, Event]
    predictions: pd.DataFrame
    models: dict
    inference: dict
    cadence_minutes: int
    availability_minutes: int


def read_predictions(path, threshold):
    frame = timestamps(read_artifact(path, pd.read_parquet, columns=PREDICTION_COLUMNS), path)
    frame["turbine_id"] = frame["turbine_id"].astype(str)
    if (frame.duplicated(KEYS).any() or frame["source_event"].isna().any()
            or not np.isfinite(frame["score"]).all() or not frame["score"].between(0, 1).all()
            or not pd.api.types.is_bool_dtype(frame["alert"]) or frame["alert"].isna().any()
            or not frame["alert"].eq(frame["score"].ge(threshold)).all()):
        raise ArtifactError(f"{path} — прогнозы, ключи или предупреждения не согласованы с порогом.")
    return frame.sort_values(KEYS).reset_index(drop=True)


def load_experiment(root):
    run = Path(root) / "training"
    config = read_json(run / "config.json")
    report = read_json(run / "metrics.json")
    coverage = read_json(run / "preparation_coverage.json")
    try:
        horizon = report["horizons"]["h7"]
        if config["horizons_days"] != [7] or report["mode"] != "care_h7" or horizon["status"] != "trained":
            raise ArtifactError(f"{run} — требуется завершённый эксперимент CARE H7.")
        fold_by_asset = {str(asset): fold for fold, summary in horizon["coverage"].items()
                         for asset in summary["turbines"]}
        anchors = {event: item["anchor_event"] for item in coverage["fault_groups"] for event in item["events"]}
        events = {}
        for item in coverage["event_summaries"]:
            event, asset = int(item["event"]), str(item["asset"])
            anchor = anchors.get(event)
            fault = pd.Timestamp(item["fault_or_normal_end"]) if anchor == event else None
            if fault is not None:
                fault = pd.to_datetime(fault, utc=True)
            events[event] = Event(event, asset, fold_by_asset[asset], fault, item.get("description"),
                                  int(anchor) if anchor is not None and anchor != event else None)

        inference = {}
        for model in MODELS:
            path = run / "h7" / model / "inference.json"
            entry = read_json(path)
            if (entry["threshold"] != horizon["models"][model]["threshold"] or entry["horizon_days"] != 7
                    or entry["score_semantics"] != "uncalibrated_failure_score"):
                raise ArtifactError(f"{path} — конфигурация прогнозов расходится с отчётом.")
            inference[model] = entry
        if inference["logistic"]["feature_names"] != inference["lightgbm"]["feature_names"]:
            raise ArtifactError(f"{run} — у моделей разные карты признаков.")

        parts = []
        for fold in ("validation", "test"):
            frames = []
            for model in MODELS:
                path = run / "h7" / model / f"{fold}_predictions.parquet"
                frame = read_predictions(path, inference[model]["threshold"])
                for asset, event in frame[["turbine_id", "source_event"]].drop_duplicates().itertuples(index=False):
                    if events[event].turbine != asset or events[event].fold != fold:
                        raise ArtifactError(f"{path} — событие не принадлежит ожидаемой выборке.")
                frames.append(frame.assign(model=model, fold=fold))
            if not frames[0][KEYS].equals(frames[1][KEYS]):
                raise ArtifactError(f"{run} — ключи прогнозов моделей в {fold} различаются.")
            parts.extend(frames)
        return Experiment(run, config["sensors"], events, pd.concat(parts, ignore_index=True), horizon["models"],
                          inference, config["prediction_every_minutes"],
                          config["cadence_minutes"] * config["measurement_delay_steps"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ArtifactError(f"{run} — неполное описание запуска CARE H7 ({exc}).") from exc


@dataclass
class Telemetry:
    frame: pd.DataFrame
    stride_minutes: int
    available_after_minutes: int
    from_raw: bool


def load_telemetry(experiment, raw_dir, event_id):
    event = experiment.events[event_id]
    path = Path(raw_dir) / "datasets" / f"{event_id}.csv"
    sensors = {item["column"]: key for key, item in experiment.sensors.items()}
    if path.is_file():
        columns = ["time_stamp", "asset_id", "train_test", *sensors]
        frame = read_artifact(path, pd.read_csv, sep=";", usecols=columns, dtype={"asset_id": str})
        if set(frame["asset_id"]) != {event.turbine}:
            raise ArtifactError(f"{path} — турбина в файле не совпадает с метаданными.")
        frame = frame.loc[frame["train_test"].eq("prediction"), ["time_stamp", *sensors]]
        frame = frame.rename(columns={"time_stamp": "timestamp", **sensors})
        stride, delay = 10, experiment.availability_minutes
        from_raw = True
    else:
        history = experiment.run / "h7" / "lightgbm" / "risk_sensor_history.parquet"
        if event.fold != "test" or not history.is_file():
            raise ArtifactError(f"{path} — локальный файл SCADA отсутствует для этого события.")
        source = read_artifact(history, pd.read_parquet)
        fields = ["timestamp", *(key for key in experiment.sensors if key in source)]
        frame = source.loc[source["source_event"].eq(event_id), fields].copy()
        stride, delay = experiment.cadence_minutes, experiment.availability_minutes
        from_raw = False
    frame = timestamps(frame, path if from_raw else history)
    if frame.empty or frame["timestamp"].duplicated().any():
        raise ArtifactError(f"{path} — пустой ряд SCADA или повторные отметки времени.")
    for sensor in set(frame).difference(["timestamp"]):
        numeric = pd.to_numeric(frame[sensor], errors="coerce")
        frame[sensor] = numeric.where(np.isfinite(numeric))
    return Telemetry(frame.sort_values("timestamp").reset_index(drop=True), stride, delay, from_raw)


def visible_rows(frame, as_of=None, delay_minutes=0):
    if as_of is None:
        return frame.copy()
    cutoff = pd.to_datetime(as_of, utc=True) - pd.Timedelta(minutes=delay_minutes)
    return frame.loc[frame["timestamp"].le(cutoff)].copy()


def visible_fault(event, as_of):
    fault = event.failure_at
    if as_of is not None and (pd.isna(as_of) or (fault is not None and fault > pd.to_datetime(as_of, utc=True))):
        return None
    return fault


def alert_summary(frame, experiment, as_of=None):
    frame = visible_rows(frame, as_of)
    summary = {}
    for model in MODELS:
        series = frame.loc[frame["model"].eq(model)].sort_values("timestamp")
        alerts = series.loc[series["alert"]]
        latest = series.iloc[-1] if len(series) else None
        stale = latest is not None and as_of is not None and pd.to_datetime(as_of, utc=True) - latest["timestamp"] >= (
            pd.Timedelta(minutes=experiment.cadence_minutes))
        summary[model] = {
            "first_alert": alerts["timestamp"].min() if len(alerts) else None,
            "latest": latest, "stale": stale, "disabled": experiment.inference[model]["threshold"] > 1,
        }
    return summary


def chart_grid(frame, column, cadence_minutes):
    """Insert missing bins so Plotly never bridges a telemetry or forecast gap."""
    series = frame.set_index("timestamp")[column].sort_index()
    if series.empty:
        return series
    index = pd.date_range(series.index[0], series.index[-1], freq=pd.Timedelta(minutes=cadence_minutes))
    return series.reindex(index.union(series.index))


def load_pipelines(experiment):
    return {model: read_artifact(experiment.run / "h7" / model / "pipeline.joblib", joblib.load)
            for model in MODELS}


def load_features(path, names):
    frame = read_artifact(path, pd.read_parquet, columns=KEYS + list(names))
    return timestamps(frame, path)


def explain_point(features, pipelines, experiment, event, at, predictions):
    """Reproduce the archived scores and explain one causal feature row."""
    key = features["turbine_id"].astype(str).eq(event.turbine) & features["source_event"].eq(event.id)
    row = features.loc[key & features["timestamp"].eq(at)]
    names = experiment.inference["lightgbm"]["feature_names"]
    saved = predictions.loc[predictions["source_event"].eq(event.id) & predictions["timestamp"].eq(at)]
    if len(row) != 1 or set(saved["model"]) != set(MODELS):
        raise ArtifactError("Для выбранного архивного прогноза нет единственной строки признаков.")
    scores = {}
    for model, pipeline in pipelines.items():
        score = float(pipeline.predict_proba(row[names])[0, 1])
        if not np.isclose(score, saved.loc[saved["model"].eq(model), "score"].iloc[0], rtol=0, atol=1e-10):
            raise ArtifactError(f"{MODELS[model]} — кеш и сохранённый прогноз расходятся.")
        scores[model] = score
    tree = pipelines["lightgbm"]
    transformed = tree.named_steps["impute"].transform(row[names])
    contributions = tree.named_steps["model"].booster_.predict(
        transformed, pred_contrib=True, validate_features=True, num_threads=1)[0]
    return pd.DataFrame({"feature": names, "shap_log_odds": contributions[:-1]}), float(contributions[-1]), scores
