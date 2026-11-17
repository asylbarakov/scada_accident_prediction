"""Purged splits, CPU model fitting, alarm metrics and TreeSHAP exports."""

from __future__ import annotations

from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .edp import KEY
from .preparation import check_cache, feature_metadata, history_span, write_json

FOLDS = ("train", "validation", "test")


def split_horizon(frame, days, config):
    """Each fold must contain both the feature history and the complete target window."""
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate turbine and timestamp key in prepared data")
    boundaries = pd.to_datetime(config["split_boundaries_utc"], utc=True)
    if len(boundaries) != 4 or boundaries.hasnans or not boundaries.is_unique or not boundaries.is_monotonic_increasing:
        raise ValueError("Expected four strictly increasing UTC split boundaries")
    stride = config["prediction_every_minutes"]
    if not isinstance(stride, int) or stride <= 0 or stride % config["cadence_minutes"]:
        raise ValueError("Prediction stride must be a positive multiple of the sampling interval")
    timestamp = frame["timestamp"]
    on_schedule = timestamp.eq(timestamp.dt.floor(pd.Timedelta(minutes=stride)))
    target = f"h{days}_label"
    event = f"h{days}_episode"
    history = history_span(config)
    forward = pd.Timedelta(days=days)
    holdout = set(config.get("holdout_turbines", []))
    if holdout and not holdout <= set(frame["turbine_id"]):
        raise ValueError("Configured holdout turbines are absent from SCADA")
    splits, coverage = {}, {}
    for fold, start, end in zip(FOLDS, boundaries[:-1], boundaries[1:]):
        in_fold = timestamp.ge(start) & timestamp.lt(end)
        if holdout:
            on_held_out_turbine = frame["turbine_id"].isin(holdout)
            in_fold &= on_held_out_turbine if fold == "test" else ~on_held_out_turbine
        scheduled = in_fold & on_schedule
        safe = scheduled & timestamp.sub(history).ge(start) & timestamp.add(forward).lt(end)
        candidate = safe & frame["history_ready"] & frame["reviewed_history_clear"]
        subset = frame.loc[candidate & frame[target].notna()]
        splits[fold] = subset
        positive = int(subset[target].eq(1).sum())
        coverage[fold] = {
            "start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat(),
            "purged_rows": int(scheduled.sum() - safe.sum()),
            "missing_feature_history": int(safe.sum() - candidate.sum()),
            "censored_rows": int((candidate & frame[target].isna()).sum()),
            "positive": positive, "negative": len(subset) - positive,
            "positive_episode_ids": sorted(subset[event].dropna().unique().tolist()),
            "turbines": sorted(subset["turbine_id"].unique().tolist()),
        }
    for earlier, later in combinations(FOLDS, 2):
        if set(coverage[earlier]["positive_episode_ids"]) & set(coverage[later]["positive_episode_ids"]):
            raise ValueError(f"Independent episode crosses {earlier} and {later}")
        if not splits[earlier].empty and not splits[later].empty:
            support_end = splits[earlier]["timestamp"].max() + forward
            support_start = splits[later]["timestamp"].min() - history
            if support_end >= support_start:
                raise ValueError(f"Feature or label support crosses {earlier} and {later}")
    if holdout and (set(coverage["train"]["turbines"]) | set(coverage["validation"]["turbines"])) & holdout:
        raise ValueError("Holdout turbine leaked into development folds")
    return splits, coverage


def alarm_metrics(predictions, threshold, cadence_minutes):
    """Count contiguous warning episodes on evaluable timestamps."""
    groups = ["turbine_id"] + (["source_event"] if "source_event" in predictions else [])
    predictions = predictions.sort_values([*groups, "timestamp"])
    y = predictions["label"].to_numpy(dtype=int)
    alert = predictions["score"].to_numpy() >= threshold
    same_series = predictions[groups].eq(predictions[groups].shift()).fillna(False).all(axis=1)
    continuous = (same_series
                  & predictions["timestamp"].diff().eq(pd.Timedelta(minutes=cadence_minutes))).to_numpy(dtype=bool)
    starts = alert & ~(np.r_[False, alert[:-1]] & continuous)
    episode_no = starts.cumsum()
    total_alarms = int(starts.sum())
    true_alarm_count = len(np.unique(episode_no[alert & (y == 1)]))
    false_alarms = total_alarms - true_alarm_count
    positive = predictions.loc[predictions["label"].eq(1)]
    first = predictions.loc[alert & (y == 1)].groupby("episode_id")["timestamp"].min()
    events = positive.groupby("episode_id", as_index=False).agg(
        turbine_id=("turbine_id", "first"), failure_at=("failure_at", "first"))
    events["first_alert_at"] = events["episode_id"].map(first)
    events["lead_time_hours"] = (events["failure_at"] - events["first_alert_at"]).dt.total_seconds() / 3600
    lead_times = events["lead_time_hours"].dropna()
    exposure = int((y == 0).sum()) * cadence_minutes / (60 * 24 * (365.25 / 12))
    metrics = {
        "eligible_rows": len(y), "positive_rows": int(y.sum()),
        "event_recall": len(lead_times) / len(events) if len(events) else None,
        "detected_events": len(lead_times), "eligible_events": len(events),
        "warning_episodes": total_alarms, "false_warning_episodes": false_alarms,
        "negative_exposure_turbine_months": exposure,
        "false_alarms_per_turbine_month": false_alarms / exposure if exposure else None,
        "median_lead_time_hours": float(lead_times.median()) if len(lead_times) else None,
    }
    return metrics, events


def choose_threshold(validation, config):
    options = np.unique(np.r_[np.quantile(validation["score"], np.linspace(0, 1, 101)),
                             np.nextafter(1.0, np.inf)])
    acceptable = []
    for threshold in options:
        metrics, _ = alarm_metrics(validation, threshold, config["prediction_every_minutes"])
        alarms = metrics["false_alarms_per_turbine_month"]
        if alarms <= config["max_validation_false_alarms_per_turbine_month"]:
            acceptable.append((metrics["event_recall"], -alarms, float(threshold)))
    return max(acceptable)[2]


def build_pipelines(config):
    def imputer():
        return SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")

    lgbm = config["lightgbm"]
    if (config["mode"] == "synthetic_smoke"
            and not 1 <= lgbm["n_estimators"] * len(config["horizons_days"]) <= 20):
        raise ValueError("Synthetic smoke is limited to 20 LightGBM trees across horizons")
    return {
        "logistic": Pipeline([
            ("impute", imputer()), ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config["seed"])),
        ]),
        "lightgbm": Pipeline([
            ("impute", imputer()),
            ("model", LGBMClassifier(objective="binary", device_type="cpu", deterministic=True,
                                     force_col_wise=True, random_state=config["seed"], n_jobs=config["threads"],
                                     verbosity=-1, **lgbm)),
        ]),
    }


def predictions_for(pipeline, frame, columns, days):
    prefix = f"h{days}"
    pred = frame[KEY + [f"{prefix}_label", f"{prefix}_episode", f"{prefix}_failure_at"]].copy()
    pred.columns = KEY + ["label", "episode_id", "failure_at"]
    if "source_event" in frame:
        pred["source_event"] = frame["source_event"]
    pred["score"] = pipeline.predict_proba(frame[columns])[:, 1]
    return pred


def explain_lightgbm(pipeline, test, predictions, metadata, raw_scada, output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    booster = pipeline.named_steps["model"].booster_
    columns = list(metadata)
    sample_index = np.linspace(0, len(test) - 1, min(len(test), config["shap_rows"]), dtype=int)
    local_index = np.argsort(-predictions["score"].to_numpy(), kind="stable")[:config["local_shap_rows"]]
    selected = np.unique(np.r_[sample_index, local_index])
    x = pipeline.named_steps["impute"].transform(test.iloc[selected][columns])
    values = booster.predict(x, pred_contrib=True, validate_features=True, num_threads=config["threads"])
    margin = booster.predict(x, raw_score=True, validate_features=True, num_threads=config["threads"])
    np.testing.assert_allclose(values.sum(axis=1), margin, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(expit(margin), predictions.iloc[selected]["score"], rtol=1e-8, atol=1e-8)
    global_values = values[np.searchsorted(selected, sample_index), :-1]
    gains = booster.feature_importance(importance_type="gain")
    importance = pd.DataFrame([{"feature": feature, **metadata[feature], "gain": float(gain),
                                "mean_abs_shap_log_odds": float(np.abs(global_values[:, i]).mean())}
                               for i, (feature, gain) in enumerate(zip(columns, gains, strict=True))])
    importance.sort_values("mean_abs_shap_log_odds", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    sensors = []
    for sensor, features in importance.groupby("sensor", sort=True):
        channel = features.iloc[0]
        combined = global_values[:, features.index].sum(axis=1)
        sensors.append({"sensor": sensor, "source_column": channel["source_column"],
                        "physical_measurement": channel["physical_measurement"], "unit": channel["sensor_unit"],
                        "gain": float(features["gain"].sum()), "mean_abs_grouped_shap_log_odds": float(np.abs(combined).mean()),
                        "sum_feature_mean_abs_shap_log_odds": float(features["mean_abs_shap_log_odds"].sum())})
    sensor_importance = pd.DataFrame(sensors).sort_values("mean_abs_grouped_shap_log_odds", ascending=False)
    sensor_importance.to_csv(output / "sensor_importance.csv", index=False)
    top = sensor_importance["sensor"].head(config["top_sensor_count"]).tolist()
    history = predictions.merge(raw_scada[KEY + top], on=KEY, how="left", validate="one_to_one")
    history.to_parquet(output / "risk_sensor_history.parquet", index=False)

    local = []
    for index in local_index:
        row = predictions.iloc[index]
        pos = np.searchsorted(selected, index)
        for number, feature in enumerate(columns):
            local.append({**{key: row[key] for key in KEY + ["score", "label"]},
                          "feature": feature, **metadata[feature], "model_input_value": x.iloc[pos, number],
                          "shap_log_odds": values[pos, number], "base_log_odds": values[pos, -1],
                          "raw_log_odds": margin[pos]})
    pd.DataFrame(local).to_csv(output / "local_shap.csv", index=False)
    ranked = importance.loc[importance["mean_abs_shap_log_odds"].gt(0)].nlargest(14, "mean_abs_shap_log_odds")
    if ranked.empty:
        ranked = importance.head(1)
    ranked = ranked.set_index("feature")
    ax = ranked["mean_abs_shap_log_odds"].sort_values().plot.barh(figsize=(9, 5))
    ax.set_xlabel("Mean absolute TreeSHAP contribution (log-odds)")
    ax.figure.savefig(output / "shap_global.png", dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    first = np.searchsorted(selected, local_index[0])
    attribution = pd.Series(values[first, :-1], index=columns)
    nonzero = attribution.loc[attribution.ne(0)]
    largest = nonzero.abs().nlargest(11).index
    displayed = attribution.loc[largest]
    if displayed.empty:
        displayed = pd.Series({"No feature contribution": 0.0})
    elif len(displayed) < len(nonzero):
        displayed = pd.concat([displayed, pd.Series({"Remaining features": attribution.drop(largest).sum()})])
    displayed = displayed.iloc[::-1]
    ax = displayed.plot.barh(figsize=(9, 5), color=np.where(displayed >= 0, "#b13e3e", "#356c92"))
    ax.set_xlabel("Signed TreeSHAP contribution (log-odds)")
    ax.set_title(f"Base {values[first, -1]:.3f} → predicted margin {margin[first]:.3f}")
    ax.figure.savefig(output / "shap_local.png", dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    gain_rank = importance.loc[importance["gain"].gt(0)].nlargest(14, "gain")
    if gain_rank.empty:
        gain_rank = importance.head(1)
    ax = gain_rank.set_index("feature")["gain"].sort_values().plot.barh(figsize=(9, 5))
    ax.set_xlabel("LightGBM split gain")
    ax.figure.savefig(output / "gain.png", dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    return {"shap_rows": len(sample_index), "local_rows": len(local_index),
            "shap_max_abs_margin_error": float(np.max(np.abs(values.sum(axis=1) - margin))),
            "actual_trees": booster.num_trees()}


def train(config, cache, output, *, audit_only=False):
    """Check fold readiness and optionally train eligible horizons."""
    output, cache = Path(output), Path(cache)
    manifest = check_cache(config, cache)
    with (cache / "feature_map.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata != feature_metadata(config):
        raise ValueError("Cached features differ from configuration")
    if config["mode"] == "synthetic_smoke" and manifest["signals"]["rows"] > 5000:
        raise ValueError("Synthetic smoke is limited to 5000 SCADA rows")
    if config["max_validation_false_alarms_per_turbine_month"] < 0:
        raise ValueError("Validation false alarm budget must be nonnegative")
    if any(config[key] <= 0 for key in ("shap_rows", "local_shap_rows", "top_sensor_count", "threads", "min_rows_per_class")):
        raise ValueError("SHAP, sensor export and CPU thread counts must be positive")
    audit_columns = KEY + ["history_ready", "reviewed_history_clear"] + [
        f"h{days}_{name}" for days in config["horizons_days"] for name in ("label", "episode")
    ]
    is_care = config["mode"] == "care_h7"
    if is_care:
        audit_columns.remove("reviewed_history_clear")
        audit_columns.extend(["source_event", "lineage_group"])
    dataset = pd.read_parquet(cache / "features.parquet", columns=audit_columns if audit_only else None)
    if config["mode"] == "synthetic_smoke" and len(dataset) > 5000:
        raise ValueError("Synthetic smoke is limited to 5000 prepared rows")
    if any(config["min_positive_episodes"][fold] < 1 for fold in FOLDS):
        raise ValueError("Each fold must require at least one confirmed failure episode")
    report = {"mode": config["mode"], "dataset_id": manifest.get("dataset_id"), "audit_only": audit_only,
              "score_semantics": "uncalibrated_failure_score",
              "cache_signature": manifest["config_signature"], "horizons": {}}
    if is_care:
        report["evaluation_scope"] = "CARE curated event windows; negative exposure is conditional on eligible observations"
    output.mkdir(parents=True, exist_ok=False)
    write_json(config, output / "config.json")
    write_json(manifest, output / "preparation_coverage.json")
    write_json({name: version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "lightgbm", "joblib", "matplotlib")},
               output / "package_versions.json")
    metrics_rows = []
    columns = list(metadata)
    for days in config["horizons_days"]:
        if is_care:
            from .care import check_raw_fold_separation, split_care

            try:
                splits, coverage = split_care(dataset, config, manifest)
                report["raw_fold_audit"] = check_raw_fold_separation(cache, config)
            except ValueError as exc:
                report["horizons"][f"h{days}"] = {"status": "skipped", "reasons": [str(exc)], "coverage": {}}
                continue
        else:
            splits, coverage = split_horizon(dataset, days, config)
        horizon = report["horizons"][f"h{days}"] = {"coverage": coverage}
        reasons = []
        for fold, counts in coverage.items():
            if min(counts["positive"], counts["negative"]) < config["min_rows_per_class"]:
                reasons.append(f"{fold}: insufficient positives or negatives")
            if len(counts["positive_episode_ids"]) < config["min_positive_episodes"][fold]:
                reasons.append(f"{fold}: insufficient independent failure episodes")
        if config["mode"] == "edp_reference":
            reasons.append("Reference fragment is restricted to the data audit")
        if reasons:
            horizon.update(status="skipped", reasons=reasons)
            continue
        if audit_only:
            horizon.update(status="eligible")
            continue
        horizon.update(status="trained", models={})
        train_frame, val_frame, test_frame = (splits[key] for key in FOLDS)
        for name, pipeline in build_pipelines(config).items():
            path = output / f"h{days}" / name
            path.mkdir(parents=True)
            target = train_frame[f"h{days}_label"].astype(int)
            pipeline.fit(train_frame[columns], target)
            validation = predictions_for(pipeline, val_frame, columns, days)
            threshold = choose_threshold(validation, config)
            original = predictions_for(pipeline, test_frame, columns, days)
            joblib.dump(pipeline, path / "pipeline.joblib")
            restored = joblib.load(path / "pipeline.joblib")
            test = predictions_for(restored, test_frame, columns, days)
            np.testing.assert_array_equal(original["score"], test["score"])
            info = {"threshold": threshold, "reload_max_abs_score_error": float(
                np.max(np.abs(original["score"] - test["score"]))), "fit_rows": len(train_frame)}
            horizon["models"][name] = info
            write_json({"threshold": threshold, "score_semantics": report["score_semantics"],
                        "feature_names": columns, "horizon_days": days}, path / "inference.json")
            for fold, predictions in (("validation", validation), ("test", test)):
                predictions["alert"] = predictions["score"].ge(threshold)
                scores, events = alarm_metrics(predictions, threshold, config["prediction_every_minutes"])
                y = predictions["label"].astype(int)
                score = predictions["score"]
                scores.update(pr_auc_average_precision=float(average_precision_score(y, score)),
                              roc_auc=float(roc_auc_score(y, score)))
                info[fold] = scores
                metrics_rows.append({"horizon_days": days, "model": name, "fold": fold, **scores})
                predictions.to_parquet(path / f"{fold}_predictions.parquet", index=False)
                events.to_csv(path / f"{fold}_event_alerts.csv", index=False)
                precision, recall, _ = precision_recall_curve(y, score)
                fpr, tpr, _ = roc_curve(y, score)
                pd.concat([pd.DataFrame({"curve": "PR", "x": recall, "y": precision}),
                           pd.DataFrame({"curve": "ROC", "x": fpr, "y": tpr})]).to_csv(
                               path / f"{fold}_curves.csv", index=False)
            if name == "lightgbm":
                scada = pd.read_parquet(cache / "scada.parquet")
                info.update(explain_lightgbm(restored, test_frame, test, metadata, scada, path, config))
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(output / "metrics.csv", index=False)
    write_json(report, output / "metrics.json")
    return report
