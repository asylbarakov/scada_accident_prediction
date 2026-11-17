"""Causal features, supervised horizon labels and a shareable Parquet cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .edp import KEY, load_failure_log, load_scada, read_intervals, review_failures, sha256_file

CACHE_VERSION = 1
SEMANTIC_CONFIG = ("mode", "cadence_minutes", "horizons_days", "windows_hours", "sensors", "operation")
REVIEW_INPUTS = ("failure_review", "log_coverage", "outages")


def write_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def config_signature(config):
    settings = {key: config[key] for key in SEMANTIC_CONFIG}
    if config.get("measurement_delay_steps", 0):
        settings["measurement_delay_steps"] = config["measurement_delay_steps"]
    if config.get("dataset_id"):
        settings["dataset_id"] = config["dataset_id"]
    if config["mode"] == "care_h7":
        settings["required_history_hours"] = config["required_history_hours"]
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def history_span(config):
    extra_steps = max(0, config.get("measurement_delay_steps", 0) - 1)
    return pd.Timedelta(hours=max(config["windows_hours"]), minutes=extra_steps * config["cadence_minutes"])


def validate_config(config):
    cadence = config["cadence_minutes"]
    horizons, windows = config["horizons_days"], config["windows_hours"]
    if not isinstance(cadence, int) or cadence <= 0 or 1440 % cadence:
        raise ValueError("cadence_minutes must divide one day")
    if not horizons or len(set(horizons)) != len(horizons) or any(h not in (7, 30) for h in horizons):
        raise ValueError("horizons_days must contain unique values from [7, 30]")
    if not windows or len(set(windows)) != len(windows) or any(w <= 0 or w * 60 % cadence for w in windows):
        raise ValueError("Feature windows must be positive multiples of the sampling interval")
    if config["mode"] not in ("edp", "edp_reference", "synthetic_smoke", "care_h7"):
        raise ValueError("Unknown data mode")
    if config["mode"] == "edp" and cadence != 10:
        raise ValueError("Original EDP SCADA uses 10-minute cadence")
    delay = config.get("measurement_delay_steps", 0)
    if not isinstance(delay, int) or delay < 0:
        raise ValueError("measurement_delay_steps must be a nonnegative integer")
    if config["mode"] == "care_h7":
        minimum = config["required_history_hours"]
        if horizons != [7] or cadence != 10 or not 0 < minimum <= max(windows) or minimum * 60 % cadence:
            raise ValueError("CARE preparation requires H7, 10-minute cadence and a valid feature history")
        return
    operation = config["operation"]
    if not {operation[key] for key in ("wind_sensor", "power_sensor", "rotor_sensor")} <= set(config["sensors"]):
        raise ValueError("Operation sensors must be configured physical channels")


def operating_state(scada, operation):
    wind = scada[operation["wind_sensor"]]
    power = scada[operation["power_sensor"]]
    rotor = scada[operation["rotor_sensor"]]
    complete = wind.notna() & power.notna() & rotor.notna()
    moving = rotor.gt(operation["min_rotor_rpm"]) | power.gt(operation["min_power_kw"])
    # An observed idle turbine can be covered. Unexplained stops at productive
    # wind speeds break follow-up until operation resumes.
    stopped = complete & ~moving & wind.between(*operation["productive_wind_mps"], inclusive="both")
    return pd.DataFrame({"observed": complete & ~stopped, "at_risk": complete & moving,
                         "suspected_stop": stopped, "operation_missing": ~complete}, index=scada.index)


def feature_metadata(config):
    metadata = {}
    for sensor, info in config["sensors"].items():
        base = {"sensor": sensor, "source_column": info["column"], "physical_measurement": info["measurement"],
                "sensor_unit": info["unit"]}
        metadata[sensor + "__now"] = {**base, "statistic": "now", "window_hours": 0, "feature_unit": info["unit"]}
        for hours in config["windows_hours"]:
            statistics = ["mean", "max", "std", "delta_mean"]
            if hours * 60 > config["cadence_minutes"]:
                statistics.append("trend")
            for statistic in statistics:
                unit = info["unit"] + "/h" if statistic == "trend" else info["unit"]
                name = f"{sensor}__{hours}h_{statistic}"
                metadata[name] = {**base, "statistic": statistic, "window_hours": hours, "feature_unit": unit}
    return metadata


def make_features(scada, config, *, state_policy=operating_state):
    """Use readings at t and earlier, restarting after sensor or cadence breaks."""
    cadence_minutes = config["cadence_minutes"]
    step = pd.Timedelta(minutes=cadence_minutes)
    label_state = state_policy(scada, config["operation"])
    channels = list(config["sensors"])
    delay = config.get("measurement_delay_steps", 0)
    available = scada[channels]
    if delay:
        lagged = scada.groupby("turbine_id", sort=False)[["timestamp", *channels]].shift(delay)
        expected = scada["timestamp"] - delay * step
        available = lagged[channels].where(lagged["timestamp"].eq(expected), np.nan, axis=0)
    state = state_policy(available, config["operation"])
    breaks = (scada["turbine_id"].ne(scada["turbine_id"].shift())
              | scada["timestamp"].diff().ne(step) | ~state["observed"]
              | ~state["observed"].shift(fill_value=False))
    segment = breaks.cumsum()
    raw = available.where(state["observed"], np.nan, axis=0)
    grouped = raw.groupby(segment, sort=False)
    columns = {sensor + "__now": raw[sensor] for sensor in channels}
    for hours in config["windows_hours"]:
        slots = hours * 60 // cadence_minutes
        rolling = grouped.rolling(slots, min_periods=slots)
        mean = rolling.mean().droplevel(0)
        stats = {"mean": mean, "max": rolling.max().droplevel(0),
                 "std": rolling.std(ddof=0).droplevel(0), "delta_mean": raw - mean}
        if slots > 1:
            elapsed_hours = (slots - 1) * cadence_minutes / 60
            stats["trend"] = ((raw - grouped.shift(slots - 1)) / elapsed_hours).where(mean.notna())
        for statistic, values in stats.items():
            for sensor in channels:
                columns[f"{sensor}__{hours}h_{statistic}"] = values[sensor]
    history_slots = config.get("required_history_hours", max(config["windows_hours"])) * 60 // cadence_minutes
    ready = scada.groupby(segment, sort=False).cumcount().ge(history_slots - 1) & state["at_risk"]
    features = pd.DataFrame(columns, index=scada.index, dtype="float32")
    return features[list(feature_metadata(config))], label_state, ready


def mark_labels(scada, state, episodes, pending, coverage, outages, config):
    """Label a next confirmed failure or an observed no-failure horizon."""
    step = pd.Timedelta(minutes=config["cadence_minutes"])
    history = history_span(config)
    frames = []
    for asset, group in scada.groupby("turbine_id", sort=False):
        t = group["timestamp"]
        observed = state.loc[group.index, "observed"].copy()
        operational = state.loc[group.index, "at_risk"]
        clear_history = pd.Series(True, index=group.index)
        interruptions = []
        asset_faults = episodes.loc[episodes["turbine_id"].eq(asset)]
        for event in asset_faults.itertuples():
            active = t.ge(event.failure_at)
            if pd.notna(event.recovery_at):
                active &= t.lt(event.recovery_at)
                clear_history &= ~(t.ge(event.failure_at) & t.sub(history).lt(event.recovery_at))
            else:
                clear_history &= ~active
            observed &= ~active
        for row in pending.loc[pending["turbine_id"].eq(asset)].itertuples():
            interruptions.append(row.timestamp)
            observed &= t.lt(row.timestamp)
        for interval in outages.loc[outages["turbine_id"].eq(asset)].itertuples():
            # An outage wholly between two samples also interrupts follow-up.
            grid_start = interval.start.ceil(step)
            grid_end = max(interval.end.ceil(step), grid_start + step)
            interruptions.append(interval.start)
            observed &= ~(t.ge(grid_start) & t.lt(grid_end))
            clear_history &= ~(t.ge(interval.start) & t.sub(history).lt(interval.end))
        # A reverse scan finds the last observed slot in each uninterrupted run.
        run_end = ~observed.shift(-1, fill_value=False) | t.shift(-1).sub(t).ne(step)
        observed_until = t.where(run_end).bfill()
        verified_until = pd.Series(pd.NaT, index=group.index, dtype=t.dtype)
        for interval in coverage.loc[coverage["turbine_id"].eq(asset)].itertuples():
            covered = t.between(interval.start, interval.end)
            verified_until.loc[covered] = interval.end

        # A known failure may break the observation run at its own timestamp.
        # Require uninterrupted telemetry up to the last grid slot strictly
        # before the independently timestamped event.
        next_failure = pd.Series(pd.NaT, index=group.index, dtype=t.dtype)
        next_episode = pd.Series(pd.NA, index=group.index, dtype="string")
        for event in reversed(list(asset_faults.itertuples())):
            future = t.lt(event.failure_at)
            next_failure.loc[future] = event.failure_at
            next_episode.loc[future] = event.episode_id
        before_failure = next_failure.dt.ceil(step) - step
        before_interruption = pd.Series(True, index=group.index)
        for interruption in interruptions:
            before_interruption &= ~(t.lt(interruption) & next_failure.gt(interruption))
        result = pd.DataFrame({"reviewed_history_clear": clear_history}, index=group.index)
        for days in config["horizons_days"]:
            prefix = f"h{days}"
            deadline = t + pd.Timedelta(days=days)
            failure_in_window = next_failure.le(deadline)
            eligible = observed & operational & verified_until.notna()
            positive = (eligible & failure_in_window & verified_until.ge(next_failure)
                        & observed_until.ge(before_failure) & before_interruption)
            negative = (eligible & ~failure_in_window & verified_until.ge(deadline)
                        & observed_until.ge(deadline))
            labels = pd.Series(pd.NA, dtype="Int8", index=group.index)
            labels.loc[positive], labels.loc[negative] = 1, 0
            reason = np.select(
                [positive, negative, ~observed, ~operational, verified_until.isna()],
                ["positive", "negative", "unavailable", "idle", "unverified_log"], default="incomplete_followup",
            )
            result[f"{prefix}_label"] = labels
            result[f"{prefix}_reason"] = reason
            result[f"{prefix}_episode"] = next_episode.where(positive)
            result[f"{prefix}_failure_at"] = next_failure.where(positive)
        frames.append(result)
    return pd.concat(frames).sort_index()


def label_coverage(labels, horizons):
    result = {}
    for days in horizons:
        key = f"h{days}"
        target = labels[f"{key}_label"]
        result[key] = {
            "positive": int(target.eq(1).sum()), "negative": int(target.eq(0).sum()),
            "censored": int(target.isna().sum()),
            "positive_episode_ids": sorted(labels[f"{key}_episode"].dropna().unique().tolist()),
            "counts_by_reason": {str(k): int(v) for k, v in labels[f"{key}_reason"].value_counts().items()},
        }
    return result


def prepare(config, cache):
    validate_config(config)
    cache = Path(cache)
    if cache.exists():
        raise FileExistsError(f"Choose a new cache directory: {cache}")
    scada, scada_audit = load_scada(config)
    if config["mode"] == "synthetic_smoke" and len(scada) > 5000:
        raise ValueError("Synthetic smoke is limited to 5000 SCADA rows")
    log, log_audit = load_failure_log(config["failure_logs"])
    review, episodes = review_failures(log, config.get("failure_review"))
    pending = review.loc[review["decision"].eq("pending")]
    coverage = read_intervals(config.get("log_coverage"), coverage=True)
    outages = read_intervals(config.get("outages"), coverage=False)
    x, state, ready = make_features(scada, config)
    labels = mark_labels(scada, state, episodes, pending, coverage, outages, config)
    dataset = pd.concat([scada[KEY], ready.rename("history_ready"), labels, x], axis=1)
    manifest = {
        "cache_version": CACHE_VERSION, "config_signature": config_signature(config), "mode": config["mode"],
        "dataset_id": config.get("dataset_id"),
        "signals": scada_audit, "failure_log": log_audit,
        "review_counts": {str(k): int(v) for k, v in review["decision"].value_counts().items()},
        "episodes": len(episodes), "episodes_without_recovery": int(episodes["recovery_at"].isna().sum()),
        "failure_turbines_missing_scada": sorted(set(log["turbine_id"]) - set(scada["turbine_id"])),
        "scada_turbines_missing_coverage": sorted(set(scada["turbine_id"]) - set(coverage["turbine_id"])),
        "same_turbine_distinct_log_rows_within_30d": int(
            log.groupby("turbine_id")["timestamp"].diff().le(pd.Timedelta(days=30)).sum()),
        "verified_coverage_intervals": len(coverage), "reviewed_outage_intervals": len(outages),
        "review_sha256": {key: sha256_file(config[key]) for key in REVIEW_INPUTS
                          if config.get(key)},
        "operation": {key: int(state[key].sum()) for key in state},
        "coverage": label_coverage(labels, config["horizons_days"]),
        "coverage_with_feature_history": label_coverage(
            labels.loc[ready & labels["reviewed_history_clear"]], config["horizons_days"]),
    }
    cache.mkdir(parents=True)
    pd.concat([scada, state], axis=1).to_parquet(cache / "scada.parquet", index=False)
    dataset.to_parquet(cache / "features.parquet", index=False)
    episodes.to_parquet(cache / "episodes.parquet", index=False)
    review.to_csv(cache / "failure_review_queue.csv", index=False)
    write_json(feature_metadata(config), cache / "feature_map.json")
    write_json(manifest, cache / "coverage.json")
    return manifest


def check_cache(config, cache):
    validate_config(config)
    cache = Path(cache)
    with (cache / "coverage.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["cache_version"] != CACHE_VERSION or manifest["config_signature"] != config_signature(config):
        raise ValueError("Cache schema or preparation settings differ from the configuration")
    hashes = manifest.get("review_sha256", {})
    for key in REVIEW_INPUTS:
        path = config.get(key)
        if bool(path) != (key in hashes):
            raise ValueError(f"Cache was prepared with different {key} settings. Prepare a new cache.")
        # A copied cache may be used with its saved configuration even when
        # the original inputs are currently unavailable on that machine.
        if path and Path(path).exists() and sha256_file(path) != hashes[key]:
            raise ValueError(f"{key} changed after preparation. Prepare a new cache.")
    return manifest
