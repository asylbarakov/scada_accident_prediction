"""Prepare the CARE v6 Wind Farm A H7 benchmark without fitting a model."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts.care_readiness_audit import load_event_info, locate_root
from scripts.fetch_care_a import fetch_farm

from .care_lineage import H7, STEP, dependency_groups, fault_aliases, fault_groups, find_overlaps, fingerprints
from .edp import KEY, sha256_file
from .preparation import CACHE_VERSION, config_signature, feature_metadata, history_span, make_features, validate_config, write_json

EVENT_IDS = {0, 3, 10, 13, 14, 17, 22, 24, 25, 26, 38, 40, 42, 45, 51, 68, 69, 71, 72, 73, 84, 92}
RECORD = "https://zenodo.org/records/15846963"
RAW_COLS = {"time_stamp": "timestamp", "asset_id": "turbine_id", "id": "row_id", "train_test": "partition"}
PHYSICAL_COLUMN = re.compile(r"(sensor|wind_speed|power|reactive_power)_\d+(?:_(avg|min|max|std))?")
FOLD_NAMES = ("train", "validation", "test")


def locate_farm(raw_dir):
    root = locate_root(Path(raw_dir), "A")
    farm = root / "Wind Farm A"
    expected = [farm / "feature_description.csv", *(farm / "datasets" / f"{i}.csv" for i in sorted(EVENT_IDS))]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CARE v6 Wind Farm A files: {', '.join(missing)}")
    return root, farm


def ensure_care(config):
    raw_dir = Path(config["raw_dir"])
    try:
        root, _ = locate_farm(raw_dir)
    except FileNotFoundError:
        if raw_dir.name != "CARE_To_Compare":
            raise ValueError("For automatic acquisition, raw_dir must end with CARE_To_Compare")
        fetch_farm(raw_dir.parent)
        root, _ = locate_farm(raw_dir)
    print(f"CARE v6 Wind Farm A ready: {root}", flush=True)
    return {"source": RECORD, "raw_dir": str(root), "events_expected": len(EVENT_IDS)}


def feature_channels(config, farm):
    description = pd.read_csv(farm / "feature_description.csv", sep=";").set_index("sensor_name")
    sensors = config["sensors"]
    columns = {channel: info["column"] for channel, info in sensors.items()}
    for column in columns.values():
        base = column.removesuffix("_avg")
        if not column.endswith("_avg") or not PHYSICAL_COLUMN.fullmatch(column) or base not in description.index:
            raise ValueError(f"CARE features must be documented physical averages: {column}")
        row = description.loc[base]
        if bool(row["is_angle"]) or bool(row["is_counter"]):
            raise ValueError(f"CARE angle/counter needs a dedicated transform: {column}")
    if len(set(columns.values())) != len(columns):
        raise ValueError("Choose each CARE source sensor only once")
    return columns


def read_event(path, row, sensor_columns):
    header = pd.read_csv(path, sep=";", nrows=0).columns
    physical = sorted(col for col in header if PHYSICAL_COLUMN.fullmatch(col))
    averages = [col for col in physical if col.endswith("_avg")]
    if not set(sensor_columns.values()).issubset(averages):
        raise ValueError(f"{path}: missing configured physical averages")
    frame = pd.read_csv(path, sep=";", usecols=[*RAW_COLS, *physical], float_precision="round_trip",
                        dtype={"asset_id": "string", "id": "int64"}).rename(columns=RAW_COLS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    dates = frame["timestamp"]
    if (frame.empty or dates.isna().any() or dates.duplicated().any()
            or not dates.is_monotonic_increasing or not dates.eq(dates.dt.floor(STEP)).all()
            or frame["row_id"].duplicated().any()
            or set(frame["turbine_id"]) != {row["asset"]}
            or set(frame["partition"]) != {"train", "prediction"}):
        raise ValueError(f"{path}: invalid cadence keys, row IDs, asset or partitions")
    prediction = frame["partition"].eq("prediction")
    if frame.loc[~prediction, "timestamp"].max() >= frame.loc[prediction, "timestamp"].min():
        raise ValueError(f"{path}: partition timeline overlaps within the file")
    for key in ("event_start", "event_end"):
        matching = frame.loc[prediction & frame["row_id"].eq(int(row[f"{key}_id"])), "timestamp"]
        if len(matching) != 1 or matching.iloc[0] != row[key]:
            raise ValueError(f"{path}: {key} and {key}_id do not match prediction telemetry")
    if row["event_label"] not in ("anomaly", "normal") or row["event_start"] >= row["event_end"]:
        raise ValueError(f"{path}: invalid event metadata")

    numeric = frame[physical].replace([np.inf, -np.inf], np.nan)
    identity = frame[["timestamp", "partition", "row_id"]].copy()
    identity["avg_fingerprint"] = fingerprints(numeric[averages].to_numpy())
    identity["full_fingerprint"] = fingerprints(numeric[physical].to_numpy())
    identity = identity.set_index("timestamp")
    scada = pd.concat([frame[KEY + ["partition"]], numeric[list(sensor_columns.values())].astype("float32")], axis=1)
    scada = scada.rename(columns={column: key for key, column in sensor_columns.items()})
    return identity, scada, physical


def care_state(values, _operation):
    observed = values.notna().any(axis=1)
    return pd.DataFrame({"observed": observed, "at_risk": observed})


def label_event(scada, row, fault, other_anchors, *, observed=None, competing_anchors=()):
    prediction = scada.loc[scada["partition"].eq("prediction"), KEY].copy()
    t = prediction["timestamp"]
    label = pd.Series(pd.NA, index=t.index, dtype="Int8")
    reason = pd.Series("unknown_failure_free_future", index=t.index, dtype="string")
    event_end = row["event_end"]
    if row["event_label"] == "anomaly":
        if fault["anchor_event"] != row["event_id"]:
            reason[:] = "ambiguous_fault_episode"
        else:
            positive = t.lt(event_end) & (t + H7).ge(event_end)
            label.loc[positive] = 1
            reason.loc[positive] = "positive"
            if any(event_end - H7 <= anchor < event_end for anchor in competing_anchors):
                label.loc[positive] = pd.NA
                reason.loc[positive] = "potential_competing_fault"
            reason.loc[t.ge(event_end)] = "at_or_after_fault"
    else:
        start = row["event_start"]
        inside = t.between(start, event_end)
        observed = pd.Series(True, index=t.index) if observed is None else observed.loc[t.index]
        run_end = t.shift(-1).sub(t).ne(STEP) | ~observed.shift(-1, fill_value=False)
        last = t.where(run_end).bfill()
        full = observed & inside & (t + H7).le(event_end) & (t + H7).le(last)
        # Equal displayed dates alone are weak cross-file evidence. Censor
        # any potentially contradictory normal window rather than joining it.
        conflicts = pd.Series(False, index=t.index)
        for anchor in other_anchors:
            conflicts |= t.le(anchor) & (t + H7).ge(anchor)
        label.loc[full & ~conflicts] = 0
        reason.loc[~inside] = "outside_annotated_normal_interval"
        reason.loc[inside] = "incomplete_normal_followup"
        reason.loc[full & conflicts] = "potential_competing_fault"
        reason.loc[full & ~conflicts] = "negative"
    prediction["source_event"] = row["event_id"]
    prediction["h7_label"] = label
    prediction["h7_reason"] = reason
    positive = label.eq(1).fillna(False)
    episode = fault["id"] if fault else None
    prediction["h7_episode"] = pd.Series(episode, index=t.index, dtype="string").where(positive)
    prediction["h7_failure_at"] = pd.Series(event_end, index=t.index, dtype=t.dtype).where(positive)
    return prediction


def resolve_prediction_reuse(labels):
    """Resolve cross-file repeated prediction candidates, preserving censor reasons."""
    result = labels.copy()
    result["pre_dedup_h7_label"] = result["h7_label"]
    missing = result["h7_label"].notna() & result["avg_fingerprint"].isna()
    result.loc[missing, ["h7_label", "h7_reason"]] = [pd.NA, "no_physical_readings"]
    for keys in (KEY, ["avg_fingerprint"]):
        candidates = result.loc[result["h7_label"].notna()].sort_values(
            ["history_ready", "source_event", "timestamp"], ascending=[False, True, True])
        repeated = candidates.loc[candidates.groupby(keys)["source_event"].transform("nunique").gt(1)]
        for _, group in repeated.groupby(keys, sort=False):
            conflict = (group["h7_label"].nunique() > 1 or group["h7_episode"].nunique() > 1
                        or group["avg_fingerprint"].nunique() > 1)
            if conflict:
                result.loc[group.index, ["h7_label", "h7_reason"]] = [pd.NA, "conflicting_repeated_observation"]
            else:
                # Keep one file for all equivalent candidates. Repeated values
                # within that file and different times remain distinct.
                keep = group.iloc[0]["source_event"]
                drop = group.index[group["source_event"].ne(keep)]
                result.loc[drop, ["h7_label", "h7_reason"]] = [pd.NA, "deduplicated_observation"]
    result["h7_episode"] = result["h7_episode"].where(result["h7_label"].eq(1).fillna(False))
    result["h7_failure_at"] = result["h7_failure_at"].where(result["h7_label"].eq(1).fillna(False))
    return result


def label_counts(frame):
    return {"positive": int(frame["h7_label"].eq(1).sum()), "negative": int(frame["h7_label"].eq(0).sum()),
            "without_label": int(frame["h7_label"].isna().sum()),
            "positive_episode_ids": sorted(frame["h7_episode"].dropna().unique().tolist()),
            "normal_events_with_negatives": int(frame.loc[frame["h7_label"].eq(0), "source_event"].nunique()),
            "by_reason": {str(k): int(v) for k, v in frame["h7_reason"].value_counts().items()}}


def prepare_care(config, cache):
    validate_config(config)
    cache = Path(cache)
    if cache.exists():
        raise FileExistsError(f"Choose a new cache directory: {cache}")
    root, farm = locate_farm(config["raw_dir"])
    events = {r["event_id"]: r for r in sorted(load_event_info(root, "A"), key=lambda r: r["event_id"])}
    if events.keys() != EVENT_IDS:
        raise ValueError("CARE metadata event IDs do not match Wind Farm A v6")
    sensor_columns = feature_channels(config, farm)
    identity, scada_slices, summaries, expected_columns = {}, {}, [], None
    for event, metadata in events.items():
        path = farm / "datasets" / f"{event}.csv"
        index, scada, physical = read_event(path, metadata, sensor_columns)
        if expected_columns is not None and expected_columns != physical:
            raise ValueError(f"{path}: physical SCADA schema differs across files")
        expected_columns = physical
        identity[event] = index
        p = index.index[index["partition"].eq("prediction")]
        start = p.min() - history_span(config)
        scada_slices[event] = scada.loc[scada["timestamp"].ge(start)].copy()
        summaries.append({"event": event, "asset": metadata["asset"], "event_label": metadata["event_label"],
                          "fault_or_normal_end": metadata["event_end"].isoformat(), "end_row_id": metadata["event_end_id"],
                          "description": metadata.get("event_description"), "rows": len(index),
                          "train_rows": len(index) - len(p), "prediction_rows": len(p),
                          "duplicate_avg_occurrences_within_file": int(index["avg_fingerprint"].dropna().duplicated().sum()),
                          "prediction_cadence_breaks": int(p.to_series().diff().iloc[1:].ne(STEP).sum()),
                          "sha256": sha256_file(path)})
    overlaps = find_overlaps(identity, events)
    lineage = dependency_groups(events, overlaps)
    episodes = fault_groups(events, overlaps)
    aliases = fault_aliases(identity, events)
    group_for = {event: group["id"] for group in lineage for event in group["events"]}
    episode_for = {event: episode for episode in episodes for event in episode["events"]}
    all_predictions, all_features, all_raw, lineage_rows = [], [], [], []
    raw_coverage = []
    for event, metadata in events.items():
        index = identity[event]
        lineage_rows.append(index.assign(source_event=event, turbine_id=metadata["asset"],
                                         lineage_group=group_for[event]).reset_index())
        other_anchors = [row["event_end"] for row in events.values()
                         if row["asset"] == metadata["asset"] and row["event_label"] == "anomaly"]
        other_anchors.extend(match["timestamp"] for match in aliases[event])
        competing = [match["timestamp"] for match in aliases[event] if match["partition"] == "prediction"
                     and episode_for.get(match["anchor_event"]) != episode_for.get(event)]
        scada = scada_slices[event]
        # Keep a reconstruction of the historical per-file counts for comparison.
        temporary_fault = {"id": f"raw-{event}", "anchor_event": event} if event in episode_for else None
        raw_coverage.append(label_event(scada, metadata, temporary_fault, []))
        observed = pd.Series(index.loc[pd.DatetimeIndex(scada["timestamp"]), "avg_fingerprint"].notna().to_numpy(),
                             index=scada.index)
        pred = label_event(scada, metadata, episode_for.get(event), other_anchors,
                           observed=observed, competing_anchors=competing)
        features, _, ready = make_features(scada, config, state_policy=care_state)
        pred["history_ready"] = ready.loc[pred.index]
        pred["lineage_group"] = group_for[event]
        pred["avg_fingerprint"] = index.loc[pd.DatetimeIndex(pred["timestamp"]), "avg_fingerprint"].to_numpy()
        all_predictions.append(pred)
        all_features.append(features.loc[pred.index])
        all_raw.append(scada.loc[pred.index, KEY + list(sensor_columns)])
    raw_labels = pd.concat(raw_coverage, ignore_index=True)
    labels = resolve_prediction_reuse(pd.concat(all_predictions, ignore_index=True))
    labels["raw_per_file_h7_label"] = raw_labels["h7_label"]
    ready = labels["history_ready"] & labels["h7_label"].notna()
    row_data = labels[KEY + ["source_event", "lineage_group", "history_ready", "h7_label", "h7_episode", "h7_failure_at"]]
    features = pd.concat([row_data, pd.concat(all_features, ignore_index=True)], axis=1).loc[ready]
    if features.duplicated(KEY).any():
        raise ValueError("Duplicate prediction key survived CARE preparation")
    row_lineage = pd.concat(lineage_rows, ignore_index=True)
    valid_identity = row_lineage.dropna(subset=["avg_fingerprint"])
    reused = valid_identity.groupby("avg_fingerprint", sort=False)["source_event"].transform("nunique").gt(1)
    row_lineage["reused_avg_across_files"] = False
    row_lineage.loc[valid_identity.index, "reused_avg_across_files"] = reused
    full_identity = row_lineage.dropna(subset=["full_fingerprint"])
    reused_full = full_identity.groupby("full_fingerprint", sort=False)["source_event"].transform("nunique").gt(1)
    row_lineage["reused_full_across_files"] = False
    row_lineage.loc[full_identity.index, "reused_full_across_files"] = reused_full
    manifest = {
        "cache_version": CACHE_VERSION, "config_signature": config_signature(config), "mode": config["mode"],
        "dataset_id": config["dataset_id"], "source": RECORD, "raw_dir": str(root),
        "signals": {"rows": len(row_lineage), "events": len(events),
                    "physical_columns": len(expected_columns), "average_columns": sum(c.endswith("_avg") for c in expected_columns)},
        "source_sha256": {name: sha256_file(farm / name) for name in ("event_info.csv", "feature_description.csv")},
        "event_summaries": summaries, "dependency_groups": lineage, "fault_groups": episodes,
        "fault_anchor_aliases": [{"in_event": event, **match, "timestamp": match["timestamp"].isoformat()}
                                 for event, matches in aliases.items() for match in matches],
        "pairs_examined": len(events) * (len(events) - 1) // 2,
        "overlap_pairs": len(overlaps), "raw_occurrences_with_reused_averages": int(reused.sum()),
        "distinct_reused_average_vectors": int(valid_identity.loc[reused, "avg_fingerprint"].nunique()),
        "raw_occurrences_with_reused_full_vectors": int(reused_full.sum()),
        "raw_h7": label_counts(raw_labels),
        "h7_after_reconciliation": label_counts(labels),
        "h7_with_feature_history": label_counts(labels.loc[labels["history_ready"]]),
        "fault_independence": "Conservative CARE benchmark episodes; repair history is unverified",
    }
    cache.mkdir(parents=True)
    labels.to_parquet(cache / "labels.parquet", index=False)
    features.to_parquet(cache / "features.parquet", index=False)
    pd.concat(all_raw, ignore_index=True).loc[ready].to_parquet(cache / "scada.parquet", index=False)
    row_lineage.to_parquet(cache / "observation_lineage.parquet", index=False)
    write_json(overlaps, cache / "overlaps.json")
    write_json(feature_metadata(config), cache / "feature_map.json")
    write_json(manifest, cache / "coverage.json")
    return manifest


def split_care(frame, config, manifest):
    """Split complete asset/lineage groups, without comparing dates across files."""
    assignment = config["asset_splits"]
    if set(assignment) != set(FOLD_NAMES):
        raise ValueError("CARE needs named train, validation and test asset assignments")
    assets = [asset for fold in FOLD_NAMES for asset in assignment[fold]]
    known = {asset for group in manifest["dependency_groups"] for asset in group["assets"]}
    if len(assets) != len(set(assets)) or set(assets) != known:
        raise ValueError("Assign each CARE turbine to exactly one fold")
    fold_for = {asset: fold for fold in FOLD_NAMES for asset in assignment[fold]}
    group_map, event_map = {}, {}
    for group in manifest["dependency_groups"]:
        folds = {fold_for[asset] for asset in group["assets"]}
        if len(folds) != 1:
            raise ValueError(f"CARE lineage {group['id']} crosses folds: {sorted(folds)}")
        group_map[group["id"]] = next(iter(folds))
        event_map.update({event: group["id"] for event in group["events"]})
    if frame.duplicated(KEY).any() or frame["lineage_group"].isna().any():
        raise ValueError("Invalid CARE prediction keys or lineage")
    assigned = frame["turbine_id"].map(fold_for)
    if (assigned.isna().any() or not assigned.eq(frame["lineage_group"].map(group_map)).all()
            or not frame["source_event"].map(event_map).eq(frame["lineage_group"]).all()):
        raise ValueError("CARE prediction lineage disagrees with turbine assignment")
    stride = config["prediction_every_minutes"]
    if not isinstance(stride, int) or stride <= 0 or stride % 10:
        raise ValueError("CARE prediction stride must be a positive multiple of 10 minutes")
    scheduled = frame["timestamp"].eq(frame["timestamp"].dt.floor(pd.Timedelta(minutes=stride)))
    splits, coverage = {}, {}
    seen_episodes = set()
    for fold in FOLD_NAMES:
        eligible = frame.loc[assigned.eq(fold) & scheduled & frame["history_ready"] & frame["h7_label"].notna()]
        splits[fold] = eligible
        counts = label_counts(eligible.assign(h7_reason=np.where(eligible["h7_label"].eq(1), "positive", "negative")))
        episode_ids = set(counts["positive_episode_ids"])
        if seen_episodes & episode_ids:
            raise ValueError("A CARE fault episode appears in multiple folds")
        seen_episodes.update(episode_ids)
        coverage[fold] = {**counts, "turbines": assignment[fold],
                          "source_events": sorted(eligible["source_event"].unique().tolist()),
                          "lineage_groups": sorted(group for group, group_fold in group_map.items() if group_fold == fold)}
    return splits, coverage


def check_raw_fold_separation(cache, config):
    columns = ["turbine_id", "source_event", "timestamp", "avg_fingerprint", "full_fingerprint"]
    raw = pd.read_parquet(Path(cache) / "observation_lineage.parquet", columns=columns)
    assignment = {asset: fold for fold, assets in config["asset_splits"].items() for asset in assets}
    raw["fold"] = raw["turbine_id"].map(assignment)
    if raw["fold"].isna().any():
        raise ValueError("A CARE raw observation has no fold")
    result = {"raw_rows_checked": len(raw), "cross_fold_repetitions": {}}
    for keys in (["source_event"], KEY, ["avg_fingerprint"], ["full_fingerprint"]):
        repeated_keys = raw.dropna(subset=keys).drop_duplicates([*keys, "fold"])
        count = int(repeated_keys.duplicated(keys).sum())
        result["cross_fold_repetitions"]["+".join(keys)] = count
    if any(result["cross_fold_repetitions"].values()):
        raise ValueError(f"CARE raw observation reuse crosses folds: {result['cross_fold_repetitions']}")
    return result
