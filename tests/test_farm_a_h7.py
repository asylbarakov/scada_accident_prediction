"""CARE H7 label boundaries, lineage and shared learning safeguards."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scada_pdm.care import care_state, check_raw_fold_separation, label_event, read_event, resolve_prediction_reuse, split_care
from scada_pdm.care_lineage import H7, STEP, dependency_groups, fault_aliases, fault_groups, find_overlaps, fingerprints
from scada_pdm.cli import load_config
from scada_pdm.learning import alarm_metrics, train
from scada_pdm.preparation import CACHE_VERSION, config_signature, feature_metadata, make_features, write_json
from scada_pdm.reporting import plot_care_scada

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "care_h7.json"
T0 = pd.Timestamp("2023-01-01", tz="UTC")


def label_row(event=1, kind="anomaly", asset="A", end=None):
    return {"event_id": event, "asset": asset, "event_label": kind,
            "event_start": T0, "event_end": T0 + H7 if end is None else end}


def raw_lineage(times, values, partition="prediction"):
    data = fingerprints(np.array(values, dtype="float64").reshape(-1, 1))
    return pd.DataFrame({"partition": partition, "avg_fingerprint": data, "full_fingerprint": data},
                        index=pd.DatetimeIndex(times, name="timestamp"))


class TestCareLabels(unittest.TestCase):
    def test_positive_uses_inclusive_h7_at_verified_end(self):
        start = T0 - STEP
        scada = pd.DataFrame({"turbine_id": "A", "timestamp": pd.date_range(start, T0 + H7, freq=STEP),
                              "partition": "prediction"})
        row = label_row()
        row["event_start"] = T0 + pd.Timedelta(days=5)
        result = label_event(scada, row, {"id": "f", "anchor_event": 1}, [])
        self.assertTrue(pd.isna(result.iloc[0]["h7_label"]))
        self.assertEqual(result.iloc[1]["h7_label"], 1)
        self.assertTrue(pd.isna(result.iloc[-1]["h7_label"]))
        self.assertEqual(result.iloc[-1]["h7_reason"], "at_or_after_fault")

    def test_normal_negative_requires_timestamp_and_observation_coverage(self):
        times = pd.date_range(T0, T0 + H7, freq=STEP)
        scada = pd.DataFrame({"turbine_id": "A", "timestamp": times, "partition": "prediction"})
        row = label_row(kind="normal")
        baseline = label_event(scada, row, None, [])
        self.assertEqual(baseline.iloc[0]["h7_label"], 0)
        self.assertEqual(baseline["h7_label"].count(), 1)
        missing = pd.Series(True, index=scada.index)
        missing.iloc[12] = False
        self.assertEqual(label_event(scada, row, None, [], observed=missing)["h7_label"].count(), 0)
        self.assertEqual(label_event(scada.drop(12), row, None, [])["h7_label"].count(), 0)
        self.assertEqual(label_event(scada, row, None, [T0 + H7]).iloc[0]["h7_reason"], "potential_competing_fault")

    def test_metadata_and_features_are_invariant_to_status(self):
        timestamps = pd.date_range(T0 - pd.Timedelta(days=2), periods=298, freq=STEP)
        frame = pd.DataFrame({"time_stamp": timestamps, "asset_id": "A", "id": np.arange(len(timestamps)),
                              "train_test": ["train"] * 280 + ["prediction"] * 18,
                              "status_type_id": 0, "sensor_0_avg": np.arange(len(timestamps), dtype=float)})
        row = label_row(end=timestamps[-1])
        row.update(event_start=timestamps[280], event_start_id="280", event_end_id="297")
        cfg = load_config(CONFIG)
        cfg["sensors"] = {"ambient": cfg["sensors"]["ambient_temperature"]}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "event.csv"
            frame.to_csv(path, sep=";", index=False)
            lineage, scada, _ = read_event(path, row, {"ambient": "sensor_0_avg"})
            frame["status_type_id"] = 4
            frame.to_csv(path, sep=";", index=False)
            changed, scada_after, _ = read_event(path, row, {"ambient": "sensor_0_avg"})
            pd.testing.assert_frame_equal(lineage, changed)
            pd.testing.assert_frame_equal(scada, scada_after)
            with self.assertRaisesRegex(ValueError, "event_end_id"):
                read_event(path, {**row, "event_end_id": "296"}, {"ambient": "sensor_0_avg"})
        before, _, _ = make_features(scada, cfg, state_policy=care_state)
        scada_after.loc[290:, "ambient"] = 999
        after, _, _ = make_features(scada_after, cfg, state_policy=care_state)
        pd.testing.assert_frame_equal(before.loc[:290], after.loc[:290])
        self.assertEqual(before.loc[290, "ambient__now"], 289)
        self.assertTrue(all(feature.startswith("ambient__") for feature in feature_metadata(cfg)))


class TestCareLineage(unittest.TestCase):
    def test_shifted_content_links_assets_and_blocks_assignment(self):
        times = pd.date_range(T0, periods=3, freq=STEP)
        files = {1: raw_lineage(times, [1, 2, 3]), 2: raw_lineage(times + H7, [1, 2, 3])}
        events = {1: label_row(1, asset="A"), 2: label_row(2, asset="B")}
        pairs = find_overlaps(files, events)
        self.assertEqual(pairs[0]["same_timestamp_rows"], 0)
        self.assertEqual(pairs[0]["shared_avg_at_different_displayed_times"], 3)
        lineage = dependency_groups(events, pairs)
        self.assertEqual(lineage[0]["assets"], ["A", "B"])
        cfg = {"asset_splits": {"train": ["A"], "validation": [], "test": ["B"]}}
        with self.assertRaisesRegex(ValueError, "lineage .* crosses folds"):
            split_care(pd.DataFrame(), cfg, {"dependency_groups": lineage})

    def test_fault_merging_requires_common_anchor_telemetry(self):
        times = pd.date_range(T0, periods=5, freq=STEP)
        files = {51: raw_lineage(times, [1, 2, 3, 4, 5]), 72: raw_lineage(times, [1, 2, 3, 4, 5])}
        events = {51: label_row(51, end=times[4]), 72: label_row(72, end=times[2])}
        groups = fault_groups(events, find_overlaps(files, events))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["anchor_event"], 72)
        self.assertEqual(groups[0]["status"], "conservative_merge")
        groups = fault_groups(events, [])
        self.assertIsNone(groups[0]["anchor_event"])
        self.assertEqual(groups[0]["status"], "unresolved")

    def test_shifted_fault_anchor_is_identified_in_another_file(self):
        times = pd.date_range(T0, periods=3, freq=STEP)
        files = {1: raw_lineage(times, [1, 2, 3]), 2: raw_lineage(times + H7, [1, 2, 3], "train")}
        events = {1: label_row(1, end=times[1]), 2: label_row(2, kind="normal")}
        alias = fault_aliases(files, events)[2][0]
        self.assertEqual(alias, {"anchor_event": 1, "timestamp": times[1] + H7, "partition": "train"})

    def test_conflicting_and_equivalent_duplicates(self):
        rows = pd.DataFrame({"turbine_id": "A", "timestamp": [T0, T0, T0 + STEP, T0 + STEP],
                             "source_event": [1, 2, 1, 2], "history_ready": [True, True, False, True],
                             "avg_fingerprint": [b"a", b"a", b"b", b"b"], "h7_label": pd.Series([1, 0, 0, 0], dtype="Int8"),
                             "h7_reason": ["positive", "negative", "negative", "negative"],
                             "h7_episode": ["f", None, None, None], "h7_failure_at": [T0 + H7, pd.NaT, pd.NaT, pd.NaT]})
        result = resolve_prediction_reuse(rows)
        self.assertEqual(result["h7_label"].count(), 1)
        self.assertEqual(result.iloc[3]["h7_label"], 0)
        self.assertEqual(result.iloc[0]["h7_reason"], "conflicting_repeated_observation")
        self.assertEqual(result.iloc[2]["h7_reason"], "deduplicated_observation")

    def test_raw_fold_check_includes_historical_rows_and_shifted_hash(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame({"turbine_id": ["A", "B"], "source_event": [1, 2],
                                  "timestamp": [T0, T0 + H7], "avg_fingerprint": [b"a", b"a"],
                                  "full_fingerprint": [b"x", b"y"]})
            frame.to_parquet(path / "observation_lineage.parquet", index=False)
            cfg = {"asset_splits": {"train": ["A"], "validation": [], "test": ["B"]}}
            with self.assertRaisesRegex(ValueError, "raw observation reuse crosses folds"):
                check_raw_fold_separation(path, cfg)

    def test_alarms_do_not_bridge_source_files(self):
        pred = pd.DataFrame({"turbine_id": "A", "source_event": [1, 2], "timestamp": [T0, T0 + STEP],
                             "label": [0, 0], "score": [1.0, 1.0], "episode_id": pd.Series(dtype="string"),
                             "failure_at": pd.Series(pd.NaT, index=[0, 1], dtype="datetime64[ns, UTC]")})
        metrics, _ = alarm_metrics(pred, 0.5, 10)
        self.assertEqual(metrics["false_warning_episodes"], 2)


class TestSyntheticCareModelSmoke(unittest.TestCase):
    def test_care_learning_reload_and_shap_on_72_artificial_rows(self):
        cfg = load_config(CONFIG)
        cfg.update(dataset_id="synthetic-care-h7-smoke", asset_splits={"train": ["A"], "validation": ["B"], "test": ["C"]},
                   min_rows_per_class=2, min_positive_episodes={"train": 1, "validation": 1, "test": 1},
                   shap_rows=10, local_shap_rows=2, top_sensor_count=1, threads=1,
                   lightgbm={"n_estimators": 6, "num_leaves": 3, "min_child_samples": 2})
        cfg["sensors"] = {"ambient": cfg["sensors"]["ambient_temperature"]}
        metadata = feature_metadata(cfg)
        parts, groups = [], []
        for i, asset in enumerate(("A", "B", "C")):
            times = pd.date_range(T0, periods=24, freq="h")
            y = np.repeat([0, 1], 12)
            part = pd.DataFrame({"turbine_id": asset, "timestamp": times, "history_ready": True,
                                 "source_event": 2 * i + y, "lineage_group": asset, "h7_label": y,
                                 "h7_episode": [None] * 12 + [f"synthetic-{asset}"] * 12,
                                 "h7_failure_at": [pd.NaT] * 12 + [times[-1] + pd.Timedelta(hours=1)] * 12})
            part = part.assign(**{feature: y * 10.0 + np.linspace(0, 1, len(y)) for feature in metadata})
            parts.append(part)
            groups.append({"id": asset, "assets": [asset], "events": [2 * i, 2 * i + 1]})
        dataset = pd.concat(parts, ignore_index=True)
        manifest = {"cache_version": CACHE_VERSION, "config_signature": config_signature(cfg),
                    "dataset_id": cfg["dataset_id"], "dependency_groups": groups}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            write_json(manifest, cache / "coverage.json")
            write_json(metadata, cache / "feature_map.json")
            dataset.to_parquet(cache / "features.parquet", index=False)
            dataset[["turbine_id", "timestamp", "ambient__now"]].rename(columns={"ambient__now": "ambient"}).to_parquet(
                cache / "scada.parquet", index=False)
            raw = dataset[["turbine_id", "timestamp", "source_event"]].assign(
                avg_fingerprint=[b"avg" + str(i).encode() for i in range(len(dataset))],
                full_fingerprint=[b"full" + str(i).encode() for i in range(len(dataset))])
            raw.to_parquet(cache / "observation_lineage.parquet", index=False)
            output = root / "models"
            models = train(cfg, cache, output)["horizons"]["h7"]["models"]
            self.assertEqual(models["logistic"]["reload_max_abs_score_error"], 0)
            self.assertEqual(models["lightgbm"]["reload_max_abs_score_error"], 0)
            self.assertLessEqual(models["lightgbm"]["actual_trees"], 6)
            self.assertLess(models["lightgbm"]["shap_max_abs_margin_error"], 1e-8)
            self.assertTrue((output / "h7" / "lightgbm" / "risk_sensor_history.parquet").is_file())
            plots = plot_care_scada(output)
            self.assertEqual(len(plots), 2)
            self.assertTrue(all(path.is_file() for path in plots))
