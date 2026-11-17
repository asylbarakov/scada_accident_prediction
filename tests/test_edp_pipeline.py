"""Focused checks for supervised labels and future-information leakage."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scada_pdm.acquisition import acquire_file
from scada_pdm.edp import load_failure_log, load_scada, review_failures
from scada_pdm.learning import alarm_metrics, split_horizon, train
from scada_pdm.preparation import (
    CACHE_VERSION, check_cache, config_signature, feature_metadata, make_features, mark_labels, operating_state, write_json,
)


def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "edp.json"
    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(periods, *, freq="10min", start="2016-01-01", asset="T01"):
    return pd.DataFrame({"turbine_id": asset, "timestamp": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
                         "wind_speed": 8.0, "power": 350.0, "rotor_speed": 12.0,
                         "generator_speed": 1150.0, "ambient_temperature": 17.0,
                         "gearbox_oil_temperature": np.arange(periods, dtype=float)})


class TestTargets(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.data = telemetry(4321)
        self.t0 = self.data["timestamp"].iloc[0]
        self.fault = self.data["timestamp"].iloc[-1]
        self.episodes = pd.DataFrame({"turbine_id": ["T01"], "episode_id": ["T01/f1"],
                                      "failure_at": [self.fault], "recovery_at": [pd.NaT]})
        self.coverage = pd.DataFrame({"turbine_id": ["T01"], "start": [self.t0], "end": [self.fault]})
        self.pending = pd.DataFrame(columns=["turbine_id", "timestamp"])
        self.outages = pd.DataFrame(columns=["turbine_id", "start", "end"])

    def labels(self, data=None):
        data = self.data if data is None else data
        state = operating_state(data, self.config["operation"])
        return mark_labels(data, state, self.episodes, self.pending, self.coverage, self.outages, self.config)

    def test_closed_7_and_30_day_endpoint_and_excluded_fault(self):
        seven = 7 * 144
        labels = self.labels()
        self.assertEqual(labels.loc[0, "h30_label"], 1)
        self.assertEqual(labels.loc[0, "h7_label"], 0)
        self.assertEqual(labels.loc[4320 - seven, "h7_label"], 1)
        self.assertEqual(labels.loc[4319 - seven, "h7_label"], 0)
        self.assertTrue(pd.isna(labels.loc[4320, "h7_label"]))

    def test_missing_slot_censors_positive_and_negative(self):
        labels = self.labels(self.data.drop(100))
        self.assertTrue(pd.isna(labels.loc[0, "h7_label"]))
        self.assertTrue(pd.isna(labels.loc[0, "h30_label"]))
        self.assertEqual(labels.loc[101, "h30_label"], 1)

    def test_right_censor_and_unverified_log(self):
        self.episodes = self.episodes.iloc[:0]
        labels = self.labels()
        self.assertEqual(labels.loc[0, "h30_label"], 0)
        self.assertTrue(pd.isna(labels.loc[1, "h30_label"]))
        self.coverage = self.coverage.iloc[:0]
        self.assertEqual(self.labels().loc[0, "h7_reason"], "unverified_log")

    def test_between_sample_outage_and_pending_failure(self):
        self.outages = pd.DataFrame({"turbine_id": ["T01"], "start": [self.t0 + pd.Timedelta(minutes=2)],
                                     "end": [self.t0 + pd.Timedelta(minutes=4)]})
        self.assertTrue(pd.isna(self.labels().loc[0, "h7_label"]))
        self.outages = self.outages.iloc[:0]
        self.pending = pd.DataFrame({"turbine_id": ["T01"], "timestamp": [self.t0 + pd.Timedelta(days=2)]})
        self.assertTrue(pd.isna(self.labels().loc[0, "h7_label"]))
        self.assertTrue(pd.isna(self.labels().loc[0, "h30_label"]))

    def test_logged_outage_between_last_slot_and_off_grid_failure(self):
        self.episodes["failure_at"] = self.t0 + pd.Timedelta(minutes=7)
        self.assertEqual(self.labels().loc[0, "h7_label"], 1)
        self.outages = pd.DataFrame({"turbine_id": ["T01"], "start": [self.t0 + pd.Timedelta(minutes=2)],
                                     "end": [self.t0 + pd.Timedelta(minutes=4)]})
        self.assertTrue(pd.isna(self.labels().loc[0, "h7_label"]))

    def test_recovery_keeps_fault_out_of_history(self):
        self.episodes["failure_at"] = self.t0 + pd.Timedelta(days=1)
        self.episodes["recovery_at"] = self.t0 + pd.Timedelta(days=2)
        labels = self.labels()
        self.assertTrue(pd.isna(labels.loc[144, "h7_label"]))
        self.assertFalse(labels.loc[288, "reviewed_history_clear"])
        self.assertTrue(labels.loc[9 * 144, "reviewed_history_clear"])


class TestFeaturesAndSplit(unittest.TestCase):
    def test_features_unchanged_by_future_or_another_turbine(self):
        cfg = config()
        data = pd.concat([telemetry(1090), telemetry(120, asset="T02")], ignore_index=True)
        before, _, ready = make_features(data, cfg)
        data.loc[1050:, "gearbox_oil_temperature"] = -900.0
        after, _, _ = make_features(data, cfg)
        pd.testing.assert_frame_equal(before.loc[:1050], after.loc[:1050])
        self.assertEqual(before.loc[100, "gearbox_oil_temperature__1h_mean"], 96.5)
        self.assertEqual(before.loc[100, "gearbox_oil_temperature__1h_trend"], 6.0)
        self.assertTrue(ready.loc[1049])
        self.assertFalse(ready.loc[1090])

    def test_purge_entire_window_and_held_out_turbine(self):
        cfg = config()
        starts = pd.date_range("2016-01-01", periods=4, freq="45D", tz="UTC")
        cfg.update(split_boundaries_utc=[t.isoformat() for t in starts], holdout_turbines=["T02"])
        stamps = pd.date_range(starts[0], starts[-1], freq="D", inclusive="left")
        frame = pd.concat([pd.DataFrame({"turbine_id": asset, "timestamp": stamps}) for asset in ("T01", "T02")],
                          ignore_index=True)
        frame = frame.assign(history_ready=True, reviewed_history_clear=True, h30_label=0, h30_episode=pd.NA)
        folds, _ = split_horizon(frame, 30, cfg)
        for n, (key, fold) in enumerate(folds.items()):
            self.assertTrue((fold["timestamp"] >= starts[n] + pd.Timedelta(days=7)).all())
            self.assertTrue((fold["timestamp"] + pd.Timedelta(days=30) < starts[n + 1]).all())
            self.assertEqual(set(fold["turbine_id"]), {"T02" if key == "test" else "T01"})

    def test_warning_episodes_and_lead_time(self):
        timestamps = pd.date_range("2016-01-01", periods=5, freq="h", tz="UTC")
        pred = pd.DataFrame({"turbine_id": "T01", "timestamp": timestamps, "label": [0, 0, 0, 1, 1],
                             "score": [0.9, 0.8, 0.1, 0.8, 0.9], "episode_id": [None, None, None, "e1", "e1"],
                             "failure_at": [pd.NaT, pd.NaT, pd.NaT, timestamps[-1] + pd.Timedelta(hours=1),
                                            timestamps[-1] + pd.Timedelta(hours=1)]})
        metrics, _ = alarm_metrics(pred, 0.5, 60)
        self.assertEqual(metrics["event_recall"], 1)
        self.assertEqual(metrics["false_warning_episodes"], 1)
        self.assertEqual(metrics["median_lead_time_hours"], 2)


class TestRealFormats(unittest.TestCase):
    def test_v1_log_keeps_off_grid_timestamps_and_pending_reviews(self):
        path = Path(__file__).resolve().parents[1] / "sources" / "edp_mendeley_v1_failures.csv"
        log, audit = load_failure_log([{"path": path}])
        self.assertEqual(audit["unique_records"], 28)
        self.assertEqual(audit["exact_duplicate_records"], 0)
        self.assertEqual(log["timestamp"].dt.year.value_counts().to_dict(), {2016: 16, 2017: 12})
        self.assertEqual(int(log["timestamp"].ne(log["timestamp"].dt.floor("10min")).sum()), 24)
        with TemporaryDirectory() as temporary:
            queue = Path(temporary) / "review.csv"
            unreviewed, _ = review_failures(log, None)
            unreviewed.to_csv(queue, index=False)
            reviewed, episodes = review_failures(log, queue)
            self.assertTrue(reviewed["decision"].eq("pending").all())
            self.assertTrue(episodes.empty)

    def test_conflicting_telemetry_leaves_gap_and_log_duplicates_collapse(self):
        cfg = config()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = telemetry(4).rename(columns={"turbine_id": "Turbine_ID", "timestamp": "Timestamp",
                                               **{key: sensor["column"] for key, sensor in cfg["sensors"].items()}})
            conflicting = raw.iloc[[2]].copy()
            conflicting.loc[:, "Grd_Prod_Pwr_Avg"] = 500.0
            raw = pd.concat([raw, raw.iloc[[0]], conflicting])
            data = root / "edp.csv"
            raw.to_csv(data, index=False, sep=";")
            cfg["signals"] = [{"path": data, "separator": ";", "year": 2016}]
            cleared, audit = load_scada(cfg)
            self.assertEqual(len(cleared), 3)
            self.assertEqual(audit["identical_duplicate_rows"], 1)
            self.assertEqual(audit["conflicting_timestamp_keys"], 1)
            self.assertEqual(audit["by_turbine"][0]["missing_grid_slots"], 1)

            source = root / "failures.xlsx"
            pd.DataFrame({"Turbine_ID": ["T01"] * 3, "Timestamp": ["2016-01-02", "2016-01-02", "2016-01-03"],
                          "Component": ["Gearbox"] * 3, "Remarks": ["failure", "failure", "same episode"]}).to_excel(
                              source, index=False)
            log, log_audit = load_failure_log([{"path": source}])
            self.assertEqual(log_audit["exact_duplicate_records"], 1)
            self.assertEqual(len(log), 2)
            reviewed = root / "review.csv"
            pd.DataFrame({"failure_key": log["failure_key"], "decision": "confirmed", "episode_id": "gb1",
                          "recovery_at": "2016-01-04", "evidence": "fixture"}).to_csv(reviewed, index=False)
            _, episodes = review_failures(log, reviewed)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes.iloc[0]["source_records"], 2)
            self.assertEqual(episodes.iloc[0]["failure_at"], pd.Timestamp("2016-01-02", tz="UTC"))


class TestColabReadiness(unittest.TestCase):
    def test_download_preflight_rejects_web_page_saved_as_workbook(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "signals_2016.xlsx"
            path.write_text("<html>Download form</html>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an XLSX workbook"):
                acquire_file({"path": path})

    def test_cache_requires_same_dataset_and_review_settings(self):
        cfg = config()
        cfg["dataset_id"] = "mendeley-v1-fixture"
        with TemporaryDirectory() as temporary:
            cache = Path(temporary)
            manifest = {"cache_version": CACHE_VERSION, "config_signature": config_signature(cfg), "review_sha256": {}}
            write_json(manifest, cache / "coverage.json")
            self.assertEqual(check_cache(cfg, cache), manifest)
            with self.assertRaisesRegex(ValueError, "Cache schema"):
                check_cache({**cfg, "dataset_id": "another-source"}, cache)
            with self.assertRaisesRegex(ValueError, "different failure_review settings"):
                check_cache({**cfg, "failure_review": "updated.csv"}, cache)
            manifest["review_sha256"]["failure_review"] = "earlier_digest"
            write_json(manifest, cache / "coverage.json")
            review = cache / "changed.csv"
            review.write_text("new review", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after preparation"):
                check_cache({**cfg, "failure_review": review}, cache)

    def test_audit_only_reports_eligible_folds_without_building_models(self):
        cfg = config()
        cfg.update(min_rows_per_class=1, min_positive_episodes={"train": 1, "validation": 1, "test": 1})
        starts = pd.to_datetime(cfg["split_boundaries_utc"], utc=True)[:-1]
        frame = pd.DataFrame({
            "turbine_id": "T01",
            "timestamp": [day + pd.Timedelta(days=offset) for day in starts for offset in (10, 11)],
            "history_ready": True, "reviewed_history_clear": True,
        })
        for days in (7, 30):
            frame[f"h{days}_label"] = [0, 1] * 3
            frame[f"h{days}_episode"] = [None, "a", None, "b", None, "c"]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            write_json({"cache_version": CACHE_VERSION, "config_signature": config_signature(cfg)},
                       cache / "coverage.json")
            write_json(feature_metadata(cfg), cache / "feature_map.json")
            frame.to_parquet(cache / "features.parquet", index=False)
            with patch("scada_pdm.learning.build_pipelines", side_effect=AssertionError("unexpected model fit")):
                report = train(cfg, cache, root / "audit", audit_only=True)
            self.assertTrue(report["audit_only"])
            self.assertEqual([horizon["status"] for horizon in report["horizons"].values()], ["eligible", "eligible"])
            self.assertEqual(list((root / "audit").rglob("*.joblib")), [])
