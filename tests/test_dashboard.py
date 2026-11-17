"""Checks for the read-only CARE H7 dashboard and historical replay."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from scada_pdm.dashboard_data import (
    ArtifactError, alert_summary, chart_grid, explain_point, load_experiment, load_features,
    load_pipelines, load_telemetry, read_predictions, visible_fault, visible_rows,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "care_h7_windows_20261116"
RAW = ROOT / "data" / "CARE_To_Compare" / "Wind Farm A"
FEATURES = ROOT / "cache" / "care_h7_v6" / "features.parquet"
HAS_EXPERIMENT = (RUN / "training" / "config.json").is_file()
HAS_PLOTLY = find_spec("plotly") is not None
HAS_STREAMLIT = find_spec("streamlit") is not None
ALERT_AT = pd.Timestamp("2023-08-12 10:00:00Z")


class DashboardDataTests(unittest.TestCase):
    def test_missing_run_reports_exact_required_file(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ArtifactError, "config.json"):
                load_experiment(folder)

    def test_prediction_read_rejects_alert_inconsistent_with_saved_threshold(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.parquet"
            frame = pd.DataFrame({"turbine_id": ["21"], "source_event": [22], "timestamp": [ALERT_AT],
                                  "score": [0.7], "alert": [True]})
            frame.to_parquet(path)
            with self.assertRaisesRegex(ArtifactError, "predictions.parquet"):
                read_predictions(path, 0.8)
            self.assertTrue(read_predictions(path, 0.7)["alert"].iloc[0])

    def test_graph_grid_preserves_missing_bins_and_as_of_respects_sensor_delay(self):
        frame = pd.DataFrame({"timestamp": [ALERT_AT, ALERT_AT + pd.Timedelta(minutes=20)], "sensor": [1.0, 2.0]})
        line = chart_grid(frame, "sensor", 10)
        self.assertEqual(len(line), 3)
        self.assertTrue(pd.isna(line.iloc[1]))
        snapshot = visible_rows(frame, ALERT_AT + pd.Timedelta(minutes=20), delay_minutes=10)
        self.assertEqual(snapshot["sensor"].tolist(), [1.0])
        self.assertTrue(visible_rows(frame, pd.NaT).empty)


@unittest.skipUnless(HAS_EXPERIMENT, "Archived CARE H7 experiment required")
class DashboardArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_experiment(RUN)
        cls.event = cls.data.events[22]
        cls.forecasts = cls.data.predictions.loc[cls.data.predictions["source_event"].eq(22)]

    def test_archived_models_and_first_real_warning(self):
        self.assertEqual(self.data.predictions.groupby(["fold", "model"]).size().to_dict(), {
            ("test", "lightgbm"): 644, ("test", "logistic"): 644,
            ("validation", "lightgbm"): 829, ("validation", "logistic"): 829,
        })
        summary = alert_summary(self.forecasts, self.data)
        self.assertEqual(summary["logistic"]["first_alert"], ALERT_AT)
        self.assertIsNone(summary["lightgbm"]["first_alert"])
        self.assertTrue(summary["lightgbm"]["disabled"])
        before = alert_summary(self.forecasts, self.data, ALERT_AT - pd.Timedelta(minutes=1))
        self.assertIsNone(before["logistic"]["first_alert"])

    def test_fault_markers_respect_event_group_and_replay_cutoff(self):
        self.assertIsNone(visible_fault(self.event, ALERT_AT))
        self.assertIsNone(visible_fault(self.event, pd.NaT))
        self.assertEqual(visible_fault(self.event, self.event.failure_at), self.event.failure_at)
        self.assertEqual(self.data.events[51].related_anchor, 72)
        self.assertIsNone(visible_fault(self.data.events[51], None))

    def test_missing_raw_scada_falls_back_only_where_telemetry_was_exported(self):
        with TemporaryDirectory() as folder:
            fallback = load_telemetry(self.data, folder, self.event.id)
            self.assertFalse(fallback.from_raw)
            self.assertEqual(fallback.stride_minutes, 60)
            self.assertEqual(len(fallback.frame.columns), 6)
            train = next(e for e in self.data.events.values() if e.fold == "train")
            with self.assertRaisesRegex(ArtifactError, f"{train.id}.csv"):
                load_telemetry(self.data, folder, train.id)

    @unittest.skipUnless(HAS_PLOTLY and (RAW / "datasets" / "22.csv").is_file(), "Plotly and raw SCADA required")
    def test_plotly_charts_expose_no_future_points_and_keep_physical_units(self):
        from scada_pdm.dashboard_charts import prediction_figure, telemetry_figure

        replay = visible_rows(self.forecasts, ALERT_AT)
        fault = visible_fault(self.event, ALERT_AT)
        figure = prediction_figure(replay, self.data.inference, self.data.cadence_minutes, fault)
        self.assertTrue(all(pd.to_datetime(point, utc=True) <= ALERT_AT for line in figure.data for point in line.x))
        self.assertEqual(sum(line.mode == "markers" for line in figure.data), 1)
        self.assertTrue(all(shape.yref != "paper" for shape in figure.layout.shapes))
        telemetry = load_telemetry(self.data, RAW, self.event.id)
        visible = visible_rows(telemetry.frame, ALERT_AT, telemetry.available_after_minutes)
        self.assertEqual(visible["timestamp"].iloc[-1], ALERT_AT - pd.Timedelta(minutes=10))
        channels = ["ambient_temperature", "wind_speed"]
        figure = telemetry_figure(visible, self.data.sensors, channels, telemetry.stride_minutes)
        self.assertEqual(figure.layout.yaxis.title.text, "°C")
        self.assertEqual(figure.layout.yaxis2.title.text, "m/s")
        self.assertEqual(figure.data[1].yaxis, "y2")

    @unittest.skipUnless(FEATURES.is_file(), "Prepared feature cache required")
    def test_saved_preprocessors_reproduce_both_scores_and_shap(self):
        models = load_pipelines(self.data)
        features = load_features(FEATURES, self.data.inference["lightgbm"]["feature_names"])
        items, baseline, scores = explain_point(features, models, self.data, self.event, ALERT_AT, self.forecasts)
        raw_score = baseline + items["shap_log_odds"].sum()
        np.testing.assert_allclose(expit(raw_score), scores["lightgbm"], rtol=0, atol=1e-10)
        self.assertEqual(len(items), len(self.data.inference["lightgbm"]["feature_names"]))
        expected = self.forecasts.loc[self.forecasts["timestamp"].eq(ALERT_AT)].set_index("model")["score"]
        np.testing.assert_allclose([scores[model] for model in expected.index], expected, rtol=0, atol=1e-10)

    @unittest.skipUnless(HAS_PLOTLY and HAS_STREAMLIT, "Streamlit and Plotly required")
    def test_streamlit_routes_and_replay(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        for page, plots in (("Overview", 1), ("SCADA Monitoring", 1), ("Failure Prediction", 1),
                            ("Model Explainability", 3)):
            with self.subTest(page=page):
                app.radio(key="page").set_value(page).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.get("plotly_chart")), plots)
        app.radio(key="mode").set_value("Воспроизведение").run()
        self.assertEqual(list(app.radio(key="explain_False").options), ["Локальный SHAP", "История датчика"])
        app.radio(key="page").set_value("Failure Prediction").run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(any("Зафиксированный отказ CARE" in entry.value for entry in app.caption))
        if FEATURES.is_file():
            app.radio(key="mode").set_value("История").run()
            app.radio(key="page").set_value("Model Explainability").run()
            app.radio(key="explain_True").set_value("Локальный SHAP").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertTrue(any("воспроизвели сохранённые score" in entry.value for entry in app.caption))


if __name__ == "__main__":
    unittest.main()
