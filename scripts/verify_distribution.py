"""Check a fresh CARE H7 distribution using local artifacts and Streamlit AppTest."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from streamlit.testing.v1 import AppTest

from scada_pdm import dashboard_data
from scada_pdm.dashboard_data import KEYS, explain_point, load_experiment, load_features, load_pipelines, load_telemetry

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "care_h7_windows_20261116"
RAW = ROOT / "data" / "CARE_To_Compare" / "Wind Farm A"
FEATURES = ROOT / "cache" / "care_h7_v6" / "features.parquet"


def require(check, message):
    if not check:
        raise AssertionError(message)


def check_artifacts():
    require(Path(dashboard_data.__file__).resolve().is_relative_to(ROOT), "Imported another project's package")
    data = load_experiment(RUN)
    names = data.inference["lightgbm"]["feature_names"]
    features = load_features(FEATURES, names)
    features["turbine_id"] = features["turbine_id"].astype(str)
    pipelines = load_pipelines(data)
    for model, pipeline in pipelines.items():
        saved = data.predictions.loc[data.predictions["model"].eq(model)]
        rows = saved.merge(features, on=KEYS, how="left", validate="one_to_one", indicator=True)
        require(len(rows) == 1473 and rows["_merge"].eq("both").all(), f"{model}: incomplete archive")
        scores = pipeline.predict_proba(rows[names])[:, 1]
        np.testing.assert_allclose(scores, rows["score"], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(scores >= data.inference[model]["threshold"], rows["alert"])
        print(f"{model}: {len(rows)} archived scores and flags reproduced")

    for event in data.events.values():
        telemetry = load_telemetry(data, RAW, event.id)
        require(telemetry.from_raw and set(data.sensors) <= set(telemetry.frame), f"Missing SCADA for {event.id}")
    print(f"SCADA: {len(data.events)} files with all {len(data.sensors)} channels")
    event = data.events[22]
    timestamp = data.predictions.loc[data.predictions["source_event"].eq(event.id), "timestamp"].min()
    contributions, base, scores = explain_point(features, pipelines, data, event, timestamp, data.predictions)
    np.testing.assert_allclose(expit(base + contributions["shap_log_odds"].sum()), scores["lightgbm"],
                               rtol=0, atol=1e-10)
    print(f"Local TreeSHAP: {len(contributions)} contributions reconstruct the LightGBM score")


def check_pages():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    for page, plots in (("Overview", 1), ("SCADA Monitoring", 1), ("Failure Prediction", 1),
                        ("Model Explainability", 3)):
        app.radio(key="page").set_value(page).run()
        require(not (app.exception or app.error), f"{page}: Streamlit error")
        require(len(app.get("plotly_chart")) == plots, f"{page}: expected {plots} Plotly charts")
        print(f"{page}: {plots} Plotly chart(s)")

    app.radio(key="explain_True").set_value("Локальный SHAP").run()
    require(len(app.get("plotly_chart")) == 1 and not (app.warning or app.exception or app.error),
            "Local SHAP did not use the packaged pipelines and feature cache")
    app.radio(key="explain_True").set_value("История датчика").run()
    require(len(app.get("plotly_chart")) == 1 and not app.exception, "No significant sensor chart")
    app.radio(key="page").set_value("SCADA Monitoring").run()
    app.selectbox(key="compare_22_ambient_temperature").set_value("wind_speed").run()
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    require(len(spec["data"]) == 2 and "yaxis2" in spec["layout"], "Sensor comparison is unavailable")
    app.radio(key="page").set_value("Failure Prediction").run()
    app.radio(key="mode").set_value("Воспроизведение").run()
    require(not (app.exception or app.error) and app.slider, "Replay is unavailable")
    require(not any("Зафиксированный отказ CARE" in entry.value for entry in app.caption), "Future fault leaked")
    print("Local SHAP, ranked sensor, two-channel comparison and initial replay: passed")


if __name__ == "__main__":
    check_artifacts()
    check_pages()
