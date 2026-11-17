"""Deterministic technical fixture, isolated from all real experiments."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .edp import load_failure_log
from .preparation import prepare, write_json


def run_smoke(base_config, output):
    from .learning import train

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "generated_inputs"
    raw.mkdir()
    config = deepcopy(base_config)
    config.pop("dataset_id", None)
    starts = pd.date_range("2016-01-01", periods=4, freq="65D", tz="UTC")
    signals = raw / "signals.csv"
    # The hourly fixture leaves enough room to exercise both horizons and all
    # three disjoint periods within 5000 rows.
    timestamps = pd.date_range(starts[0], starts[-1], freq="h", inclusive="left")
    steps = np.arange(len(timestamps)) % (65 * 24)
    hours_to_failure = 44 * 24 - steps
    approach = np.exp(-np.abs(hours_to_failure) / 150.0)
    rng = np.random.default_rng(config["seed"])
    frame = pd.DataFrame({"Turbine_ID": "T01", "Timestamp": timestamps})
    bases = {"wind_speed": 8, "power": 450, "rotor_speed": 12, "generator_speed": 1200,
             "gearbox_oil_temperature": 55, "ambient_temperature": 17}
    for key, sensor in config["sensors"].items():
        level = bases[key]
        frame[sensor["column"]] = level * (1 + 0.25 * approach + rng.normal(0, 0.02, len(frame)))
    frame.to_csv(signals, index=False)
    failures = raw / "failure_log.xlsx"
    events = starts[:3] + pd.Timedelta(days=44)
    pd.DataFrame({"Turbine_ID": "T01", "Timestamp": events.tz_localize(None), "Component": "synthetic_gearbox",
                  "Remarks": "Synthetic fixture only"}).to_excel(failures, index=False)
    source = [{"path": str(failures), "source": "synthetic technical fixture"}]
    log, _ = load_failure_log(source)
    review_path = raw / "failure_review.csv"
    pd.DataFrame({"failure_key": log["failure_key"], "decision": "confirmed",
                  "episode_id": [f"fixture_{i}" for i in range(len(log))],
                  "recovery_at": log["timestamp"] + pd.Timedelta(days=1),
                  "evidence": "Known from synthetic generator"}).to_csv(review_path, index=False)
    coverage_path = raw / "log_coverage.csv"
    pd.DataFrame([{"turbine_id": "T01", "start": starts[0], "end": timestamps[-1],
                   "evidence": "Known complete synthetic generator"}]).to_csv(coverage_path, index=False)
    config.update(mode="synthetic_smoke", signals=[{"path": str(signals), "source": "synthetic technical fixture"}],
                  failure_logs=source, failure_review=str(review_path), log_coverage=str(coverage_path), outages=None,
                  split_boundaries_utc=[value.isoformat() for value in starts], holdout_turbines=[],
                  cadence_minutes=60, measurement_delay_steps=0, prediction_every_minutes=60, threads=2,
                  min_rows_per_class=5, min_positive_episodes={"train": 1, "validation": 1, "test": 1},
                  lightgbm={"n_estimators": 10, "learning_rate": 0.15, "num_leaves": 7, "min_child_samples": 12},
                  shap_rows=256, local_shap_rows=3, top_sensor_count=4)
    write_json(config, output / "config.json")
    cache = output / "cache"
    prepare(config, cache)
    report = train(config, cache, output / "run")
    if any(h["status"] != "trained" for h in report["horizons"].values()):
        raise RuntimeError("Synthetic smoke did not exercise every configured horizon")
    return report
