"""Plot saved CARE H7 scores and observed SCADA channels."""

import json
from pathlib import Path

import pandas as pd

from .edp import KEY


def plot_care_scada(run, output=None):
    """Export one figure per test file without fitting or scoring a model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run = Path(run)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    horizon = report["horizons"].get("h7", {})
    if report["mode"] != "care_h7" or horizon.get("status") != "trained":
        raise ValueError("Select a completed CARE H7 training output")
    lgbm = run / "h7" / "lightgbm"
    logistic = run / "h7" / "logistic"
    rank = pd.read_csv(lgbm / "sensor_importance.csv").head(config["top_sensor_count"])
    history = pd.read_parquet(lgbm / "risk_sensor_history.parquet").rename(columns={"score": "lightgbm"})
    keys = [*KEY, "source_event"]
    baseline = pd.read_parquet(logistic / "test_predictions.parquet", columns=[*keys, "score"])
    history = history.merge(baseline.rename(columns={"score": "logistic"}), on=keys, how="left", validate="one_to_one")
    if len(history) != len(baseline) or history["logistic"].isna().any():
        raise ValueError("Saved models have different CARE H7 test keys")

    directory = Path(output) if output is not None else run / "scada_plots"
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    colors = {"lightgbm": "#a43f1c", "logistic": "#385993"}
    names = {"lightgbm": "LightGBM", "logistic": "Logistic Regression"}
    for (asset, event), rows in history.groupby(["turbine_id", "source_event"], sort=True):
        timestamps = pd.date_range(rows["timestamp"].min(), rows["timestamp"].max(),
                                   freq=pd.Timedelta(minutes=config["prediction_every_minutes"]))
        timeline = rows.set_index("timestamp").reindex(timestamps)
        figure, axes = plt.subplots(1 + len(rank), 1, sharex=True, layout="constrained",
                                    figsize=(11, 2.5 + len(rank) * 1.35),
                                    gridspec_kw={"height_ratios": [1.65, *([1] * len(rank))]})
        disabled = []
        for model, color in colors.items():
            axes[0].plot(timeline.index, timeline[model], color=color, label=names[model], lw=1.35)
            threshold = horizon["models"][model]["threshold"]
            if threshold <= 1:
                axes[0].axhline(threshold, linestyle="--", color=color, lw=0.9, alpha=0.7,
                                label=f"{names[model]} threshold")
            else:
                disabled.append(names[model])
        if disabled:
            axes[0].set_title("Alerts disabled by validation threshold · " + ", ".join(disabled), loc="left", fontsize=9)
        axes[0].set_ylim(-0.025, 1.025)
        axes[0].set_ylabel("H7 score")
        for axis, item in zip(axes[1:], rank.itertuples(index=False), strict=True):
            axis.plot(timeline.index, timeline[item.sensor], color="#1c6c71", lw=1.2)
            axis.set_ylabel(item.unit)
            axis.set_title(f"{item.physical_measurement} · {item.source_column}", loc="left", fontsize=10)
        fault = rows["failure_at"].dropna().unique()
        if len(fault) > 1:
            raise ValueError(f"More than one fault anchor in CARE test event {event}")
        if len(fault):
            for position, axis in enumerate(axes):
                axis.axvline(fault[0], lw=1, color="#793480", linestyle=":",
                             label="Recorded fault" if position == 0 else None)
        axes[0].legend(loc="upper left", ncols=2, fontsize=8)
        for axis in axes:
            axis.grid(axis="x", color="#e6eaed", linewidth=0.65)
            axis.spines[["top", "right"]].set_visible(False)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1].set_xlabel("CARE timestamp (UTC)")
        figure.suptitle(f"Wind Farm A · turbine {asset} · test event {event}")
        path = directory / f"turbine_{asset}_event_{event}.png"
        figure.savefig(path, dpi=145)
        plt.close(figure)
        paths.append(path)
    return paths
