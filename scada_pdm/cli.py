"""Commands used locally and by the Colab notebook."""

import argparse
import json
from pathlib import Path

from .acquisition import acquire
from .edp import audit_operation_logs
from .preparation import prepare, write_json


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    base = config.pop("extends", None)
    if base:
        parent = json.loads((path.parent / base).read_text(encoding="utf-8"))
        return {**parent, **config}
    return config


def run_prediction(model_dir, features, output):
    import joblib
    import pandas as pd

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "inference.json").read_text(encoding="utf-8"))
    names = metadata["feature_names"]
    frame = pd.read_parquet(features, columns=["turbine_id", "timestamp", "history_ready", *names])
    frame = frame.loc[frame["history_ready"]]
    pred = frame[["turbine_id", "timestamp"]].copy()
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    pred["score"] = pipeline.predict_proba(frame[names])[:, 1] if len(frame) else pd.Series(dtype=float)
    pred["alert"] = pred["score"].ge(metadata["threshold"])
    pred["horizon_days"] = metadata["horizon_days"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(output, index=False)
    return len(pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "prepare", "train", "smoke", "audit-operations"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, default=Path("configs/edp.json"))
        if command in ("prepare", "train"):
            sub.add_argument("--cache", type=Path, required=True)
        if command in ("download", "train", "smoke", "audit-operations"):
            sub.add_argument("--output", type=Path, required=True)
        if command == "train":
            sub.add_argument("--audit-only", action="store_true", help="Report split readiness without fitting any model")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model-dir", type=Path, required=True)
    prediction.add_argument("--features", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    plots = commands.add_parser("plot-scada", help="Plot saved CARE H7 test scores and sensor values")
    plots.add_argument("--run", type=Path, required=True, help="Completed output directory from train")
    plots.add_argument("--output", type=Path, help="New directory for the figures")
    args = parser.parse_args()
    if args.command == "predict":
        rows = run_prediction(args.model_dir, args.features, args.output)
        print(f"Wrote {rows} scores to {args.output}")
        return 0
    if args.command == "plot-scada":
        from .reporting import plot_care_scada

        for path in plot_care_scada(args.run, args.output):
            print(f"Wrote {path}")
        return 0
    config = load_config(args.config)
    if args.command == "download":
        if config["mode"] == "care_h7":
            from .care import ensure_care

            report = ensure_care(config)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(report, args.output)
            return 0
        records = acquire(config, args.output)
        for record in records:
            print(record["status"], record["path"], record.get("error", ""))
        return int(any(record["status"] != "available" for record in records))
    if args.command == "audit-operations":
        report = audit_operation_logs(config.get("operation_logs", []))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(report, args.output)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "prepare":
        if config["mode"] == "care_h7":
            from .care import prepare_care

            report = prepare_care(config, args.cache)
            print(json.dumps(report["h7_with_feature_history"], indent=2))
        else:
            report = prepare(config, args.cache)
            print(json.dumps(report["coverage_with_feature_history"], indent=2))
    elif args.command == "train":
        from .learning import train

        report = train(config, args.cache, args.output, audit_only=args.audit_only)
        print({key: value["status"] for key, value in report["horizons"].items()})
    else:
        from .smoke import run_smoke

        report = run_smoke(config, args.output)
        print({key: value["status"] for key, value in report["horizons"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
