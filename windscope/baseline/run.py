"""Sequential inference CLI. Event metadata is deliberately not an argument."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from baseline.inference import FEATURE_NAMES, ScadaXGBoostAdapter, iter_measurements


def csv_record(prediction):
    result = prediction.to_dict()
    features = result.pop('features')
    for name in FEATURE_NAMES:
        result[name] = features.get(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--input', required=True, action='append', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--partition', choices=['all', 'train', 'prediction'], default='all')
    parser.add_argument('--limit', type=int, default=0, help='Pilot only; 0 means all records')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    adapter = ScadaXGBoostAdapter(args.model)
    inspection = adapter.inspection()
    (args.output/'model_inspection.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost-cpu', 'joblib',
                'threadpoolctl', 'matplotlib']
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={p: importlib.metadata.version(p) for p in packages},
                       uid=os.getuid(), network_interfaces=sorted(os.listdir('/sys/class/net')))
    (args.output/'runtime.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    scenarios = []
    for source in args.input:
        count = invalid = 0
        source_start = time.perf_counter()
        output = args.output/f'event_{source.stem}_predictions.csv'
        writer = None
        with output.open('w', encoding='utf-8', newline='') as stream:
            for measurement in iter_measurements(source, args.partition):
                if args.limit and count >= args.limit:
                    break
                result = adapter.predict(measurement)
                record = csv_record(result)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                count += 1
                invalid += result.status != 'ok'
                if count % 10000 == 0:
                    stream.flush()
                    print(json.dumps(dict(scenario=source.stem, rows=count,
                                          elapsed_seconds=round(time.perf_counter()-source_start, 2))), flush=True)
        elapsed = time.perf_counter()-source_start
        scenarios.append(dict(scenario=source.stem, input_path=str(source),
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=count,
            invalid_rows=invalid, elapsed_seconds=elapsed,
            prediction_csv=output.name, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        print(json.dumps(scenarios[-1]), flush=True)
    run = dict(partition=args.partition, limit=args.limit, sequential=True,
               future_measurements_used=False, annotations_available_to_inference=False,
               total_elapsed_seconds=time.perf_counter()-started, scenarios=scenarios)
    (args.output/'run_manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
