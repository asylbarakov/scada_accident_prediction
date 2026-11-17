"""Build the local KEGOC handoff ZIPs from the saved CARE H7 experiment."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from scada_pdm.dashboard_data import KEYS, load_experiment, load_features

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RUN = Path("results/care_h7_windows_20261116")
TRAINING = RUN / "training"
FARM = Path("data/CARE_To_Compare/Wind Farm A")
CACHE = Path("cache/care_h7_v6")
COMMON = (
    "app.py", ".streamlit/config.toml", "requirements-ml.txt", "requirements-dashboard.txt",
    "requirements-windows-lock.txt", "DATA_LICENSES.md", "scripts/verify_distribution.py",
)
DEMO_TRAINING = (
    "config.json", "metrics.json", "metrics.csv", "preparation_coverage.json", "package_versions.json",
    "h7/lightgbm/feature_importance.csv", "h7/lightgbm/sensor_importance.csv",
    "h7/lightgbm/local_shap.csv", "h7/lightgbm/risk_sensor_history.parquet",
)
DEMO_MODEL = ("pipeline.joblib", "inference.json", "validation_predictions.parquet", "test_predictions.parquet")


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_zip(target, members):
    staged = target.with_suffix(".part")
    try:
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for path, source in sorted(members.items()):
                archive.write(source, path)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def project_members(paths):
    return {Path(path).as_posix(): ROOT / path for path in paths}


def make_demo_data(staging):
    experiment = load_experiment(ROOT / RUN)
    columns = ["time_stamp", "asset_id", "train_test", *(item["column"] for item in experiment.sensors.values())]
    raw = staging / FARM / "datasets"
    raw.mkdir(parents=True)
    for event in experiment.events.values():
        path = ROOT / FARM / "datasets" / f"{event.id}.csv"
        frame = pd.read_csv(path, sep=";", usecols=columns, dtype=str, keep_default_na=False)
        frame = frame.loc[frame["train_test"].eq("prediction"), columns]
        if frame.empty:
            raise ValueError(f"No prediction telemetry in {path}")
        frame.to_csv(raw / path.name, sep=";", index=False)

    keyset = experiment.predictions[KEYS].drop_duplicates()
    names = experiment.inference["lightgbm"]["feature_names"]
    features = load_features(ROOT / CACHE / "features.parquet", names)
    features["turbine_id"] = features["turbine_id"].astype(str)
    selected = keyset.merge(features, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not selected.pop("_merge").eq("both").all():
        raise ValueError("Missing features for archived CARE H7 predictions")
    features_path = staging / CACHE / "features.parquet"
    features_path.parent.mkdir(parents=True)
    selected.to_parquet(features_path, index=False)
    return {path.relative_to(staging).as_posix(): path for path in [*raw.glob("*.csv"), features_path]}


def full_members():
    paths = [*COMMON, "README.md", "README_TEAM.md", ".gitignore", "scripts/build_distribution.py"]
    for directory, suffixes in {
        "scada_pdm": {".py"}, "tests": {".py"}, "scripts": {".py"},
        "configs": {".json"}, "docs": {".md"}, "notebooks": {".ipynb"}, "sources": {".csv"},
    }.items():
        paths.extend(path.relative_to(ROOT) for path in (ROOT / directory).glob("*") if path.suffix in suffixes)
    paths.extend(path.relative_to(ROOT) for path in (ROOT / FARM).rglob("*.csv"))
    paths.extend(path.relative_to(ROOT) for path in (ROOT / CACHE).iterdir()
                 if path.suffix in {".json", ".parquet"})
    paths.extend(path.relative_to(ROOT) for path in (ROOT / "results").rglob("*")
                 if path.is_file() and path.suffix in {".json", ".csv", ".parquet", ".png", ".md", ".txt", ".joblib", ".xlsx"}
                 and not {"synthetic_plot_fixture", "cache", "generated_inputs"}.intersection(path.parts))
    return project_members(paths)


def manifest_record(package, archive_path, entries):
    lines = [f"### {package}", "", "| Путь в ZIP | Байт | SHA-256 |", "| --- | ---: | --- |"]
    with ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(entries) or len(archive.namelist()) != len(entries):
            raise ValueError(f"Unexpected files in {archive_path}")
        for item in archive.infolist():
            with archive.open(item) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hash_file(entries[item.filename]):
                raise ValueError(f"Archived bytes differ: {item.filename}")
            lines.append(f"| `{item.filename}` | {item.file_size:,} | `{digest}` |")
    return "\n".join(lines)


def main():
    DIST.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="build-", dir=DIST) as directory:
        demo = project_members([
            *COMMON, "scada_pdm/__init__.py", "scada_pdm/dashboard_charts.py", "scada_pdm/dashboard_data.py",
            *(TRAINING / name for name in DEMO_TRAINING),
            *(TRAINING / "h7" / model / name for model in ("logistic", "lightgbm") for name in DEMO_MODEL),
        ])
        demo["README.md"] = ROOT / "docs" / "demo_readme.md"
        demo.update(make_demo_data(Path(directory)))
        packages = {"demo": demo, "full": full_members()}
        summary = ["# Манифест передачи SCADA Predictive Maintenance", "",
                   "Размеры приведены в байтах. SHA-256 отдельных записей рассчитаны по распакованным байтам.", "",
                   "| Архив | Файлов | Размер ZIP, байт | SHA-256 ZIP |", "| --- | ---: | ---: | --- |"]
        details = []
        for kind, entries in packages.items():
            archive = DIST / f"scada-pdm-mvp-{kind}.zip"
            add_zip(archive, entries)
            digest = hash_file(archive)
            summary.append(f"| [{archive.name}]({archive.name}) | {len(entries)} | {archive.stat().st_size:,} | `{digest}` |")
            print(f"{archive.name}: {archive.stat().st_size:,} bytes, {len(entries)} files, SHA-256 {digest}", flush=True)
            details.append(manifest_record(archive.name, archive, entries))
        (DIST / "MANIFEST.md").write_text("\n\n".join(["\n".join(summary), *details]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
