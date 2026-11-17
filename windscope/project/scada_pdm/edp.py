"""Load original EDP tables and explicit reviews of failure-log evidence."""

from __future__ import annotations

import hashlib
import re
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

KEY = ["turbine_id", "timestamp"]
LOG_COLUMNS = KEY + ["component", "remarks"]
INPUT_COLUMNS = {
    "turbine_id": ("Turbine_ID", "Turbine"),
    "timestamp": ("Timestamp", "DateTime"),
    "component": ("Component",),
    "remarks": ("Remarks", "Description"),
}


def normal_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def read_table(source, aliases, *, limit=None, optional=()):
    """Read configured columns from CSV or XLSX, filtering during ingestion."""
    path = Path(source["path"])
    patterns = {key: {normal_name(name) for name in names} for key, names in aliases.items()}

    def resolve(header):
        columns = {}
        for key, names in patterns.items():
            matches = [index for index, name in enumerate(header) if normal_name(name) in names]
            if len(matches) == 1:
                columns[key] = matches[0]
            elif matches or key not in optional:
                raise ValueError(f"{path}: expected one column for {key}, found {len(matches)}")
        return columns

    if path.suffix.lower() == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook[source["sheet"]] if "sheet" in source else workbook.worksheets[0]
            rows = sheet.iter_rows(min_row=source.get("header_row", 1), values_only=True)
            columns = resolve(next(rows, ()))
            return pd.DataFrame.from_records(
                (tuple(row[i] for i in columns.values()) for row in islice(rows, limit) if any(v is not None for v in row)),
                columns=list(columns),
            )
        finally:
            workbook.close()
    if path.suffix.lower() == ".csv":
        options = {"sep": source.get("separator", ","), "encoding": "utf-8-sig",
                   "skiprows": source.get("header_row", 1) - 1}
        header = pd.read_csv(path, nrows=0, **options).columns
        columns = resolve(header)
        selected = {header[i]: key for key, i in columns.items()}
        return pd.read_csv(path, usecols=list(selected), nrows=limit, **options).rename(columns=selected)
    raise ValueError(f"{path}: expected CSV or XLSX")


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_keys(frame, source):
    frame = frame.copy()
    frame["timestamp"] = utc(frame["timestamp"])
    frame["turbine_id"] = frame["turbine_id"].astype("string").str.strip().str.upper()
    valid = frame["timestamp"].notna() & frame["turbine_id"].str.fullmatch(r"T\d+").fillna(False)
    if "year" in source:
        valid &= frame["timestamp"].dt.year.eq(source["year"])
    return frame.loc[valid].copy(), int((~valid).sum())


def source_record(source, rows, invalid):
    path = Path(source["path"])
    checksum = sha256_file(path)
    if source.get("sha256") and checksum != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")
    record = {"file": path.name, "year": source.get("year"), "declared_source": source.get("source"),
              "sha256": checksum, "rows_read": rows, "invalid_keys": invalid}
    if "representation" in source:
        record["representation"] = source["representation"]
    return record


def load_scada(config):
    sensors = config["sensors"]
    aliases = {key: INPUT_COLUMNS[key] for key in KEY}
    aliases.update({key: (sensor["column"],) for key, sensor in sensors.items()})
    parts, records = [], []
    for source in config["signals"]:
        table = read_table(source, aliases, limit=source.get("limit_rows"))
        frame, invalid = read_keys(table, source)
        parser = lambda col: pd.to_numeric(col, errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[list(sensors)] = frame[list(sensors)].apply(parser).astype("float32")
        parts.append(frame)
        records.append(source_record(source, len(table), invalid))
    frame = pd.concat(parts, ignore_index=True)
    cadence = pd.Timedelta(minutes=config["cadence_minutes"])
    on_grid = frame["timestamp"].eq(frame["timestamp"].dt.floor(cadence))
    frame = frame.loc[on_grid]
    unique = frame.drop_duplicates()
    # Conflicting measurements at one key leave a gap in the timeline.
    conflicts = unique.duplicated(KEY, keep=False)
    frame = unique.loc[~conflicts].sort_values(KEY).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid, nonconflicting SCADA rows")
    per_asset = []
    for asset, series in frame.groupby("turbine_id", sort=True):
        stamps = series["timestamp"]
        expected = int((stamps.iloc[-1] - stamps.iloc[0]) / cadence) + 1
        per_asset.append({"turbine_id": asset, "rows": len(series), "start": stamps.iloc[0].isoformat(),
                          "end": stamps.iloc[-1].isoformat(), "missing_grid_slots": expected - len(series),
                          "gap_count": int(stamps.diff().gt(cadence).sum()),
                          "missing_by_sensor": {key: int(series[key].isna().sum()) for key in sensors}})
    audit = {"sources": records, "off_grid_dropped": int((~on_grid).sum()),
             "identical_duplicate_rows": int(on_grid.sum()) - len(unique),
             "conflicting_timestamp_keys": len(unique.loc[conflicts, KEY].drop_duplicates()),
             "by_turbine": per_asset, "rows": len(frame)}
    return frame, audit


def load_failure_log(sources):
    parts, records = [], []
    for source in sources:
        table = read_table(source, INPUT_COLUMNS, optional=("remarks",))
        frame, invalid = read_keys(table, source)
        frame["component"] = frame["component"].fillna("").astype(str).str.strip()
        frame["remarks"] = frame.get("remarks", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
        parts.append(frame[LOG_COLUMNS])
        records.append(source_record(source, len(table), invalid))
    frame = pd.concat(parts, ignore_index=True)
    if any(record["invalid_keys"] for record in records):
        raise ValueError(f"Invalid dated keys in failure log: {records}")
    frame["failure_key"] = [
        hashlib.sha256("\x1f".join((asset, timestamp.isoformat(), component.casefold(), remarks.casefold())).encode()).hexdigest()
        for asset, timestamp, component, remarks in frame.itertuples(index=False, name=None)
    ]
    distinct = frame.drop_duplicates("failure_key").sort_values(KEY).reset_index(drop=True)
    audit = {"sources": records, "unique_records": len(distinct),
             "exact_duplicate_records": len(frame) - len(distinct),
             "same_turbine_timestamp_records": int(distinct.duplicated(KEY, keep=False).sum()),
             "turbines": sorted(distinct["turbine_id"].unique().tolist())}
    return distinct, audit


def audit_operation_logs(sources):
    """Count usable alarm reset timestamps without inferring physical recovery."""
    if not sources:
        raise ValueError("Configure operation_logs to audit the alarm records")
    aliases = {"turbine_id": ("Turbine_Identifier", "UnitTitle"),
               "timestamp": ("Time_Detected", "TimeDetected"), "reset": ("Time_Reset", "TimeReset")}
    reports = []
    for source in sources:
        table = read_table(source, aliases)
        frame, invalid = read_keys(table, source)
        reset = utc(frame["reset"])
        present = frame["reset"].notna() & frame["reset"].astype(str).str.strip().ne("")
        reports.append({
            **source_record(source, len(table), invalid),
            "turbines": sorted(frame["turbine_id"].unique().tolist()),
            "nonempty_reset_fields": int(present.sum()),
            "parseable_reset_timestamps": int(reset.notna().sum()),
            "reset_at_or_after_detection": int(reset.ge(frame["timestamp"]).sum()),
        })
    return reports


def read_intervals(path, *, coverage):
    columns = ["turbine_id", "start", "end", "evidence"]
    if not path:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)[columns]
    frame["turbine_id"] = frame["turbine_id"].str.strip().str.upper()
    frame["start"], frame["end"] = utc(frame["start"]), utc(frame["end"])
    invalid = (frame[["start", "end"]].isna().any(axis=1) | frame["start"].ge(frame["end"])
               | ~frame["turbine_id"].str.fullmatch(r"T\d+") | frame["evidence"].str.strip().eq(""))
    if invalid.any():
        raise ValueError(f"{path}: invalid interval or missing evidence")
    frame = frame.sort_values(["turbine_id", "start"])
    if coverage:
        previous = frame.groupby("turbine_id")["end"].shift()
        if frame["start"].le(previous).any():
            raise ValueError(f"{path}: merge overlapping coverage intervals for each turbine")
    return frame


def review_failures(log, path):
    """Unresolved rows block follow-up. Confirmed rows share a reviewed episode ID."""
    columns = ["failure_key", "decision", "episode_id", "recovery_at", "evidence"]
    review = pd.read_csv(path, dtype=str, keep_default_na=False)[columns] if path else pd.DataFrame(columns=columns)
    if review["failure_key"].duplicated().any() or not review["failure_key"].isin(log["failure_key"]).all():
        raise ValueError("Failure review contains duplicate or unknown failure keys")
    review["decision"] = review["decision"].str.strip().str.lower()
    needs_evidence = review["decision"].isin(["confirmed", "ignored"])
    if (not review["decision"].isin(["confirmed", "ignored", "pending"]).all()
            or (needs_evidence & review["evidence"].str.strip().eq("")).any()):
        raise ValueError("Provide evidence and decision confirmed or ignored for every reviewed failure")
    joined = log.merge(review, on="failure_key", how="left", validate="one_to_one")
    joined["decision"] = joined["decision"].fillna("pending")
    confirmed = joined.loc[joined["decision"].eq("confirmed")].copy()
    if confirmed["episode_id"].str.strip().eq("").any():
        raise ValueError("Confirmed failures require an episode_id")
    confirmed["recovery"] = utc(confirmed["recovery_at"])
    if (confirmed["recovery_at"].ne("") & confirmed["recovery"].isna()).any():
        raise ValueError("Unparseable recovery timestamp")
    episodes = []
    for (asset, episode), group in confirmed.groupby(["turbine_id", "episode_id"], sort=True):
        recoveries = group["recovery"].dropna().unique()
        if len(recoveries) > 1:
            raise ValueError(f"{asset}/{episode}: inconsistent recovery timestamps")
        start = group["timestamp"].min()
        recovery = recoveries[0] if len(recoveries) else pd.NaT
        if pd.notna(recovery) and recovery <= group["timestamp"].max():
            raise ValueError(f"{asset}/{episode}: recovery must follow all episode records")
        episodes.append((asset, f"{asset}/{episode}", start, recovery, len(group)))
    episodes = pd.DataFrame(episodes, columns=["turbine_id", "episode_id", "failure_at", "recovery_at", "source_records"])
    episodes["failure_at"], episodes["recovery_at"] = utc(episodes["failure_at"]), utc(episodes["recovery_at"])
    episodes = episodes.sort_values(["turbine_id", "failure_at"])
    for asset, group in episodes.groupby("turbine_id"):
        for earlier, later in zip(group.itertuples(), islice(group.itertuples(), 1, None)):
            if pd.isna(earlier.recovery_at) or earlier.recovery_at > later.failure_at:
                raise ValueError(f"{asset}: resolve overlapping or unresolved confirmed episodes before labeling")
    return joined, episodes
