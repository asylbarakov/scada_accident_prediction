#!/usr/bin/env python3
"""Read-only CARE-to-Compare metadata/coverage audit.

Standard-library only. Deliberately does not train models, download data,
use status_type_id as a predictor, or join anonymized event timelines.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Optional

FARMS = ("A", "B", "C")
HORIZONS = (7, 30)
CADENCE = timedelta(minutes=10)
LABEL_COUNTS = ("positive", "negative", "censored", "excluded")


def parse_timestamp(value: str) -> datetime:
    """Normalise timestamps to UTC. Naive CARE dates are relative/anonymized."""
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None
            else dt.astimezone(timezone.utc))


def horizon_label(
    when: datetime,
    horizon_days: int,
    *,
    failure_at: Optional[datetime] = None,
    normal_observed_until: Optional[datetime] = None,
) -> Optional[int]:
    """Label (when, when + horizon], or None when future is unknown.

    For a failure event, we assume the event contains one documented upcoming
    failure and no other prior faults. Confirm this before using labels.
    For a normal event, lack of recorded failure is only usable up to the end
    of continuously observed normal event data (gaps checked by caller).
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if failure_at is not None:
        if when >= failure_at:
            return None  # after failure: not a pre-failure forecasting sample
        return int(when < failure_at <= when + timedelta(days=horizon_days))
    if normal_observed_until is None:
        return None
    return 0 if when + timedelta(days=horizon_days) <= normal_observed_until else None


def locate_root(raw_dir: Path, farm: str = "A") -> Path:
    raw_dir = raw_dir.resolve()
    candidates = [raw_dir, raw_dir / "CARE_To_Compare"]
    for candidate in candidates:
        if (candidate / f"Wind Farm {farm}" / "event_info.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Can't find Wind Farm {farm}/event_info.csv under {raw_dir}. "
        "Pass the extracted CARE_To_Compare directory."
    )


def load_event_info(root: Path, farm: str) -> list[dict]:
    path = root / f"Wind Farm {farm}" / "event_info.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        required = {"event_id", "event_label", "event_start", "event_end"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing metadata columns {sorted(missing)}")
        records = list(reader)
    for row in records:
        row["event_id"] = int(row["event_id"])
        row["event_label"] = row["event_label"].strip().lower()
        for key in ("event_start", "event_end"):
            row[key] = parse_timestamp(row[key]) if row.get(key, "").strip() else None
    if len({r["event_id"] for r in records}) != len(records):
        raise ValueError(f"{path}: duplicate event_id")
    return records


@dataclass
class SeriesScan:
    prediction_ids: dict[datetime, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ("train", "prediction", "other", "bad_timestamp"), 0))
    assets: set[str] = field(default_factory=set)
    train_start: datetime | None = None
    train_end: datetime | None = None
    duplicates: int = 0


def scan_series(path: Path) -> SeriesScan:
    """Read partition, timestamp, row ID and asset. Sensor values and status are ignored."""
    scan = SeriesScan()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";", strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path}: empty file")
        try:
            t_idx = header.index("time_stamp")
            kind_idx = header.index("train_test")
        except ValueError as e:
            raise ValueError(f"{path}: expected time_stamp/train_test columns") from e
        id_idx = header.index("id") if "id" in header else None
        asset_idx = next((header.index(key) for key in ("asset_id", "asset") if key in header), None)
        for parts in reader:
            if len(parts) != len(header):
                scan.counts["bad_timestamp"] += 1
                continue
            kind = parts[kind_idx].strip().lower()
            if kind == "test":
                kind = "prediction"
            if kind not in ("train", "prediction"):
                scan.counts["other"] += 1
                continue
            scan.counts[kind] += 1
            if asset_idx is not None:
                scan.assets.add(parts[asset_idx].strip())
            try:
                ts = parse_timestamp(parts[t_idx])
            except ValueError:
                scan.counts["bad_timestamp"] += 1
                continue
            if kind == "train":
                scan.train_start = ts if scan.train_start is None else min(ts, scan.train_start)
                scan.train_end = ts if scan.train_end is None else max(ts, scan.train_end)
            else:
                scan.duplicates += ts in scan.prediction_ids
                scan.prediction_ids[ts] = parts[id_idx].strip() if id_idx is not None else ""
    return scan


def scan_prediction_timestamps(path: Path) -> tuple[list[datetime], dict]:
    scan = scan_series(path)
    return sorted(scan.prediction_ids), scan.counts


def check_boundary(row: dict, key: str, prediction_ids: dict[datetime, str]) -> str:
    timestamp = row[key]
    row_id = row.get(f"{key}_id")
    if timestamp is None:
        return "missing_timestamp"
    if timestamp not in prediction_ids:
        return "timestamp_absent_from_prediction"
    if row_id is None or not str(row_id).strip():
        return "timestamp_only"
    expected = str(row_id).strip()
    if prediction_ids[timestamp] != expected or sum(value == expected for value in prediction_ids.values()) != 1:
        return "row_id_mismatch"
    return "verified"


def summarize_event(row: dict, farm_dir: Path, horizons: tuple[int, ...], scan: bool) -> dict:
    eid = row["event_id"]
    result = {
        "event_id": eid,
        "asset": row.get("asset_id") or row.get("asset") or "unknown",
        "event_label": row["event_label"],
        "event_start": row["event_start"].isoformat() if row["event_start"] else None,
        "event_end": row["event_end"].isoformat() if row["event_end"] else None,
        "event_start_id": row.get("event_start_id"),
        "event_end_id": row.get("event_end_id"),
        "event_description": row.get("event_description"),
        "fault_anchor": "event_end" if row["event_label"] == "anomaly" else None,
        "issues": [],
    }
    known_label = row["event_label"] in ("anomaly", "normal")
    ordered_bounds = not (row["event_start"] and row["event_end"] and row["event_start"] > row["event_end"])
    if not known_label:
        result["issues"].append("unknown event_label")
    if not ordered_bounds:
        result["issues"].append("event_start > event_end")
    path = farm_dir / "datasets" / f"{eid}.csv"
    result["csv_exists"] = path.is_file()
    if not result["csv_exists"]:
        result["issues"].append("missing event CSV")
    if not scan or not result["csv_exists"]:
        return result

    try:
        series = scan_series(path)
    except (ValueError, csv.Error) as exc:
        result["issues"].append(f"cannot scan event CSV: {exc}")
        return result
    dates = sorted(series.prediction_ids)
    result["row_counts"] = series.counts
    result["train_start"] = series.train_start.isoformat() if series.train_start else None
    result["train_end"] = series.train_end.isoformat() if series.train_end else None
    result["prediction_start"] = dates[0].isoformat() if dates else None
    result["prediction_end"] = dates[-1].isoformat() if dates else None
    if not dates:
        result["issues"].append("no prediction timestamps")
        return result
    breaks = [a for a, b in zip(dates, dates[1:]) if b - a != CADENCE]
    result["duplicates"] = series.duplicates
    result["cadence_breaks"] = len(breaks)
    if series.duplicates:
        result["issues"].append("duplicated timestamps")
    if series.counts["bad_timestamp"] or series.counts["other"]:
        result["issues"].append("unusable CSV rows")
    if series.train_end is not None and series.train_end >= dates[0]:
        result["issues"].append("train and prediction overlap within event")
    if result["asset"] == "unknown" and len(series.assets) == 1 and "" not in series.assets:
        result["asset"] = next(iter(series.assets))
    asset_mismatch = bool(series.assets and series.assets != {result["asset"]})
    if asset_mismatch:
        result["issues"].append("SCADA asset does not match event metadata")
    result["boundary_checks"] = {
        key: check_boundary(row, key, series.prediction_ids) for key in ("event_start", "event_end")
    }
    for key, check in result["boundary_checks"].items():
        if check != "verified":
            result["issues"].append(f"{key}: {check}")

    failure_at = row["event_end"] if row["event_label"] == "anomaly" else None
    normal_start = row["event_start"] if row["event_label"] == "normal" else None
    normal_until = row["event_end"] if row["event_label"] == "normal" else None
    usable = (known_label and ordered_bounds and not series.duplicates and not asset_mismatch
              and (series.train_end is None or series.train_end < dates[0])
              and result["boundary_checks"]["event_end"] == "verified"
              and (failure_at is not None or result["boundary_checks"]["event_start"] == "verified"))
    result["usable_for_coverage"] = usable
    result["unconfirmed_negative_candidates_in_anomaly_events"] = {}
    for h in horizons:
        coverage = dict.fromkeys(LABEL_COUNTS, 0)
        unconfirmed = 0
        endpoint_delta = timedelta(days=h)
        for ts in dates:
            if (not usable or (failure_at is not None and ts >= failure_at)
                    or (normal_start is not None and not normal_start <= ts <= normal_until)):
                coverage["excluded"] += 1
                continue
            label = horizon_label(ts, h, failure_at=failure_at, normal_observed_until=normal_until)
            if label == 0:
                j = bisect_left(breaks, ts)
                if j < len(breaks) and breaks[j] < ts + endpoint_delta:
                    label = None
                elif failure_at is not None:
                    # One annotated failure does not prove the earlier window was fault-free.
                    unconfirmed += 1
                    label = None
            coverage[{1: "positive", 0: "negative", None: "censored"}[label]] += 1
        result[f"h{h}d"] = coverage
        result["unconfirmed_negative_candidates_in_anomaly_events"][f"h{h}d"] = unconfirmed
    return result


def overlapping_event_pairs(summaries: list[dict], *, prediction_only: bool) -> list[list[int]]:
    pairs = []
    for a, b in combinations(summaries, 2):
        if a["asset"] == "unknown" or a["asset"] != b["asset"]:
            continue
        if not a.get("prediction_start") or not b.get("prediction_start"):
            continue
        start_a = a["prediction_start"] if prediction_only else a["train_start"] or a["prediction_start"]
        start_b = b["prediction_start"] if prediction_only else b["train_start"] or b["prediction_start"]
        if max(start_a, start_b) <= min(a["prediction_end"], b["prediction_end"]):
            pairs.append([a["event_id"], b["event_id"]])
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True, help="Extracted CARE_To_Compare directory")
    p.add_argument("--farm", choices=FARMS, default="A")
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--scan-series", action="store_true", help="Scan event CSVs; can take time")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = p.parse_args()
    if any(h <= 0 for h in args.horizons) or len(args.horizons) != len(set(args.horizons)):
        p.error("--horizons requires distinct positive integers")
    root = locate_root(args.raw_dir, args.farm)
    rows = load_event_info(root, args.farm)
    summaries = [summarize_event(r, root / f"Wind Farm {args.farm}", tuple(args.horizons), args.scan_series)
                 for r in rows]
    report = {
        "source": str(root), "farm": args.farm, "scanned_series": args.scan_series,
        "warning": "Preliminary supervised label coverage. Confirm fault episodes and recovery "
                   "against the original log before training. Negative coverage comes only from "
                   "annotated normal intervals with complete 10-minute sampling. "
                   "WF A status_type_id must NOT be used as an inference feature/filter. "
                   "Anonymized event files must not be joined by timestamp.",
        "events": len(rows),
        "anomaly_events": sum(r["event_label"] == "anomaly" for r in rows),
        "normal_events": sum(r["event_label"] == "normal" for r in rows),
        "missing_csvs": sum(not r["csv_exists"] for r in summaries),
        "assets": sorted({r["asset"] for r in summaries if r["asset"] != "unknown"}),
        "per_event": summaries,
    }
    if args.scan_series:
        report["label_totals"] = {
            f"h{h}d": {c: sum(r.get(f"h{h}d", {}).get(c, 0) for r in summaries)
                         for c in LABEL_COUNTS}
            for h in args.horizons
        }
        report["horizon_events"] = {}
        for h in args.horizons:
            key = f"h{h}d"
            positive = [r for r in summaries if r.get(key, {}).get("positive")]
            negative = [r for r in summaries if r.get(key, {}).get("negative")]
            report["horizon_events"][key] = {
                "positive_events": len(positive),
                "positive_assets": sorted({r["asset"] for r in positive if r["asset"] != "unknown"}),
                "negative_events": len(negative),
            }
        report["unconfirmed_negative_candidates_in_anomaly_events"] = {
            f"h{h}d": sum(r.get("unconfirmed_negative_candidates_in_anomaly_events", {}).get(f"h{h}d", 0)
                          for r in summaries)
            for h in args.horizons
        }
        report["events_with_label_counts"] = sum(f"h{args.horizons[0]}d" in r for r in summaries)
        report["events_usable_for_coverage"] = sum(bool(r.get("usable_for_coverage")) for r in summaries)
        report["same_asset_overlapping_file_ranges"] = overlapping_event_pairs(summaries, prediction_only=False)
        report["same_asset_overlapping_prediction_ranges"] = overlapping_event_pairs(summaries, prediction_only=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Audit saved to {args.output}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
