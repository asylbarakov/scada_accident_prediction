"""Content and timestamp relationships between CARE event files."""

from hashlib import blake2b
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=10)
H7 = pd.Timedelta(days=7)


def fingerprints(values):
    """Fingerprint numeric content independently of row ID, status and timestamp."""
    numbers = np.array(values, dtype="<f8", order="C", copy=True)
    numbers[numbers == 0] = 0.0  # Canonicalize signed zero.
    numbers[~np.isfinite(numbers)] = np.nan
    return [blake2b(row, digest_size=16).digest() if present else None
            for row, present in zip(numbers, np.isfinite(numbers).any(axis=1), strict=True)]


def partition_counts(left, right):
    names = left.astype(str) + "/" + right.astype(str)
    return {str(k): int(v) for k, v in names.value_counts().sort_index().items()}


def find_overlaps(files, events):
    """Check every file pair, including telemetry whose displayed time shifted."""
    profiles, full_sets, prediction_sets = {}, {}, {}
    for event, frame in files.items():
        rows = frame.reset_index()
        profiles[event] = rows.dropna(subset=["avg_fingerprint"]).groupby("avg_fingerprint")["timestamp"].agg(
            ["min", "max"])
        full_sets[event] = pd.Index(rows["full_fingerprint"].dropna().unique())
        prediction_sets[event] = pd.Index(rows.loc[rows["partition"].eq("prediction"), "avg_fingerprint"].dropna().unique())
    results = []
    for a, b in combinations(files, 2):
        left, right = files[a], files[b]
        same_asset = events[a]["asset"] == events[b]["asset"]
        start, end = max(left.index.min(), right.index.min()), min(left.index.max(), right.index.max())
        overlapping = same_asset and start <= end
        shared = profiles[a].index.intersection(profiles[b].index)
        shared_full = full_sets[a].intersection(full_sets[b])
        if not overlapping and shared.empty and shared_full.empty:
            continue
        common = left.index.intersection(right.index)
        first, second = left.loc[common], right.loc[common]
        same_avg = first["avg_fingerprint"].notna() & first["avg_fingerprint"].eq(second["avg_fingerprint"])
        same_full = first["full_fingerprint"].notna() & first["full_fingerprint"].eq(second["full_fingerprint"])
        in_both_predictions = first["partition"].eq("prediction") & second["partition"].eq("prediction")
        synchronized_pred = common[same_avg & in_both_predictions]
        l_profile, r_profile = profiles[a].loc[shared], profiles[b].loc[shared]
        shifted = l_profile["min"].ne(r_profile["max"]) | l_profile["max"].ne(r_profile["min"])
        p1 = left.index[left["partition"].eq("prediction")]
        p2 = right.index[right["partition"].eq("prediction")]
        results.append({
            "events": [a, b], "assets": [events[a]["asset"], events[b]["asset"]],
            "same_asset_timestamp_range_overlap": [start.isoformat(), end.isoformat()] if overlapping else None,
            "same_asset_prediction_range_overlap": bool(same_asset and max(p1.min(), p2.min()) <= min(p1.max(), p2.max())),
            "same_timestamp_rows": len(common),
            "same_timestamp_partition_counts": partition_counts(first["partition"], second["partition"]),
            "same_timestamp_matching_avg": partition_counts(first.loc[same_avg, "partition"], second.loc[same_avg, "partition"]),
            "same_timestamp_matching_full": partition_counts(first.loc[same_full, "partition"], second.loc[same_full, "partition"]),
            "shared_avg_fingerprints": len(shared),
            "shared_full_fingerprints": len(shared_full),
            "shared_avg_at_different_displayed_times": int(shifted.sum()),
            "shared_prediction_avg_fingerprints": len(prediction_sets[a].intersection(prediction_sets[b])),
            "both_fault_anchors_match_in_prediction": bool(
                events[a]["event_end"] in synchronized_pred and events[b]["event_end"] in synchronized_pred),
        })
    return results


def components(nodes, edges):
    neighbors = {node: set() for node in nodes}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    groups = []
    while neighbors:
        pending, group = [min(neighbors)], []
        while pending:
            node = pending.pop()
            if node in neighbors:
                group.append(node)
                pending.extend(neighbors.pop(node))
        groups.append(sorted(group))
    return groups


def dependency_groups(events, overlaps):
    edges = [(a, b) for a, b in combinations(events, 2) if events[a]["asset"] == events[b]["asset"]]
    edges.extend(pair["events"] for pair in overlaps
                 if pair["shared_avg_fingerprints"] or pair["shared_full_fingerprints"])
    groups = components(events, edges)
    return [{"id": f"care-a-group-{members[0]}", "events": members,
             "assets": sorted({events[event]["asset"] for event in members})} for members in groups]


def fault_groups(events, overlaps):
    anomalies = {event for event, row in events.items() if row["event_label"] == "anomaly"}
    evidence = {tuple(sorted(pair["events"])): pair for pair in overlaps}
    possible_repeats, unresolved = [], []
    for a, b in combinations(sorted(anomalies), 2):
        left, right = events[a], events[b]
        if left["asset"] != right["asset"]:
            continue
        delta = abs(left["event_end"] - right["event_end"])
        pair = evidence.get((a, b), {})
        if delta <= H7 or pair.get("shared_prediction_avg_fingerprints"):
            possible_repeats.append((a, b))
            if not (delta <= H7 and pair.get("both_fault_anchors_match_in_prediction")):
                unresolved.append((a, b))
    members = components(anomalies, possible_repeats)
    groups = []
    for group in members:
        uncertain = any(a in group and b in group for a, b in unresolved)
        anchor = None if uncertain else min(group, key=lambda event: (events[event]["event_end"], event))
        groups.append({"id": f"care-a-fault-{group[0]}", "events": group, "anchor_event": anchor,
                       "anchor_timestamp": events[anchor]["event_end"].isoformat() if anchor is not None else None,
                       "status": "unresolved" if uncertain else "conservative_merge" if len(group) > 1 else "single"})
    return groups


def fault_aliases(files, events):
    """Locate telemetry identical to a fault anchor within other files."""
    anchors = defaultdict(list)
    for event, row in events.items():
        if row["event_label"] == "anomaly":
            fingerprint = files[event].at[row["event_end"], "avg_fingerprint"]
            if fingerprint is not None:
                anchors[fingerprint].append(event)
    aliases = defaultdict(list)
    for event, frame in files.items():
        matching = frame.loc[frame["avg_fingerprint"].isin(anchors)]
        for timestamp, row in matching.iterrows():
            for anchor in anchors[row["avg_fingerprint"]]:
                if anchor != event and events[anchor]["asset"] == events[event]["asset"]:
                    aliases[event].append({"anchor_event": anchor, "timestamp": timestamp,
                                           "partition": row["partition"]})
    return aliases
