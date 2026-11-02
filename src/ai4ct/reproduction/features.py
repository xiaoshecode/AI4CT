"""Versioned single-layer adaptation of Table 1, not the paper's dual-layer features.

Definitions and units: docs/paper_baseline_protocol.md. No fitted transforms,
labels, feature selection, or legacy metric functions are used here.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pydicom

FEATURE_VERSION = "taichi_single_layer_geometry_v1"
FEATURE_NAMES = ("AAV", "LSV", "MCS", "BA", "UAA", "PI", "PM", "CA", "CLS",
                 "MAD", "SAS2", "SAS5", "SAS10", "SAS15", "ALG", "SLG", "MU")
FEATURE_UNITS = {name: "dimensionless" for name in FEATURE_NAMES}
FEATURE_UNITS.update(BA="cm^2", UAA="cm^2", CA="cm^-1", MAD="mm", ALG="mm", SLG="mm", MU="MU")
EPS_MM = 1e-6
WEIGHT_TOL = 1e-7


class FeatureExtractionError(ValueError):
    """Invalid or unsupported geometry; never silently manufacture features."""


def _array(values, size=None, name="values"):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (size is not None and len(array) != size) or not np.isfinite(array).all():
        raise FeatureExtractionError(f"Invalid {name}: expected {size or 'a finite vector'} values")
    return array


def _jaw(state, devices, axis):
    keys = [key for key in (axis, "ASYM" + axis) if key in devices]
    if len(keys) > 1:
        raise FeatureExtractionError(f"Ambiguous {axis} jaw devices")
    if not keys:
        return (-np.inf, np.inf)  # No declared jaw: aperture limited by the MLC.
    if keys[0] not in state:
        raise FeatureExtractionError(f"Missing initial {axis} jaw position")
    lo, hi = _array(state[keys[0]], 2, axis + " jaw")
    if lo >= hi:
        raise FeatureExtractionError(f"Invalid {axis} jaw opening")
    return float(lo), float(hi)


def _aperture(boundaries, positions, jaw_parallel=(-np.inf, np.inf), jaw_perpendicular=(-np.inf, np.inf)):
    """Clip one leaf-row aperture to jaws. All internal distances are millimetres."""
    n = len(boundaries) - 1
    pos = _array(positions, 2 * n, "MLC positions")
    low = np.maximum(boundaries[:-1], jaw_perpendicular[0])
    high = np.minimum(boundaries[1:], jaw_perpendicular[1])
    widths = np.maximum(high - low, 0)
    left = np.maximum(pos[:n], jaw_parallel[0])
    right = np.minimum(pos[n:], jaw_parallel[1])
    gaps = np.maximum(right - left, 0)
    effective = widths > EPS_MM
    opened = effective & (gaps > EPS_MM)
    gaps = np.where(opened, gaps, 0.0)
    area = float(np.dot(gaps, widths))
    # Union of disjoint leaf-row rectangles: shared interfaces cancel twice.
    perimeter = float(np.sum(2 * (gaps[opened] + widths[opened])))
    touching = opened[:-1] & opened[1:] & (np.abs(high[:-1] - low[1:]) <= EPS_MM)
    overlap = np.maximum(np.minimum(right[:-1], right[1:]) - np.maximum(left[:-1], left[1:]), 0)
    perimeter -= float(2 * np.sum(overlap[touching]))
    indices = np.flatnonzero(opened)
    def bank_lsv(bank):
        if len(indices) < 2:
            return 1.0
        span = float(np.ptp(bank[indices]))
        adjacent = np.diff(indices) == 1
        if span <= EPS_MM or not adjacent.any():
            return 1.0
        differences = np.abs(np.diff(bank[indices]))[adjacent]
        return float(np.clip(1.0 - differences.mean() / span, 0.0, 1.0))
    lsv = bank_lsv(left) * bank_lsv(right) if opened.any() else 0.0
    values = {
        "BA": area / 100.0,
        "PI": perimeter**2 / (4 * math.pi * area) if area > 0 else 0.0,
        "CA": 10.0 * perimeter / area if area > 0 else 0.0,
        "LSV": lsv,
        "CLS": float(np.count_nonzero(effective & ~opened) / np.count_nonzero(effective)) if effective.any() else 0.0,
        "MAD": float(np.abs((left[opened] + right[opened]) / 2).mean()) if opened.any() else 0.0,
        "ALG": float(gaps[opened].mean()) if opened.any() else 0.0,
        "SLG": float(gaps[opened].std(ddof=0)) if opened.any() else 0.0,
    }
    for threshold in (2, 5, 10, 15):
        values[f"SAS{threshold}"] = float(np.mean(gaps[opened] < threshold)) if opened.any() else 0.0
    return {"low": low, "high": high, "left": left, "right": right, "opened": opened, "values": values}


def _union_area(apertures):
    """Exact union of sampled CP rectangles, including disconnected intervals.

    This is a sampled geometric union, not a claim about unsampled trajectories
    or the virtual maximum-opening denominator in other AAV implementations.
    """
    total = 0.0
    n = len(apertures[0]["opened"])
    for leaf in range(n):
        rects = [(a["low"][leaf], a["high"][leaf], a["left"][leaf], a["right"][leaf])
                 for a in apertures if a["opened"][leaf]]
        if not rects:
            continue
        rects = np.asarray(rects)
        ys = np.unique(rects[:, :2])
        for lo, hi in zip(ys[:-1], ys[1:]):
            middle = (lo + hi) / 2
            selected = rects[(rects[:, 0] < middle) & (rects[:, 1] > middle)]
            intervals = sorted((float(row[2]), float(row[3])) for row in selected)
            if not intervals:
                continue
            start, end = intervals[0]
            span = 0.0
            for left, right in intervals[1:]:
                if left > end:
                    span += end - start
                    start, end = left, right
                else:
                    end = max(end, right)
            total += (span + end - start) * (hi - lo)
    return total / 100.0


def _beam_features(beam):
    devices = {}
    for device in getattr(beam, "BeamLimitingDeviceSequence", []):
        key = str(device.RTBeamLimitingDeviceType)
        if key in devices:
            raise FeatureExtractionError("Duplicate beam-limiting device type; possible unsupported multilayer MLC")
        devices[key] = device
    mlcs = [key for key in devices if key.startswith("MLC")]
    if len(mlcs) != 1 or mlcs[0] not in {"MLCX", "MLCY"}:
        raise FeatureExtractionError("Exactly one conventional MLCX or MLCY layer is required")
    mlc = mlcs[0]
    device = devices[mlc]
    n = int(device.NumberOfLeafJawPairs)
    if n < 1:
        raise FeatureExtractionError("Invalid MLC leaf count")
    boundaries = _array(getattr(device, "LeafPositionBoundaries", []), n + 1, "LeafPositionBoundaries")
    if np.any(np.diff(boundaries) <= 0):
        raise FeatureExtractionError("LeafPositionBoundaries must strictly increase")
    cps = list(getattr(beam, "ControlPointSequence", []))
    if len(cps) < 2:
        raise FeatureExtractionError("At least two control points are required")
    if hasattr(beam, "NumberOfControlPoints") and int(beam.NumberOfControlPoints) != len(cps):
        raise FeatureExtractionError("Control-point count does not match metadata")
    state, apertures, cumulative = {}, [], []
    for expected_index, cp in enumerate(cps):
        if hasattr(cp, "ControlPointIndex") and int(cp.ControlPointIndex) != expected_index:
            raise FeatureExtractionError("Nonsequential control-point indices")
        updated = set()
        for entry in getattr(cp, "BeamLimitingDevicePositionSequence", []):
            key = str(entry.RTBeamLimitingDeviceType)
            if key in updated or key not in devices:
                raise FeatureExtractionError("Duplicate or undeclared control-point device")
            updated.add(key)
            state[key] = _array(entry.LeafJawPositions, name=key)
        if mlc not in state:
            raise FeatureExtractionError("Missing initial MLC positions")
        if not hasattr(cp, "CumulativeMetersetWeight") or cp.CumulativeMetersetWeight is None:
            raise FeatureExtractionError("Missing CumulativeMetersetWeight")
        cumulative.append(float(cp.CumulativeMetersetWeight))
        parallel_axis, perpendicular_axis = ("X", "Y") if mlc == "MLCX" else ("Y", "X")
        apertures.append(_aperture(boundaries, state[mlc],
                                  _jaw(state, devices, parallel_axis), _jaw(state, devices, perpendicular_axis)))
    cumulative = _array(cumulative, len(cps), "CumulativeMetersetWeight")
    final = float(getattr(beam, "FinalCumulativeMetersetWeight", cumulative[-1]))
    if not np.isfinite(final) or final <= 0 or abs(cumulative[0]) > WEIGHT_TOL:
        raise FeatureExtractionError("Meterset weights must start at zero and finish above zero")
    if not np.isclose(cumulative[-1], final, rtol=1e-6, atol=WEIGHT_TOL):
        raise FeatureExtractionError("Final cumulative meterset weight mismatch")
    delta = np.diff(cumulative)
    if np.any(delta < -WEIGHT_TOL):
        raise FeatureExtractionError("Cumulative meterset weights decrease")
    weights = np.maximum(delta, 0)
    if weights.sum() <= 0:
        raise FeatureExtractionError("No positive-weight segment")
    weights /= weights.sum()
    supported_cp = np.r_[weights > 0, False] | np.r_[False, weights > 0]
    uaa = _union_area([a for a, supported in zip(apertures, supported_cp) if supported])
    if uaa <= 0:
        raise FeatureExtractionError("No open aperture in delivered segments")
    result = {}
    for key in apertures[0]["values"]:
        vals = np.array([a["values"][key] for a in apertures])
        result[key] = float(np.dot((vals[:-1] + vals[1:]) / 2, weights))
    aav = np.array([a["values"]["BA"] / uaa for a in apertures])
    lsv = np.array([a["values"]["LSV"] for a in apertures])
    result["UAA"] = uaa
    result["AAV"] = float(np.dot((aav[:-1] + aav[1:]) / 2, weights))
    result["MCS"] = float(np.dot((aav[:-1] + aav[1:]) * (lsv[:-1] + lsv[1:]) / 4, weights))
    result["PM"] = 1.0 - result["BA"] / uaa
    return result


def extract_features(path: str | Path) -> dict[str, float]:
    """Return 17 finite plan-level single-layer features, without patient fields.

    Treatment-beam metrics are averaged by BeamMeterset; MU is summed. Refuse
    multilayer MLCs, ambiguous fraction groups and malformed geometry/weights.
    """
    ds = pydicom.dcmread(path)
    if str(getattr(ds, "Modality", "")) != "RTPLAN":
        raise FeatureExtractionError("Input is not an RTPLAN")
    groups = list(getattr(ds, "FractionGroupSequence", []))
    if len(groups) != 1:
        raise FeatureExtractionError("Exactly one fraction group is required")
    mu_by_beam = {}
    for rb in getattr(groups[0], "ReferencedBeamSequence", []):
        number = int(rb.ReferencedBeamNumber)
        if number in mu_by_beam:
            raise FeatureExtractionError("Duplicate beam MU reference")
        if not hasattr(rb, "BeamMeterset"):
            continue  # Setup beams can have no meterset.
        mu = float(rb.BeamMeterset)
        if not np.isfinite(mu) or mu < 0:
            raise FeatureExtractionError("Invalid BeamMeterset")
        mu_by_beam[number] = mu
    accumulated = {key: 0.0 for key in FEATURE_NAMES if key != "MU"}
    total_mu = 0.0
    seen = set()
    for beam in getattr(ds, "BeamSequence", []):
        if str(getattr(beam, "TreatmentDeliveryType", "")) != "TREATMENT":
            continue
        number = int(beam.BeamNumber)
        if number in seen:
            raise FeatureExtractionError("Duplicate treatment beam number")
        seen.add(number)
        if number not in mu_by_beam:
            raise FeatureExtractionError("Treatment beam lacks MU reference")
        mu = mu_by_beam[number]
        if mu == 0:
            continue
        values = _beam_features(beam)
        for key in accumulated:
            accumulated[key] += values[key] * mu
        total_mu += mu
    if total_mu <= 0:
        raise FeatureExtractionError("No positive-MU treatment beams")
    result = {key: (total_mu if key == "MU" else accumulated[key] / total_mu) for key in FEATURE_NAMES}
    if not all(np.isfinite(value) for value in result.values()):
        raise FeatureExtractionError("Non-finite extracted feature")
    return result
