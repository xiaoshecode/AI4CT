#!/usr/bin/env python
# ai4ct.features.metrics_expanded
"""
Expanded plan complexity metrics (30+ features) extracted from DICOM RT Plan.

For single-layer MLC (60 pairs / 120 leaves), computes per-bank, combined,
and plan-level features.  Designed to approximate the 33-metric set used in
PSQA prediction literature (dual-layer → single-layer adaptation).

Core DICOM data path per Control Point:
  BeamSequence[i].ControlPointSequence[j]
    └ BeamLimitingDevicePositionSequence[MLCX].LeafJawPositions[0..119]
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple


# ================================================================
#  Low-level leaf geometry functions
# ================================================================

def _bank_a(pos) -> np.ndarray:
    """First half of leaf array (bank A)."""
    n = len(pos) // 2
    return np.asarray(pos[:n], dtype=float)

def _bank_b(pos) -> np.ndarray:
    """Second half of leaf array (bank B)."""
    n = len(pos) // 2
    return np.asarray(pos[n:], dtype=float)

def _openings(pos) -> np.ndarray:
    """Per-pair aperture opening = B - A (mm)."""
    n = len(pos) // 2
    return _bank_b(pos) - _bank_a(pos)


# ================================================================
#  Metric functions (operate on a list of leaf positions in mm)
# ================================================================

def compute_lsv(pos) -> float:
    """Leaf Sequence Variability."""
    p = np.asarray(pos, dtype=float)
    pmax = float(p.max() - p.min())
    if pmax == 0:
        return 0.0
    N = len(p) // 2
    bl = sum(pmax - abs(p[i] - p[i+1]) for i in range(N-1))
    br = sum(pmax - abs(p[i+N] - p[i+N+1]) for i in range(N-1))
    denom = (N - 1) * pmax
    return (bl / denom) * (br / denom)

def compute_aa(pos) -> float:
    """Aperture Area (cm²)."""
    N = len(pos) // 2
    return sum(abs(pos[i+N] - pos[i]) for i in range(N)) * 0.1 * 0.5

def compute_ac(pos) -> float:
    """Aperture Circumference (cm)."""
    N = len(pos) // 2
    p = 0.0
    for i in range(N):
        if i == 0 or pos[i-1] >= pos[i+N] or pos[i+N-1] <= pos[i]:
            top = pos[i+N] - pos[i]
        else:
            top = max(pos[i-1]-pos[i], 0) + max(pos[i+N]-pos[i+N-1], 0)
        if i == N-1 or pos[i+1] >= pos[i+N] or pos[i+N+1] <= pos[i]:
            bot = pos[i+N] - pos[i]
        else:
            bot = max(pos[i+1]-pos[i], 0) + max(pos[i+N]-pos[i+N+1], 0)
        p += (top + bot + 10) * 0.1
    return p

def compute_sas(pos, threshold_mm: float) -> float:
    """Small Aperture Score at given threshold."""
    gaps = _openings(pos)
    open_mask = gaps > 0
    if open_mask.sum() == 0:
        return 0.0
    return float((gaps[open_mask] < threshold_mm).sum()) / float(open_mask.sum())

def compute_cls(pos) -> float:
    """Closed Leaf Score."""
    N = len(pos) // 2
    if N == 0:
        return 0.0
    closed = sum(1 for i in range(N) if pos[i+N] - pos[i] == 0)
    return float(closed) / float(N)

def compute_mad(pos) -> float:
    """Mean Asymmetry Distance (mm)."""
    N = len(pos) // 2
    return sum(abs(pos[i+N] + pos[i]) for i in range(N)) / 2.0 * 0.1

def compute_max_opening(pos) -> float:
    """Maximum single-pair opening (mm)."""
    return float(_openings(pos).max())

def compute_min_opening(pos) -> float:
    """Minimum non-zero single-pair opening (mm)."""
    g = _openings(pos)
    g_pos = g[g > 0]
    return float(g_pos.min()) if len(g_pos) > 0 else 0.0

def compute_mean_opening(pos) -> float:
    """Mean opening across pairs (mm)."""
    g = _openings(pos)
    g_pos = g[g > 0]
    return float(g_pos.mean()) if len(g_pos) > 0 else 0.0

def compute_std_opening(pos) -> float:
    """Std dev of openings (mm)."""
    g = _openings(pos)
    g_pos = g[g > 0]
    return float(g_pos.std()) if len(g_pos) > 1 else 0.0

def compute_leaf_speed(pos_prev, pos_curr) -> float:
    """Mean absolute leaf speed between consecutive CPs (mm/CP)."""
    return float(np.abs(np.asarray(pos_curr) - np.asarray(pos_prev)).mean())

def compute_aav_over_uaa(cp_areas: List[float], uaa: float) -> float:
    """Aperture Area Variability = mean(AA_cp) / UAA."""
    if uaa == 0 or len(cp_areas) == 0:
        return 0.0
    return float(np.mean(cp_areas)) / uaa


# ================================================================
#  Per-beam metric extraction
# ================================================================

def extract_beam_metrics(cp_positions: List[List[float]],
                         cp_weights: List[float]) -> Dict[str, float]:
    """Extract all metrics for one beam given a list of per-CP MLC positions.

    Parameters
    ----------
    cp_positions : list of list of float
        MLC LeafJawPositions for each CP (len = N_CPs, each = 120 floats).
    cp_weights : list of float
        Differential MU weights between consecutive CPs (len = N_CPs - 1).

    Returns
    -------
    dict with metric_name → float
    """
    n_cp = len(cp_positions)
    n_seg = len(cp_weights)

    # --- per-CP metrics ---
    cp_aa = []
    cp_lsv = []
    cp_sas2 = []; cp_sas5 = []; cp_sas10 = []; cp_sas15 = []
    cp_cls = []; cp_mad = []; cp_ca = []
    cp_max_open = []; cp_mean_open = []; cp_std_open = []
    cp_leaf_speed = []

    # Track union aperture across CPs
    half = len(cp_positions[0]) // 2 if cp_positions else 60
    max_left  = np.zeros(half)
    max_right = np.zeros(half)

    for k, pos in enumerate(cp_positions):
        cp_aa.append(compute_aa(pos))
        cp_lsv.append(compute_lsv(pos))
        cp_sas2.append(compute_sas(pos, 2))
        cp_sas5.append(compute_sas(pos, 5))
        cp_sas10.append(compute_sas(pos, 10))
        cp_sas15.append(compute_sas(pos, 15))
        cp_cls.append(compute_cls(pos))
        cp_mad.append(compute_mad(pos))
        cp_ca.append(compute_ac(pos) / cp_aa[-1] if cp_aa[-1] > 0 else 0.0)
        cp_max_open.append(compute_max_opening(pos))
        cp_mean_open.append(compute_mean_opening(pos))
        cp_std_open.append(compute_std_opening(pos))

        # Track union aperture
        a = _bank_a(pos); b = _bank_b(pos)
        for i in range(min(half, len(a))):
            vl = a[i] * 0.05
            if vl < max_left[i]: max_left[i] = vl
            vr = b[i] * 0.05
            if vr > max_right[i]: max_right[i] = vr

        # Leaf speed (relative to previous CP)
        if k > 0:
            cp_leaf_speed.append(compute_leaf_speed(cp_positions[k-1], pos))

    uaa = float(sum(max_right - max_left))

    # --- MU-weighted averages ---
    def _w_avg(vals):
        if len(vals) < 2 or n_seg == 0:
            return float(np.mean(vals)) if vals else 0.0
        return float(np.average(vals[:-1], weights=cp_weights[:len(vals)-1]))

    def _w_sum(vals):
        if len(vals) < 2 or n_seg == 0:
            return float(np.sum(vals)) if vals else 0.0
        return float(np.dot(vals[:-1], cp_weights[:len(vals)-1]))

    metrics = {
        # Core plan complexity (14 metrics, per-beam)
        "AAV":   _w_avg(cp_aa) / uaa if uaa > 0 else 0.0,
        "LSV":   _w_avg(cp_lsv),
        "MCS":   (_w_avg(cp_aa) / uaa if uaa > 0 else 0.0) * _w_avg(cp_lsv),
        "BA":    _w_sum(cp_aa),
        "CA":    _w_avg(cp_ca),
        "UAA":   uaa,
        "SAS2":  _w_avg(cp_sas2),
        "SAS5":  _w_avg(cp_sas5),
        "SAS10": _w_avg(cp_sas10),
        "SAS15": _w_avg(cp_sas15),
        "CLS":   _w_avg(cp_cls),
        "MAD":   _w_avg(cp_mad),
        "BM":    1.0 - (_w_sum(cp_aa) / uaa if uaa > 0 else 0.0),

        # Opening statistics
        "MaxOpening":  _w_avg(cp_max_open),
        "MeanOpening": _w_avg(cp_mean_open),
        "StdOpening":  _w_avg(cp_std_open),
        "LeafSpeed":   _w_avg(cp_leaf_speed) if cp_leaf_speed else 0.0,
    }
    return metrics


# ================================================================
#  Per-plan metric extraction
# ================================================================

def get_bld(cp, rt_type: str):
    for bld in cp.BeamLimitingDevicePositionSequence:
        if bld.RTBeamLimitingDeviceType == rt_type:
            return bld
    return None


def extract_plan_metrics(dicom_path: str) -> Dict[str, float]:
    """Extract 30+ plan complexity metrics from a DICOM RT Plan file.

    Returns
    -------
    dict with ~30 plan-level metric_name → float values.
    """
    import pydicom
    ds = pydicom.dcmread(dicom_path)
    patient_id = str(ds.PatientID)

    # --- MU per beam ---
    mu_by_beam = {}
    for rb in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if hasattr(rb, 'BeamMeterset'):
            mu_by_beam[int(rb.ReferencedBeamNumber)] = float(rb.BeamMeterset)

    beam_metrics_list = []
    beam_mu_list = []
    total_cp_count = 0
    gantry_ranges = []

    for beam in ds.BeamSequence:
        if beam.TreatmentDeliveryType != "TREATMENT":
            continue

        bn = int(beam.BeamNumber) if hasattr(beam, 'BeamNumber') else -1
        beam_mu = mu_by_beam.get(bn, 0.0)
        beam_mu_list.append(beam_mu)

        # --- MLC positions per CP ---
        cp_positions = []
        cp_cum_weights = []
        cp_gantry = []

        for cp in beam.ControlPointSequence:
            mlc = get_bld(cp, 'MLCX')
            if mlc is None:
                continue
            pos = [float(v) for v in mlc.LeafJawPositions]
            cp_positions.append(pos)
            cp_cum_weights.append(float(cp.CumulativeMetersetWeight))
            if hasattr(cp, 'GantryAngle'):
                cp_gantry.append(float(cp.GantryAngle))

        # Differential segment weights
        seg_weights = [cp_cum_weights[i] - cp_cum_weights[i-1]
                       for i in range(1, len(cp_cum_weights))]

        # Extract per-beam metrics
        bm = extract_beam_metrics(cp_positions, seg_weights)
        bm["BeamMU"] = beam_mu
        bm["NumCPs"] = len(cp_positions)
        bm["NumSegments"] = len(seg_weights)

        if len(cp_gantry) >= 2:
            gantry_ranges.append(abs(cp_gantry[-1] - cp_gantry[0]))

        beam_metrics_list.append(bm)
        total_cp_count += len(cp_positions)

    if not beam_metrics_list:
        return {"patient_id": patient_id}

    total_mu = sum(beam_mu_list)

    # --- Aggregate across beams (MU-weighted average) ---
    agg_metrics: Dict[str, float] = {}
    for key in beam_metrics_list[0].keys():
        if key in ("BeamMU", "NumCPs", "NumSegments"):
            continue
        vals = [bm[key] for bm in beam_metrics_list]
        weights = beam_mu_list
        w_sum = sum(weights)
        agg_metrics[key] = (sum(v * w for v, w in zip(vals, weights)) / w_sum
                            if w_sum > 0 else float(np.mean(vals)))

    # --- Per-beam breakdown: add beam-0 and beam-1 metrics individually ---
    n_beams = len(beam_metrics_list)
    if n_beams >= 1:
        for key, val in beam_metrics_list[0].items():
            if key not in ("BeamMU", "NumCPs", "NumSegments"):
                agg_metrics[f"B0_{key}"] = val
    if n_beams >= 2:
        for key, val in beam_metrics_list[1].items():
            if key not in ("BeamMU", "NumCPs", "NumSegments"):
                agg_metrics[f"B1_{key}"] = val

    # --- Beam asymmetry (difference between beams) ---
    if n_beams >= 2:
        for key in ['AAV','LSV','MCS','BA','CA','UAA','SAS5','SAS10','SAS15','CLS','MAD','BM',
                     'MaxOpening','MeanOpening','StdOpening','LeafSpeed']:
            v0 = beam_metrics_list[0].get(key, 0)
            v1 = beam_metrics_list[1].get(key, 0)
            agg_metrics[f"Diff_{key}"] = abs(v0 - v1)
            agg_metrics[f"Ratio_{key}"] = v0 / v1 if v1 != 0 else 0.0

    # --- Plan-level features ---
    agg_metrics["TotalMU"] = total_mu
    agg_metrics["NumBeams"] = len(beam_metrics_list)
    agg_metrics["TotalCPs"] = total_cp_count
    agg_metrics["AvgBeamMU"] = total_mu / len(beam_metrics_list) if beam_metrics_list else 0.0
    agg_metrics["GantryRange"] = max(gantry_ranges) if gantry_ranges else 0.0
    agg_metrics["patient_id"] = patient_id

    # Compute per-bank (bank A / bank B) metrics if we want more features
    # For single-layer MLC, banks A and B are NOT independent layers,
    # but we can compute some bank-specific statistics

    return agg_metrics
