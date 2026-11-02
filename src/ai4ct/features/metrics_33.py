#!/usr/bin/env python
# ai4ct.features.metrics_33
"""
Historical 33-feature adaptation; see docs/feature_definitions_review.md.

For single-layer MLC (60 pairs / 120 leaves), splits leaves into
two groups to approximate distal (pairs 0-29) and proximal (pairs 30-59).

Metrics per group (16 × 2 = 32) + Total MU (1) = 33 features.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict


# ================================================================
#  Low-level helpers
# ================================================================

def _openings(pos) -> np.ndarray:
    """Per-pair aperture = B - A (mm)."""
    n = len(pos) // 2
    return np.asarray(pos[n:], dtype=float) - np.asarray(pos[:n], dtype=float)

def _bank_a(pos) -> np.ndarray:
    return np.asarray(pos[:len(pos)//2], dtype=float)

def _bank_b(pos) -> np.ndarray:
    return np.asarray(pos[len(pos)//2:], dtype=float)


# ================================================================
#  Individual metric functions (operate on leaf position list)
# ================================================================

def compute_lsv(pos) -> float:
    """Leaf Sequence Variability (McNiven 2010)."""
    p = np.asarray(pos, dtype=float)
    pmax = float(p.max() - p.min())
    if pmax == 0: return 0.0
    N = len(p) // 2
    bl = sum(pmax - abs(p[i] - p[i+1]) for i in range(N-1))
    br = sum(pmax - abs(p[i+N] - p[i+N+1]) for i in range(N-1))
    denom = (N - 1) * pmax
    return (bl / denom) * (br / denom)

def compute_aa(pos) -> float:
    """Aperture Area (cm²)."""
    return float(_openings(pos).clip(0).sum()) * 0.1 * 0.5

def compute_ac(pos) -> float:
    """Aperture Circumference (cm)."""
    N = len(pos) // 2; p = 0.0
    for i in range(N):
        pN = pos[i+N]; pi = pos[i]
        top = pN - pi
        if i > 0 and pos[i-1] < pN and pos[i+N-1] > pi:
            top = max(pos[i-1]-pi, 0) + max(pN-pos[i+N-1], 0)
        bot = pN - pi
        if i < N-1 and pos[i+1] < pN and pos[i+N+1] > pi:
            bot = max(pos[i+1]-pi, 0) + max(pN-pos[i+N+1], 0)
        p += (top + bot + 10) * 0.1
    return p

def compute_sas(pos, threshold_mm: float) -> float:
    """Small Aperture Score at threshold (Crowe 2015)."""
    g = _openings(pos); open_mask = g > 0
    if open_mask.sum() == 0: return 0.0
    return float((g[open_mask] < threshold_mm).sum()) / float(open_mask.sum())

def compute_cls(pos) -> float:
    """Closed Leaf Score (Crowe 2014)."""
    N = len(pos) // 2
    if N == 0: return 0.0
    return float(sum(1 for i in range(N) if pos[i+N] - pos[i] == 0)) / float(N)

def compute_mad(pos) -> float:
    """Mean Asymmetry Distance (mm) (Crowe 2014)."""
    N = len(pos) // 2
    return sum(abs(pos[i+N] + pos[i]) for i in range(N)) / 2.0 * 0.1

def compute_alg(pos) -> float:
    """Average Leaf Gap (mm) (Nauta 2011)."""
    g = _openings(pos); g = g[g > 0]
    return float(g.mean()) if len(g) > 0 else 0.0

def compute_slg(pos) -> float:
    """Standard deviation of Leaf Gap (mm) (Nauta 2011)."""
    g = _openings(pos); g = g[g > 0]
    return float(g.std()) if len(g) > 1 else 0.0


# ================================================================
#  Beam-level: compute 16 metrics for one MLC layer
# ================================================================

def _compute_16(pos_list: List[np.ndarray], weights: List[float]) -> Dict[str, float]:
    """Compute 16 MU-weighted metrics from a list of per-CP MLC positions."""
    cp_aa = [compute_aa(p) for p in pos_list]
    cp_lsv = [compute_lsv(p) for p in pos_list]
    cp_ca = [compute_ac(p) / (compute_aa(p) if compute_aa(p) > 0 else 1.0) for p in pos_list]
    cp_sas2 = [compute_sas(p,2) for p in pos_list]
    cp_sas5 = [compute_sas(p,5) for p in pos_list]
    cp_sas10 = [compute_sas(p,10) for p in pos_list]
    cp_sas15 = [compute_sas(p,15) for p in pos_list]
    cp_cls = [compute_cls(p) for p in pos_list]
    cp_mad = [compute_mad(p) for p in pos_list]
    cp_alg = [compute_alg(p) for p in pos_list]
    cp_slg = [compute_slg(p) for p in pos_list]

    n_seg = len(weights)
    def _wavg(vals):
        return float(np.average(vals[:n_seg], weights=weights)) if n_seg > 0 and len(vals) >= n_seg else float(np.mean(vals))

    aa_avg = _wavg(cp_aa)

    # Track UAA
    half = len(pos_list[0]) // 2 if pos_list else 30
    ml = np.zeros(half); mr = np.zeros(half)
    for p in pos_list:
        a = _bank_a(p); b = _bank_b(p)
        for i in range(min(half, len(a))):
            vl = float(a[i]) * 0.05
            if vl < ml[i]: ml[i] = vl
            vr = float(b[i]) * 0.05
            if vr > mr[i]: mr[i] = vr
    uaa = float(sum(mr - ml))

    pi_val = _wavg(cp_ca)  # PI = C/A
    aav_val = aa_avg / uaa if uaa > 0 else 0.0
    lsv_val = _wavg(cp_lsv)
    mcs_val = aav_val * lsv_val

    return {
        "AAV": aav_val, "LSV": lsv_val, "MCS": mcs_val,
        "BA": aa_avg, "UAA": uaa,
        "PI": pi_val, "PM": 1.0 - aa_avg / uaa if uaa > 0 else 0.0,
        "CA": pi_val,
        "CLS": _wavg(cp_cls), "MAD": _wavg(cp_mad),
        "SAS2": _wavg(cp_sas2), "SAS5": _wavg(cp_sas5),
        "SAS10": _wavg(cp_sas10), "SAS15": _wavg(cp_sas15),
        "ALG": _wavg(cp_alg), "SLG": _wavg(cp_slg),
    }


# ================================================================
#  Top-level: extract 33 plan metrics from DICOM
# ================================================================

def get_bld(cp, rt_type: str):
    for b in cp.BeamLimitingDevicePositionSequence:
        if b.RTBeamLimitingDeviceType == rt_type:
            return b
    return None


def extract_33_metrics(dicom_path: str) -> Dict[str, float]:
    """Extract exactly 33 complexity metrics from one DICOM RT Plan.

    Returns dict with 33 keys:
      d-AAV, d-LSV, d-MCS, d-BA, d-UAA, d-PI, d-PM, d-CA,
      d-CLS, d-MAD, d-SAS2, d-SAS5, d-SAS10, d-SAS15, d-ALG, d-SLG,
      p-AAV, p-LSV, p-MCS, p-BA, p-UAA, p-PI, p-PM, p-CA,
      p-CLS, p-MAD, p-SAS2, p-SAS5, p-SAS10, p-SAS15, p-ALG, p-SLG,
      TotalMU
    """
    import pydicom
    ds = pydicom.dcmread(dicom_path)

    # MU per beam
    mu_by_beam = {}
    for rb in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if hasattr(rb, 'BeamMeterset'):
            mu_by_beam[int(rb.ReferencedBeamNumber)] = float(rb.BeamMeterset)

    all_d_metrics = {}  # distal: pairs 0-29
    all_p_metrics = {}  # proximal: pairs 30-59
    total_mu = 0.0
    beam_count = 0

    for beam in ds.BeamSequence:
        if beam.TreatmentDeliveryType != "TREATMENT":
            continue
        bn = int(beam.BeamNumber) if hasattr(beam, 'BeamNumber') else -1
        beam_mu = mu_by_beam.get(bn, 0.0)
        total_mu += beam_mu

        # Collect per-CP positions, split into distal/proximal
        d_positions = []  # pairs 0-29
        p_positions = []  # pairs 30-59
        cp_weights = []

        for cp in beam.ControlPointSequence:
            mlc = get_bld(cp, 'MLCX')
            if mlc is None: continue
            pos = [float(v) for v in mlc.LeafJawPositions]  # 120 values
            n_total = len(pos)  # 120
            half = n_total // 2   # 60

            # Distal: first 30 pairs → leaves 0-29 + 60-89
            d_pos = pos[:30] + pos[half:half+30]
            d_positions.append(np.array(d_pos))
            # Proximal: last 30 pairs → leaves 30-59 + 90-119
            p_pos = pos[30:60] + pos[half+30:]
            p_positions.append(np.array(p_pos))

        cp_cum = [float(cp.CumulativeMetersetWeight)
                  for cp in beam.ControlPointSequence
                  if get_bld(cp, 'MLCX') is not None]
        seg_w = [cp_cum[i] - cp_cum[i-1] for i in range(1, len(cp_cum))]

        # Compute 16 metrics for distal and proximal separately
        d_metrics = _compute_16(d_positions, seg_w)
        p_metrics = _compute_16(p_positions, seg_w)

        # Per-beam MU-weighted accumulation
        for k, v in d_metrics.items():
            all_d_metrics[k] = all_d_metrics.get(k, 0.0) + v * beam_mu
        for k, v in p_metrics.items():
            all_p_metrics[k] = all_p_metrics.get(k, 0.0) + v * beam_mu
        beam_count += 1

    # Normalize by total MU
    if total_mu > 0:
        for k in all_d_metrics: all_d_metrics[k] /= total_mu
        for k in all_p_metrics: all_p_metrics[k] /= total_mu

    # Build flat 33-feature dict
    result = {}
    for k, v in all_d_metrics.items():
        result[f"d-{k}"] = v
    for k, v in all_p_metrics.items():
        result[f"p-{k}"] = v
    result["TotalMU"] = total_mu
    result["patient_id"] = str(ds.PatientID)

    return result


# ================================================================
#  Quick test
# ================================================================
if __name__ == "__main__":
    import argparse
    from ai4ct.paths import DICOM_DIR

    parser = argparse.ArgumentParser(description="Inspect the historical 33-feature extraction.")
    parser.add_argument("dicom_path", nargs="?", help="DICOM file (default: first RP_pID*.dcm in the configured DICOM directory)")
    args = parser.parse_args()
    files = sorted(DICOM_DIR.glob("RP_pID*.dcm")) if not args.dicom_path else []
    if not args.dicom_path and not files:
        parser.error(f"No RP_pID*.dcm files found in {DICOM_DIR}")
    f = args.dicom_path or files[0]
    m = extract_33_metrics(f)
    print(f"File: {f}")
    print(f"Total keys: {len(m)} (expect 34 = 33 metrics + patient_id)")
    for k, v in sorted(m.items()):
        print(f"  {k:>20s}: {v:.4f}" if isinstance(v, float) else f"  {k:>20s}: {v}")
