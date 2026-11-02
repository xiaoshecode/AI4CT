#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
DICOM RT Plan Complexity Metrics Exporter
=============================================================================
Reads DICOM RT Plan files (UIH Linac / RayStation TPS) and computes
14 plan complexity metrics per patient, exported as JSON.

Usage:
    python -m ai4ct.features.export_metrics [input_dir] [--output <output_dir>]

Example:
    python -m ai4ct.features.export_metrics
    python -m ai4ct.features.export_metrics data/raw/taichi --output outputs/metrics

Metrics computed:
    MU    - Monitor Units (total)
    MCS   - Modulation Complexity Score
    LSV   - Leaf Sequence Variability
    AAV   - Aperture Area Variability
    BM    - Beam Modulation
    BA    - Beam Area (cm²)
    UAA   - Union Aperture Area (cm²)
    CA    - Circumference / Area ratio
    SAS2  - Small Aperture Score (< 2 mm)
    SAS5  - Small Aperture Score (< 5 mm)
    SAS10 - Small Aperture Score (< 10 mm)
    SAS15 - Small Aperture Score (< 15 mm)
    CLS   - Closed Leaf Score
    MAD   - Mean Asymmetry Distance (mm)

References:
    [1] McNiven et al., "A new metric for assessing IMRT modulation
        complexity and plan deliverability", Med Phys 37(2), 2010.
    [2] Du et al., "Quantification of beam complexity in intensity-modulated
        radiation therapy", Med Phys 41(2), 2014.
    [3] Younge et al., "Predicting deliverability of volumetric-modulated
        arc therapy (VMAT) plans using aperture complexity analysis",
        JACMP 17(4), 2016.
    [4] Crowe et al., "Examination of the properties of IMRT and VMAT
        beams and evaluation against pre-treatment quality assurance
        results", PMB 60(6), 2015.
=============================================================================
"""

import os
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import pydicom

from ai4ct.paths import DICOM_DIR, METRICS_OUTPUT


# ================================================================
#  Helper: find a Beam Limiting Device by type tag
# ================================================================
def get_bld(control_point, rt_type):
    """Return the BLD entry matching `rt_type` (e.g. 'MLCX', 'ASYMY'),
    or None if not found in this control point."""
    for bld in control_point.BeamLimitingDevicePositionSequence:
        if bld.RTBeamLimitingDeviceType == rt_type:
            return bld
    return None


# ================================================================
#  Metric functions  (all operate on leaf-position lists in mm)
# ================================================================

def compute_lsv(pos):
    """Leaf Sequence Variability — how smoothly adjacent leaves move."""
    pos_max = max(pos) - min(pos)
    n_pairs = len(pos) // 2

    b_left = b_right = 0.0
    for i in range(n_pairs - 1):
        b_left  += pos_max - abs(pos[i] - pos[i + 1])
        b_right += pos_max - abs(pos[i + n_pairs] - pos[i + n_pairs + 1])

    denom = (n_pairs - 1) * pos_max
    if denom == 0:
        return 0.0
    return (b_left / denom) * (b_right / denom)


def compute_aa(pos):
    """Aperture Area — sum of open area across leaf pairs (cm²)."""
    n_pairs = len(pos) // 2
    area = 0.0
    for i in range(n_pairs):
        # leaf width = 0.5 cm, DICOM units = mm → cm factor = 0.1
        area += abs(pos[i] - pos[i + n_pairs]) * 0.1 * 0.5
    return area


def track_max_openings(pos, max_left, max_right):
    """Update per-pair maximum left/right openings (for Union Aperture Area)."""
    n_pairs = len(pos) // 2
    for i in range(min(n_pairs, len(max_left))):
        val = float(pos[i]) * 0.1 * 0.5      # mm → cm² factor
        if val < max_left[i]:
            max_left[i] = val
        val_r = float(pos[i + n_pairs]) * 0.1 * 0.5
        if val_r > max_right[i]:
            max_right[i] = val_r
    return max_left, max_right


def compute_ac(pos):
    """Aperture Circumference — perimeter of the open-aperture shape (cm)."""
    n_pairs = len(pos) // 2
    perimeter = 0.0
    for i in range(n_pairs):
        # top edge
        if i == 0 or pos[i - 1] >= pos[i + n_pairs] or pos[i + n_pairs - 1] <= pos[i]:
            top = pos[i + n_pairs] - pos[i]
        else:
            top = max(pos[i - 1] - pos[i], 0) + max(pos[i + n_pairs] - pos[i + n_pairs - 1], 0)
        # bottom edge
        if i == n_pairs - 1 or pos[i + 1] >= pos[i + n_pairs] or pos[i + n_pairs + 1] <= pos[i]:
            bot = pos[i + n_pairs] - pos[i]
        else:
            bot = max(pos[i + 1] - pos[i], 0) + max(pos[i + n_pairs] - pos[i + n_pairs + 1], 0)
        perimeter += (top + bot + 10) * 0.1
    return perimeter


def compute_sas(pos, threshold_mm):
    """Small Aperture Score — fraction of open leaf pairs with gap < threshold."""
    n_pairs = len(pos) // 2
    n_open, n_small = 0, 0
    for i in range(n_pairs):
        gap = pos[i + n_pairs] - pos[i]
        if gap > 0:
            n_open += 1
            if gap < threshold_mm:
                n_small += 1
    return float(n_small) / float(n_open) if n_open > 0 else 0.0


def compute_cls(pos):
    """Closed Leaf Score — fraction of leaf pairs that are fully closed."""
    n_pairs = len(pos) // 2
    if n_pairs == 0:
        return 0.0
    closed = sum(1 for i in range(n_pairs) if pos[i + n_pairs] - pos[i] == 0)
    return float(closed) / float(n_pairs)


def compute_mad(pos):
    """Mean Asymmetry Distance — average distance from isocenter (mm)."""
    n_pairs = len(pos) // 2
    total = 0.0
    for i in range(n_pairs):
        total += abs(pos[i + n_pairs] + pos[i]) / 2.0 * 0.1  # mm → cm (×0.1)
    return total


# ================================================================
#  Split 1-cm-wide leaves into two 0.5-cm entries
# ================================================================
def split_wide_leaves(pos1, start_idx):
    """Duplicate 1-cm-wide MLC leaves so every entry represents 0.5 cm.

    For a 60-pair MLC: pairs 0-19 and 40-59 are 1.0 cm → need splitting;
    pairs 20-39 are 0.5 cm → leave as-is.

    Parameters
    ----------
    pos1 : list
        Leaf positions inside the jaw opening.
    start_idx : int
        Original MLC pair index of the first element in pos1.

    Returns
    -------
    list : pos2 with 1-cm leaves duplicated.
    """
    n_left = len(pos1) // 2           # number of leaf pairs in pos1
    pos2 = []
    for i in range(len(pos1)):
        pos2.append(pos1[i])
        # Map back to original MLC pair index
        pair_idx = start_idx + (i % n_left)
        # Outer pairs (0-19, 40-59) are 1.0 cm wide → split into two
        if pair_idx < 20 or pair_idx >= 40:
            pos2.append(pos1[i])
    return pos2


# ================================================================
#  Core: process one DICOM RT Plan file
# ================================================================
def process_plan(filepath):
    """Read a single DICOM RT Plan and return a dict of plan metrics.

    Returns
    -------
    dict  {patient_id: str, metrics: {metric_name: float}}
    """
    ds = pydicom.dcmread(filepath)
    patient_id = str(ds.PatientID)

    # ---- Extract MU per beam (keyed by BeamNumber) ----
    mu_by_beam = {}
    for ref_beam in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if not hasattr(ref_beam, 'BeamMeterset'):
            continue
        mu_by_beam[ref_beam.ReferencedBeamNumber] = ref_beam.BeamMeterset

    # ---- Ordered MU list (aligned with treatment beams) ----
    mu_list = []
    for beam in ds.BeamSequence:
        if beam.TreatmentDeliveryType != "TREATMENT":
            continue
        bn = beam.BeamNumber if hasattr(beam, 'BeamNumber') else None
        mu_list.append(mu_by_beam.get(bn, 0.0))

    total_mu = sum(mu_list)

    # ---- Accumulators for MU-weighted plan metrics ----
    plan_metrics = defaultdict(float)

    beam_idx = 0
    for beam in ds.BeamSequence:
        if beam.TreatmentDeliveryType != "TREATMENT":
            continue

        cp0 = beam.ControlPointSequence[0]

        # Jaw Y positions (ASYMY) — constant across all CPs for VMAT
        asy_my = get_bld(cp0, 'ASYMY')
        if asy_my is not None and len(asy_my.LeafJawPositions) >= 2:
            JY1 = abs(asy_my.LeafJawPositions[0])
            JY2 = abs(asy_my.LeafJawPositions[1])
        else:
            JY1 = JY2 = 200.0   # assume full jaw opening

        # MLC leaf count
        mlc0 = get_bld(cp0, 'MLCX')
        if mlc0 is None:
            continue
        total_leaves = len(mlc0.LeafJawPositions)
        half = total_leaves // 2

        # Jaw leaf-count mapping
        if JY1 <= 100:
            N1 = math.ceil(JY1 / 5)
        else:
            N1 = 20 + math.ceil((JY1 - 100) / 10)
        if JY2 <= 100:
            N2 = math.ceil(JY2 / 5)
        else:
            N2 = 20 + math.ceil((JY2 - 100) / 10)

        jaw_start = half // 2 - N1
        jaw_end   = half // 2 + N2

        # Per-beam metric accumulators  (MU-weighted)
        sum_AAV = 0.0; sum_LSV = 0.0; sum_MCS = 0.0
        sum_BA  = 0.0; sum_CA  = 0.0
        sum_SAS2  = 0.0; sum_SAS5  = 0.0
        sum_SAS10 = 0.0; sum_SAS15 = 0.0
        sum_CLS = 0.0; sum_MAD = 0.0

        # Track union aperture area per pair
        max_left  = np.zeros(half, dtype=float)
        max_right = np.zeros(half, dtype=float)

        # Store per-CP values for MU weighting
        cp_aav  = []; cp_lsv  = []; cp_aa = []; cp_ca = []
        cp_sas2 = []; cp_sas5 = []; cp_sas10 = []; cp_sas15 = []
        cp_cls  = []; cp_mad = []
        cp_mu_wrt = []    # cumulative meterset weights

        # ---- Iterate control points ----
        for cp in beam.ControlPointSequence:
            mlc = get_bld(cp, 'MLCX')
            if mlc is None:
                continue
            pos = mlc.LeafJawPositions

            # Filter leaves inside jaw opening
            pos1 = []
            for i in range(len(pos)):
                if i < jaw_start or (i > jaw_end and i < jaw_start + half) or i > jaw_end + half:
                    continue
                pos1.append(float(pos[i]))

            if len(pos1) < 2:
                continue   # all leaves under jaws

            # Split 1-cm leaves to 0.5-cm for AA / CA calculation
            pos2 = split_wide_leaves(pos1, jaw_start)

            # Individual CP metrics
            lsv_val = compute_lsv(pos1)
            aa_val  = compute_aa(pos2)
            ac_val  = compute_ac(pos2)
            ca_val  = ac_val / aa_val if aa_val > 0 else 0.0

            cp_lsv.append(lsv_val)
            cp_aa.append(aa_val)
            cp_ca.append(ca_val)
            cp_sas2.append(compute_sas(pos1, 2))
            cp_sas5.append(compute_sas(pos1, 5))
            cp_sas10.append(compute_sas(pos1, 10))
            cp_sas15.append(compute_sas(pos1, 15))
            cp_cls.append(compute_cls(pos1))
            cp_mad.append(compute_mad(pos1))
            cp_mu_wrt.append(cp.CumulativeMetersetWeight)

            max_left, max_right = track_max_openings(pos1, max_left, max_right)

        # Union Aperture Area for this beam
        UAA = float(sum(max_right - max_left))

        # Normalize AA by UAA → AAV
        cp_aav = (np.array(cp_aa, dtype=float) / UAA).tolist() if UAA > 0 else [0.0] * len(cp_aa)

        # MU weights (differential)
        mu_weights = [cp_mu_wrt[i] - cp_mu_wrt[i - 1] for i in range(1, len(cp_mu_wrt))]

        # ---- MU-weighted sum across CPs ----
        n_seg = len(mu_weights)
        for i in range(min(n_seg, len(cp_aav))):
            w = mu_weights[i]
            sum_AAV   += cp_aav[i] * w
            sum_LSV   += cp_lsv[i] * w
            sum_MCS   += cp_aav[i] * cp_lsv[i] * w
            sum_BA    += cp_aa[i] * w
            sum_CA    += cp_ca[i] * w
            sum_SAS2  += cp_sas2[i] * w
            sum_SAS5  += cp_sas5[i] * w
            sum_SAS10 += cp_sas10[i] * w
            sum_SAS15 += cp_sas15[i] * w
            sum_CLS   += cp_cls[i] * w
            sum_MAD   += cp_mad[i] * w

        # Beam Modulation
        BM = 1.0 - sum_BA / UAA if UAA > 0 else 0.0

        # Per-beam MU for cross-beam weighting
        beam_mu = mu_list[beam_idx] if beam_idx < len(mu_list) else 0.0

        # Store per-beam values (MU-weighted within beam, then × beam_MU for cross-beam weighting)
        plan_metrics['UAA']  += UAA * beam_mu
        plan_metrics['AAV']  += sum_AAV * beam_mu
        plan_metrics['LSV']  += sum_LSV * beam_mu
        plan_metrics['MCS']  += sum_MCS * beam_mu
        plan_metrics['BA']   += sum_BA * beam_mu
        plan_metrics['CA']   += sum_CA * beam_mu
        plan_metrics['SAS2']  += sum_SAS2 * beam_mu
        plan_metrics['SAS5']  += sum_SAS5 * beam_mu
        plan_metrics['SAS10'] += sum_SAS10 * beam_mu
        plan_metrics['SAS15'] += sum_SAS15 * beam_mu
        plan_metrics['CLS']  += sum_CLS * beam_mu
        plan_metrics['MAD']  += sum_MAD * beam_mu
        plan_metrics['BM']   += BM * beam_mu

        beam_idx += 1

    # ---- Normalize: MU-weighted average across beams ----
    # All metrics (including UAA) are beam-MU-weighted in plan_metrics;
    # divide by total MU to get the plan-level weighted average.
    result = {}
    for key in ['AAV', 'LSV', 'MCS', 'BA', 'CA', 'UAA',
                'SAS2', 'SAS5', 'SAS10', 'SAS15', 'CLS', 'MAD', 'BM']:
        result[key] = plan_metrics[key] / total_mu if total_mu > 0 else 0.0
    result['MU'] = total_mu

    return {'patient_id': patient_id, 'metrics': result}


# ================================================================
#  Batch processing & JSON export
# ================================================================
def process_directory(input_dir, output_dir=METRICS_OUTPUT):
    """Process all *.dcm files under `input_dir` and write JSON results."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all DICOM files
    dcm_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                dcm_files.append(os.path.join(root, f))

    if not dcm_files:
        print(f"[ERROR] No .dcm files found in {input_dir}")
        return

    print(f"Found {len(dcm_files)} DICOM file(s) in '{input_dir}'")

    # Process each file
    all_patients = {}
    errors = []
    for i, filepath in enumerate(sorted(dcm_files), 1):
        try:
            result = process_plan(filepath)
            pid = result['patient_id']
            all_patients[pid] = result['metrics']

            if i % 100 == 0 or i == len(dcm_files):
                print(f"  [{i}/{len(dcm_files)}] processed ...")
        except Exception as e:
            errors.append((os.path.basename(filepath), str(e)))
            print(f"  [WARN] {os.path.basename(filepath)}: {e}")

    # Report errors
    if errors:
        print(f"\n{len(errors)} file(s) had errors:")
        for fname, msg in errors[:10]:
            print(f"  - {fname}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # ---- Write per-metric JSON files ----
    metric_names = ['AAV', 'BA', 'BM', 'CA', 'CLS', 'LSV', 'MAD', 'MCS',
                    'MU', 'SAS2', 'SAS5', 'SAS10', 'SAS15', 'UAA']

    for metric in metric_names:
        out_path = os.path.join(output_dir, f"{metric}.json")

        # Load existing data if present
        existing = []
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as fh:
                existing = json.load(fh)

        # Build lookup of existing entries
        existing_map = {e['patient']: e for e in existing}

        # Update or insert
        for pid, metrics in all_patients.items():
            entry = {'patient': pid, 'value': metrics[metric]}
            existing_map[pid] = entry

        merged = sorted(existing_map.values(), key=lambda e: int(e['patient']))

        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)

    # ---- Write combined JSON ----
    combined = [{'patient': pid, **metrics}
                for pid, metrics in sorted(all_patients.items(),
                                           key=lambda x: int(x[0]))]
    combined_path = os.path.join(output_dir, '_all_patients.json')
    with open(combined_path, 'w', encoding='utf-8') as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)

    # ---- Summary ----
    print(f"\nDone. {len(all_patients)} patient(s) exported → '{output_dir}/'")
    print(f"  Individual metrics: {len(metric_names)} JSON files")
    print(f"  Combined:           _all_patients.json")
    return all_patients


# ================================================================
#  CLI entry point
# ================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export DICOM RT Plan complexity metrics to JSON")
    parser.add_argument(
        'input_dir', nargs='?', default=str(DICOM_DIR),
        help=f"Directory containing DICOM RT Plan files (*.dcm; default: {DICOM_DIR})")
    parser.add_argument(
        '--output', '-o', default=str(METRICS_OUTPUT),
        help=f"Output directory for JSON files (default: {METRICS_OUTPUT})")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] '{args.input_dir}' is not a valid directory")
        sys.exit(1)

    process_directory(args.input_dir, args.output)


if __name__ == "__main__":
    main()
