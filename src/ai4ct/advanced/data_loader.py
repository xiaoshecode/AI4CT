# ai4ct/advanced/data_loader.py
"""Load DICOM metrics, raw sequences (362 CP × 124 values), and JSON labels."""
import os, json, glob, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pydicom

from ai4ct.features.metrics_expanded import extract_plan_metrics

from .config import SEED, DICOM_DIR, LABEL_PATH, N_SAMPLES, TEST_SIZE, VAL_SIZE

def _get_bld(cp, rt):
    for b in cp.BeamLimitingDevicePositionSequence:
        if b.RTBeamLimitingDeviceType == rt: return b
    return None

def _load_labels():
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(
            f"Labels not found: {LABEL_PATH}. Restore data/raw/taichi or set AI4CT_DATA_DIR."
        )
    with open(LABEL_PATH, encoding='utf-8') as f:
        data = json.load(f)
    # Historical assumption: first 644 rows match DICOM positionally (not verified).
    return data[:N_SAMPLES]

def _extract_raw_sequence(dcm_path):
    """Return variable-length CP rows: 120 MLC + 2 jaws + gantry + MU weight."""
    ds = pydicom.dcmread(dcm_path)
    all_cps = []
    for beam in ds.BeamSequence:
        if beam.TreatmentDeliveryType != "TREATMENT": continue
        # Cache CP0 jaw
        cp0 = beam.ControlPointSequence[0]
        jaw_y0 = _get_bld(cp0, 'ASYMY')
        jy1 = abs(jaw_y0.LeafJawPositions[0]) if jaw_y0 else 200.0
        jy2 = abs(jaw_y0.LeafJawPositions[1]) if jaw_y0 else 200.0
        for cp in beam.ControlPointSequence:
            mlc = _get_bld(cp, 'MLCX')
            if mlc is None: continue
            leaf = [float(v) for v in mlc.LeafJawPositions]
            gantry = float(cp.GantryAngle) if hasattr(cp, 'GantryAngle') else 0.0
            cum_w  = float(cp.CumulativeMetersetWeight)
            all_cps.append(leaf + [jy1, jy2, gantry, cum_w])
    # Truncate/pad to 362 CPs (2×181) or keep variable
    return all_cps  # list of lists

def load_full_dataset():
    """Return (X_feat, X_raw_sequences, y, feature_names, labels_meta)"""
    labels = _load_labels()
    dcm_files = sorted(glob.glob(os.path.join(DICOM_DIR, "RP_pID*.dcm")))[:N_SAMPLES]
    if not dcm_files:
        raise FileNotFoundError(
            f"No RP_pID*.dcm files found in {DICOM_DIR}. "
            "Restore data/raw/taichi or set AI4CT_DATA_DIR."
        )
    if len(labels) < len(dcm_files):
        raise ValueError("Fewer label rows than DICOM files; verify the cohort mapping before training.")

    X_feat_list = []; X_raw_list = []; y_list = []
    all_keys = None

    for i, f in enumerate(dcm_files):
        try:
            m = extract_plan_metrics(f)
        except Exception:
            continue
        pid = m.pop('patient_id', str(i))
        feat = {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and not np.isnan(v)}
        if all_keys is None:
            all_keys = sorted(feat.keys())
        X_feat_list.append([feat.get(k, 0.0) for k in all_keys])

        raw_seq = _extract_raw_sequence(f)
        # pad/truncate to 362 CPs
        target_len = 362
        if len(raw_seq) > target_len:
            raw_seq = raw_seq[:target_len]
        else:
            seq_len = len(raw_seq[0]) if raw_seq else 124
            raw_seq += [[0.0]*seq_len] * (target_len - len(raw_seq))
        X_raw_list.append(np.array(raw_seq, dtype=np.float32))

        gpr = labels[i]['gpr_2mm_2pct']
        y_list.append(1 if gpr >= 95.0 else 0)  # 1=Pass, 0=Fail

    if not X_feat_list:
        raise RuntimeError("No DICOM features could be extracted; inspect the input cohort before training.")
    X_feat = np.array(X_feat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded {len(y)} samples, {X_feat.shape[1]} features, "
          f"raw_seq shape=({len(X_raw_list)}, {target_len}, {X_raw_list[0].shape[1]})")
    print(f"  Pass (≥95%): {(y==1).sum()}  |  Fail (<95%): {(y==0).sum()}")

    return X_feat, X_raw_list, y, all_keys, labels[:N_SAMPLES]

def split_data(X_feat, X_raw_list, y):
    """Historical split: approximately 67.5/12.5/20 with the current config.

    The original 70/10/20 description did not match VAL_SIZE/(1-TEST_SIZE).
    See docs/research_review.md before changing the experimental protocol.
    """
    # First split: train+val (80%) vs test (20%)
    indices = np.arange(len(y))
    trval_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE,
                                            random_state=SEED, stratify=y)
    # Historical second split uses VAL_SIZE as a fraction of the full cohort.
    y_trval = y[trval_idx]
    tr_idx_part, val_idx_part = train_test_split(
        np.arange(len(trval_idx)), test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=SEED, stratify=y_trval)
    train_idx = trval_idx[tr_idx_part]
    val_idx   = trval_idx[val_idx_part]

    def _subset(idx_list):
        return (X_feat[idx_list],
                [X_raw_list[i] for i in idx_list],
                y[idx_list])

    return (_subset(train_idx), _subset(val_idx), _subset(test_idx),
            train_idx, val_idx, test_idx)
