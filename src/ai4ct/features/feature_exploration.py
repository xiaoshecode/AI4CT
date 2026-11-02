#!/usr/bin/env python
# ai4ct.features.feature_exploration
"""
Comprehensive feature exploration for PSQA prediction.

Answers:
  Q1: What unused DICOM features correlate with GPR?
  Q2: Would regression (predict GPR value) work better than classification?
  Q3: Would more unlabeled data help? (learning curve analysis)
  Q4: What's the theoretical upper bound with current features?
"""
from __future__ import annotations
import os, sys, json, glob, math, time, warnings, numpy as np, pandas as pd
import pydicom
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from ai4ct.paths import BENCHMARK_OUTPUT, DICOM_DIR, LABEL_PATH

SEED = 42

# ================================================================
#  NEW feature extractors — things we haven't used yet
# ================================================================

def _get_bld(cp, rt):
    for b in cp.BeamLimitingDevicePositionSequence:
        if b.RTBeamLimitingDeviceType == rt: return b
    return None

def extract_new_features(dcm_path: str) -> dict:
    """Extract features NOT in the 33 original metrics."""
    ds = pydicom.dcmread(dcm_path)
    feats = {}

    # --- Beam-level ---
    for i, beam in enumerate(ds.BeamSequence):
        if beam.TreatmentDeliveryType != "TREATMENT": continue
        btag = f"B{i}"

        # 1. Nominal beam energy
        cp0 = beam.ControlPointSequence[0]
        feats[f"{btag}_Energy"] = float(cp0.NominalBeamEnergy) if hasattr(cp0,'NominalBeamEnergy') else 0.0
        feats[f"{btag}_DoseRate"] = float(cp0.DoseRateSet) if hasattr(cp0,'DoseRateSet') else 0.0

        # 2. Jaw X range & asymmetry
        jaw_x = _get_bld(cp0, 'ASYMX')
        if jaw_x and len(jaw_x.LeafJawPositions) >= 2:
            jx = [float(v) for v in jaw_x.LeafJawPositions]
            feats[f"{btag}_JawX_Width"] = abs(jx[1] - jx[0])
            feats[f"{btag}_JawX_Asymmetry"] = abs(jx[0] + jx[1])

        # 3. Jaw Y width
        jaw_y = _get_bld(cp0, 'ASYMY')
        if jaw_y and len(jaw_y.LeafJawPositions) >= 2:
            jy = [float(v) for v in jaw_y.LeafJawPositions]
            feats[f"{btag}_JawY_Width"] = abs(jy[1] - jy[0])

        # 4. Gantry trajectory statistics
        gantry_angles = []
        for cp in beam.ControlPointSequence:
            if hasattr(cp, 'GantryAngle'):
                gantry_angles.append(float(cp.GantryAngle))
        if len(gantry_angles) >= 2:
            ga = np.array(gantry_angles)
            feats[f"{btag}_Gantry_Range"] = abs(ga[-1] - ga[0])
            feats[f"{btag}_Gantry_MeanSpeed"] = np.mean(np.abs(np.diff(ga)))

        # 5. SSD variation
        ssd_vals = []
        for cp in beam.ControlPointSequence:
            if hasattr(cp, 'SourceToSurfaceDistance'):
                v = cp.SourceToSurfaceDistance
                if v is not None and str(v).strip() != '':
                    try: ssd_vals.append(float(v))
                    except: pass
        if len(ssd_vals) >= 2:
            ssd = np.array(ssd_vals)
            feats[f"{btag}_SSD_Mean"] = ssd.mean()
            feats[f"{btag}_SSD_Range"] = ssd.max() - ssd.min()
            feats[f"{btag}_SSD_Std"] = ssd.std()

        # 6. MLC leaf speed & acceleration dynamics
        mlc_cps = []
        for cp in beam.ControlPointSequence:
            mlc = _get_bld(cp, 'MLCX')
            if mlc is None: continue
            mlc_cps.append([float(v) for v in mlc.LeafJawPositions])
        if len(mlc_cps) >= 2:
            mlc_arr = np.array(mlc_cps)
            leaf_speed = np.abs(np.diff(mlc_arr, axis=0))
            feats[f"{btag}_MLC_MeanSpeed"] = leaf_speed.mean()
            feats[f"{btag}_MLC_MaxSpeed"] = leaf_speed.max()
            feats[f"{btag}_MLC_SpeedStd"] = leaf_speed.std()
            # Acceleration (2nd derivative)
            leaf_accel = np.abs(np.diff(leaf_speed, axis=0))
            feats[f"{btag}_MLC_MeanAccel"] = leaf_accel.mean() if len(leaf_accel) > 0 else 0.0
            feats[f"{btag}_MLC_MaxAccel"] = leaf_accel.max() if len(leaf_accel) > 0 else 0.0

        # 7. CP weight entropy (how "peaked" the MU distribution is)
        cp_weights = []
        for k in range(1, len(beam.ControlPointSequence)):
            w = float(beam.ControlPointSequence[k].CumulativeMetersetWeight) - \
                float(beam.ControlPointSequence[k-1].CumulativeMetersetWeight)
            cp_weights.append(w)
        if cp_weights:
            w_arr = np.array(cp_weights); w_arr = w_arr / w_arr.sum()
            feats[f"{btag}_CPWeight_Entropy"] = -np.sum(w_arr * np.log(w_arr + 1e-10))
            feats[f"{btag}_CPWeight_Std"] = w_arr.std()

        # 8. Leaf position at beam start/end (aperture opening/closing pattern)
        if len(mlc_cps) >= 2:
            start_pos = np.array(mlc_cps[0]); end_pos = np.array(mlc_cps[-1])
            n = len(start_pos)//2
            start_open = (start_pos[n:] - start_pos[:n]).mean()
            end_open = (end_pos[n:] - end_pos[:n]).mean()
            feats[f"{btag}_MLC_StartOpening"] = start_open
            feats[f"{btag}_MLC_EndOpening"] = end_open
            feats[f"{btag}_MLC_OpeningChange"] = end_open - start_open

    # --- Plan-level ---
    # 9. Beam asymmetry
    beam_keys = ['Energy','DoseRate','JawX_Width','SSD_Mean','MLC_MeanSpeed','MLC_MaxSpeed']
    for key in beam_keys:
        vals = [feats.get(f"B{i}_{key}", np.nan) for i in range(2) if f"B{i}_{key}" in feats]
        if len(vals) == 2:
            feats[f"Asymmetry_{key}"] = abs(vals[0] - vals[1])

    # 10. Fractionation
    fg = ds.FractionGroupSequence[0]
    feats["NumFractions"] = float(fg.NumberOfFractionsPlanned) if hasattr(fg,'NumberOfFractionsPlanned') \
                            else float(fg.NumberOfFractions if hasattr(fg,'NumberOfFractions') else 25)
    feats["PrescriptionDose"] = 0.0
    if hasattr(ds, 'DoseReferenceSequence'):
        for dr in ds.DoseReferenceSequence:
            if hasattr(dr, 'TargetPrescriptionDose'):
                feats["PrescriptionDose"] = max(feats["PrescriptionDose"], float(dr.TargetPrescriptionDose))

    # Fill NaN
    for k in list(feats.keys()):
        if np.isnan(feats[k]): feats[k] = 0.0

    return feats


# ================================================================
#  Combined feature extraction
# ================================================================

def extract_all_features(dcm_path: str) -> dict:
    """Combine original 33 metrics + new features."""
    from ai4ct.features.metrics_33 import extract_33_metrics
    m = extract_33_metrics(dcm_path)
    m.pop('patient_id', None)
    # Remove redundant features
    for k in ['d-PI','p-PI','d-CLS','p-CLS','d-SAS2','p-SAS2']:
        m.pop(k, None)
    # Add new features
    new = extract_new_features(dcm_path)
    m.update(new)
    return m


# ================================================================
#  Analysis
# ================================================================

def main():
    with open(LABEL_PATH, encoding='utf-8') as f: labels = json.load(f)
    files = sorted(glob.glob(os.path.join(DICOM_DIR, "RP_pID*.dcm")))[:644]

    print("Extracting features from 644 DICOM files...")
    all_feats = []
    for i, f in enumerate(files):
        m = extract_all_features(f)
        all_feats.append(m)
        if (i+1) % 200 == 0: print(f"  {i+1}/644...")

    # Align feature keys
    all_keys = sorted(all_feats[0].keys())
    X = np.array([[feat.get(k, 0.0) for k in all_keys] for feat in all_feats], dtype=np.float32)
    y_cls = np.array([1 if l['gpr_2mm_2pct'] >= 95 else 0 for l in labels[:644]], dtype=np.int64)
    y_reg = np.array([l['gpr_2mm_2pct'] for l in labels[:644]], dtype=np.float32)

    # Separate original (27) vs new features
    orig_keys = [k for k in all_keys if not any(k.startswith(p) for p in ['B0_','B1_','Asymmetry_','NumFractions','PrescriptionDose'])]
    new_keys = [k for k in all_keys if k not in orig_keys]
    print(f"\nTotal features: {len(all_keys)} ({len(orig_keys)} original + {len(new_keys)} new)")

    # ================================================================
    #  Q1: Feature correlation with GPR
    # ================================================================
    print("\n" + "="*70)
    print("  Q1: TOP FEATURES BY |CORRELATION| WITH GPR")
    print("="*70)
    corrs = [(abs(np.corrcoef(X[:,i], y_cls)[0,1]), all_keys[i], 'NEW' if all_keys[i] in new_keys else 'orig')
             for i in range(len(all_keys))]
    corrs.sort(reverse=True)
    print(f"  {'Feature':<30s} {'|r|':>6s}  {'Source':>6s}  {'AUC':>6s}")
    print("  " + "-"*52)
    for r, k, src in corrs[:15]:
        auc = roc_auc_score(y_cls, X[:, all_keys.index(k)])
        print(f"  {k:<30s} {r:>6.4f}  {src:>6s}  {auc:>6.4f}")

    # ================================================================
    #  Q2: Regression vs Classification
    # ================================================================
    print("\n" + "="*70)
    print("  Q2: REGRESSION vs CLASSIFICATION")
    print("="*70)

    # Classification
    gbdt_cls = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=SEED)
    cls_scores = cross_val_score(gbdt_cls, X, y_cls, cv=5, scoring='roc_auc')
    print(f"  Classification (predict Pass/Fail):")
    print(f"    5-fold CV AUC = {cls_scores.mean():.4f} +/- {cls_scores.std():.4f}")
    print(f"    Fold AUCs: {[f'{s:.4f}' for s in cls_scores]}")

    # Regression → then threshold
    gbdt_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=SEED)
    reg_scores_r2 = cross_val_score(gbdt_reg, X, y_reg, cv=5, scoring='r2')
    # For regression: predict GPR value, then classify with threshold=95
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    reg_cls_aucs = []
    for tr, te in kf.split(X):
        gbdt_reg.fit(X[tr], y_reg[tr])
        pred_reg = gbdt_reg.predict(X[te])
        pred_cls = (pred_reg >= 95).astype(int)
        reg_cls_aucs.append(roc_auc_score(y_cls[te], pred_cls > 0))
    print(f"\n  Regression -> threshold classification:")
    print(f"    R2 = {reg_scores_r2.mean():.4f} +/- {reg_scores_r2.std():.4f}")
    print(f"    Regression->Cls AUC = {np.mean(reg_cls_aucs):.4f} +/- {np.std(reg_cls_aucs):.4f}")

    # Direct regression baseline (always predict mean)
    baseline_mse = ((y_reg - y_reg.mean())**2).mean()
    print(f"    Baseline MSE (predict mean): {baseline_mse:.2f}")
    reg_mse_scores = cross_val_score(gbdt_reg, X, y_reg, cv=5, scoring='neg_mean_squared_error')
    print(f"    Model MSE: {-reg_mse_scores.mean():.2f} vs baseline {baseline_mse:.2f}")

    # ================================================================
    #  Q3: Learning curve — would more data help?
    # ================================================================
    print("\n" + "="*70)
    print("  Q3: LEARNING CURVE ANALYSIS")
    print("="*70)
    train_sizes, train_scores, test_scores = learning_curve(
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        X, y_cls, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc', n_jobs=-1, random_state=SEED)

    print(f"  {'Train Size':>10s}  {'Train AUC':>10s}  {'Test AUC':>10s}  {'Gap':>8s}")
    print("  " + "-"*42)
    for ts, tr, te in zip(train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)):
        print(f"  {ts:>10d}  {tr:>10.4f}  {te:>10.4f}  {tr-te:>8.4f}")

    # Extrapolate: if test score asymptotes, more data won't help
    from scipy.optimize import curve_fit
    def asymp_func(x, a, b, c): return a - b * np.exp(-c * x)
    try:
        popt, _ = curve_fit(asymp_func, train_sizes, test_scores.mean(axis=1),
                            p0=[0.6, 0.1, 0.001], maxfev=5000)
        asymptote = popt[0]
        print(f"\n  Estimated asymptote (infinite data): AUC = {asymptote:.4f}")
        if asymptote < 0.65:
            print(f"  ==> More data ALONE will not solve the problem (ceiling ~{asymptote:.3f})")
            print(f"  ==> Need BETTER features, not just MORE data")
    except:
        print(f"  Could not fit asymptote")

    # ================================================================
    #  Q4: Feature importance from GBDT
    # ================================================================
    print("\n" + "="*70)
    print("  Q4: GBDT FEATURE IMPORTANCE (Top 10)")
    print("="*70)
    gbdt = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=SEED)
    gbdt.fit(X, y_cls)
    imp = gbdt.feature_importances_
    idx = np.argsort(imp)[-10:][::-1]
    print(f"  {'Feature':<35s} {'Importance':>10s}")
    print("  " + "-"*47)
    for i in idx:
        src = 'NEW' if all_keys[i] in new_keys else 'orig'
        print(f"  {all_keys[i]:<35s} {imp[i]:>10.4f}  [{src}]")

    # ================================================================
    #  Learning curve plot
    # ================================================================
    os.makedirs(BENCHMARK_OUTPUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Learning curve
    ax = axes[0]
    ax.fill_between(train_sizes, train_scores.mean(axis=1)-train_scores.std(axis=1),
                    train_scores.mean(axis=1)+train_scores.std(axis=1), alpha=0.2, color='#2196F3')
    ax.fill_between(train_sizes, test_scores.mean(axis=1)-test_scores.std(axis=1),
                    test_scores.mean(axis=1)+test_scores.std(axis=1), alpha=0.2, color='#FF9800')
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#2196F3', label='Train AUC')
    ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='#FF9800', label='Test AUC')
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Training Samples'); ax.set_ylabel('AUC'); ax.set_title('Learning Curve (GBDT)')
    ax.legend(); ax.grid(alpha=0.3)

    # Feature importance
    ax = axes[1]
    top15_idx = np.argsort(imp)[-15:][::-1]
    ax.barh(range(15), imp[top15_idx], color=['#FF9800' if all_keys[i] in new_keys else '#2196F3' for i in top15_idx])
    ax.set_yticks(range(15)); ax.set_yticklabels([all_keys[i] for i in top15_idx], fontsize=8)
    ax.set_xlabel('Feature Importance'); ax.set_title('Top 15 Features (Orange = NEW)')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(BENCHMARK_OUTPUT / "feature_analysis.png", dpi=200)
    plt.close()
    print(f"\n  Charts saved to {BENCHMARK_OUTPUT / 'feature_analysis.png'}")

    return corrs, all_keys, X, y_cls, y_reg


if __name__ == "__main__":
    main()
