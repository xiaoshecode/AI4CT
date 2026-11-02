#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
DICOM RT Plan Reader — Full Data Model & Access Interface
=============================================================================
Usage:
    from dicom_rtplan import DcmRTPlan

    from ai4ct.paths import DICOM_DIR
    plan = DcmRTPlan(DICOM_DIR / 'RP_pID001.dcm')
    print(plan.patient_id)
    print(plan.beams[0].name)
    print(plan.beams[0].control_points[0].mlc_positions)

All DICOM data elements are parsed into typed Python objects with
descriptive attribute names.  No data is left inaccessible.
=============================================================================
"""

from __future__ import annotations
import pydicom
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ================================================================
#  Leaf-level data classes (bottom-up)
# ================================================================

@dataclass
class BeamLimitingDevice:
    """A single beam-limiting device definition (jaw or MLC)."""
    device_type: str          # 'ASYMX' | 'ASYMY' | 'MLCX'
    num_leaf_pairs: int       # Number of Leaf/Jaw Pairs
    leaf_boundaries: List[float] = field(default_factory=list)  # MLC only


@dataclass
class DevicePosition:
    """Leaf/jaw positions for one device at one control point."""
    device_type: str                    # 'ASYMX' | 'ASYMY' | 'MLCX'
    positions: List[float] = field(default_factory=list)


@dataclass
class DoseRefCoefficient:
    """Cumulative dose reference at a control point."""
    ref_number: int
    coefficient: float


@dataclass
class ControlPoint:
    """A single control point within a beam."""
    index: int
    # Beam-limiting device positions (keyed by device type)
    device_positions: Dict[str, DevicePosition] = field(default_factory=dict)
    # Geometry
    gantry_angle: float = 0.0
    gantry_direction: str = 'NONE'          # CW | CC | NONE
    collimator_angle: float = 0.0
    collimator_direction: str = 'NONE'
    patient_support_angle: float = 0.0
    patient_support_direction: str = 'NONE'
    table_top_eccentric_angle: float = 0.0
    table_top_eccentric_direction: str = 'NONE'
    table_top_vertical: Optional[float] = None
    table_top_longitudinal: Optional[float] = None
    table_top_lateral: Optional[float] = None
    table_top_pitch_angle: float = 0.0
    table_top_pitch_direction: str = 'NONE'
    table_top_roll_angle: float = 0.0
    table_top_roll_direction: str = 'NONE'
    gantry_pitch_angle: float = 0.0
    gantry_pitch_direction: str = 'NONE'
    isocenter: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ssd: float = 0.0
    cumulative_meterset_weight: float = 0.0
    nominal_beam_energy: float = 0.0
    dose_rate_set: float = 0.0
    dose_refs: List[DoseRefCoefficient] = field(default_factory=list)

    # ---- convenience accessors ----
    @property
    def mlc_positions(self) -> List[float]:
        """Return MLCX leaf positions (120 values for 60-pair MLC)."""
        dp = self.device_positions.get('MLCX')
        return dp.positions if dp else []

    @property
    def jaw_x(self) -> List[float]:
        """Return ASYMX jaw positions [X1, X2] in mm."""
        dp = self.device_positions.get('ASYMX')
        return dp.positions if dp else []

    @property
    def jaw_y(self) -> List[float]:
        """Return ASYMY jaw positions [Y1, Y2] in mm."""
        dp = self.device_positions.get('ASYMY')
        return dp.positions if dp else []

    @property
    def mlc_pairs(self) -> int:
        """Number of MLC leaf pairs at this CP."""
        return len(self.mlc_positions) // 2


@dataclass
class FluenceMode:
    """Primary fluence mode of a beam."""
    mode: str        # STANDARD | NON_STANDARD
    mode_id: str     # e.g. 'FFF' for flattening-filter-free


@dataclass
class ReferencedBeam:
    """Beam reference from FractionGroupSequence (dose & MU per beam)."""
    beam_number: int
    beam_meterset: float = 0.0          # MU
    beam_dose: float = 0.0              # Gy
    dose_spec_point: List[float] = field(default_factory=list)
    dose_ref_uid: str = ''
    dose_point_depth: float = 0.0        # mm
    dose_point_equiv_depth: float = 0.0  # mm
    dose_point_ssd: float = 0.0          # mm
    dose_type: str = ''                  # PHYSICAL | EFFECTIVE | ...


@dataclass
class Beam:
    """A single treatment / setup beam."""
    number: int
    name: str
    beam_type: str                # DYNAMIC | STATIC
    radiation_type: str           # PHOTON | ELECTRON | ...
    delivery_type: str            # TREATMENT | SETUP | ...
    dosimeter_unit: str = 'MU'
    sad: float = 1000.0           # Source-Axis Distance (mm)
    final_cum_weight: float = 1.0
    num_control_points: int = 0
    num_wedges: int = 0
    num_compensators: int = 0
    num_boli: int = 0
    num_blocks: int = 0
    fluence: Optional[FluenceMode] = None
    beam_limiting_devices: List[BeamLimitingDevice] = field(default_factory=list)
    control_points: List[ControlPoint] = field(default_factory=list)
    # convenience key for matching ReferencedBeam
    referenced_setup_number: int = 1

    # ---- convenience ----
    @property
    def is_treatment(self) -> bool:
        return self.delivery_type == 'TREATMENT'

    @property
    def mlc_leaf_count(self) -> int:
        """Total MLC leaves (2 × pairs)."""
        for bld in self.beam_limiting_devices:
            if bld.device_type == 'MLCX':
                return bld.num_leaf_pairs * 2
        return 0

    @property
    def mlc_leaf_boundaries(self) -> List[float]:
        for bld in self.beam_limiting_devices:
            if bld.device_type == 'MLCX':
                return bld.leaf_boundaries
        return []

    @property
    def gantry_angles(self) -> List[float]:
        return [cp.gantry_angle for cp in self.control_points]

    @property
    def cum_weights(self) -> List[float]:
        return [cp.cumulative_meterset_weight for cp in self.control_points]


@dataclass
class DoseReference:
    """Prescription dose reference."""
    number: int
    uid: str
    ref_roi_number: int
    structure_type: str            # VOLUME | POINT | ...
    dose_type: str                 # TARGET | ORGAN_AT_RISK | ...
    target_prescription_dose: float = 0.0   # Gy
    target_underdose_vol_frac: float = 0.0  # percent


@dataclass
class FractionGroup:
    """A fraction group."""
    number: int
    num_fractions: int = 0
    num_beams: int = 0
    num_brachy_setups: int = 0
    referenced_beams: List[ReferencedBeam] = field(default_factory=list)
    ref_dose_ref_numbers: List[int] = field(default_factory=list)


@dataclass
class PatientSetup:
    """Patient positioning."""
    number: int
    position: str = ''      # HFS | HFP | FFS | FFP | ...


# ================================================================
#  Top-level DICOM RT Plan class
# ================================================================

class DcmRTPlan:
    """
    Complete DICOM RT Plan data model.

    Usage
    -----
    >>> plan = DcmRTPlan('path/to/RP_plan.dcm')
    >>> plan.patient_id
    '1'
    >>> plan.patient_name
    '1'
    >>> plan.prescription_dose_gy
    45.0
    >>> for b in plan.beams:
    ...     for cp in b.control_points:
    ...         print(cp.mlc_positions)
    """

    # ============================================================
    #  Constructor — parses everything from a DICOM file
    # ============================================================
    def __init__(self, filepath: str):
        self._filepath = filepath
        ds = pydicom.dcmread(filepath)

        # ---- top-level identity ----
        self.sop_class_uid: str            = _get_str(ds, 'SOPClassUID')
        self.sop_instance_uid: str         = _get_str(ds, 'SOPInstanceUID')
        self.modality: str                 = _get_str(ds, 'Modality')
        self.manufacturer: str             = _get_str(ds, 'Manufacturer')
        self.manufacturer_model: str       = _get_str(ds, 'ManufacturerModelName')
        self.software_versions: List[str]  = _get_str_list(ds, 'SoftwareVersions')
        self.instance_creation_date: str   = _get_str(ds, 'InstanceCreationDate')
        self.instance_creation_time: str   = _get_str(ds, 'InstanceCreationTime')
        self.study_instance_uid: str       = _get_str(ds, 'StudyInstanceUID')
        self.series_instance_uid: str      = _get_str(ds, 'SeriesInstanceUID')
        self.frame_of_reference_uid: str   = _get_str(ds, 'FrameOfReferenceUID')
        self.series_number: str           = _get_str(ds, 'SeriesNumber')
        self.approval_status: str          = _get_str(ds, 'ApprovalStatus')
        self.review_date: str              = _get_str(ds, 'ReviewDate')
        self.review_time: str              = _get_str(ds, 'ReviewTime')

        # ---- patient ----
        self.patient_name: str      = _get_str(ds, 'PatientName')
        self.patient_id: str        = _get_str(ds, 'PatientID')
        self.patient_birth_date: str = _get_str(ds, 'PatientBirthDate')
        self.patient_sex: str       = _get_str(ds, 'PatientSex')
        self.patient_identity_removed: str = _get_str(ds, 'PatientIdentityRemoved')
        self.deidentification_method: str  = _get_str(ds, 'DeidentificationMethod')

        # ---- RT Plan ----
        self.plan_label: str         = _get_str(ds, 'RTPlanLabel')
        self.plan_name: str          = _get_str(ds, 'RTPlanName')
        self.plan_intent: str        = _get_str(ds, 'PlanIntent')
        self.plan_geometry: str      = _get_str(ds, 'RTPlanGeometry')
        self.treatment_protocols: str = _get_str(ds, 'TreatmentProtocols')

        # ---- dose references ----
        self.dose_references: List[DoseReference] = []
        if hasattr(ds, 'DoseReferenceSequence'):
            for item in ds.DoseReferenceSequence:
                self.dose_references.append(DoseReference(
                    number=_get_int(item, 'DoseReferenceNumber'),
                    uid=_get_str(item, 'DoseReferenceUID'),
                    ref_roi_number=_get_int(item, 'ReferencedROINumber'),
                    structure_type=_get_str(item, 'DoseReferenceStructureType'),
                    dose_type=_get_str(item, 'DoseReferenceType'),
                    target_prescription_dose=_get_float(item, 'TargetPrescriptionDose'),
                    target_underdose_vol_frac=_get_float(item, 'TargetUnderdoseVolumeFraction'),
                ))

        # ---- fraction groups ----
        self.fraction_groups: List[FractionGroup] = []
        if hasattr(ds, 'FractionGroupSequence'):
            for fg_item in ds.FractionGroupSequence:
                fg = FractionGroup(
                    number=_get_int(fg_item, 'FractionGroupNumber'),
                    num_fractions=_get_int(fg_item, 'NumberOfFractionsPlanned'),
                    num_beams=_get_int(fg_item, 'NumberOfBeams'),
                    num_brachy_setups=_get_int(fg_item, 'NumberOfBrachyApplicationSetups'),
                )
                # referenced beams
                if hasattr(fg_item, 'ReferencedBeamSequence'):
                    for rb_item in fg_item.ReferencedBeamSequence:
                        fg.referenced_beams.append(ReferencedBeam(
                            beam_number=_get_int(rb_item, 'ReferencedBeamNumber'),
                            beam_meterset=_get_float(rb_item, 'BeamMeterset'),
                            beam_dose=_get_float(rb_item, 'BeamDose'),
                            dose_spec_point=_get_float_list(rb_item, 'BeamDoseSpecificationPoint'),
                            dose_ref_uid=_get_str(rb_item, 'ReferencedDoseReferenceUID'),
                            dose_point_depth=_get_float(rb_item, 'BeamDosePointDepth'),
                            dose_point_equiv_depth=_get_float(rb_item, 'BeamDosePointEquivalentDepth'),
                            dose_point_ssd=_get_float(rb_item, 'BeamDosePointSSD'),
                            dose_type=_get_str(rb_item, 'BeamDoseType'),
                        ))
                # referenced dose references
                if hasattr(fg_item, 'ReferencedDoseReferenceSequence'):
                    fg.ref_dose_ref_numbers = [
                        _get_int(r, 'ReferencedDoseReferenceNumber')
                        for r in fg_item.ReferencedDoseReferenceSequence
                    ]
                self.fraction_groups.append(fg)

        # ---- beams ----
        self.beams: List[Beam] = []
        if hasattr(ds, 'BeamSequence'):
            for beam_item in ds.BeamSequence:
                b = Beam(
                    number=_get_int(beam_item, 'BeamNumber'),
                    name=_get_str(beam_item, 'BeamName'),
                    beam_type=_get_str(beam_item, 'BeamType'),
                    radiation_type=_get_str(beam_item, 'RadiationType'),
                    delivery_type=_get_str(beam_item, 'TreatmentDeliveryType'),
                    dosimeter_unit=_get_str(beam_item, 'PrimaryDosimeterUnit'),
                    sad=_get_float(beam_item, 'SourceAxisDistance'),
                    final_cum_weight=_get_float(beam_item, 'FinalCumulativeMetersetWeight'),
                    num_control_points=_get_int(beam_item, 'NumberOfControlPoints'),
                    num_wedges=_get_int(beam_item, 'NumberOfWedges'),
                    num_compensators=_get_int(beam_item, 'NumberOfCompensators'),
                    num_boli=_get_int(beam_item, 'NumberOfBoli'),
                    num_blocks=_get_int(beam_item, 'NumberOfBlocks'),
                    referenced_setup_number=_get_int(beam_item, 'ReferencedPatientSetupNumber'),
                )

                # fluence mode
                if hasattr(beam_item, 'PrimaryFluenceModeSequence'):
                    fm = beam_item.PrimaryFluenceModeSequence[0]
                    b.fluence = FluenceMode(
                        mode=_get_str(fm, 'FluenceMode'),
                        mode_id=_get_str(fm, 'FluenceModeID'),
                    )

                # beam limiting devices (static definitions)
                if hasattr(beam_item, 'BeamLimitingDeviceSequence'):
                    for bld_item in beam_item.BeamLimitingDeviceSequence:
                        bld = BeamLimitingDevice(
                            device_type=_get_str(bld_item, 'RTBeamLimitingDeviceType'),
                            num_leaf_pairs=_get_int(bld_item, 'NumberOfLeafJawPairs'),
                        )
                        if hasattr(bld_item, 'LeafPositionBoundaries'):
                            bld.leaf_boundaries = _to_float_list(bld_item.LeafPositionBoundaries)
                        b.beam_limiting_devices.append(bld)

                # control points
                if hasattr(beam_item, 'ControlPointSequence'):
                    for cp_item in beam_item.ControlPointSequence:
                        cp = ControlPoint(
                            index=_get_int(cp_item, 'ControlPointIndex'),
                            gantry_angle=_get_float(cp_item, 'GantryAngle'),
                            gantry_direction=_get_str(cp_item, 'GantryRotationDirection'),
                            collimator_angle=_get_float(cp_item, 'BeamLimitingDeviceAngle'),
                            collimator_direction=_get_str(cp_item, 'BeamLimitingDeviceRotationDirection'),
                            patient_support_angle=_get_float(cp_item, 'PatientSupportAngle'),
                            patient_support_direction=_get_str(cp_item, 'PatientSupportRotationDirection'),
                            table_top_eccentric_angle=_get_float(cp_item, 'TableTopEccentricAngle'),
                            table_top_eccentric_direction=_get_str(cp_item, 'TableTopEccentricRotationDirection'),
                            table_top_vertical=_get_opt_float(cp_item, 'TableTopVerticalPosition'),
                            table_top_longitudinal=_get_opt_float(cp_item, 'TableTopLongitudinalPosition'),
                            table_top_lateral=_get_opt_float(cp_item, 'TableTopLateralPosition'),
                            table_top_pitch_angle=_get_float(cp_item, 'TableTopPitchAngle'),
                            table_top_pitch_direction=_get_str(cp_item, 'TableTopPitchRotationDirection'),
                            table_top_roll_angle=_get_float(cp_item, 'TableTopRollAngle'),
                            table_top_roll_direction=_get_str(cp_item, 'TableTopRollRotationDirection'),
                            gantry_pitch_angle=_get_float(cp_item, 'GantryPitchAngle'),
                            gantry_pitch_direction=_get_str(cp_item, 'GantryPitchRotationDirection'),
                            isocenter=_get_float_list(cp_item, 'IsocenterPosition'),
                            ssd=_get_float(cp_item, 'SourceToSurfaceDistance'),
                            cumulative_meterset_weight=_get_float(cp_item, 'CumulativeMetersetWeight'),
                            nominal_beam_energy=_get_float(cp_item, 'NominalBeamEnergy'),
                            dose_rate_set=_get_float(cp_item, 'DoseRateSet'),
                        )

                        # BLD positions at this CP
                        if hasattr(cp_item, 'BeamLimitingDevicePositionSequence'):
                            for bldp_item in cp_item.BeamLimitingDevicePositionSequence:
                                dtype = _get_str(bldp_item, 'RTBeamLimitingDeviceType')
                                cp.device_positions[dtype] = DevicePosition(
                                    device_type=dtype,
                                    positions=_to_float_list(bldp_item.LeafJawPositions),
                                )

                        # dose reference coefficients
                        if hasattr(cp_item, 'ReferencedDoseReferenceSequence'):
                            for drc_item in cp_item.ReferencedDoseReferenceSequence:
                                cp.dose_refs.append(DoseRefCoefficient(
                                    ref_number=_get_int(drc_item, 'ReferencedDoseReferenceNumber'),
                                    coefficient=_get_float(drc_item, 'CumulativeDoseReferenceCoefficient'),
                                ))

                        b.control_points.append(cp)

                self.beams.append(b)

        # ---- patient setup ----
        self.patient_setups: List[PatientSetup] = []
        if hasattr(ds, 'PatientSetupSequence'):
            for ps_item in ds.PatientSetupSequence:
                self.patient_setups.append(PatientSetup(
                    number=_get_int(ps_item, 'PatientSetupNumber'),
                    position=_get_str(ps_item, 'PatientPosition'),
                ))

        # ---- referenced structure set ----
        self.ref_structure_set_uid: str = ''
        if hasattr(ds, 'ReferencedStructureSetSequence'):
            rss = ds.ReferencedStructureSetSequence[0]
            self.ref_structure_set_uid = _get_str(rss, 'ReferencedSOPInstanceUID')

    # ============================================================
    #  High-level accessors
    # ============================================================

    @property
    def treatment_beams(self) -> List[Beam]:
        """Return only TREATMENT beams (exclude SETUP)."""
        return [b for b in self.beams if b.is_treatment]

    @property
    def total_mu(self) -> float:
        """Total Monitor Units across all treatment beams."""
        if self.fraction_groups:
            return sum(
                rb.beam_meterset
                for fg in self.fraction_groups
                for rb in fg.referenced_beams
            )
        return 0.0

    @property
    def beam_mu_map(self) -> Dict[int, float]:
        """Return {BeamNumber: MU} for all referenced beams."""
        result = {}
        for fg in self.fraction_groups:
            for rb in fg.referenced_beams:
                result[rb.beam_number] = rb.beam_meterset
        return result

    @property
    def prescription_dose_gy(self) -> float:
        """Target prescription dose in Gy (first TARGET reference)."""
        for dr in self.dose_references:
            if dr.dose_type == 'TARGET':
                return dr.target_prescription_dose
        return 0.0

    @property
    def num_fractions(self) -> int:
        """Number of fractions from the first fraction group."""
        if self.fraction_groups:
            return self.fraction_groups[0].num_fractions
        return 0

    @property
    def patient_position(self) -> str:
        """Patient position string, e.g. 'HFS'."""
        if self.patient_setups:
            return self.patient_setups[0].position
        return ''

    @property
    def is_approved(self) -> bool:
        return self.approval_status == 'APPROVED'

    @property
    def is_fff(self) -> bool:
        """True if any beam uses flattening-filter-free mode."""
        for b in self.beams:
            if b.fluence and b.fluence.mode_id == 'FFF':
                return True
        return False

    # ============================================================
    #  Bulk data access (lists / numpy-friendly)
    # ============================================================

    def get_all_mlc_positions(self, beam_index: int = 0) -> List[List[float]]:
        """Return MLC positions for every CP in a beam.
        Result: list of lists, shape (n_cp, 120) for 60-pair MLC.
        """
        beam = self.beams[beam_index]
        return [cp.mlc_positions for cp in beam.control_points]

    def get_all_jaw_positions(self, beam_index: int = 0) -> List[Dict[str, List[float]]]:
        """Return jaw X/Y positions for every CP in a beam.
        Since jaws are typically only in CP0, later CPs inherit via lookup.
        """
        beam = self.beams[beam_index]
        # inherit jaws from most recent CP that has them
        jaw_x, jaw_y = [], []
        last_x, last_y = [], []
        for cp in beam.control_points:
            jx = cp.jaw_x
            jy = cp.jaw_y
            if jx: last_x = jx
            if jy: last_y = jy
            jaw_x.append(last_x)
            jaw_y.append(last_y)
        return [{'ASYMX': x, 'ASYMY': y} for x, y in zip(jaw_x, jaw_y)]

    def get_gantry_trajectory(self, beam_index: int = 0) -> List[float]:
        """Return gantry angles for every CP (degrees)."""
        return self.beams[beam_index].gantry_angles

    def get_cumulative_weights(self, beam_index: int = 0) -> List[float]:
        """Return cumulative meterset weights for every CP."""
        return self.beams[beam_index].cum_weights

    def get_segment_weights(self, beam_index: int = 0) -> List[float]:
        """Return differential MU weights between consecutive CPs."""
        w = self.get_cumulative_weights(beam_index)
        return [w[i] - w[i - 1] for i in range(1, len(w))]

    def get_mlc_pair_openings(self, beam_index: int = 0,
                               cp_index: int = 0) -> List[float]:
        """Return aperture opening (B - A) per leaf pair for one CP, in mm."""
        mlc = self.beams[beam_index].control_points[cp_index].mlc_positions
        n = len(mlc) // 2
        return [mlc[i + n] - mlc[i] for i in range(n)]

    def to_dataframe(self) -> 'pandas.DataFrame':
        """Export all CP-level data as a flat pandas DataFrame (one row per CP)."""
        import pandas as pd
        rows = []
        for beam in self.beams:
            beam_mu = self.beam_mu_map.get(beam.number, 0.0)
            for cp in beam.control_points:
                rows.append({
                    'beam_number': beam.number,
                    'beam_name': beam.name,
                    'beam_type': beam.beam_type,
                    'delivery_type': beam.delivery_type,
                    'cp_index': cp.index,
                    'gantry_angle': cp.gantry_angle,
                    'gantry_direction': cp.gantry_direction,
                    'collimator_angle': cp.collimator_angle,
                    'ssd': cp.ssd,
                    'cum_weight': cp.cumulative_meterset_weight,
                    'isocenter_x': cp.isocenter[0] if len(cp.isocenter) > 0 else None,
                    'isocenter_y': cp.isocenter[1] if len(cp.isocenter) > 1 else None,
                    'isocenter_z': cp.isocenter[2] if len(cp.isocenter) > 2 else None,
                    'jaw_x1': cp.jaw_x[0] if len(cp.jaw_x) > 0 else None,
                    'jaw_x2': cp.jaw_x[1] if len(cp.jaw_x) > 1 else None,
                    'jaw_y1': cp.jaw_y[0] if len(cp.jaw_y) > 0 else None,
                    'jaw_y2': cp.jaw_y[1] if len(cp.jaw_y) > 1 else None,
                    'num_mlc_leaves': len(cp.mlc_positions),
                    'mlc_mean': sum(cp.mlc_positions) / len(cp.mlc_positions) if cp.mlc_positions else None,
                    'beam_mu': beam_mu,
                })
        return pd.DataFrame(rows)

    # ============================================================
    #  Summary
    # ============================================================

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"DICOM RT Plan: {self._filepath}",
            f"  Patient: {self.patient_name} (ID: {self.patient_id})",
            f"  Plan:    {self.plan_label}  Intent: {self.plan_intent}",
            f"  TPS:     {self.manufacturer} {self.manufacturer_model} "
            f"v{self.software_versions[0] if self.software_versions else '?'}",
            f"  Status:  {self.approval_status}  ({self.instance_creation_date})",
            f"  Rx:      {self.prescription_dose_gy} Gy / {self.num_fractions} fx",
            f"  Beams:   {len(self.beams)} total, {len(self.treatment_beams)} treatment",
            f"  Total MU: {self.total_mu:.2f}",
            f"  FFF:     {self.is_fff}",
            f"  Position: {self.patient_position}",
        ]
        for b in self.beams:
            mu = self.beam_mu_map.get(b.number, 0)
            lines.append(
                f"  Beam {b.number} ({b.name}): {b.beam_type} {b.delivery_type}, "
                f"{b.radiation_type}, {b.num_control_points} CPs, "
                f"{b.mlc_leaf_count} MLC leaves, {mu:.2f} MU"
            )
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"DcmRTPlan('{self._filepath}')"


# ================================================================
#  Internal helpers — safe DICOM attribute access
# ================================================================

def _get_str(ds, attr: str) -> str:
    val = getattr(ds, attr, None)
    return str(val) if val is not None else ''


def _get_int(ds, attr: str) -> int:
    val = getattr(ds, attr, None)
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _get_float(ds, attr: str) -> float:
    val = getattr(ds, attr, None)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _get_opt_float(ds, attr: str) -> Optional[float]:
    val = getattr(ds, attr, None)
    if val is None or (isinstance(val, str) and val == ''):
        return None
    try:
        v = float(val)
        return v if v == v else None  # NaN check
    except (ValueError, TypeError):
        return None


def _get_str_list(ds, attr: str) -> List[str]:
    val = getattr(ds, attr, None)
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    return [str(val)]


def _get_float_list(ds, attr: str) -> List[float]:
    val = getattr(ds, attr, None)
    if val is None:
        return []
    return _to_float_list(val)


def _to_float_list(val) -> List[float]:
    """Convert pydicom MultiValue / list / single value to plain float list."""
    try:
        # MultiValue and other iterables (but not str/bytes)
        if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
            return [float(v) for v in val]
    except (ValueError, TypeError):
        pass
    try:
        return [float(val)]
    except (ValueError, TypeError):
        return []


# ================================================================
#  Quick test
# ================================================================
if __name__ == '__main__':
    import argparse
    from ai4ct.paths import DICOM_DIR

    parser = argparse.ArgumentParser(description="Inspect a DICOM RT Plan.")
    parser.add_argument("dicom_path", nargs="?", default=str(DICOM_DIR / 'RP_pID001.dcm'))
    path = parser.parse_args().dicom_path
    plan = DcmRTPlan(path)
    print(plan.summary())
    print()

    # Example: access MLC positions
    if plan.beams:
        b0 = plan.beams[0]
        cp0 = b0.control_points[0]
        mlc = cp0.mlc_positions
        print(f"Beam 0, CP 0: {len(mlc)} MLC leaves")
        print(f"  First 5: {mlc[:5]}")
        print(f"  Jaw X: {cp0.jaw_x}")
        print(f"  Jaw Y: {cp0.jaw_y}")
        print(f"  Gantry: {cp0.gantry_angle}°")
        print(f"  Isocenter: {cp0.isocenter}")
        print(f"  SSD: {cp0.ssd}")
