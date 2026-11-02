import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from pydicom.dataset import Dataset

from ai4ct.reproduction.features import (FEATURE_NAMES, FeatureExtractionError,
                                         _aperture, extract_features)


def item(**kwargs):
    ds = Dataset()
    for key, value in kwargs.items():
        setattr(ds, key, value)
    return ds


def beam(number=1, positions=None, boundaries=(-10, 0, 5), weights=(0, 1), jaws=None):
    if positions is None:
        positions = [[-10, -10, 10, 10]] + [None] * (len(weights) - 1)
    devices = [item(RTBeamLimitingDeviceType='MLCX', NumberOfLeafJawPairs=len(boundaries)-1,
                    LeafPositionBoundaries=list(boundaries))]
    if jaws:
        devices += [item(RTBeamLimitingDeviceType=k, NumberOfLeafJawPairs=1) for k in jaws]
    cps = []
    for i, (weight, pos) in enumerate(zip(weights, positions)):
        entries = []
        if pos is not None:
            entries.append(item(RTBeamLimitingDeviceType='MLCX', LeafJawPositions=pos))
        if i == 0 and jaws:
            entries += [item(RTBeamLimitingDeviceType=k, LeafJawPositions=v) for k,v in jaws.items()]
        cps.append(item(ControlPointIndex=i, CumulativeMetersetWeight=weight,
                        BeamLimitingDevicePositionSequence=entries))
    return item(BeamNumber=number, TreatmentDeliveryType='TREATMENT',
                BeamLimitingDeviceSequence=devices, ControlPointSequence=cps,
                NumberOfControlPoints=len(cps), FinalCumulativeMetersetWeight=weights[-1])


def plan(beams=None, references=None):
    return item(Modality='RTPLAN', BeamSequence=beams or [beam()], FractionGroupSequence=[
        item(ReferencedBeamSequence=references or [item(ReferencedBeamNumber=1, BeamMeterset=100)])])


def features(ds):
    with patch('ai4ct.reproduction.features.pydicom.dcmread', return_value=ds):
        return extract_features('synthetic.dcm')


class ReproductionFeatureTests(unittest.TestCase):
    def test_rectangle_true_unequal_leaf_width_and_inheritance(self):
        result = features(plan())
        self.assertEqual(tuple(result), FEATURE_NAMES)
        self.assertEqual(len(result), 17)
        self.assertAlmostEqual(result['BA'], 3)
        self.assertAlmostEqual(result['UAA'], 3)
        self.assertAlmostEqual(result['PI'], 49/(12*math.pi))
        self.assertAlmostEqual(result['CA'], 7/3)
        self.assertAlmostEqual(result['ALG'], 20)
        for key in ['AAV', 'LSV', 'MCS']:
            self.assertAlmostEqual(result[key], 1)
        for key in ['MAD', 'CLS', 'SLG', 'PM', 'SAS2', 'SAS5', 'SAS10', 'SAS15']:
            self.assertAlmostEqual(result[key], 0)
        self.assertEqual(result['MU'],100)

    def test_partial_leaf_jaw_clipping_both_axes(self):
        result = features(plan([beam(jaws={'ASYMX':[-3,7], 'ASYMY':[-2,3]})]))
        self.assertAlmostEqual(result['BA'], .5)
        self.assertAlmostEqual(result['CA'], 6)
        self.assertAlmostEqual(result['PI'], 4.5/math.pi)
        self.assertAlmostEqual(result['MAD'], 2)
        self.assertAlmostEqual(result['ALG'], 10)
        self.assertEqual(result['SAS10'], 0)
        self.assertEqual(result['SAS15'], 1)

    def test_nonuniform_segment_mu_endpoint_integration(self):
        b=beam(boundaries=(-5,0,5),weights=(0,.25,1),positions=[[-5,-5,5,5],[-10,-10,10,10],[-20,-20,20,20]])
        result=features(plan([b]))
        self.assertAlmostEqual(result['BA'],2.625)
        self.assertAlmostEqual(result['UAA'],4)
        self.assertAlmostEqual(result['AAV'],.65625)
        self.assertAlmostEqual(result['MCS'],.65625)
        self.assertAlmostEqual(result['PM'],.34375)

    def test_reference_beam_number_and_setup_filter(self):
        a=beam(7,positions=[[-5,-5,5,5],None],boundaries=(-5,0,5))
        b=beam(3,positions=[[-20,-20,20,20],None],boundaries=(-5,0,5))
        setup=item(BeamNumber=99,TreatmentDeliveryType='SETUP')
        refs=[item(ReferencedBeamNumber=3,BeamMeterset=300),item(ReferencedBeamNumber=99),item(ReferencedBeamNumber=7,BeamMeterset=100)]
        result=features(plan([setup,a,b],refs))
        self.assertAlmostEqual(result['BA'],3.25)
        self.assertAlmostEqual(result['MU'],400)

    def test_disconnected_aperture_perimeter(self):
        ap=_aperture(np.array([0,5,10]),[-10,0,-5,5])
        self.assertAlmostEqual(ap['values']['BA'],.5)
        self.assertAlmostEqual(ap['values']['CA'],8)
        self.assertAlmostEqual(ap['values']['MAD'],5)

    def test_union_does_not_fill_disconnected_intervals(self):
        b=beam(boundaries=(0,5), positions=[[-10,-5],[5,10]])
        result=features(plan([b]))
        self.assertAlmostEqual(result['UAA'],.5)
        self.assertAlmostEqual(result['BA'],.25)
        self.assertAlmostEqual(result['AAV'],.5)

    def test_rejects_decreasing_weights(self):
        b=beam(weights=(0,.8,.7,1))
        with self.assertRaisesRegex(FeatureExtractionError,'decrease'):
            features(plan([b]))

    def test_rejects_multilayer_and_unknown_geometry(self):
        b=beam()
        b.BeamLimitingDeviceSequence.append(copy.deepcopy(b.BeamLimitingDeviceSequence[0]))
        with self.assertRaisesRegex(FeatureExtractionError,'Duplicate'):
            features(plan([b]))
        b=beam();del b.BeamLimitingDeviceSequence[0].LeafPositionBoundaries
        with self.assertRaisesRegex(FeatureExtractionError,'LeafPositionBoundaries'):
            features(plan([b]))

    def test_rejects_missing_initial_positions_and_no_aperture(self):
        with self.assertRaisesRegex(FeatureExtractionError,'initial MLC'):
            features(plan([beam(positions=[None,None])]))
        with self.assertRaisesRegex(FeatureExtractionError,'No open aperture'):
            features(plan([beam(positions=[[0,0,0,0],None])]))

    def test_rejects_ambiguous_fraction_groups_and_missing_mu(self):
        ds=plan();ds.FractionGroupSequence.append(copy.deepcopy(ds.FractionGroupSequence[0]))
        with self.assertRaisesRegex(FeatureExtractionError,'fraction group'):
            features(ds)
        ds=plan();ds.FractionGroupSequence[0].ReferencedBeamSequence=[]
        with self.assertRaisesRegex(FeatureExtractionError,'MU reference'):
            features(ds)


if __name__=='__main__':
    unittest.main()
