import unittest
import numpy as np

from ai4ct.reproduction.evaluation import (
    make_folds, classification_metrics, group_bootstrap_accuracy,
)


class EvaluationTests(unittest.TestCase):
    def test_shared_group_never_crosses_folds(self):
        rates = np.tile([85., 92., 98.], 30)
        groups = np.repeat(np.arange(45).astype(str), 2)
        folds = make_folds(rates, groups)
        np.testing.assert_array_equal(folds, make_folds(rates, groups))
        self.assertEqual(set(folds), set(range(5)))
        for group in np.unique(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)

    def test_invalid_rates_rejected(self):
        with self.assertRaises(ValueError):
            make_folds([float("nan"), 99], ["a", "b"])

    def test_failure_recall_is_specificity(self):
        result = classification_metrics([0, 0, 0, 1, 1], [0.1, 0.8, 0.9, 0.3, 0.9])
        self.assertEqual(result["confusion_matrix_fail_pass"], [[1, 2], [1, 1]])
        self.assertAlmostEqual(result["accuracy"], 0.4)
        self.assertAlmostEqual(result["specificity_fail"], 1/3)
        self.assertAlmostEqual(result["sensitivity_pass"], 0.5)

    def test_probability_tie_matches_argmax(self):
        self.assertEqual(classification_metrics([0], [0.5])["accuracy"], 1.)

    def test_invalid_model_output_cannot_be_reported(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0.1, float("nan")])
        with self.assertRaises(ValueError):
            classification_metrics([0.5, 1], [0.1, 0.9])

    def test_perfect_group_bootstrap(self):
        result = group_bootstrap_accuracy([0, 1, 0, 1], [.1, .9, .1, .9], ["a", "a", "b", "c"])
        self.assertEqual((result["low"], result["high"]), (1., 1.))


if __name__ == "__main__":
    unittest.main()
