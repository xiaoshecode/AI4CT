"""Synthetic identity audit tests; no patient records are used or disclosed."""
import json
import unittest
from unittest.mock import patch

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from ai4ct.reproduction.identity_audit import IdentityLookup, inspect_dataset, normalized, main


class IdentityAuditTests(unittest.TestCase):
    def lookup(self):
        return IdentityLookup([{"sheet": "Synthetic", "row": 2,
                                "name": "Example Person", "patient_id": "AB123456"}])

    def test_recursive_original_attributes_are_examined(self):
        old = Dataset()
        old.PatientID = "AB123456"
        old.PatientName = "Example^Person"
        attrs = Dataset()
        attrs.ModifiedAttributesSequence = Sequence([old])
        ds = Dataset()
        ds.PatientID = "pID001"
        ds.OriginalAttributesSequence = Sequence([attrs])
        counts, _, _, candidates = inspect_dataset(ds, self.lookup())
        self.assertEqual(counts["maximum_sequence_depth"], 2)
        self.assertEqual({c["match_type"] for c in candidates}, {"exact_name", "exact_id"})
        self.assertTrue(all("04000561[0]/04000550[0]" in c["tag_path"] for c in candidates))
        text = json.dumps(candidates)
        self.assertNotIn("AB123456", text)
        self.assertNotIn("Example", text)

    def test_uids_and_uid_substrings_do_not_match_numeric_identity(self):
        lookup = IdentityLookup([{"sheet": "Synthetic", "row": 3, "patient_id": "123456"}])
        self.assertEqual(lookup.match("123456", "UI"), [])
        self.assertEqual(lookup.match("1.2.123456.789", "LO"), [])
        self.assertEqual(lookup.match("01234567", "LO"), [])
        self.assertEqual(lookup.match("123456", "DS"), [])
        self.assertEqual(lookup.match("ID=123456;", "LO")[0]["match_type"], "bounded_id_in_text")

    def test_private_text_is_counted_and_compared_without_exposing_values(self):
        ds = Dataset()
        ds.add_new((0x0011, 0x1010), "LO", "AB123456")
        counts, _, _, candidates = inspect_dataset(ds, self.lookup())
        self.assertEqual(counts["private_element_instances"], 1)
        self.assertTrue(candidates[0]["private"])
        self.assertEqual(candidates[0]["tag"], "00111010")
        self.assertNotIn("AB123456", json.dumps(candidates))

    def test_private_unknown_bytes_are_decoded_locally_without_retaining_identity(self):
        ds = Dataset()
        ds.add_new((0x0011, 0x1010), "UN", b"AB123456")
        counts, _, _, candidates = inspect_dataset(ds, self.lookup())
        self.assertEqual(counts["private_binary_element_instances"], 1)
        self.assertEqual(counts["private_binary_total_bytes"], 8)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_type"], "decoded_private_exact_id")
        self.assertNotIn("AB123456", json.dumps(candidates))

    def test_numeric_surrogates_are_separate_from_literal_pid_patterns(self):
        ds = Dataset()
        ds.PatientName = "7"
        ds.PatientID = "7"
        counts, _, _, candidates = inspect_dataset(ds, self.lookup(), {"7"})
        self.assertEqual(counts["PN_numeric_surrogate_values"], 1)
        self.assertEqual(counts["LO_numeric_surrogate_values"], 1)
        self.assertEqual(counts["PN_anonymous_pid_values"], 0)
        self.assertEqual(candidates, [])

    def test_anonymous_names_are_counted_but_not_equated_to_real_identities(self):
        ds = Dataset()
        ds.PatientName = "pID001"
        ds.PatientID = "pID001"
        counts, _, _, candidates = inspect_dataset(ds, self.lookup())
        self.assertEqual(counts["PN_anonymous_pid_values"], 1)
        self.assertEqual(counts["LO_anonymous_pid_values"], 1)
        self.assertEqual(candidates, [])

    def test_identifier_leading_zeros_are_preserved(self):
        self.assertEqual(normalized(" 00123 "), "00123")
        self.assertEqual(normalized(123.0), "123")
        lookup = IdentityLookup([{"sheet": "Synthetic", "row": 4, "patient_id": "00123"}])
        self.assertEqual(lookup.match("123", "LO"), [])

    def test_existing_output_is_rejected_before_inputs_are_read(self):
        with patch("ai4ct.reproduction.identity_audit.Path.exists", return_value=True), \
                patch("ai4ct.reproduction.identity_audit.audit") as scan:
            with self.assertRaises(FileExistsError):
                main(["--dicom-dir", "unused", "--workbook", "unused.xlsx", "--output", "exists.json"])
            scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
