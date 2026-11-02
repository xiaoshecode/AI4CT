"""Synthetic workbook-value preservation and explicitly sourced label records."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from openpyxl import Workbook

from ai4ct.reproduction.excel_labels import (
    SHEET_SCHEMAS, convert_arccheck_workbook, parse_gamma,
    standardize_rows, typed_value,
)


class ExcelValueTests(unittest.TestCase):
    def test_json_types_preserve_dates_boolean_null_and_formula(self):
        self.assertEqual(typed_value(True), (True, "boolean"))
        self.assertEqual(typed_value(None), (None, "null"))
        self.assertEqual(typed_value(12), (12, "integer"))
        self.assertEqual(typed_value(datetime(2024, 1, 2, 3, 4)), ("2024-01-02T03:04:00", "datetime"))
        self.assertEqual(typed_value("=A1", "f"), ("=A1", "formula"))

    def test_decimal_comma_is_explicit_and_invalid_gamma_fails(self):
        self.assertEqual(parse_gamma("91,9"), (91.9, "decimal_comma_to_decimal_point"))
        self.assertEqual(parse_gamma(None), (None, None))
        for value in (True, float("nan"), -1, 101, "not measured", "1,2.3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gamma(value)

    def test_standard_rows_preserve_original_row_after_empty_row(self):
        rows = [SHEET_SCHEMAS["Sheet2"]["headers"],
                [None] * 7,
                [2, "synthetic", "", "", datetime(2024, 1, 2), "91,9", 99.0]]
        document, summary = standardize_rows("Sheet2", rows, "synthetic-source")
        record = document["records"][0]
        self.assertEqual(record["source_index"], 2)
        self.assertEqual(record["source_row"], 3)
        self.assertEqual(record["gpr_2mm_2pct"], 91.9)
        self.assertEqual(document["alignment_status"], "unverified")
        self.assertEqual(summary["threshold_counts"]["90"], {"pass": 1, "fail": 0})
        self.assertEqual(summary["threshold_counts"]["95"], {"pass": 0, "fail": 1})
        self.assertEqual(document["normalization_events"][0]["original_value"], "91,9")

    def test_unknown_standard_schema_fails_instead_of_assuming_columns(self):
        with self.assertRaises(ValueError):
            standardize_rows("Sheet2", [["ID", "Gamma"], [1, 90]], "hash")


class WorkbookConversionTests(unittest.TestCase):
    def test_complete_json_preserves_all_sheets_types_rows_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.xlsx"
            book = Workbook()
            sheet = book.active
            sheet.title = "Sheet2"
            sheet.append(SHEET_SCHEMAS["Sheet2"]["headers"])
            sheet.append([1, "synthetic", True, "", datetime(2024, 1, 2), "91,9", 99.0])
            sheet.append([None] * 7)
            sheet.append([2, "synthetic", False, "", None, 96.0, 100.0])
            notes = book.create_sheet("Notes")
            notes.append(["=1+2", True, None, datetime(2024, 3, 4)])
            book.save(source)
            # Keep a declared trailing empty range, as found in the real source.
            with zipfile.ZipFile(source) as archive:
                members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
            members["xl/worksheets/sheet1.xml"] = members["xl/worksheets/sheet1.xml"].replace(b'ref="A1:G4"', b'ref="A1:G6"')
            with zipfile.ZipFile(source, "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
            before = source.read_bytes()
            result = convert_arccheck_workbook(
                source, root / "raw", root / "processed",
                root / "missing_previous.xlsx", root / "missing_previous.json")
            repeated = convert_arccheck_workbook(
                source, root / "raw", root / "processed",
                root / "missing_previous.xlsx", root / "missing_previous.json")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result.raw_copy_path.read_bytes(), before)
            self.assertEqual(result.audit["source_sha256"], hashlib.sha256(before).hexdigest())
            complete = json.loads(result.workbook_json_path.read_text(encoding="utf-8"))
            self.assertEqual(complete["sheet_order"], ["Sheet2", "Notes"])
            values = complete["sheets"][0]["rows"]
            self.assertEqual(len(values), 6)
            self.assertEqual(values[2]["values"], [None] * 7)
            self.assertEqual(values[5]["values"], [None] * 7)
            self.assertEqual(values[1]["values"][5], "91,9")
            self.assertEqual(values[1]["types"][2], "boolean")
            self.assertEqual(values[1]["types"][4], "datetime")
            notes_row = complete["sheets"][1]["rows"][0]
            self.assertEqual(notes_row["values"][0], "=1+2")
            self.assertEqual(notes_row["types"][0], "formula")
            self.assertIn("A1", notes_row["formula_cached_values"])
            records = json.loads(result.records_paths["Sheet2"].read_text(encoding="utf-8"))["records"]
            self.assertEqual([row["source_row"] for row in records], [2, 4])
            self.assertEqual(records[0]["gpr_2mm_2pct"], 91.9)
            self.assertFalse(result.audit["training_manifest_created"])
            self.assertFalse((root / "processed" / "manifest.json").exists())
            self.assertEqual(repeated.audit, result.audit)
            # A new source version cannot silently replace the retained original.
            sheet["F2"] = 90.0
            book.save(source)
            with self.assertRaises(FileExistsError):
                convert_arccheck_workbook(source, root / "raw", root / "processed")
            self.assertEqual(result.raw_copy_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
