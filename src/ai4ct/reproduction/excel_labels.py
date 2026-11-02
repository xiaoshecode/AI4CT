"""Lossless worksheet-value JSON plus explicitly sourced ArcCheck records.

This is a source conversion, not a DICOM-to-QA mapping. All worksheets and row
numbers are retained, and unnamed ordinal columns are never called patient IDs.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

from ai4ct import paths


SCHEMA_VERSION = "arccheck-excel-json-v1"
SHEET_SCHEMAS = {
    "Sheet1": {
        "headers": [None, "姓名", "ID号", "计划类型", "主管大夫", "病种", "病程", "照射部位", "计划设计", None, "2mm/2%", "3mm/3%"],
        "fields": ["source_index", "patient_name", "patient_id", "plan_type", "supervising_physician", "diagnosis", "stage", "site", "plan_designer", "date", "gpr_2mm_2pct", "gpr_3mm_3pct"],
    },
    "Sheet2": {
        "headers": [None, "病种", "病程", "照射部位", None, "2mm/2%", "3mm/3%"],
        "fields": ["source_index", "diagnosis", "stage", "site", "date", "gpr_2mm_2pct", "gpr_3mm_3pct"],
    },
}


@dataclass(frozen=True)
class ExcelConversion:
    raw_copy_path: Path
    workbook_json_path: Path
    records_paths: dict[str, Path]
    audit_path: Path
    audit: dict


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def typed_value(value, excel_type=None) -> tuple[object, str]:
    """Represent values in JSON without making dates/booleans into numbers."""
    if value is None:
        return None, "null"
    if excel_type == "f":
        return str(value), "formula"
    if excel_type == "e":
        return str(value), "excel_error"
    if isinstance(value, datetime):
        return value.isoformat(), "datetime"
    if isinstance(value, date):
        return value.isoformat(), "date"
    if isinstance(value, time):
        return value.isoformat(), "time"
    if isinstance(value, timedelta):
        return value.total_seconds(), "duration_seconds"
    if isinstance(value, bool):
        return value, "boolean"
    if isinstance(value, int):
        return value, "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value), "nonfinite_number"
        return value, "number"
    if isinstance(value, str):
        return value, "string"
    raise TypeError(f"Unsupported Excel value type: {type(value).__name__}.")


def parse_gamma(value) -> tuple[float | None, str | None]:
    """Parse only numeric percentages and unambiguous decimal text.

    Comma decimal text is recognized explicitly and recorded in conversion
    events. Missing stays null; booleans, units other than percent and invalid
    ranges fail rather than silently becoming zero.
    """
    if value is None or value == "":
        return None, None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a Gamma percentage.")
    reason = None
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"\d+(?:[.,，]\d+)?\s*%?", text):
            raise ValueError("Gamma text is not an unambiguous decimal percentage.")
        reason = "decimal_comma_to_decimal_point" if "," in text or "，" in text else "numeric_text_to_number"
        number = float(text.rstrip("%").strip().replace(",", ".").replace("，", "."))
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ValueError("Unsupported Gamma value type.")
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError("Gamma percentage must be finite and within 0..100.")
    return number, reason


def _counts(records: list[dict]) -> dict:
    values = [row["gpr_2mm_2pct"] for row in records if row["gpr_2mm_2pct"] is not None]
    return {
        "measured_records": len(values), "missing_gpr_records": len(records) - len(values),
        "threshold_counts": {
            str(threshold): {"pass": sum(value >= threshold for value in values),
                             "fail": sum(value < threshold for value in values)}
            for threshold in (90, 95)
        },
    }


def standardize_rows(sheet_name: str, rows: list[list], source_sha256: str) -> tuple[dict, dict]:
    """Return records for one named schema; keep nonempty rows and source IDs."""
    schema = SHEET_SCHEMAS.get(sheet_name)
    if schema is None or not rows or list(rows[0]) != schema["headers"]:
        raise ValueError("Worksheet does not match the explicitly declared ArcCheck schema.")
    records, events, blank_rows = [], [], []
    fields = schema["fields"]
    for row_number, values in enumerate(rows[1:], 2):
        if all(value is None for value in values):
            blank_rows.append(row_number)
            continue
        if len(values) != len(fields):
            raise ValueError(f"Worksheet row {row_number}: unexpected column count.")
        record = {"source_sheet": sheet_name, "source_row": row_number}
        for field, value in zip(fields, values):
            if field.startswith("gpr_"):
                parsed, reason = parse_gamma(value)
                record[field] = parsed
                if reason:
                    events.append({"source_sheet": sheet_name, "source_row": row_number,
                                   "field": field, "reason": reason,
                                   "original_value": value, "converted_value": parsed})
            else:
                record[field] = typed_value(value)[0]
        records.append(record)
    indices = [row["source_index"] for row in records]
    index_counts = Counter(str(value) for value in indices if value is not None)
    summary = {
        "records": len(records), "blank_rows_omitted_from_standard_records": blank_rows,
        "numbered_records": sum(value is not None for value in indices),
        "duplicate_source_index_count": sum(count - 1 for count in index_counts.values()),
        "normalization_events": events, **_counts(records),
    }
    document = {
        "schema_version": SCHEMA_VERSION, "source_sha256": source_sha256,
        "source_sheet": sheet_name, "header_row": 1,
        "source_index_meaning": "Original unnamed first column; DICOM correspondence is unverified.",
        "gamma_criteria": {"gpr_2mm_2pct": "2mm/2%", "gpr_3mm_3pct": "3mm/3%"},
        "gpr_unit": "percent", "date_representation": "ISO 8601; original Excel type is retained in workbook.json",
        "alignment_status": "unverified", "records": records,
        "normalization_events": events,
    }
    return document, summary


def _save_unchanged(destination: Path, content: bytes, root: Path) -> None:
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output path escapes the requested directory.")
    if destination.exists():
        if _sha(destination.read_bytes()) != _sha(content):
            raise FileExistsError("Existing output differs; choose a new version directory.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)


def _cell_comparison(left, right, left_dimensions: dict, right_dimensions: dict) -> dict:
    result = {}
    for name in left.sheetnames:
        if name not in right.sheetnames:
            result[name] = {"present_in_previous": False}
            continue
        a, b = left[name], right[name]
        a_dimensions = left_dimensions[name]
        b_dimensions = right_dimensions[name]
        max_row, max_column = max(a_dimensions[0], b_dimensions[0]), max(a_dimensions[1], b_dimensions[1])
        changed_by_column = Counter()
        for row in range(1, max_row + 1):
            for column in range(1, max_column + 1):
                ac, bc = a.cell(row, column), b.cell(row, column)
                if typed_value(ac.value, ac.data_type) != typed_value(bc.value, bc.data_type):
                    changed_by_column[str(column)] += 1
        result[name] = {"present_in_previous": True,
                        "same_dimensions": a_dimensions == b_dimensions,
                        "changed_typed_value_cells": sum(changed_by_column.values()),
                        "changed_cells_by_column_number": dict(changed_by_column)}
    return result


def convert_arccheck_workbook(source_path, raw_dir=None, output_dir=None,
                              previous_workbook=None, previous_labels=None) -> ExcelConversion:
    """Copy the exact source and export complete worksheets and separate records.

    All source files are read-only. Outputs are versioned; changed existing
    outputs are refused. Source data remain local and are never printed.
    """
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    def declared_dimensions(path):
        # Normal-mode worksheets omit trailing empty XML rows from max_row.
        # Read-only mode retains the source dimension, so complete JSON keeps
        # these rows as null instead of truncating the workbook's used range.
        dimension_book = load_workbook(path, read_only=True, data_only=False)
        try:
            return {sheet.title: (sheet.max_row, sheet.max_column) for sheet in dimension_book}
        finally:
            dimension_book.close()

    source_path = Path(source_path).resolve()
    raw_dir = Path(raw_dir or paths.DATA_DIR / "raw" / "taichi_label_update_20261027").resolve()
    output_dir = Path(output_dir or paths.DATA_DIR / "processed" / "taichi_legacy644_label_update_20261027").resolve()
    previous_workbook = Path(previous_workbook or paths.ARCCHECK_PATH).resolve()
    previous_labels = Path(previous_labels or paths.LABEL_PATH).resolve()
    source_content = source_path.read_bytes()
    source_hash = _sha(source_content)
    raw_copy = raw_dir / source_path.name
    _save_unchanged(raw_copy, source_content, raw_dir)
    if _sha(raw_copy.read_bytes()) != source_hash:
        raise OSError("Copied source failed SHA-256 verification.")

    book = load_workbook(source_path, read_only=False, data_only=False)
    cached = load_workbook(source_path, read_only=False, data_only=True)
    dimensions = declared_dimensions(source_path)
    sheets, standardized, summaries = [], {}, {}
    comparison = None
    try:
        for sheet in book:
            serialized_rows, raw_rows = [], []
            max_row, max_column = dimensions[sheet.title]
            for row in sheet.iter_rows(min_row=1, max_row=max_row,
                                       min_col=1, max_col=max_column):
                values, types, formula_cache = [], [], {}
                raw_rows.append([cell.value for cell in row])
                for cell in row:
                    value, kind = typed_value(cell.value, cell.data_type)
                    values.append(value)
                    types.append(kind)
                    if kind == "formula":
                        cv, ct = typed_value(cached[sheet.title][cell.coordinate].value)
                        formula_cache[cell.coordinate] = {"value": cv, "type": ct}
                item = {"row_number": row[0].row, "values": values, "types": types}
                if formula_cache:
                    item["formula_cached_values"] = formula_cache
                serialized_rows.append(item)
            sheets.append({"name": sheet.title, "state": sheet.sheet_state,
                           "max_row": max_row, "max_column": max_column,
                           "column_letters": [get_column_letter(i) for i in range(1, max_column + 1)],
                           "merged_ranges": sorted(str(area) for area in sheet.merged_cells.ranges),
                           "rows": serialized_rows})
            if sheet.title in SHEET_SCHEMAS:
                document, summary = standardize_rows(sheet.title, raw_rows, source_hash)
                standardized[sheet.title] = document
                summaries[sheet.title] = summary
        if previous_workbook.is_file():
            prior = load_workbook(previous_workbook, read_only=False, data_only=False)
            try:
                comparison = {"previous_sha256": _sha(previous_workbook.read_bytes()),
                              "byte_identical": source_content == previous_workbook.read_bytes(),
                              "sheet_values": _cell_comparison(book, prior, dimensions, declared_dimensions(previous_workbook))}
            finally:
                prior.close()
        complete = {
            "schema_version": SCHEMA_VERSION, "source_filename": source_path.name,
            "source_sha256": source_hash, "source_bytes": len(source_content),
            "excel_epoch": book.epoch.isoformat(),
            "representation": "Complete worksheet cell values and value types, including headers and empty rows. Dates/times use ISO 8601. Formatting is retained in the unchanged XLSX source.",
            "sheet_order": book.sheetnames, "sheets": sheets,
        }
    finally:
        book.close()
        cached.close()

    old_json_comparison = None
    if previous_labels.is_file() and "Sheet2" in standardized:
        old_records = json.loads(previous_labels.read_text(encoding="utf-8"))
        new_records = standardized["Sheet2"]["records"]
        count = min(len(old_records), len(new_records))
        old_json_comparison = {
            "previous_sha256": _sha(previous_labels.read_bytes()), "previous_records": len(old_records),
            "compared_rows": count,
            "equal_gpr_at_same_position": sum(new_records[index]["gpr_2mm_2pct"] == old_records[index].get("gpr_2mm_2pct") for index in range(count)),
            "interpretation": "Descriptive comparison only; source row order is not a validated DICOM mapping.",
        }
    audit = {
        "schema_version": SCHEMA_VERSION, "source_filename": source_path.name,
        "source_sha256": source_hash, "source_bytes": len(source_content),
        "raw_copy_sha256_verified": True, "sheets": summaries,
        "comparison_with_previous_workbook": comparison,
        "comparison_with_previous_json": old_json_comparison,
        "alignment_status": "unverified", "training_manifest_created": False,
        "mapping_requirement": "Confirm and document whether Sheet2's original first-column index n is the anonymization/export key for RP_pID{n:03d}.dcm; workbook authenticity alone does not establish this relation.",
    }
    complete_path = output_dir / "workbook.json"
    _save_unchanged(complete_path, _json(complete), output_dir)
    record_paths = {}
    for name, document in standardized.items():
        path = output_dir / f"labels_{name.lower()}.json"
        _save_unchanged(path, _json(document), output_dir)
        record_paths[name] = path
    audit_path = output_dir / "conversion_audit.json"
    _save_unchanged(audit_path, _json(audit), output_dir)
    return ExcelConversion(raw_copy, complete_path, record_paths, audit_path, audit)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Copy ArcCheck XLSX and export complete, sourced JSON without inferring DICOM alignment.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    conversion = convert_arccheck_workbook(args.source, args.raw_dir, args.output_dir)
    # Aggregate audit only: no patient records are printed.
    print(json.dumps(conversion.audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
