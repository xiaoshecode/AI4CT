"""Read-only environment and aggregate data checks; never print patient records."""
import argparse
from collections import Counter
from importlib import metadata
import json
import math
from pathlib import Path

from . import paths

CORE_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pydicom", "matplotlib", "openpyxl")
OPTIONAL_PACKAGES = ("torch", "lightgbm", "shap", "pyvqnet")


def inspect_data(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / "taichi"
    files = list(raw.glob("RP_pID*.dcm"))
    errors, warnings = [], []
    report = {"dicom_count": len(files), "errors": errors, "warnings": warnings}
    if not files:
        errors.append("No RP_pID*.dcm files found in raw/taichi.")
    label_file = raw / "arccheck_labels.json"
    labels = None
    try:
        labels = json.loads(label_file.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(row, dict) for row in labels):
            errors.append("Labels must be a JSON list of objects.")
            labels = None
    except (OSError, ValueError):
        errors.append("Labels file is missing or is not readable JSON.")
    if labels is not None:
        report["label_count"] = len(labels)
        if len(labels) != len(files):
            errors.append("DICOM and label counts differ; positional matching is unsafe.")
        rates = [row.get("gpr_2mm_2pct") for row in labels]
        valid = [value for value in rates if isinstance(value, (int, float))
                 and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100]
        report["invalid_gpr_count"] = len(rates) - len(valid)
        if len(valid) != len(rates):
            errors.append("Some GPR values are missing, nonnumeric, nonfinite or outside 0..100.")
        report["all_labels_descriptive_counts"] = {
            str(threshold): {"pass": sum(v >= threshold for v in valid),
                             "fail": sum(v < threshold for v in valid)}
            for threshold in (90, 95)
        }
    warnings.append("Equal row counts do not prove plan-label alignment. Verify a stable-ID mapping before training.")

    tables = {}
    table_paths = sorted((data_dir / "processed").glob("radiotherapy_features*.xlsx"))
    if not table_paths:
        warnings.append("No historical feature workbooks found in processed/.")
    try:
        import openpyxl
    except ImportError:
        warnings.append("openpyxl is unavailable; feature workbook checks skipped.")
    else:
        for file in table_paths:
            try:
                workbook = openpyxl.load_workbook(file, read_only=True, data_only=True)
                try:
                    rows = workbook.active.iter_rows(values_only=True)
                    headers = list(next(rows))
                    if (any(not isinstance(header, str) or not header.strip() for header in headers)
                            or len(set(headers)) != len(headers)):
                        errors.append(f"{file.name}: headers must be nonempty, unique strings.")
                        continue
                    records = [dict(zip(headers, row)) for row in rows if any(v is not None for v in row)]
                finally:
                    workbook.close()
                if "label" not in headers:
                    errors.append(f"{file.name}: missing label column.")
                    continue
                tables[file.name] = (headers, records)
            except Exception as exc:
                # Exception text may include workbook contents; report only the error type.
                errors.append(f"{file.name}: cannot inspect workbook ({type(exc).__name__}).")
    report["feature_tables"] = {
        name: {"rows": len(rows), "columns": len(headers),
               "label_counts": dict(Counter(str(row["label"]) for row in rows))}
        for name, (headers, rows) in tables.items()
    }
    comparisons = []
    names = list(tables)
    for left, right in zip(names, names[1:]):
        left_headers, left_rows = tables[left]
        right_headers, right_rows = tables[right]
        common = sorted((set(left_headers) & set(right_headers)) - {"label", "meta1", "meta2", "meta3", "m1", "m2", "m3"})
        if len(left_rows) != len(right_rows):
            errors.append(f"{left} vs {right}: feature table row counts differ.")
        paired = list(zip(left_rows, right_rows))
        mismatches = sum(a["label"] != b["label"] for a, b in paired)
        identical_features = sum(all(a[k] == b[k] for k in common) for a, b in paired) if common else 0
        comparisons.append({"left": left, "right": right, "paired_rows": len(paired),
                            "shared_feature_columns": len(common),
                            "identical_feature_rows": identical_features,
                            "different_label_rows": mismatches})
        if mismatches:
            errors.append(f"{left} vs {right}: historical labels differ at {mismatches} row positions; provenance must be reconciled.")
    report["feature_version_comparisons"] = comparisons
    report["alignment_status"] = "unverified"
    return report


def inspect_environment() -> dict:
    versions = {}
    for name in CORE_PACKAGES + OPTIONAL_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions, "missing_core": [name for name in CORE_PACKAGES if versions[name] is None]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only, aggregate workspace checks; no training or patient-record output")
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR, help="Data root containing raw/ and processed/")
    parser.add_argument("--json", action="store_true", help="Emit aggregate JSON")
    args = parser.parse_args(argv)
    report = {"environment": inspect_environment(), "data": inspect_data(args.data_dir)}
    failed = bool(report["environment"]["missing_core"] or report["data"]["errors"])
    report["status"] = "needs_attention" if failed else "checks_passed_alignment_unverified"
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"AI4CT doctor: {report['status']}")
        print(f"DICOM files: {report['data']['dicom_count']}; labels: {report['data'].get('label_count', 'unavailable')}")
        for name in report["environment"]["missing_core"]:
            print(f"ERROR: Missing core dependency: {name}")
        for level in ("errors", "warnings"):
            for message in report["data"][level]:
                print(f"{level.upper()}: {message}")
        print("See docs/data_inventory.md and docs/research_review.md before training.")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
