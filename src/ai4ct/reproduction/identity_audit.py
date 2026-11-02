"""Read-only, recursive DICOM identity evidence audit against a local workbook.

No raw identity values are logged or saved. Candidates identify source sheet/row
and DICOM tag paths only, and never constitute an automatic label mapping.
UIDs, numeric measurements, row order and model performance are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import warnings

import pydicom
from pydicom.multival import MultiValue


TEXT_VRS = {"AE", "CS", "LO", "LT", "PN", "SH", "ST", "UC", "UR", "UT"}
WATCH_TAGS = {
    "00100010": "PatientName", "00100020": "PatientID",
    "00100021": "IssuerOfPatientID", "00101000": "OtherPatientIDs",
    "00101001": "OtherPatientNames", "00101002": "OtherPatientIDsSequence",
    "00101090": "MedicalRecordLocator", "00120062": "PatientIdentityRemoved",
    "00120063": "DeidentificationMethod", "00120064": "DeidentificationMethodCodeSequence",
    "00200010": "StudyID", "00081030": "StudyDescription",
    "0008103e": "SeriesDescription", "300a0002": "RTPlanLabel",
    "300a0003": "RTPlanName", "300a0004": "RTPlanDescription",
    "04000550": "ModifiedAttributesSequence", "04000551": "NonconformingModifiedAttributesSequence",
    "04000561": "OriginalAttributesSequence",
}
ANONYMOUS = re.compile(r"^p[ _-]?id[ _-]?\d+$", re.IGNORECASE)
NAME_HEADERS = {"姓名", "患者姓名", "病人姓名", "patientname", "name"}
ID_HEADERS = {"id", "id号", "病案号", "病历号", "患者id", "病人id", "patientid"}


def normalized(value):
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return unicodedata.normalize("NFKC", str(value)).strip().strip("\x00")


def normalized_name(value):
    return re.sub(r"[\s^]+", "", normalized(value)).casefold()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class IdentityLookup:
    """Identity values exist in memory only; emitted references contain no values."""
    def __init__(self, records):
        self.names = defaultdict(list)
        self.ids = defaultdict(list)
        for record in records:
            source = {"sheet": record["sheet"], "row": int(record["row"])}
            name = normalized_name(record.get("name"))
            identifier = normalized(record.get("patient_id"))
            if name:
                self.names[name].append(source)
            if identifier:
                self.ids[identifier].append(source)
        # Literal names in descriptive fields are only review candidates. They
        # may refer to staff or another person and are never accepted mappings.
        names = [re.escape(name) for name in self.names if len(name) >= 2]
        identifiers = [re.escape(value) for value in self.ids if len(value) >= 4]
        self.name_pattern = re.compile("|".join(sorted(names, key=len, reverse=True))) if names else None
        self.id_pattern = (re.compile(r"(?<![\w.])(?:" + "|".join(sorted(identifiers, key=len, reverse=True))
                                      + r")(?![\w.])") if identifiers else None)
        self.cache = {}

    def match(self, value, vr):
        if vr not in TEXT_VRS:
            return []
        text = normalized(value)
        key = (text, vr)
        if key in self.cache:
            return self.cache[key]
        matches = []
        ids = [(text, "exact_id")] if text in self.ids else []
        if not ids and self.id_pattern:
            ids = [(m.group(), "bounded_id_in_text") for m in self.id_pattern.finditer(text)]
        for token, mode in ids:
            for source in self.ids[token]:
                matches.append({**source, "match_type": mode})
        name_text = normalized_name(text)
        names = [(name_text, "exact_name")] if name_text in self.names else []
        if not names and self.name_pattern:
            names = [(m.group(), "literal_name_in_text") for m in self.name_pattern.finditer(name_text)]
        for token, mode in names:
            for source in self.names[token]:
                matches.append({**source, "match_type": mode})
        self.cache[key] = matches
        return matches


def read_workbook_identities(path):
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records, sheets = [], []
    try:
        for sheet in workbook:
            source_rows = sheet.iter_rows(values_only=True)
            header_row = None
            name_column = id_column = None
            for number, values in enumerate(source_rows, 1):
                headers = [re.sub(r"\s+", "", normalized(value)).casefold() for value in values]
                name_column = next((i for i, value in enumerate(headers) if value in NAME_HEADERS), None)
                id_column = next((i for i, value in enumerate(headers) if value in ID_HEADERS), None)
                if name_column is not None or id_column is not None:
                    header_row = number
                    break
                if number >= 10:
                    break
            if header_row is None:
                sheets.append({"sheet": sheet.title, "identity_columns_found": False})
                continue
            sheet_records = []
            for number, values in enumerate(source_rows, header_row + 1):
                name = values[name_column] if name_column is not None and name_column < len(values) else None
                identifier = values[id_column] if id_column is not None and id_column < len(values) else None
                if normalized(name) or normalized(identifier):
                    sheet_records.append({"sheet": sheet.title, "row": number,
                                          "name": name, "patient_id": identifier})
            sheets.append({"sheet": sheet.title, "identity_columns_found": True,
                           "header_row": header_row, "name_column_1based": None if name_column is None else name_column + 1,
                           "id_column_1based": None if id_column is None else id_column + 1,
                           "rows_with_identity": len(sheet_records),
                           "distinct_names": len({normalized_name(r["name"]) for r in sheet_records if normalized_name(r["name"])}),
                           "distinct_ids": len({normalized(r["patient_id"]) for r in sheet_records if normalized(r["patient_id"])})})
            records.extend(sheet_records)
    finally:
        workbook.close()
    return IdentityLookup(records), sheets


def inspect_dataset(dataset, lookup, numeric_surrogate_values=()):
    """Walk every nested element; return only counts, tag paths and references."""
    counts = Counter()
    vrs = Counter()
    fields = defaultdict(Counter)
    candidates = []

    def walk(current, prefix="", depth=0):
        counts["maximum_sequence_depth"] = max(counts["maximum_sequence_depth"], depth)
        for element in current:
            tag = f"{int(element.tag):08x}"
            path = f"{prefix}/{tag}" if prefix else tag
            counts["element_instances"] += 1
            counts["nested_element_instances"] += int(depth > 0)
            counts["private_element_instances"] += int(element.tag.is_private)
            vrs[element.VR] += 1
            if tag in WATCH_TAGS or element.VR in {"PN", "LO"} or element.tag.is_private:
                fields[(tag, element.keyword or "private_or_unknown", element.VR)]["instances"] += 1
            if element.VR == "SQ":
                counts["sequence_element_instances"] += 1
                counts["sequence_item_instances"] += len(element.value)
                for index, item in enumerate(element.value):
                    walk(item, f"{path}[{index}]", depth + 1)
                continue
            counts["scalar_element_instances"] += 1
            if element.tag.is_private and isinstance(element.value, bytes):
                payload = element.value
                counts["private_binary_element_instances"] += 1
                counts["private_binary_total_bytes"] += len(payload)
                counts["private_binary_ascii_printable_instances"] += bool(payload) and all(32 <= v < 127 for v in payload)
                field = fields[(tag, element.keyword or "private_or_unknown", element.VR)]
                field["binary_payload_bytes"] += len(payload)
                matched = set()
                for encoding in ("utf-8", "utf-16-le", "gb18030", "latin-1"):
                    try:
                        decoded = payload.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                    counts["private_binary_successful_text_decodings"] += 1
                    for match in lookup.match(decoded, "UT"):
                        key = (match["sheet"], match["row"], match["match_type"])
                        if key in matched:
                            continue
                        matched.add(key)
                        candidates.append({"tag": tag, "keyword": element.keyword or "private_or_unknown",
                                           "vr": element.VR, "tag_path": path, "private": True,
                                           "source_sheet": match["sheet"], "source_row": match["row"],
                                           "match_type": "decoded_private_" + match["match_type"],
                                           "decoding": encoding, "mapping_status": "candidate_only"})
                continue
            if element.VR == "UI":
                counts["uid_elements_excluded_from_identity_matching"] += 1
                continue
            if element.VR not in TEXT_VRS:
                continue
            values = element.value if isinstance(element.value, MultiValue) else [element.value]
            for value in values:
                text = normalized(value)
                counts["text_value_instances"] += 1
                field = fields[(tag, element.keyword or "private_or_unknown", element.VR)]
                field["nonempty_values"] += bool(text)
                field["empty_values"] += not bool(text)
                anonymous = bool(ANONYMOUS.fullmatch(normalized_name(text)))
                field["anonymous_pid_values"] += anonymous
                numeric_surrogate = text in numeric_surrogate_values
                field["numeric_surrogate_values"] += numeric_surrogate
                if element.VR in {"PN", "LO"}:
                    counts[f"{element.VR}_value_instances"] += 1
                    counts[f"{element.VR}_nonempty_values"] += bool(text)
                    counts[f"{element.VR}_anonymous_pid_values"] += anonymous
                    counts[f"{element.VR}_numeric_surrogate_values"] += numeric_surrogate
                for match in lookup.match(text, element.VR):
                    candidates.append({"tag": tag, "keyword": element.keyword or "private_or_unknown",
                                       "vr": element.VR, "tag_path": path, "private": element.tag.is_private,
                                       "source_sheet": match["sheet"], "source_row": match["row"],
                                       "match_type": match["match_type"], "mapping_status": "candidate_only"})
    walk(dataset)
    if hasattr(dataset, "file_meta"):
        walk(dataset.file_meta, "file_meta")
    return counts, vrs, fields, candidates


def audit(dicom_dir, workbook_path):
    dicom_dir, workbook_path = Path(dicom_dir), Path(workbook_path)
    lookup, sheets = read_workbook_identities(workbook_path)
    counts, vrs = Counter(), Counter()
    field_totals = defaultdict(Counter)
    candidates, failures = [], []
    patient_ids, patient_names = set(), set()
    files = sorted(dicom_dir.glob("*.dcm"))
    for path in files:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dataset = pydicom.dcmread(path, stop_before_pixels=False)
                pid = normalized(dataset.get("PatientID", ""))
                pname = normalized_name(dataset.get("PatientName", ""))
                file_number = re.fullmatch(r"RP_pID([0-9]+)", path.stem)
                numeric_surrogate = (bool(file_number) and pid.isdecimal()
                                     and int(pid) == int(file_number.group(1)))
                surrogate_values = {pid} if numeric_surrogate else set()
                local_counts, local_vrs, fields, matches = inspect_dataset(dataset, lookup, surrogate_values)
                counts["dicom_warning_count"] += len(caught)
        except Exception as exc:
            failures.append({"file": path.name, "error_type": type(exc).__name__})
            continue
        counts["files_read"] += 1
        for key, value in local_counts.items():
            if key == "maximum_sequence_depth":
                counts[key] = max(counts[key], value)
            else:
                counts[key] += value
        vrs.update(local_vrs)
        for key, values in fields.items():
            field_totals[key].update(values)
            field_totals[key]["files_containing"] += 1
        for match in matches:
            candidates.append({"dicom_file": path.name, **match})
        pid = normalized(dataset.get("PatientID", ""))
        pname = normalized_name(dataset.get("PatientName", ""))
        if pid:
            patient_ids.add(pid)
        if pname:
            patient_names.add(pname)
        counts["files_with_anonymous_patient_id"] += bool(ANONYMOUS.fullmatch(pid))
        counts["files_with_anonymous_patient_name"] += bool(ANONYMOUS.fullmatch(pname))
        counts["files_patient_id_equals_patient_name"] += pid.casefold() == pname
        counts["files_with_numeric_surrogate_patient_id"] += numeric_surrogate
        counts["files_numeric_patient_id_matches_filename_number"] += numeric_surrogate
        counts["files_numeric_patient_name_matches_filename_number"] += numeric_surrogate and pname == pid
    watch_report = {}
    for tag, keyword in WATCH_TAGS.items():
        matches = [(key, value) for key, value in field_totals.items() if key[0] == tag]
        watch_report[tag] = {"keyword": keyword,
                             "element_instances": sum(value["instances"] for _, value in matches),
                             "files_containing": sum(value["files_containing"] for _, value in matches)}
    rows_by_file = defaultdict(set)
    for candidate in candidates:
        rows_by_file[candidate["dicom_file"]].add((candidate["source_sheet"], candidate["source_row"]))
    return {
        "audit_version": 2, "workbook_file": workbook_path.name, "workbook_sha256": digest(workbook_path),
        "dicom_directory": str(dicom_dir.resolve()), "dicom_file_count": len(files),
        "workbook_identity_sources": sheets, "structure_counts": dict(counts),
        "vr_element_counts": dict(sorted(vrs.items())), "watched_tags": watch_report,
        "text_and_identity_fields": [{"tag": tag, "keyword": keyword, "vr": vr, **dict(values)}
                                     for (tag, keyword, vr), values in sorted(field_totals.items())],
        "top_level_distinct_patient_ids": len(patient_ids), "top_level_distinct_patient_names": len(patient_names),
        "identity_matches": {"candidate_evidence_count": len(candidates), "matched_dicom_files": len(rows_by_file),
                             "files_with_multiple_source_rows": sum(len(rows) > 1 for rows in rows_by_file.values()),
                             "match_type_counts": dict(Counter(c["match_type"] for c in candidates))},
        "candidates": candidates, "read_failures": failures, "mapping_status": "not_established",
        "independent_real_patients_established": False,
        "interpretation": "Distinct anonymous IDs prove distinct pseudonyms only, not distinct real patients.",
        "numeric_surrogate_interpretation": "A numeric PatientID equal to the anonymous filename number is counted only as a surrogate-format observation, never as evidence for an Excel row/ordinal label mapping.",
        "matching_rules": ["Unicode NFKC and edge whitespace normalization only for IDs; leading zeros preserved.",
                           "Names additionally ignore whitespace/PN component separators and letter case; no transliteration.",
                           "Full-value exact IDs/names and bounded full ID tokens in text are candidates only.",
                           "Literal full names in descriptive text are reviewed as weak candidates, never accepted mappings.",
                           "No UID fields, numeric measurements, row order, model performance or short numeric substrings used."],
        "limitations": ["No inverse mapping for irreversible anonymization is inferred.",
                        "Private byte payloads are additionally checked using strict UTF-8, UTF-16LE, GB18030 and Latin-1 decoding. No compressed/encrypted proprietary payload semantics are inferred.",
                        "A matching name can identify a physician or a homonym; candidate evidence needs provenance review."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Audit output already exists; choose a new output file.")
    report = audit(args.dicom_dir, args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"dicom_file_count": report["dicom_file_count"],
                      "structure_counts": report["structure_counts"],
                      "identity_matches": report["identity_matches"],
                      "read_failure_count": len(report["read_failures"]),
                      "mapping_status": report["mapping_status"],
                      "independent_real_patients_established": False}, indent=2))
    return 2 if report["read_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
