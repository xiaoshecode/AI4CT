"""Read and verify the frozen cohort without deriving new labels or folds."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .atomic import replace_with_retry

import numpy as np

from ai4ct.paths import DATA_DIR, PROJECT_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return DATA_DIR.joinpath(*path.parts[1:])
    return PROJECT_ROOT / path


def checked_path(value, expected_hash, label):
    if not expected_hash:
        raise ValueError(f"Missing frozen hash for {label}")
    path = resolve_path(value)
    if sha256(path) != expected_hash:
        raise ValueError(f"Frozen hash differs: {label}")
    return path


@dataclass
class Dataset:
    samples: list
    x: np.ndarray
    fold: np.ndarray
    schema: dict
    provenance: dict

    @property
    def rates(self):
        return np.asarray([row["gpr"] for row in self.samples], dtype=float)

    @property
    def groups(self):
        return np.asarray([row["group_id"] for row in self.samples])


def load_dataset(config):
    """Require pinned input hashes; join saved folds by ID, never regenerate."""
    cohort = config["cohorts"]["new"]
    evaluation = config["evaluation"]
    if evaluation.get("outer_split_regeneration_allowed", False):
        raise ValueError("Outer fold regeneration must be disabled")
    manifest_path = checked_path(cohort["manifest"], cohort["manifest_sha256"], "manifest")
    manifest = read_json(manifest_path)
    source_samples = manifest["samples"] if isinstance(manifest, dict) else manifest
    if not source_samples:
        raise ValueError("Empty cohort")
    samples = []
    for row in source_samples:
        if row["mapping_status"] != "explicit_serial_and_internal_id_agree":
            raise ValueError("Unverified cohort mapping")
        rate = float(row["gpr"])
        if not np.isfinite(rate) or not 0 <= rate <= 100:
            raise ValueError("Invalid GPR")
        if not row["sample_id"] or not row["group_id"]:
            raise ValueError("Missing sample/group ID")
        relative = row.get("dicom_relative_path")
        dicom_path = manifest_path.parent / relative if relative else resolve_path(row["dicom_path"])
        if sha256(dicom_path) != row["dicom_sha256"]:
            raise ValueError("Source DICOM hash differs")
        samples.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                        "gpr": rate, "dicom_sha256": row["dicom_sha256"]})
    ids = [row["sample_id"] for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cohort IDs")
    if len(samples) != cohort["samples"] or len({s["group_id"] for s in samples}) != cohort["groups"]:
        raise ValueError("Cohort sample/group counts differ")
    split_path = checked_path(evaluation["outer_split_source"], evaluation["outer_split_sha256"], "outer splits")
    split_rows = read_json(split_path)
    split_index = {s["sample_id"]: s for s in split_rows}
    if len(split_index) != len(split_rows) or set(split_index) != set(ids):
        raise ValueError("Split coverage differs from manifest")
    folds = []
    for sample in samples:
        row = split_index[sample["sample_id"]]
        if row["group_id"] != sample["group_id"] or type(row["test_fold"]) is not int:
            raise ValueError("Split group or fold differs")
        folds.append(row["test_fold"])
    folds = np.asarray(folds, dtype=int)
    if set(folds) != set(range(evaluation["outer_folds"])):
        raise ValueError("Invalid saved outer fold set")
    groups = np.asarray([s["group_id"] for s in samples])
    if any(len(set(folds[groups == g])) != 1 for g in set(groups)):
        raise ValueError("A patient/duplicate group crosses outer folds")
    feature = config.get("features", cohort)
    feature_path = checked_path(feature["features_path"], feature["features_sha256"], "features")
    schema_path = checked_path(feature["feature_schema_path"], feature["feature_schema_sha256"], "feature schema")
    schema = read_json(schema_path)
    names = schema["ordered_names"]
    if len(names) != cohort["features"] or len(set(names)) != len(names) or schema["version"] != cohort["feature_version"]:
        raise ValueError("Feature schema differs from protocol")
    with feature_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", *names]:
            raise ValueError("Feature column order differs")
        rows = list(reader)
    if [r["sample_id"] for r in rows] != ids:
        raise ValueError("Feature sample order differs from manifest")
    x = np.asarray([[float(r[n]) for n in names] for r in rows], dtype=np.float64)
    if x.shape != (len(ids), len(names)) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    for threshold in evaluation["gpr_thresholds"]:
        y = np.asarray([s["gpr"] >= float(threshold) for s in samples], dtype=int)
        for f in range(evaluation["outer_folds"]):
            if len(set(y[folds == f])) != 2 or len(set(y[folds != f])) != 2:
                raise ValueError("Every outer training/test partition must contain both classes")
    provenance = {"manifest_sha256": cohort["manifest_sha256"],
                  "outer_split_sha256": evaluation["outer_split_sha256"],
                  "features_sha256": feature["features_sha256"],
                  "feature_schema_sha256": feature["feature_schema_sha256"],
                  "sample_records_sha256": digest_json(samples), "source_dicoms_verified": len(samples)}
    return Dataset(samples, x, folds, schema, provenance)
