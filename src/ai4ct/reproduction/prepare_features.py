"""Prepare unlabeled feature evidence while a cohort's QA mapping is unresolved."""
import argparse
from pathlib import Path

from .run import build_features, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    files = sorted(args.dicom_dir.glob("*.dcm"))
    if not files:
        parser.error("No DICOM files found")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new, empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records = [{"sample_id": p.stem, "dicom_path": str(p.resolve()), "dicom_sha256": sha256(p)} for p in files]
    write_json(args.output / "unlabeled_sources.json", records)
    write_json(args.output / "status.json", {"status": "extracting", "n_sources": len(records)})
    try:
        x, names = build_features(records, args.output)
    except Exception as exc:
        write_json(args.output / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        raise
    write_json(args.output / "status.json", {
        "status": "features_ready_mapping_required", "n_samples": len(x), "n_features": len(names),
        "labels_assigned": False, "models_trained": False,
        "next_input": "Audited anonymous-pID to QA-record mapping and patient grouping keys",
    })


if __name__ == "__main__":
    main()
