"""Check the Git index or a commit tree, never upload or print matched secrets."""
from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess


ROOT_FILES = {".gitignore", "README.md", "pyproject.toml", "requirements.txt"}
EXACT_FILES = {
    "scripts/check_public_release.py",
    "configs/public/algorithm_catalog.json", "configs/public/model_configurations.json",
    "requirements/locks/README.md", "requirements/locks/validate_qbench_environment.py",
    "requirements/locks/qbench-win-cpu-py312-20261028.txt",
    "requirements/locks/qbench-win-cpu-py312-20261028.wheels.json",
    "src/ai4ct/qbench/LICENSE_Xanadu.txt", "tests/fixtures/LICENSE_Xanadu.txt",
    "examples/transformer_imdb/requirements.txt",
}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "api_secret": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "aws_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "personal_directory": re.compile(r"(?:[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s]+|/(?:Users|home)/[^/\s]+)", re.I),
}
PRIVATE_JSON_KEYS = {"dicom_sha256", "manifest_sha256", "features_sha256", "patient_name",
                     "patient_id", "patient_group_id", "sample_id", "gpr", "coverage_sha256"}


def allowed_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if name in ROOT_FILES or name in EXACT_FILES:
        return True
    if path.suffix == ".py" and name.startswith(("src/ai4ct/", "tests/", "examples/")):
        return True
    return path.suffix == ".md" and name.startswith("docs/public/")


def git(*args):
    return subprocess.check_output(["git", *args])


def private_keys(value):
    if isinstance(value, dict):
        return bool(PRIVATE_JSON_KEYS & set(value)) or any(private_keys(v) for v in value.values())
    if isinstance(value, list):
        return any(private_keys(v) for v in value)
    return False


def audit(staged=False, ref="HEAD"):
    listing = git("ls-files", "--stage", "-z") if staged else git("ls-tree", "-rz", ref)
    violations, count, total = [], 0, 0
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, raw_name = entry.split(b"\t", 1)
        name = raw_name.decode("utf-8")
        parts = meta.decode("ascii").split()
        mode, oid = parts[0], parts[1] if staged else parts[2]
        count += 1
        if mode not in {"100644", "100755"} or (staged and parts[2] != "0"):
            violations.append({"path": name, "reason": "symlink/submodule/unmerged entry"})
            continue
        if not allowed_path(name):
            violations.append({"path": name, "reason": "outside public allowlist"})
            continue
        content = git("cat-file", "blob", oid)
        total += len(content)
        if len(content) > 2_000_000 or b"\0" in content:
            violations.append({"path": name, "reason": "large or binary content"})
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            violations.append({"path": name, "reason": "non-UTF8 content"})
            continue
        for reason, pattern in PATTERNS.items():
            if pattern.search(text):
                violations.append({"path": name, "reason": reason})
        if name.startswith("configs/public/") and private_keys(json.loads(text)):
            violations.append({"path": name, "reason": "private cohort fields in configuration"})
    if not count:
        violations.append({"path": "", "reason": "empty publication tree"})
    result = {"status": "passed" if not violations else "failed", "files": count,
              "bytes": total, "violations": violations,
              "scope": "allowlist and common-pattern checks; human privacy review is still required"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if violations else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    return audit(args.staged, args.ref)


if __name__ == "__main__":
    raise SystemExit(main())
