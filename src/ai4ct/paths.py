"""Shared paths, independent of the process working directory.

Environment overrides must be absolute. Importing this module never creates files.
AI4CT_DATA_DIR refers to the data root (containing raw/ and processed/).
"""
from pathlib import Path
import os


def _absolute_override(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {value!r}")
    return path.resolve()


def _project_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        default = candidate
    else:
        # Installed wheel: data live in a user workspace, not site-packages.
        default = Path.home() / "AI4CT"
    return _absolute_override("AI4CT_PROJECT_ROOT", default)


PROJECT_ROOT = _project_root()
DATA_DIR = _absolute_override("AI4CT_DATA_DIR", PROJECT_ROOT / "data")
DICOM_DIR = DATA_DIR / "raw" / "taichi"
LABEL_PATH = DICOM_DIR / "arccheck_labels.json"
ARCCHECK_PATH = DICOM_DIR / "匿名化—太极ArcCheck验证患者.xlsx"
FEATURE_TABLE = DATA_DIR / "processed" / "radiotherapy_features_v3.xlsx"
OUTPUT_ROOT = _absolute_override("AI4CT_OUTPUT_DIR", PROJECT_ROOT / "outputs")
BENCHMARK_OUTPUT = OUTPUT_ROOT / "benchmarks"
ADVANCED_OUTPUT = OUTPUT_ROOT / "advanced"
PAPER_OUTPUT = OUTPUT_ROOT / "paper"
METRICS_OUTPUT = OUTPUT_ROOT / "metrics"
