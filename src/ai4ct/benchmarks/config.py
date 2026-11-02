# ai4ct/benchmarks/config.py
"""
Central configuration for all benchmark models.

All hyperparameters are gathered here so that experiments can be
reproduced by changing values in one place.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional

from ai4ct.paths import FEATURE_TABLE, BENCHMARK_OUTPUT

# ================================================================
#  Random seed (fixed for reproducibility)
# ================================================================
SEED = 42

# ================================================================
#  Data
# ================================================================
DATA_PATH = FEATURE_TABLE
FEATURE_COLS: Tuple[int, int] = (3, 16)   # 13 feature columns (excluding MU)
LABEL_COL: int = 16                       # label column (0-indexed)                        # AL column
TEST_SIZE: float = 0.2

# ================================================================
#  Output
# ================================================================
OUTPUT_DIR = BENCHMARK_OUTPUT / "baseline"


# ================================================================
#  Per-model hyperparameters
# ================================================================

@dataclass
class GBDTConfig:
    name: str = "GBDT"
    n_estimators: int = 200
    learning_rate: float = 0.2
    max_depth: int = 3
    subsample: float = 0.7


@dataclass
class MLPConfig:
    name: str = "Classical-MLP"
    input_dim: int = 13
    hidden_dim1: int = 16
    hidden_dim2: int = 8
    output_dim: int = 2
    activation: str = "leaky_relu"    # "leaky_relu" | "relu" | "gelu"
    lr: float = 0.005
    epochs: int = 65
    optimizer: str = "adam"           # "adam" | "adamw"


@dataclass
class PQCHybridConfig:
    name: str = "PQC-HybridNet"
    input_dim: int = 13
    hidden_dim1: int = 24
    hidden_dim2: int = 12           # must be even (gives 6 qubits)
    output_dim: int = 2
    num_qubits: Optional[int] = None # derived: hidden_dim2 // 2
    num_layer: int = 2               # PQC entanglement layers
    activation: str = "leaky_relu"
    lr: float = 0.005
    epochs: int = 205
    optimizer: str = "adamw"
    backend: str = "torch"           # pure PyTorch backend

    def __post_init__(self):
        if self.num_qubits is None:
            self.num_qubits = self.hidden_dim2 // 2


@dataclass
class QFMVQCConfig:
    name: str = "QFM-VQCNet"
    input_dim: int = 13
    hidden_dim1: int = 16
    hidden_dim2: int = 5             # maps to num_qubits
    output_dim: int = 2
    num_qubits: int = 5
    reps: int = 3                    # VQC ansatz repetitions
    lr: float = 0.005
    epochs: int = 12
    batch_size: int = 48
    optimizer: str = "adamw"


# ================================================================
#  Active configuration (change this dict to run different setups)
# ================================================================
MODELS = {
    "gbdt":       GBDTConfig(),
    "mlp":        MLPConfig(),
    "pqc_hybrid": PQCHybridConfig(),
    "qfm_vqc":    QFMVQCConfig(),
}
