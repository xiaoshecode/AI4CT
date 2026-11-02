"""
Central configuration for the Transformer learning project.

All hyperparameters are collected here so you can tweak them in one place
and understand how each one affects the model and training.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parent


# =============================================================================
# Transformer Architecture
# =============================================================================

class TransformerConfig:
    """Hyperparameters for the Transformer encoder architecture.

    The original paper used d_model=512, n_heads=8, n_layers=6, d_ff=2048
    for the base model. We use slightly smaller values to keep training fast
    while still demonstrating the full architecture.
    """

    # --- Token & Embedding ---
    vocab_size: int = 10000       # Vocabulary size (top-K most frequent words)
    max_seq_len: int = 512        # Maximum sequence length (pad/truncate to this)
    d_model: int = 256            # Embedding dimension — every token becomes a vector of this size
    pad_idx: int = 0              # Index of <pad> token in vocabulary
    cls_idx: int = 2              # Index of <cls> token (used for classification)

    # --- Attention ---
    n_heads: int = 8              # Number of parallel attention heads
    # Each head works with d_k = d_model / n_heads = 32 dimensions
    # Multiple heads let the model attend to different linguistic patterns
    # simultaneously (e.g., one head for syntax, another for semantics)

    # --- Feed-Forward ---
    d_ff: int = 1024              # Hidden dimension of FFN (typically 4 × d_model)
    activation: str = "gelu"      # GELU tends to work slightly better than ReLU for NLP

    # --- Encoder Stack ---
    n_layers: int = 6             # Number of stacked encoder layers
    # Each layer = MultiHeadAttention + FeedForward with residual connections
    # More layers = deeper understanding but slower training

    # --- Regularization ---
    dropout: float = 0.1          # Dropout probability (applied to attention weights,
    # FFN hidden activations, and embeddings)


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig:
    """Hyperparameters for training the Transformer classifier."""

    # --- Data ---
    batch_size: int = 128         # Samples per batch (GPU-optimized; reduce to 32 for CPU)
    num_workers: int = 0          # DataLoader workers (0 = main process only; set 0 on Windows)
    val_split: float = 0.1        # Fraction of training data to use as validation

    # --- Optimization ---
    lr: float = 3e-4              # Peak learning rate (scaled from 1e-4 base for batch_size=128)
    # Rule of thumb: lr ∝ sqrt(batch_size) — larger batches → larger stable LR
    beta1: float = 0.9            # Adam β₁ (first moment decay)
    beta2: float = 0.98           # Adam β₂ (second moment decay — 0.98 is standard for NLP)
    eps: float = 1e-9             # Adam ε (numerical stability)
    weight_decay: float = 0.01    # L2 regularization strength (AdamW decouples this from LR)

    # --- LR Schedule (Warmup + Cosine Decay) ---
    warmup_steps: int = 1000      # Number of steps to linearly increase LR from 0 → peak
    # Warmup prevents early training instability — the model starts with random
    # attention weights, and large gradients early on can cause divergence.
    # After warmup, cosine decay smoothly reduces LR to near zero by the end.

    # --- Gradient Control ---
    grad_clip_norm: float = 1.0   # Clip gradients to this max L2 norm
    # Prevents "gradient explosion" where a single batch causes a huge update

    # --- Training Loop ---
    epochs: int = 10              # Maximum number of training epochs
    early_stop_patience: int = 3  # Stop if val loss doesn't improve for this many epochs
    log_interval: int = 50        # Print training loss every N batches

    # --- Mixed Precision (GPU only) ---
    use_amp: bool = True          # Use Automatic Mixed Precision (FP16) on GPU
    # AMP runs most operations in FP16 for 1.5-2× speedup, while keeping
    # numerically-sensitive ops (softmax, layer norm) in FP32.
    # Set to False if you encounter NaN losses.


# =============================================================================
# System Configuration
# =============================================================================

class SystemConfig:
    """System-level settings and paths."""

    # --- Device ---
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Reproducibility ---
    seed: int = 42

    # --- Output Paths ---
    checkpoint_dir: str = str(EXAMPLE_DIR / "checkpoints")
    plot_dir: str = str(EXAMPLE_DIR / "plots")
    data_dir: str = str(EXAMPLE_DIR / "data_cache")

    # --- Dataset URLs ---
    imdb_url: str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


# =============================================================================
# Convenience access
# =============================================================================

# Create singleton instances for easy importing
model = TransformerConfig()
train = TrainingConfig()
system = SystemConfig()
