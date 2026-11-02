#!/usr/bin/env python
"""
Transformer for IMDB Sentiment Classification — Main Entry Point
================================================================
This script runs the complete pipeline:
  1. Downloads and preprocesses the IMDB dataset
  2. Builds the Transformer model
  3. Trains on GPU (with AMP) or CPU (with warning)
  4. Evaluates on the test set
  5. Generates visualization plots

Usage:
    python examples/transformer_imdb/run.py              # Train with default config
    python examples/transformer_imdb/run.py --epochs 20  # Override epochs
    python examples/transformer_imdb/run.py --cpu        # Force CPU training

After training, check:
    examples/transformer_imdb/checkpoints/best_model.pt   # Best model weights
    examples/transformer_imdb/plots/                      # Confusion matrix, curves, ROC
"""

from __future__ import annotations

import os
# Fix OpenMP conflict on Windows (libiomp5md.dll loaded twice)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import sys
import os
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.transformer_imdb.config import TransformerConfig, TrainingConfig, SystemConfig, model, train, system
from examples.transformer_imdb.model import TransformerClassifier, print_model_summary
from examples.transformer_imdb.data import create_dataloaders
from examples.transformer_imdb.trainer import Trainer
from examples.transformer_imdb.visualize import generate_all_plots


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility.

    NOTE: Even with fixed seeds, GPU training may not be 100% deterministic
    due to:
    - cuDNN benchmark auto-tuning
    - Floating-point non-associativity in parallel reductions
    - AMP casting order

    For full determinism, set: torch.backends.cudnn.deterministic = True
    (but this significantly slows down training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Uncomment for full determinism (slower):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# =============================================================================
# Parse Arguments
# =============================================================================

def parse_args():
    """Parse command line arguments.

    Any config parameter can be overridden via command line.
    """
    parser = argparse.ArgumentParser(
        description="Train a Transformer on IMDB sentiment classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/transformer_imdb/run.py                     # Default training
  python examples/transformer_imdb/run.py --epochs 20          # Train longer
  python examples/transformer_imdb/run.py --batch_size 64      # Smaller batch (less VRAM)
  python examples/transformer_imdb/run.py --cpu                # Force CPU training
  python examples/transformer_imdb/run.py --lr 1e-4            # Lower learning rate
        """,
    )
    # Model
    parser.add_argument("--d_model", type=int, default=model.d_model,
                        help=f"Embedding dimension (default: {model.d_model})")
    parser.add_argument("--n_heads", type=int, default=model.n_heads,
                        help=f"Attention heads (default: {model.n_heads})")
    parser.add_argument("--n_layers", type=int, default=model.n_layers,
                        help=f"Encoder layers (default: {model.n_layers})")
    parser.add_argument("--d_ff", type=int, default=model.d_ff,
                        help=f"FFN hidden dim (default: {model.d_ff})")
    parser.add_argument("--dropout", type=float, default=model.dropout,
                        help=f"Dropout rate (default: {model.dropout})")
    # Data
    parser.add_argument("--vocab_size", type=int, default=model.vocab_size,
                        help=f"Vocabulary size (default: {model.vocab_size})")
    parser.add_argument("--max_seq_len", type=int, default=model.max_seq_len,
                        help=f"Max sequence length (default: {model.max_seq_len})")
    # Training
    parser.add_argument("--epochs", type=int, default=train.epochs,
                        help=f"Training epochs (default: {train.epochs})")
    parser.add_argument("--batch_size", type=int, default=train.batch_size,
                        help=f"Batch size (default: {train.batch_size})")
    parser.add_argument("--lr", type=float, default=train.lr,
                        help=f"Peak learning rate (default: {train.lr})")
    parser.add_argument("--warmup_steps", type=int, default=train.warmup_steps,
                        help=f"LR warmup steps (default: {train.warmup_steps})")
    # System
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training even if GPU is available")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--seed", type=int, default=system.seed,
                        help=f"Random seed (default: {system.seed})")
    parser.add_argument("--data_dir", type=str, default=system.data_dir,
                        help=f"Dataset cache directory (default: {system.data_dir})")
    parser.add_argument("--checkpoint_dir", type=str, default=system.checkpoint_dir,
                        help=f"Model save directory (default: {system.checkpoint_dir})")
    parser.add_argument("--plot_dir", type=str, default=system.plot_dir,
                        help=f"Plot output directory (default: {system.plot_dir})")

    return parser.parse_args()


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    args = parse_args()

    # --- Setup ---
    set_seed(args.seed)

    # Configure device
    if args.cpu:
        device = torch.device("cpu")
        args.no_amp = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[WARNING] No GPU detected — training on CPU will be slow (~30min/epoch)")
            args.no_amp = True

    # --- Print System Info ---
    print("\n" + "=" * 60)
    print("Transformer for IMDB Sentiment Classification")
    print("=" * 60)
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA:       {torch.version.cuda}")
        print(f"  cuDNN:      {torch.backends.cudnn.version()}")
    print("=" * 60)

    # --- Model Config ---
    model_cfg = TransformerConfig()
    model_cfg.vocab_size = args.vocab_size
    model_cfg.max_seq_len = args.max_seq_len
    model_cfg.d_model = args.d_model
    model_cfg.n_heads = args.n_heads
    model_cfg.n_layers = args.n_layers
    model_cfg.d_ff = args.d_ff
    model_cfg.dropout = args.dropout

    # --- Training Config ---
    train_cfg = TrainingConfig()
    train_cfg.epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.lr = args.lr
    train_cfg.warmup_steps = args.warmup_steps
    train_cfg.use_amp = not args.no_amp

    # --- System Config ---
    sys_cfg = SystemConfig()
    sys_cfg.device = device
    sys_cfg.seed = args.seed
    sys_cfg.data_dir = args.data_dir
    sys_cfg.checkpoint_dir = args.checkpoint_dir
    sys_cfg.plot_dir = args.plot_dir

    # --- Data ---
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        data_dir=sys_cfg.data_dir,
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
        val_split=train_cfg.val_split,
        num_workers=train_cfg.num_workers,
        seed=sys_cfg.seed,
    )

    # Update config based on actual vocabulary
    actual_vocab_size = len(vocab)
    if actual_vocab_size != model_cfg.vocab_size:
        print(f"  Note: vocab_size adjusted from {model_cfg.vocab_size} to {actual_vocab_size}")
        model_cfg.vocab_size = actual_vocab_size
        # Recreate model config with correct vocab size
        model_cfg = TransformerConfig()
        model_cfg.vocab_size = actual_vocab_size
        model_cfg.max_seq_len = args.max_seq_len
        model_cfg.d_model = args.d_model
        model_cfg.n_heads = args.n_heads
        model_cfg.n_layers = args.n_layers
        model_cfg.d_ff = args.d_ff
        model_cfg.dropout = args.dropout

    # --- Model ---
    print("\n[2/4] Building model...")
    model = TransformerClassifier(model_cfg)
    print_model_summary(model)

    # --- Train ---
    print("\n[3/4] Training...")
    trainer = Trainer(model, train_loader, val_loader, train_cfg, sys_cfg)
    history = trainer.train()

    # --- Evaluate ---
    print("\n[4/4] Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    print(f"\n{'='*60}")
    print(f"Test Set Results")
    print(f"{'='*60}")
    print(f"  Loss:        {test_metrics['loss']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score:    {test_metrics['f1']:.4f}")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={int(test_metrics['tn']):5d}  FP={int(test_metrics['fp']):5d}")
    print(f"  FN={int(test_metrics['fn']):5d}  TP={int(test_metrics['tp']):5d}")
    print(f"{'='*60}")

    # --- Visualize ---
    generate_all_plots(history, test_metrics, plot_dir=sys_cfg.plot_dir)

    print(f"\nDone! Plots: {sys_cfg.plot_dir}")
    print(f"Model: {Path(sys_cfg.checkpoint_dir) / 'best_model.pt'}\n")


if __name__ == "__main__":
    main()
