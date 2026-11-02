"""
Training Pipeline for Transformer Classification
================================================
GPU-accelerated training loop with Automatic Mixed Precision (AMP),
warmup + cosine LR scheduling, gradient clipping, early stopping,
and best-model checkpointing.

Key Features:
- AMP (FP16): 1.5-2× faster training on modern GPUs with no accuracy loss
- Warmup: Gradually increase LR to prevent early instability
- Cosine Decay: Smoothly reduce LR for better convergence
- Early Stopping: Stop training when validation loss plateaus
- Checkpointing: Save best model (including optimizer & scaler state)

Usage:
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()
    test_metrics = trainer.evaluate(test_loader)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Learning Rate Scheduler (Warmup + Cosine Decay)
# =============================================================================

class WarmupCosineScheduler:
    """Learning rate schedule: linear warmup → cosine decay.

    LR
    ^
    |     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    |    /                              \
    |   /                                \
    |  /                                  \\___
    | /                                        ‾‾‾
    +----------------------------------------------> steps
         <-warmup->  <------ cosine decay ------->

    WHY WARMUP?
    At initialization, the model's attention weights are random, so gradients
    can be very large. Using the full learning rate immediately can cause the
    model to diverge. Warmup gradually increases LR, giving the model time to
    settle into a reasonable state before full training begins.

    WHY COSINE DECAY?
    Cosine decay smoothly reduces LR to near zero by the end of training.
    This is more effective than step decay because:
    1. No sudden LR drops that can destabilize training
    2. The model can "fine-tune" at low LR for a long period
    3. Empirically better final performance than linear or step schedules
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        total_steps: int,
    ):
        """
        Args:
            optimizer: The optimizer whose LR to schedule
            d_model: Model dimension (used in the warmup formula)
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

        # Store initial LR for reference
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        lr = self._compute_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        """Compute LR for the given step."""
        if step <= self.warmup_steps:
            # Linear warmup: LR goes from 0 → base_lr
            # Also includes the d_model scaling from the original paper:
            # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
            # Simplified version: just linear warmup
            return self.base_lr * step / max(1, self.warmup_steps)
        else:
            # Cosine decay: smoothly from base_lr → 0
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Metrics Computer
# =============================================================================

def compute_metrics(
    labels: torch.Tensor,
    preds: torch.Tensor,
) -> Dict[str, float]:
    """Compute classification metrics from predictions and labels.

    Args:
        labels: Ground truth, shape (N,)
        preds: Binary predictions (0 or 1), shape (N,)
    Returns:
        Dictionary of metric_name → value
    """
    labels = labels.long()
    preds = preds.long()

    # True/False Positives/Negatives
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = len(labels)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """GPU-accelerated training loop for Transformer classification.

    Handles:
    - Training loop with AMP (FP16 mixed precision)
    - Validation after each epoch
    - LR scheduling (warmup + cosine decay)
    - Gradient clipping
    - Best model checkpointing
    - Early stopping

    Example:
        trainer = Trainer(model, train_loader, val_loader, config, device)
        history = trainer.train()  # Returns loss/accuracy history
        test_metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: "TrainingConfig",  # noqa: F821
        system_config: "SystemConfig",      # noqa: F821
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = training_config
        self.sys_cfg = system_config
        self.device = system_config.device

        # --- Loss Function ---
        # BCEWithLogitsLoss = Sigmoid + BCE in one numerically stable operation
        # More stable than applying sigmoid first then computing BCE
        self.criterion = nn.BCEWithLogitsLoss()

        # --- Optimizer ---
        # AdamW: Adam with decoupled weight decay
        # Decoupled weight decay means L2 regularization doesn't interact with
        # the adaptive learning rates — better generalization than standard Adam
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.lr,
            betas=(training_config.beta1, training_config.beta2),
            eps=training_config.eps,
            weight_decay=training_config.weight_decay,
        )

        # --- LR Scheduler ---
        total_steps = len(train_loader) * training_config.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            d_model=256,  # d_model
            warmup_steps=training_config.warmup_steps,
            total_steps=total_steps,
        )

        # --- AMP Scaler (GPU only) ---
        # GradScaler prevents underflow in FP16 gradients by scaling the loss
        # before backprop and unscaling gradients before the optimizer step.
        # If gradients contain inf/NaN, the scaler skips the update and reduces
        # the scale factor (dynamic loss scaling).
        self.scaler = torch.amp.GradScaler("cuda", enabled=training_config.use_amp)
        self.use_amp = training_config.use_amp and self.device.type == "cuda"

        # --- Training State ---
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Move model to device
        self.model.to(self.device)

    # -------------------------------------------------------------------------
    # Single Training Epoch
    # -------------------------------------------------------------------------

    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)  # (B,) → (B,1)

            # --- Forward pass with AMP ---
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            # --- Backward pass with gradient scaling ---
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            self.scaler.scale(loss).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_cfg.grad_clip_norm
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update LR
            self.scheduler.step()

            # --- Track metrics ---
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

            # Update progress bar
            if batch_idx % self.train_cfg.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / labels.size(0):.3f}",
                    "lr": f"{self.scheduler.get_lr():.2e}",
                })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            data_loader: DataLoader for evaluation data
        Returns:
            Dictionary with loss, accuracy, precision, recall, f1, specificity,
            and raw confusion matrix counts (tp, tn, fp, fn)
        """
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        pbar = tqdm(data_loader, desc="[eval]", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_labels.append(batch["label"])
            all_probs.append(torch.sigmoid(logits).cpu().squeeze(-1))

        # Concatenate all batches
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        all_preds = (all_probs > 0.5).float()

        avg_loss = total_loss / len(data_loader)
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = avg_loss
        metrics["probs"] = all_probs  # Store for ROC curve
        metrics["labels"] = all_labels  # Store for confusion matrix

        return metrics

    # -------------------------------------------------------------------------
    # Full Training Loop
    # -------------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            History dictionary with per-epoch metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"  Device:        {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU:           {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"  CUDA:          {torch.version.cuda}")
            print(f"  AMP (FP16):    {'enabled' if self.use_amp else 'disabled'}")
        else:
            print(f"  [WARNING] Running on CPU — training will be slow (~30min/epoch)")
        print(f"  Epochs:        {self.train_cfg.epochs}")
        print(f"  Batch size:    {self.train_cfg.batch_size}")
        print(f"  Peak LR:       {self.train_cfg.lr:.2e}")
        print(f"  Warmup steps:  {self.train_cfg.warmup_steps}")
        print(f"  Early stop:    patience={self.train_cfg.early_stop_patience}")
        print(f"  Total steps:   {len(self.train_loader) * self.train_cfg.epochs:,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.train_cfg.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # --- Train ---
            train_loss, train_acc = self._train_epoch()

            # --- Validate ---
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            epoch_time = time.time() - epoch_start
            current_lr = self.scheduler.get_lr()

            # --- Record history ---
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # --- Print epoch summary ---
            print(
                f"Epoch {epoch:2d}/{self.train_cfg.epochs} | "
                f"Time {epoch_time:.0f}s | "
                f"LR {current_lr:.2e} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} | "
                f"F1 {val_metrics['f1']:.3f}"
            )

            # --- Checkpoint (best model) ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
                print(f"  [best] Model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.train_cfg.early_stop_patience:
                    print(f"\n  [STOP] Early stopping after {epoch} epochs "
                          f"(no improvement for {self.train_cfg.early_stop_patience} epochs)")
                    break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        # Load best model for final evaluation
        self._load_best_model()

        return self.history

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint to disk.

        Includes everything needed to resume training: model weights,
        optimizer state, AMP scaler state, epoch, and training history.
        """
        checkpoint_dir = Path(self.sys_cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Always save the latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pt")

        # Save best model separately
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")

    def _load_best_model(self):
        """Load the best model checkpoint."""
        best_path = Path(self.sys_cfg.checkpoint_dir) / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best model from epoch {checkpoint['epoch']}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        print(f"  Resumed from epoch {self.current_epoch}")


# =============================================================================
# Quick Test (mock data)
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from examples.transformer_imdb.config import TransformerConfig, TrainingConfig, SystemConfig
    from examples.transformer_imdb.model import TransformerClassifier

    print("Testing trainer with mock data...\n")

    # Create a tiny model for testing
    model_cfg = TransformerConfig()
    model_cfg.vocab_size = 100
    model_cfg.max_seq_len = 64
    model_cfg.d_model = 64
    model_cfg.n_heads = 4
    model_cfg.n_layers = 2
    model_cfg.d_ff = 256

    model = TransformerClassifier(model_cfg)

    # Create mock data
    from torch.utils.data import DataLoader, TensorDataset

    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 256

        def __getitem__(self, idx):
            seq_len = torch.randint(10, 64, (1,)).item()
            # Random token IDs (mostly non-pad)
            input_ids = torch.randint(3, 100, (64,))
            input_ids[seq_len:] = 0  # Pad rest
            input_ids[0] = 2  # CLS token
            attention_mask = torch.zeros(64, dtype=torch.long)
            attention_mask[:seq_len] = 1
            label = torch.tensor(float(torch.randint(0, 2, (1,)).item()))
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,
            }

    train_loader = DataLoader(MockDataset(), batch_size=16, shuffle=True)
    val_loader = DataLoader(MockDataset(), batch_size=32, shuffle=False)

    train_cfg = TrainingConfig()
    train_cfg.batch_size = 16
    train_cfg.epochs = 2
    train_cfg.lr = 1e-3
    train_cfg.warmup_steps = 10
    train_cfg.use_amp = False  # Disable for mock test

    sys_cfg = SystemConfig()
    if sys_cfg.device.type == "cuda":
        sys_cfg.device = torch.device("cpu")  # Force CPU for quick test

    trainer = Trainer(model, train_loader, val_loader, train_cfg, sys_cfg)
    history = trainer.train()
    print("Trainer test passed")
