"""
Transformer Architecture — From Scratch in PyTorch
==================================================
Based on "Attention Is All You Need" (Vaswani et al., 2017).

This file builds the Transformer layer by layer, with extensive comments
explaining not just WHAT each line does, but WHY it's designed that way.

Architecture Overview (Encoder-Only for Classification):
--------------------------------------------------------
Input Tokens:  ["<cls>", "this", "movie", "was", "great", "<pad>", ...]
       │
       ▼
  ┌─────────────────┐
  │ Token Embedding  │  ← nn.Embedding(vocab_size, d_model)
  └────────┬────────┘
       │   +
  ┌─────────────────┐
  │Positional Encoding│  ← Sinusoidal: sin/cos at different frequencies
  └────────┬────────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │         Encoder Layer × N               │
  │  ┌──────────────────────────────────┐   │
  │  │  LayerNorm → MultiHeadAttention  │   │  ← Pre-LN architecture
  │  │       │  +  residual ──────────  │   │
  │  │       ▼                          │   │
  │  │  LayerNorm → FeedForward         │   │
  │  │       │  +  residual ──────────  │   │
  │  └──────────────────────────────────┘   │
  └────────────────────┬────────────────────┘
       │
       ▼ (take [CLS] token)
  ┌─────────────────┐
  │ Classification   │  ← Linear → GELU → Dropout → Linear → Sigmoid
  │     Head         │
  └────────┬────────┘
       │
       ▼
  P(positive) ∈ [0, 1]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# 1. Positional Encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    WHY WE NEED THIS:
    Self-attention is permutation-invariant — shuffling the input tokens
    would produce the same output (just shuffled). The model has no inherent
    notion of word order. Positional encoding injects position information
    so the model can distinguish "dog bites man" from "man bites dog".

    THE FORMULA (for position pos and dimension i):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    WHY SINUSOIDAL?
    1. Values are bounded in [-1, 1] — numerically stable
    2. Different dimensions have different wavelengths (2π to 10000·2π)
       — the model can learn to attend to relative positions
    3. For any fixed offset k, PE(pos+k) can be expressed as a linear
       function of PE(pos) — this helps the model learn relative position patterns
    4. Can handle sequences longer than any seen during training (extrapolation)
    5. No learnable parameters — the pattern is fixed by the formula
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length to pre-compute encodings for
            dropout: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the entire positional encoding matrix: shape (1, max_len, d_model)
        # We use shape (1, max_len, d_model) so it broadcasts over batch dimension
        pe = torch.zeros(max_len, d_model)

        # Create position indices: 0, 1, 2, ..., max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # div_term = 1 / (10000^(2i/d_model)) for each dimension i
        # This gives us the wavelength for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        # Register as buffer (saved with model but not a learnable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings, shape (batch_size, seq_len, d_model)
        Returns:
            Embeddings with position information, same shape as input
        """
        # Add positional encoding to the input embeddings
        # pe[:, :seq_len, :] automatically broadcasts over batch dimension
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 2. Multi-Head Self-Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Attention.

    THE CORE IDEA:
    Attention allows each token to "look at" all other tokens in the sequence
    and gather relevant information from them. Instead of processing tokens
    one-by-one (like RNNs), attention processes the entire sequence in parallel.

    SCALED DOT-PRODUCT ATTENTION:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    Step by step for a single head:
    1. Q·K^T:     Compute "compatibility" scores between every pair of tokens
                  → (seq_len, seq_len) matrix
    2. / √d_k:    Scale to prevent softmax from saturating (gradient vanishing)
                  — without scaling, large d_k leads to extreme softmax outputs
    3. softmax:   Convert to attention weights (sum to 1 across each row)
    4. · V:       Weighted sum of value vectors — tokens attend to relevant context

    WHY MULTIPLE HEADS?
    A single attention head can only focus on one pattern at a time. Multiple
    heads allow the model to jointly attend to information from different
    representation subspaces. For example:
    - Head 1 might track syntactic dependencies (subject-verb agreement)
    - Head 2 might track semantic similarity (synonyms, topics)
    - Head 3 might track positional relationships (nearby adjectives)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Total model dimension (must be divisible by n_heads)
            n_heads: Number of parallel attention heads
            dropout: Dropout applied to attention weights
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Combined Q, K, V projection (more efficient than 3 separate layers)
        # Instead of 3 separate (d_model × d_model) matrices, we use one
        # (d_model × 3*d_model) matrix and split the result.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection: combine all heads back into d_model space
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization (optional)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from (B, S, d_model) to (B, n_heads, S, d_k).

        This splits the d_model dimension into n_heads separate "views",
        each of dimension d_k. The batch and sequence dimensions are preserved.
        """
        B, S, _ = x.shape
        # (B, S, d_model) → (B, S, n_heads, d_k) → (B, n_heads, S, d_k)
        return x.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse of _split_heads: (B, n_heads, S, d_k) → (B, S, d_model)."""
        B, _, S, _ = x.shape
        # (B, n_heads, S, d_k) → (B, S, n_heads, d_k) → (B, S, d_model)
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            mask: Optional attention mask, shape (batch_size, seq_len)
                  True/1 = attend, False/0 = ignore (will be set to -inf before softmax)
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        B, S, _ = x.shape

        # --- Step 1: Linear projection ---
        qkv = self.qkv_proj(x)  # (B, S, 3*d_model)
        Q, K, V = qkv.chunk(3, dim=-1)  # each: (B, S, d_model)

        # --- Step 2: Split into multiple heads ---
        Q = self._split_heads(Q)  # (B, n_heads, S, d_k)
        K = self._split_heads(K)  # (B, n_heads, S, d_k)
        V = self._split_heads(V)  # (B, n_heads, S, d_k)

        # --- Step 3: Flash Attention (PyTorch 2.0+) ---
        # F.scaled_dot_product_attention uses Flash Attention on supported GPUs,
        # which is 10-100x faster than manual attention and uses less memory.
        # It automatically selects the best backend (FlashAttention2, Memory-Efficient,
        # or the standard math implementation).
        if mask is not None:
            # Convert mask from (B, S) → (B, 1, 1, S) for broadcasting
            # F.scaled_dot_product_attention expects bool mask: True=keep, False=mask
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()  # (B, 1, 1, S)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, n_heads, S, d_k)

        # --- Step 4: Merge heads and project back ---
        output = self._merge_heads(attn_output)  # (B, S, d_model)
        output = self.out_proj(output)

        return output


# =============================================================================
# 3. Position-wise Feed-Forward Network
# =============================================================================

class PositionWiseFeedForward(nn.Module):
    """Two-layer feed-forward network applied independently to each position.

    FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂

    WHERE: W₁: (d_model → d_ff), W₂: (d_ff → d_model)

    WHY?
    Self-attention is a linear operation (weighted sum). The FFN adds
    non-linear processing capacity at each position. Without it, stacking
    multiple attention layers would still be essentially linear.

    WHY TWO LAYERS WITH A BOTTLENECK?
    The expansion (d_model → d_ff, typically 4×) followed by contraction
    (d_ff → d_model) is a common pattern. The wide hidden layer gives the
    model capacity to learn complex features, while the narrow output keeps
    the residual connection dimension consistent.

    WHY GELU INSTEAD OF RELU?
    GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU
    that works slightly better for NLP. Unlike ReLU which has a hard zero
    cutoff, GELU allows small negative values to pass through, providing
    useful gradient signal even for "inactive" neurons.
        GELU(x) = x · Φ(x) ≈ x · sigmoid(1.702x)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # Expand
        self.linear2 = nn.Linear(d_ff, d_model)    # Contract
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply first linear + GELU activation, then dropout, then second linear
        x = self.linear1(x)
        x = F.gelu(x)  # GELU is built into PyTorch
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# =============================================================================
# 4. Encoder Layer
# =============================================================================

class EncoderLayer(nn.Module):
    """A single Transformer encoder layer.

    Structure (Pre-LN variant):
        x ──→ LayerNorm → MultiHeadAttention ──→ [+] ──→ LayerNorm → FFN ──→ [+]
              ↑                                  ↑                          ↑
              │──────────────────────────────────┘                          │
              │─────────────────────────────────────────────────────────────┘

    WHY PRE-LN (LayerNorm BEFORE each sublayer)?
    The original paper used Post-LN (LayerNorm AFTER), but Pre-LN has been
    found to be more stable during training:
    - Gradients flow more smoothly through the residual path
    - No need for learning rate warmup in some cases
    - More robust to large learning rates
    Most modern Transformer implementations (GPT, BERT variants) use Pre-LN.

    WHY RESIDUAL CONNECTIONS (the [+] operations)?
    Residual connections (He et al., 2015) solve the degradation problem:
    as networks get deeper, training becomes harder. By providing a direct
    path x → output, gradients can flow unimpeded through the network.
    The sublayer only needs to learn the "residual" — what to add/change.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed-forward sublayer
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, S, d_model)
            mask: Attention mask, shape (B, S) — 1=attend, 0=ignore
        Returns:
            Output, shape (B, S, d_model)
        """
        # Sublayer 1: Self-Attention with residual connection
        residual = x
        x = self.norm1(x)                         # Pre-LN
        x = self.self_attn(x, mask)               # Multi-head attention
        x = self.dropout(x)
        x = residual + x                          # Residual connection

        # Sublayer 2: Feed-Forward with residual connection
        residual = x
        x = self.norm2(x)                         # Pre-LN
        x = self.ffn(x)                           # Position-wise FFN
        x = self.dropout(x)
        x = residual + x                          # Residual connection

        return x


# =============================================================================
# 5. Transformer Encoder (Stack of EncoderLayers)
# =============================================================================

class TransformerEncoder(nn.Module):
    """Full Transformer Encoder: Embedding + PositionalEncoding + N layers.

    This is the core of BERT-like models. It takes token indices as input
    and produces contextualized representations for each token. The final
    hidden state of each token contains information about that token in
    the context of the entire sequence.
    """

    def __init__(self, config: "TransformerConfig"):
        """
        Args:
            config: TransformerConfig with architecture hyperparameters
        """
        super().__init__()
        self.d_model = config.d_model
        self.pad_idx = config.pad_idx
        self.max_seq_len = config.max_seq_len

        # Token embedding: map token index → dense vector
        # nn.Embedding is a lookup table: each row is a learnable vector
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        # padding_idx ensures that <pad> tokens have embedding = 0
        # and don't receive gradients during training

        # Scale embedding by √d_model
        # WHY? In the original paper, the embedding weights are multiplied
        # by √d_model so that the embedding values are comparable in scale
        # to the positional encoding (which ranges [-1, 1]).
        self.embed_scale = math.sqrt(config.d_model)

        # Positional encoding: inject sequence order information
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        # Stack of identical encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer normalization (applied after all encoder layers)
        self.final_norm = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices, shape (B, S)
            attention_mask: 1=attend, 0=ignore, shape (B, S). If None, no masking.
        Returns:
            Contextualized token representations, shape (B, S, d_model)
        """
        # --- Token Embedding ---
        x = self.token_embedding(input_ids) * self.embed_scale  # (B, S, d_model)

        # --- Positional Encoding ---
        x = self.positional_encoding(x)  # (B, S, d_model)

        # Add dropout after embedding for regularization
        x = self.dropout(x)

        # --- Pass through each encoder layer ---
        for layer in self.layers:
            x = layer(x, attention_mask)

        # --- Final LayerNorm ---
        x = self.final_norm(x)

        return x


# =============================================================================
# 6. Transformer Classifier (Encoder + Classification Head)
# =============================================================================

class TransformerClassifier(nn.Module):
    """Complete Transformer model for text classification.

    Uses the [CLS] token strategy (same as BERT):
    - A special [CLS] token is prepended to every input sequence
    - After encoding, the hidden state of [CLS] is used for classification
    - [CLS] "aggregates" information from all tokens via self-attention

    WHY [CLS] INSTEAD OF POOLING?
    - Mean pooling treats all tokens equally (including padding)
    - Max pooling loses information
    - [CLS] attends to the most relevant tokens for the classification task
      because its attention weights are learned during training
    """

    def __init__(self, config: "TransformerConfig"):
        """
        Args:
            config: TransformerConfig with architecture hyperparameters
        """
        super().__init__()
        self.config = config
        self.cls_idx = config.cls_idx

        # The Transformer encoder (shared with the full seq2seq architecture)
        self.encoder = TransformerEncoder(config)

        # Classification head: [CLS] vector → sentiment probability
        # Two-layer MLP with residual-style design
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),  # Single logit output
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following the original Transformer paper.

        - Linear layers: Xavier uniform (good for tanh/sigmoid-like activations)
        - LayerNorm: weight=1, bias=0
        - Embedding: normal distribution N(0, d_model^-0.5)
        """
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "norm" not in name.lower():
                nn.init.xavier_uniform_(param)
            elif "norm" in name.lower() and "weight" in name:
                nn.init.ones_(param)
            elif "norm" in name.lower() and "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices INCLUDING [CLS] at position 0, shape (B, S)
            attention_mask: 1=attend, 0=ignore, shape (B, S)
        Returns:
            Raw logits (not probabilities), shape (B, 1)
            Use torch.sigmoid() to convert to probabilities.
        """
        # Encode the entire sequence
        hidden_states = self.encoder(input_ids, attention_mask)  # (B, S, d_model)

        # Extract [CLS] token (first position)
        cls_hidden = hidden_states[:, 0, :]  # (B, d_model)

        # Classify
        logits = self.classifier(cls_hidden)  # (B, 1)

        return logits

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method: return binary predictions (0 or 1)."""
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(-1)
        return (probs > 0.5).long()

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method: return probability of positive class."""
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits).squeeze(-1)


# =============================================================================
# 7. Utility: Model Summary
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: PyTorch model
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module):
    """Print a human-readable summary of the model's parameters."""
    total, trainable = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Model Parameter Summary")
    print(f"{'='*60}")
    print(f"Total parameters:      {total:>10,}")
    print(f"Trainable parameters:  {trainable:>10,}")
    print(f"{'='*60}")

    # Per-component breakdown
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<25s} {n:>10,}")
    print(f"{'='*60}\n")


# =============================================================================
# 8. Quick Test
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    # Add parent dir to path so we can import examples.transformer_imdb.config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from examples.transformer_imdb.config import TransformerConfig as TC

    config = TC()
    config.vocab_size = 10000
    config.max_seq_len = 512

    print("Building TransformerClassifier with config:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    model = TransformerClassifier(config)
    print_model_summary(model)

    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 128
    dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_mask = torch.ones(batch_size, seq_len)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_ids, dummy_mask)
        probs = torch.sigmoid(logits)
        print(f"Dummy input shape:  {dummy_ids.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Probabilities:       {probs.squeeze(-1).tolist()}")
        print(f"\nForward pass OK")
