"""
IMDB Sentiment Analysis Dataset Pipeline
========================================
Downloads the IMDB movie review dataset, builds a vocabulary, and creates
efficient PyTorch DataLoaders for training.

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative
(25,000 train, 25,000 test). It's the "hello world" of NLP sentiment analysis.

Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
Paper:   "Learning Word Vectors for Sentiment Analysis" (Maas et al., 2011)

Pipeline:
  1. Download & extract raw text files from Stanford mirror (~80 MB)
  2. Tokenize: lowercase → remove punctuation → split on whitespace
  3. Build vocabulary: top 10,000 words + special tokens (<pad>, <unk>, <cls>)
  4. Convert tokens → indices, pad/truncate to max_seq_len
  5. Return PyTorch DataLoader with (input_ids, attention_mask, label)
"""

from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

# Special tokens — these get special indices in the vocabulary
PAD_TOKEN = "<pad>"   # Index 0: padding token (fills short sequences)
UNK_TOKEN = "<unk>"   # Index 1: unknown token (replaces out-of-vocabulary words)
CLS_TOKEN = "<cls>"   # Index 2: classification token (prepended to every sequence)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]
PAD_IDX, UNK_IDX, CLS_IDX = 0, 1, 2

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


# =============================================================================
# Tokenizer
# =============================================================================

def simple_tokenizer(text: str) -> List[str]:
    """A minimal word-level tokenizer.

    Steps:
    1. Lowercase — "Great" and "great" should be the same word
    2. Remove HTML tags (<br /> is common in IMDB reviews)
    3. Replace punctuation with spaces (keep only letters and spaces)
    4. Split on whitespace
    5. Filter out empty strings

    WHY NOT USE A SUBWORD TOKENIZER (BPE, WordPiece)?
    For learning purposes, a word-level tokenizer is much simpler to understand.
    Subword tokenizers (used in BERT, GPT) handle unknown words by splitting
    into smaller pieces, but they add complexity. For a first Transformer
    implementation, a word tokenizer gets the point across.

    Args:
        text: Raw review text
    Returns:
        List of lowercase word tokens
    """
    text = text.lower()
    # Remove HTML break tags
    text = re.sub(r"<br\s*/?>", " ", text)
    # Replace non-alphabetic characters with spaces
    text = re.sub(r"[^a-z']", " ", text)
    # Split and filter
    tokens = [t.strip("'") for t in text.split() if t.strip("'")]
    return tokens


# =============================================================================
# Dataset Download & Loading
# =============================================================================

def download_imdb(data_dir: str = str(Path(__file__).resolve().parent / "data_cache")) -> str:
    """Download and extract the IMDB dataset.

    Only downloads if the data isn't already extracted (checks for aclImdb/ directory).

    Args:
        data_dir: Directory to store the dataset
    Returns:
        Path to the extracted dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_path / "aclImdb"
    if extracted_dir.exists():
        print(f"[data] IMDB dataset already exists at {extracted_dir}")
        return str(extracted_dir)

    # Download
    tar_path = data_path / "aclImdb_v1.tar.gz"
    if not tar_path.exists():
        print(f"[data] Downloading IMDB dataset...")
        print(f"      URL: {IMDB_URL}")
        try:
            urllib.request.urlretrieve(IMDB_URL, tar_path)
            print(f"      Saved to: {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.0f} MB)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download IMDB dataset: {e}\n"
                f"Please download manually from {IMDB_URL} and extract to {extracted_dir}"
            )

    # Extract
    print(f"[data] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_path, filter="data")
    print(f"[data] Extracted to {extracted_dir}")

    return str(extracted_dir)


def load_imdb_reviews(data_dir: str, split: str) -> List[Tuple[str, int]]:
    """Load IMDB reviews from disk.

    Directory structure:
        aclImdb/{split}/pos/  → label 1 (positive)
        aclImdb/{split}/neg/  → label 0 (negative)
        aclImdb/{split}/unsup/ → unlabeled (ignored)

    Args:
        data_dir: Path to aclImdb/ directory
        split: "train" or "test"
    Returns:
        List of (review_text, label) tuples
    """
    base = Path(data_dir) / split
    reviews = []

    for label, subdir in [(1, "pos"), (0, "neg")]:
        subdir_path = base / subdir
        if not subdir_path.exists():
            print(f"[data] Warning: {subdir_path} not found, skipping")
            continue

        for file_path in tqdm(
            sorted(subdir_path.glob("*.txt")),
            desc=f"  Loading {split}/{subdir}",
            leave=False,
        ):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            reviews.append((text, label))

    print(f"  Loaded {len(reviews):,} reviews from {split}")
    return reviews


# =============================================================================
# Vocabulary Builder
# =============================================================================

def build_vocabulary(
    texts: List[str],
    vocab_size: int = 10000,
    min_freq: int = 2,
) -> Dict[str, int]:
    """Build word-to-index vocabulary from a corpus of texts.

    Process:
    1. Count word frequencies across all texts
    2. Keep the top (vocab_size - len(SPECIAL_TOKENS)) most frequent words
    3. Assign indices: special tokens first (0,1,2), then by frequency

    WHY MIN_FREQ?
    Words that appear only once are often typos or rare names — they add
    vocabulary size without helping generalization. Requiring at least 2
    occurrences filters out most noise while keeping useful rare words.

    Args:
        texts: List of tokenized text strings (already joined)
        vocab_size: Maximum vocabulary size (including special tokens)
        min_freq: Minimum frequency for a word to be included
    Returns:
        {word: index} dictionary
    """
    # Count word frequencies
    counter: Counter = Counter()
    for text in tqdm(texts, desc="  Counting words", leave=False):
        tokens = simple_tokenizer(text)
        counter.update(tokens)

    print(f"  Unique words in corpus: {len(counter):,}")
    print(f"  Most common: {counter.most_common(20)}")

    # Filter by frequency and take top K
    most_common = [
        (word, count)
        for word, count in counter.most_common()
        if count >= min_freq
    ][:vocab_size - len(SPECIAL_TOKENS)]

    print(f"  Words with freq >= {min_freq}: {len(most_common):,}")

    # Build vocabulary: special tokens first
    vocab: Dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    for word, count in most_common:
        vocab[word] = len(vocab)

    print(f"  Final vocabulary size: {len(vocab):,}")
    return vocab


# =============================================================================
# PyTorch Dataset
# =============================================================================

class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB sentiment classification.

    Each sample is converted to:
    - input_ids: [CLS] + token_indices + [PAD]...   (padded to max_len)
    - attention_mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]  (1 for real tokens, 0 for padding)
    - label: 0 (negative) or 1 (positive)

    WHY [CLS] AT THE FRONT?
    [CLS] (classification token) is a special token whose final hidden state
    is used as the aggregate sequence representation for classification. This
    is the BERT strategy — the [CLS] token attends to all other tokens during
    self-attention and learns to aggregate information relevant to the task.
    """

    def __init__(
        self,
        reviews: List[Tuple[str, int]],
        vocab: Dict[str, int],
        max_seq_len: int = 512,
    ):
        """
        Args:
            reviews: List of (text, label) tuples
            vocab: Word-to-index mapping
            max_seq_len: Maximum sequence length (shorter → pad, longer → truncate)
        """
        self.reviews = reviews
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.reviews[idx]

        # Tokenize and convert to indices
        tokens = simple_tokenizer(text)
        token_ids = [self.vocab.get(token, UNK_IDX) for token in tokens]

        # Truncate to max_seq_len - 1 (leave room for [CLS])
        token_ids = token_ids[:self.max_seq_len - 1]

        # Prepend [CLS] token
        input_ids = [CLS_IDX] + token_ids

        # Build attention mask (1 for real tokens, 0 for padding)
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        input_ids.extend([PAD_IDX] * pad_len)
        attention_mask.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(
    data_dir: str = str(Path(__file__).resolve().parent / "data_cache"),
    vocab_size: int = 10000,
    max_seq_len: int = 512,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Download IMDB, build vocabulary, create train/val/test DataLoaders.

    This is the main entry point — call this once to get everything you need
    for training.

    Args:
        data_dir: Directory for caching downloaded data
        vocab_size: Maximum vocabulary size
        max_seq_len: Maximum sequence length for padding/truncation
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use as validation
        num_workers: DataLoader worker processes
        seed: Random seed for train/val split

    Returns:
        (train_loader, val_loader, test_loader, vocab_dict)
    """
    print("\n" + "=" * 60)
    print("IMDB Data Pipeline")
    print("=" * 60)

    # --- Step 1: Download & load ---
    extracted_dir = download_imdb(data_dir)
    train_reviews = load_imdb_reviews(extracted_dir, "train")
    test_reviews = load_imdb_reviews(extracted_dir, "test")

    # --- Step 2: Build vocabulary (from TRAINING data only) ---
    # WHY ONLY TRAINING DATA? In a real scenario, you don't have access to
    # test data at training time. Building vocab from test data would "cheat"
    # by leaking information about the test distribution.
    print(f"\n[data] Building vocabulary (size={vocab_size})...")
    train_texts = [text for text, _ in train_reviews]
    vocab = build_vocabulary(train_texts, vocab_size=vocab_size)

    # Report vocabulary coverage on test set
    test_texts = [text for text, _ in test_reviews]
    test_tokens = [t for text in test_texts for t in simple_tokenizer(text)]
    test_known = sum(1 for t in test_tokens if t in vocab)
    test_coverage = test_known / len(test_tokens) * 100 if test_tokens else 0
    print(f"  Test set vocab coverage: {test_coverage:.1f}% "
          f"({test_known:,} / {len(test_tokens):,} tokens known)")

    # --- Step 3: Train/validation split ---
    # Shuffle deterministically with seed
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_reviews), generator=generator).tolist()
    val_size = int(len(train_reviews) * val_split)
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    train_data = [train_reviews[i] for i in train_indices]
    val_data = [train_reviews[i] for i in val_indices]
    print(f"\n[data] Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_reviews):,}")

    # --- Step 4: Create PyTorch Datasets ---
    train_dataset = IMDBDataset(train_data, vocab, max_seq_len)
    val_dataset = IMDBDataset(val_data, vocab, max_seq_len)
    test_dataset = IMDBDataset(test_reviews, vocab, max_seq_len)

    # --- Step 5: Create DataLoaders ---
    # pin_memory=True: speeds up CPU→GPU transfer by using pinned (page-locked) memory
    # num_workers>0: loads data in background processes so GPU doesn't wait for I/O
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete last batch for cleaner batch norm stats
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batches for eval (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print("=" * 60 + "\n")

    return train_loader, val_loader, test_loader, vocab


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing data pipeline...\n")

    loaders = create_dataloaders(
        data_dir=str(Path(__file__).resolve().parent / "data_cache"),
        vocab_size=5000,   # Smaller for quick test
        max_seq_len=256,   # Shorter for quick test
        batch_size=8,
        num_workers=0,     # 0 for debugging (easier stack traces)
    )

    train_loader, val_loader, test_loader, vocab = loaders

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  label shape:          {batch['label'].shape}")
    print(f"  Positive ratio:       {batch['label'].mean().item():.2f}")
    print(f"  Avg sequence length:  {batch['attention_mask'].sum(dim=1).mean().item():.0f}")

    # Show a sample review (converted back to text)
    idx_to_word = {i: w for w, i in vocab.items()}
    sample_ids = batch["input_ids"][0]
    sample_mask = batch["attention_mask"][0]
    real_len = sample_mask.sum().item()
    words = [idx_to_word.get(i.item(), "<?>") for i in sample_ids[:real_len]]
    print(f"\n  Sample review (first 30 tokens):")
    print(f"  {' '.join(words[:30])}")
    print(f"  Label: {'positive' if batch['label'][0].item() == 1 else 'negative'}")
