"""Data pipeline for loading and preprocessing text data.

Implements a streaming data pipeline using tf.data for:
- Efficient text loading and tokenization
- Async prefetching to device
- Batching and sequence padding
"""

from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


class CharacterTokenizer:
    """Simple character-level tokenizer.

    For initial experiments, character-level tokenization is simpler
    and requires no external dependencies. Can be replaced with BPE later.
    """

    def __init__(self, text: str | None = None, vocab_size: int = 256) -> None:
        """Initialize tokenizer.

        Args:
            text: Optional text to build vocabulary from.
            vocab_size: Maximum vocabulary size (default: 256 for ASCII).
        """
        self.vocab_size = vocab_size

        if text is not None:
            # Build vocabulary from text
            chars = sorted(set(text))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = dict(enumerate(chars))
            self.vocab_size = len(chars)
        else:
            # Default to ASCII
            self.char_to_idx = {chr(i): i for i in range(vocab_size)}
            self.idx_to_char = {i: chr(i) for i in range(vocab_size)}

        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, token_ids: list[int] | np.ndarray) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Sequence of token IDs.

        Returns:
            Decoded text string.
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return "".join(self.idx_to_char.get(idx, "?") for idx in token_ids)


class TextDataset:
    """Simple text dataset for character-level language modeling.

    Loads text from a file and creates training examples by
    sliding a window over the text.
    """

    def __init__(
        self,
        text: str,
        seq_len: int,
        tokenizer: CharacterTokenizer | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            text: Raw text data.
            seq_len: Sequence length for training examples.
            tokenizer: Optional tokenizer (creates one if not provided).
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer or CharacterTokenizer(text)
        self.tokens = np.array(self.tokenizer.encode(text), dtype=np.int32)
        self.num_examples = max(0, len(self.tokens) - seq_len)

    def __len__(self) -> int:
        """Return number of training examples."""
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a training example.

        Args:
            idx: Example index.

        Returns:
            Dictionary with 'input_ids' and 'targets'.
        """
        if idx < 0 or idx >= self.num_examples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_examples})")

        # Extract sequence: inputs are [idx:idx+seq_len], targets are [idx+1:idx+seq_len+1]
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "targets": chunk[1:],
        }

    def batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Iterate over batches.

        Args:
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle indices.
            rng: Random number generator for shuffling.

        Yields:
            Batches as dictionaries.
        """
        indices = np.arange(self.num_examples)

        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)

        for start in range(0, self.num_examples, batch_size):
            batch_indices = indices[start : start + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete batches

            input_ids = np.stack([self.tokens[i : i + self.seq_len] for i in batch_indices])
            targets = np.stack([self.tokens[i + 1 : i + self.seq_len + 1] for i in batch_indices])

            yield {
                "input_ids": input_ids,
                "targets": targets,
            }


def load_text_file(path: str | Path) -> str:
    """Load text from a file.

    Args:
        path: Path to text file.

    Returns:
        File contents as string.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def download_tiny_shakespeare(cache_dir: str | Path = ".data") -> str:
    """Download the Tiny Shakespeare dataset.

    Args:
        cache_dir: Directory to cache the downloaded file.

    Returns:
        Path to the downloaded file.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = cache_dir / "tiny_shakespeare.txt"

    if not path.exists():
        print(f"Downloading Tiny Shakespeare to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

    return str(path)
