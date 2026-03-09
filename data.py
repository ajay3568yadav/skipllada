"""
Data loading and preprocessing for LayerSkip-LLaDA training.
Supports both pre-training (raw text → packed sequences) and SFT
(instruction → response with prompt masking).
"""

import logging

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """
    Pre-training dataset: tokenize raw text, pack to max_seq_length,
    return token ids. Masking is applied dynamically in the trainer.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = self._pack_texts(texts)

    def _pack_texts(self, texts: List[str]) -> List[torch.Tensor]:
        all_ids = []
        buffer = []

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_length:
                all_ids.append(torch.tensor(buffer[: self.max_seq_length], dtype=torch.long))
                buffer = buffer[self.max_seq_length :]

        if len(buffer) > 0:
            padded = buffer + [self.tokenizer.pad_token_id or 0] * (self.max_seq_length - len(buffer))
            all_ids.append(torch.tensor(padded[: self.max_seq_length], dtype=torch.long))

        return all_ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.examples[idx]}


class SFTDataset(Dataset):
    """
    SFT dataset: each example has a prompt and response.
    Returns input_ids and prompt_length so the trainer knows
    which tokens to mask (response only).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        examples: List[Dict[str, str]],
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.processed = self._process(examples)

    def _process(self, examples: List[Dict[str, str]]) -> List[Dict[str, torch.Tensor]]:
        processed = []
        for ex in examples:
            prompt = ex.get("prompt", ex.get("instruction", ""))
            response = ex.get("response", ex.get("output", ""))

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )
            except Exception:
                full_text = f"{prompt}\n{response}"

            prompt_messages = [{"role": "user", "content": prompt}]
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                prompt_text = f"{prompt}\n"

            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            prompt_length = len(prompt_ids)

            if len(full_ids) > self.max_seq_length:
                full_ids = full_ids[: self.max_seq_length]
                prompt_length = min(prompt_length, self.max_seq_length)

            pad_length = self.max_seq_length - len(full_ids)
            if pad_length > 0:
                eos_id = self.tokenizer.eos_token_id or 0
                full_ids = full_ids + [eos_id] * pad_length

            processed.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
            })
        return processed

    def __len__(self) -> int:
        return len(self.processed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed[idx]


class StreamingPretrainingDataset(IterableDataset):
    """
    Streaming pre-training dataset that pulls from a HuggingFace dataset
    (e.g. monology/pile-uncopyrighted), tokenizes on the fly, and packs
    tokens into fixed-length sequences with zero padding waste.

    Uses an internal token buffer that accumulates across streamed documents,
    emitting packed sequences as the buffer fills.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "monology/pile-uncopyrighted",
        dataset_split: str = "train",
        text_column: str = "text",
        max_seq_length: int = 4096,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.text_column = text_column
        self.max_seq_length = max_seq_length
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        ds = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed + worker_id, buffer_size=10000)

        buffer: List[int] = []
        docs_seen = 0

        for example in ds:
            text = example.get(self.text_column, "")
            if not text:
                continue

            docs_seen += 1
            if num_workers > 1 and (docs_seen % num_workers) != worker_id:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_length:
                ids = torch.tensor(buffer[: self.max_seq_length], dtype=torch.long)
                buffer = buffer[self.max_seq_length :]
                yield {"input_ids": ids}


def create_dummy_dataset(tokenizer: AutoTokenizer, num_samples: int = 16, max_seq_length: int = 512) -> PretrainingDataset:
    """Create a small dummy dataset for smoke testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 50
    ] * num_samples
    return PretrainingDataset(tokenizer, texts, max_seq_length)


def create_dummy_sft_dataset(tokenizer: AutoTokenizer, num_samples: int = 16, max_seq_length: int = 512) -> SFTDataset:
    """Create a small dummy SFT dataset for smoke testing."""
    examples = [
        {"prompt": "What is 2+2?", "response": "The answer is 4."},
        {"prompt": "What is the capital of France?", "response": "Paris is the capital of France."},
    ] * (num_samples // 2)
    return SFTDataset(tokenizer, examples, max_seq_length)
