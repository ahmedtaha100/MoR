from typing import Optional, Dict, List, Iterator, Iterable

import torch

from torch.utils.data import Dataset, IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, seq_len: int = 2048):
        self.input_ids = input_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[idx]
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
        labels = input_ids.clone()
        attention_mask = torch.ones(len(input_ids))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class StreamingTextDataset(IterableDataset):
    def __init__(self, data_iterator: Iterator[torch.Tensor], seq_len: int = 2048, buffer_size: int = 10000):
        self.data_iterator = data_iterator
        self.seq_len = seq_len
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        for sequence in self.data_iterator:
            if len(sequence) >= self.seq_len:
                for i in range(0, len(sequence) - self.seq_len + 1, self.seq_len):
                    chunk = sequence[i:i + self.seq_len]
                    buffer.append(chunk)
                    if len(buffer) >= self.buffer_size:
                        indices = torch.randperm(len(buffer))
                        for idx in indices:
                            input_ids = buffer[idx]
                            yield {
                                "input_ids": input_ids,
                                "labels": input_ids.clone(),
                                "attention_mask": torch.ones(len(input_ids)),
                            }
                        buffer = []
        if buffer:
            indices = torch.randperm(len(buffer))
            for idx in indices:
                input_ids = buffer[idx]
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones(len(input_ids)),
                }


class TokenStreamDataset(IterableDataset):
    def __init__(
        self,
        dataset: Iterable,
        tokenizer,
        seq_len: int,
        text_field: str,
        add_bos: bool,
        add_eos: bool,
        max_samples: Optional[int],
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        produced = 0
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        for ex in self.dataset:
            text = ex[self.text_field] if self.text_field in ex else ex.get("text", "")
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if self.add_bos and bos_id is not None:
                ids = [bos_id] + ids
            if self.add_eos and eos_id is not None:
                ids = ids + [eos_id]
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones(len(input_ids)),
                }
                produced += 1
                if self.max_samples is not None and produced >= self.max_samples:
                    return


def create_random_dataset(num_samples: int, seq_len: int, vocab_size: int) -> TextDataset:
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TextDataset(input_ids, seq_len)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention = [item.get("attention_mask", torch.ones(len(item["input_ids"]))) for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    for ids, labs, attn in zip(input_ids, labels, attention):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
            padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)]))
            attention_masks.append(torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)]))
        else:
            padded_input_ids.append(ids)
            padded_labels.append(labs)
            attention_masks.append(attn)
    return {
        "input_ids": torch.stack(padded_input_ids).long(),
        "labels": torch.stack(padded_labels).long(),
        "attention_mask": torch.stack(attention_masks).float(),
    }


def load_tokenizer(name: str, use_fast: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def create_hf_dataloader(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    text_field: str = "text",
    streaming: bool = True,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    add_bos: bool = False,
    add_eos: bool = False,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
):
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    if streaming and shuffle_buffer and shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    tokenizer = load_tokenizer(tokenizer_name)
    stream = TokenStreamDataset(ds, tokenizer, seq_len, text_field, add_bos, add_eos, max_samples)
    return DataLoader(stream, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn), tokenizer
