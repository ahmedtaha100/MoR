from typing import Optional, Dict, List, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset


class TextDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        seq_len: int = 2048,
    ):
        self.input_ids = input_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[idx]

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        data_iterator: Iterator[torch.Tensor],
        seq_len: int = 2048,
        buffer_size: int = 10000,
    ):
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
                            }
                        buffer = []

        if buffer:
            indices = torch.randperm(len(buffer))
            for idx in indices:
                input_ids = buffer[idx]
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


def create_random_dataset(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> TextDataset:
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TextDataset(input_ids, seq_len)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_input_ids.append(
                torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
            )
            attention_masks.append(
                torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
            )
        else:
            padded_input_ids.append(ids)
            padded_labels.append(labs)
            attention_masks.append(torch.ones(len(ids)))

    return {
        "input_ids": torch.stack(padded_input_ids).long(),
        "labels": torch.stack(padded_labels).long(),
        "attention_mask": torch.stack(attention_masks).float(),
    }
