"""
KV Cache utilities for Mixture-of-Recursions.

Current implementation:
- attention.py KVCache: Standard full-sequence caching for first/last unique layers
- attention.py SelectiveKVCache: True selective KV compaction - only stores K/V for
  tokens selected by the router at each recursion level, reducing memory usage
- "selective" strategy: Uses SelectiveKVCache per-recursion with true compaction
- "shared" strategy: Reuses r=0 cache for all recursions (full caching)

NOTE: MoRKVCache and ContinuousDepthBatcher below are UNUSED legacy code.
They were originally designed for a different caching approach and are kept
for reference but not actively used.
"""
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn


class RecursionKVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.key_caches: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_caches: List[Optional[torch.Tensor]] = [None] * num_layers
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_caches[layer_idx] is None:
            self.key_caches[layer_idx] = key
            self.value_caches[layer_idx] = value
        else:
            self.key_caches[layer_idx] = torch.cat(
                [self.key_caches[layer_idx], key], dim=2
            )
            self.value_caches[layer_idx] = torch.cat(
                [self.value_caches[layer_idx], value], dim=2
            )

        self.seq_len = self.key_caches[layer_idx].shape[2]
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def get_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def get_seq_len(self) -> int:
        return self.seq_len

    def reset(self):
        self.key_caches = [None] * self.num_layers
        self.value_caches = [None] * self.num_layers
        self.seq_len = 0


class MoRKVCache:
    def __init__(
        self,
        num_layers: int,
        num_recursions: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        strategy: str = "selective",
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_recursions = num_recursions
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.strategy = strategy
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        if strategy == "selective":
            self.caches: Dict[int, RecursionKVCache] = {
                r: RecursionKVCache(num_layers)
                for r in range(num_recursions)
            }
        else:
            self.caches = {0: RecursionKVCache(num_layers)}

        self.exit_depths: Optional[torch.Tensor] = None

    def update(
        self,
        recursion_step: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.strategy == "shared":
            if recursion_step == 0:
                return self.caches[0].update(layer_idx, key, value, active_mask)
            else:
                cached_key, cached_value = self.caches[0].get_cache(layer_idx)
                if cached_key is not None:
                    return cached_key, cached_value
                return key, value
        else:
            return self.caches[recursion_step].update(
                layer_idx, key, value, active_mask
            )

    def get_cache(
        self,
        recursion_step: int,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.strategy == "shared":
            return self.caches[0].get_cache(layer_idx)
        else:
            return self.caches[recursion_step].get_cache(layer_idx)

    def get_seq_len(self, recursion_step: int = 0) -> int:
        if self.strategy == "shared":
            return self.caches[0].get_seq_len()
        else:
            return self.caches.get(recursion_step, RecursionKVCache(1)).get_seq_len()

    def set_exit_depths(self, exit_depths: torch.Tensor):
        self.exit_depths = exit_depths

    def get_attention_mask_for_recursion(
        self,
        recursion_step: int,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        if self.strategy == "shared" or self.exit_depths is None:
            return None

        active_mask = self.exit_depths >= recursion_step
        return active_mask

    def reset(self):
        for cache in self.caches.values():
            cache.reset()
        self.exit_depths = None


class ContinuousDepthBatcher:
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_recursions: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_recursions = num_recursions

        self.active_sequences: List[Dict] = []
        self.pending_sequences: List[Dict] = []

    def add_sequence(self, sequence_info: Dict):
        self.pending_sequences.append(sequence_info)

    def get_batch(self) -> Optional[Dict]:
        while len(self.active_sequences) < self.max_batch_size and self.pending_sequences:
            self.active_sequences.append(self.pending_sequences.pop(0))

        if not self.active_sequences:
            return None

        batch_by_depth = {r: [] for r in range(self.num_recursions)}

        for seq_info in self.active_sequences:
            depth = seq_info.get("current_depth", 0)
            batch_by_depth[depth].append(seq_info)

        return {
            "sequences": self.active_sequences,
            "by_depth": batch_by_depth,
        }

    def update_after_step(self, exit_masks: Dict[int, torch.Tensor]):
        still_active = []
        for i, seq_info in enumerate(self.active_sequences):
            if i in exit_masks and exit_masks[i]:
                pass
            else:
                seq_info["current_depth"] = seq_info.get("current_depth", 0) + 1
                if seq_info["current_depth"] < self.num_recursions:
                    still_active.append(seq_info)

        self.active_sequences = still_active
