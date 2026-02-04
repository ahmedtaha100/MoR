import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 100))

    return perplexity


def compute_routing_statistics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_depths = []
    all_weights = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if hasattr(model, "get_recursion_depths"):
                depths = model.get_recursion_depths(input_ids, attention_mask)
                if depths is not None:
                    all_depths.append(depths)

            outputs = model(input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "routing_info") and outputs.routing_info is not None:
                routing_info = outputs.routing_info
                if "router_weights" in routing_info:
                    for weights in routing_info["router_weights"]:
                        if weights is not None:
                            all_weights.append(weights)

    stats = {}

    if all_depths:
        depths_cat = torch.cat([d.flatten() for d in all_depths])
        stats["avg_depth"] = depths_cat.float().mean().item()
        stats["min_depth"] = depths_cat.min().item()
        stats["max_depth"] = depths_cat.max().item()
        stats["std_depth"] = depths_cat.float().std().item()

        max_depth = int(depths_cat.max().item())
        for d in range(1, max_depth + 1):
            stats[f"depth_{d}_fraction"] = (depths_cat == d).float().mean().item()

    if all_weights:
        weights_cat = torch.cat([w.flatten() for w in all_weights])
        stats["avg_router_weight"] = weights_cat.mean().item()
        stats["std_router_weight"] = weights_cat.std().item()

    return stats


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")

        ppl = compute_perplexity(model, dataloader, device)

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        results[name] = {
            "perplexity": ppl,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
        }

        if hasattr(model, "get_recursion_depths"):
            routing_stats = compute_routing_statistics(model, dataloader, device)
            results[name].update(routing_stats)

    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())

    print("\n" + "=" * 80)
    print(f"{'Model':<20}", end="")
    for metric in sorted(all_metrics):
        print(f"{metric:<15}", end="")
    print("\n" + "=" * 80)

    for name, metrics in results.items():
        print(f"{name:<20}", end="")
        for metric in sorted(all_metrics):
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                if value > 1000:
                    print(f"{value:>14,.0f}", end=" ")
                else:
                    print(f"{value:>14.4f}", end=" ")
            else:
                print(f"{value:>14}", end=" ")
        print()

    print("=" * 80)
