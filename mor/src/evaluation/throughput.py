import time
import random
from collections import deque
import torch
from data.dataset import load_tokenizer


def sample_lengths(num: int, mean: float, std: float, min_len: int = 1):
    out = []
    for _ in range(num):
        val = int(random.gauss(mean, std))
        if val < min_len:
            val = min_len
        out.append(val)
    return out


def run_throughput(
    model,
    tokenizer_name: str,
    num_queries: int = 1000,
    batch_size: int = 32,
    mean_len: float = 256.0,
    std_len: float = 32.0,
    device: str = None,
    prompts=None,
    track_depth: bool = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_name)
    start_token = tokenizer.bos_token_id
    if start_token is None:
        start_token = tokenizer.eos_token_id
    if start_token is None:
        start_token = 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lengths = sample_lengths(num_queries, mean_len, std_len, 1)
    queue = deque(range(num_queries))
    active = []
    total_tokens = 0
    total_depth = 0
    max_len = getattr(model.config, "max_seq_len", 2048)
    start_time = time.time()
    while queue or active:
        while len(active) < batch_size and queue:
            idx = queue.popleft()
            target = lengths[idx]
            if prompts is None:
                prefix = [start_token]
            else:
                p = prompts[idx] if idx < len(prompts) else []
                prefix = p if len(p) > 0 else [start_token]
            active.append({"tokens": prefix, "target": target, "generated": 0})
        if not active:
            break
        seqs = []
        for q in active:
            if len(q["tokens"]) > max_len:
                q["tokens"] = q["tokens"][-max_len:]
            seqs.append(q["tokens"])
        max_seq = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_seq), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(seqs), max_seq), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attention_mask[i, :len(s)] = 1
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = outputs.logits
        if track_depth and hasattr(outputs, "routing_info") and outputs.routing_info is not None:
            if hasattr(model, "model") and hasattr(model.model, "recursive_block"):
                depths = model.model.recursive_block.get_recursion_depth_per_token(outputs.routing_info)
                if depths is not None:
                    depth_last = depths[:, -1]
                    total_depth += depth_last.sum().item()
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for q, tok in zip(active, next_tokens):
            q["tokens"].append(tok)
            q["generated"] += 1
        total_tokens += len(active)
        active = [q for q in active if q["generated"] < q["target"]]
    elapsed = time.time() - start_time
    tps = total_tokens / max(elapsed, 1e-8)
    result = {"tokens_per_sec": tps, "total_tokens": total_tokens, "elapsed_sec": elapsed}
    if track_depth and total_tokens > 0:
        avg_depth = total_depth / total_tokens
        depth_units = total_depth / max(batch_size, 1)
        depthwise_tokens_per_unit = total_tokens / max(depth_units, 1e-8)
        result["avg_recursion_depth"] = avg_depth
        result["depthwise_tokens_per_unit"] = depthwise_tokens_per_unit
    return result
