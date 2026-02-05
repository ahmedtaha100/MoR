import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from lm_eval.api.model import LM
except Exception as e:
    LM = object


class MoRLMEval(LM):
    def __init__(self, model, tokenizer_name: str, batch_size: int = 1, device: str = None, max_length: int = None):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        if max_length is None:
            max_length = getattr(self.model.config, "max_seq_len", 2048)
        self.max_length = max_length
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    @property
    def eot_token_id(self):
        return self.eos_id

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens)

    def _get_args(self, req):
        return req.args if hasattr(req, "args") else req

    def _prepare_tokens(self, context: str, continuation: str):
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        if len(ctx_ids) == 0 and self.bos_id is not None:
            tokens = [self.bos_id] + cont_ids
            ctx_len = 1
        else:
            tokens = ctx_ids + cont_ids
            ctx_len = len(ctx_ids)
        if len(tokens) > self.max_length:
            overflow = len(tokens) - self.max_length
            tokens = tokens[overflow:]
            ctx_len = max(0, ctx_len - overflow)
        return tokens, ctx_len, cont_ids

    def _loglikelihood_tokens(self, tokens: List[int]) -> Tuple[float, bool]:
        if len(tokens) <= 1:
            return 0.0, True
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        target = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        total = token_logprobs.sum().item()
        greedy = torch.argmax(logits, dim=-1)
        is_greedy = bool(torch.all(greedy == target).item())
        return total, is_greedy

    def loglikelihood(self, requests):
        results = []
        batch = []
        for req in requests:
            batch.append(req)
            if len(batch) >= self.batch_size:
                results.extend(self._loglikelihood_batch(batch))
                batch = []
        if batch:
            results.extend(self._loglikelihood_batch(batch))
        return results

    def _loglikelihood_batch(self, batch):
        token_lists = []
        ctx_lens = []
        cont_ids_list = []
        for req in batch:
            context, continuation = self._get_args(req)[:2]
            tokens, ctx_len, cont_ids = self._prepare_tokens(context, continuation)
            token_lists.append(tokens)
            ctx_lens.append(ctx_len)
            cont_ids_list.append(cont_ids)
        max_len = max(len(t) for t in token_lists) if token_lists else 0
        input_ids = torch.full((len(token_lists), max_len), self.pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(token_lists), max_len), dtype=torch.long, device=self.device)
        for i, tokens in enumerate(token_lists):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
            attention_mask[i, :len(tokens)] = 1
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        out = []
        for i, tokens in enumerate(token_lists):
            ctx_len = ctx_lens[i]
            cont_ids = cont_ids_list[i]
            if len(cont_ids) == 0:
                out.append((0.0, True))
                continue
            start = max(ctx_len - 1, 0)
            end = start + len(cont_ids)
            lp = log_probs[i, start:end, :]
            target = torch.tensor(cont_ids, device=self.device).unsqueeze(-1)
            token_logprobs = lp.gather(1, target).squeeze(-1)
            total = token_logprobs.sum().item()
            greedy = torch.argmax(lp, dim=-1)
            is_greedy = bool(torch.all(greedy == target.squeeze(-1)).item())
            out.append((total, is_greedy))
        return out

    def loglikelihood_rolling(self, requests):
        results = []
        stride = max(self.max_length - 1, 1)
        for req in requests:
            text = self._get_args(req)[0]
            tokens = self.tok_encode(text)
            if len(tokens) <= 1:
                results.append((0.0,))
                continue
            total = 0.0
            start = 0
            while start < len(tokens):
                end = min(start + self.max_length, len(tokens))
                if start == 0:
                    window = tokens[start:end]
                else:
                    window = tokens[start - 1:end]
                lp, _ = self._loglikelihood_tokens(window)
                total += lp
                start += stride
            results.append((total,))
        return results

    def generate_until(self, requests):
        results = []
        for req in requests:
            args = self._get_args(req)
            context = args[0]
            until = args[1] if len(args) > 1 else []
            max_gen_toks = args[2] if len(args) > 2 else 64
            input_ids = torch.tensor(self.tok_encode(context), dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                output = self.model.generate(input_ids, max_new_tokens=max_gen_toks, do_sample=False, use_cache=True)
            gen_tokens = output[0, input_ids.shape[1]:].tolist()
            text = self.tok_decode(gen_tokens)
            if until:
                cut = len(text)
                for u in until:
                    idx = text.find(u)
                    if idx != -1 and idx < cut:
                        cut = idx
                text = text[:cut]
            results.append(text)
        return results
