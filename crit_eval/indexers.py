"""Faster or leaner forms of modeling code the reference would otherwise run as a Python loop per token, or in a
tensor larger than the device.

transformers' qwen4_exp sparse-attention indexer (Qwen4ExpTextQSAIndexer.forward) selects, for every query token, the
top blocks of pooled keys among the complete blocks it sees, in a loop over the queries: on a 32K-token text that is
about 33,000 iterations of a host-synchronising nonzero, a pooling of every visible block, RoPE, a matmul and a top-k
per full-attention layer, which made the reference of Qwen3.8 Flash-Next take hours. `qsa_indexer_forward` selects
exactly what that loop selects, ties included, when every query sees a prefix of the keys (a causal mask without
padding, as the reference's chunks are); any other mask runs transformers' own loop.

Exactly, because it does the loop's arithmetic: a query's keys are its complete blocks pooled, normed and rotated in a
tensor of that query's shape, its scores the matmul of its own heads with them, and its blocks the top-k of that row,
each op as the loop has it on the same device. Only what cannot change a value is left out: with a prefix mask, the
queries that see the same number of complete blocks pool the same tokens into the same keys, so those are pooled once
per block count, not once per query; a query that sees no more blocks than its budget keeps all of them (a top-k of all
of a row), so its scores are not computed; and no step reads anything back to the host. Batching queries or pooling all
blocks at once would be faster, but rounds in tensors of other shapes, and the loop's top-k breaks ties its own way
(which of several blocks scoring exactly 0 after the ReLU it keeps, common with few heads, depends on the device's top-k
over that query's own row): an earlier form did both and kept other blocks than the loop on some queries (0.43 nats on
a target of the tiny qwen4exp model's 23-token test text; tests/test_eval_indexers.py).

transformers' deepseek_v4 indexer scorer (DeepseekV4IndexerScorer.forward) scores every query against every
compressed key in one FP32 tensor of [batch, queries, index heads, keys] and two more of its size: for an 8192-token
chunk of a 32K-token text against its 8192 compressed keys that is 17 GiB each, more than a 32 GB device holds.
`indexer_scorer_forward` computes the same ops over blocks of query rows, each block's tensor at most SCORER_BYTES, so
every value is the same matmul of the same rows, ReLU, scale and sum over the heads: equal bit for bit to the whole
tensor on an RTX 5090 at the model's shapes (blocks of 512 to 4,194 rows) and on the CPU at the test's
(tests/test_eval_indexers.py). Blocks of a few rows can round otherwise, since a batched matmul may pick its kernel by
the batch count, so a block is never smaller than SCORER_BYTES allows; a tensor within SCORER_BYTES runs transformers'
own forward.

`exact_forms(model)` binds these forms to a model's modules whose forward is the code they were checked against
(by a hash of its source, so a changed transformers keeps its own) and returns their names.
"""
import hashlib
import inspect
import math
import sys
import types

# sha256 of the source of each forward a form replaces, as the transformers release the form was checked against
# (tests/test_eval_indexers.py) has it.
CHECKED = {
    "Qwen4ExpTextQSAIndexer": "cc3b0347832752a530a86a4b69e5147daea9b58d29f756e8f60a580bf519f093",   # transformers 5.20.2
    "DeepseekV4IndexerScorer": "7e1489dad5000d0243a37d6d2bb97a4ee1017d13a63a01d7b941c947dcd4b092",  # transformers 5.20.2
}
# Bytes of one block's FP32 scores in indexer_scorer_forward (three such tensors are alive at once).
SCORER_BYTES = 1 << 30


def source_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def qsa_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Qwen4ExpTextQSAIndexer.forward without its per-query pooling and host reads, selecting exactly as it does
    (module docstring)."""
    import torch
    rope = sys.modules[type(self).__module__].apply_rotary_pos_emb
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = rope(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    kv_length = attention_mask.shape[-1]
    counts = visible[:, 0].sum(-1)   # (batch, queries): every query sees a prefix of this many keys, or the loop runs
    positions = torch.arange(kv_length, device=visible.device)
    if not torch.equal(positions < counts[..., None], visible[:, 0]):
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    ratio = self.compress_ratio
    selected = torch.zeros((batch_size, seq_length, kv_length), dtype=torch.bool, device=attention_mask.device)
    complete = counts // ratio                                    # each query's complete blocks
    # Its tail: the visible tokens after its complete blocks.
    selected |= (positions >= (complete * ratio)[..., None]) & (positions < counts[..., None])
    # Up to its budget a query keeps every complete block it sees: topk(n, dim=0) of n blocks returns them all.
    within = complete <= self.block_topk
    selected |= within[..., None] & (positions < (complete * ratio)[..., None])
    for batch in range(batch_size):
        counted = complete[batch].tolist()
        by_count = {}
        for query, n in enumerate(counted):
            if n > self.block_topk:
                by_count.setdefault(n, []).append(query)
        for n, queries in by_count.items():
            # The loop's keys for any query that sees n complete blocks of a prefix: its visible indices are 0, 1, ...
            block_token_indices = torch.arange(n * ratio, device=raw_keys.device).view(n, ratio)
            key_groups = raw_keys[batch].index_select(0, block_token_indices.flatten())
            key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
            pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
            pooled_keys = self.k_layernorm(pooled_keys)
            group_starts = block_token_indices[:, 0]
            block_key_states = rope(
                pooled_keys.unsqueeze(1),
                cos=full_cos[batch].index_select(0, group_starts),
                sin=full_sin[batch].index_select(0, group_starts),
            ).squeeze(1)
            for query in queries:
                scores = torch.matmul(q[batch, query].float(), block_key_states.float().transpose(-1, -2)).transpose(
                    -1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                selected_block_indices = scores.topk(min(self.block_topk, n), dim=0).indices
                selected[batch, query, block_token_indices.index_select(0, selected_block_indices).flatten()] = True
    selected_token_mask = selected.unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)
    return selected_token_mask


def indexer_scorer_forward(self, q, compressed_kv, hidden_states):
    """DeepseekV4IndexerScorer.forward over blocks of query rows whose scores take at most SCORER_BYTES each, the same
    values (module docstring)."""
    import torch
    batch, rows, heads, _ = q.shape
    width = batch * heads * compressed_kv.shape[1] * 4   # bytes of one query row's scores
    if rows * width <= SCORER_BYTES:
        return type(self).forward(self, q, compressed_kv, hidden_states)
    keys = compressed_kv.transpose(-1, -2).float().unsqueeze(1)
    weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
    out = torch.empty((batch, rows, compressed_kv.shape[1]), dtype=torch.float32, device=q.device)
    block = max(1, SCORER_BYTES // width)
    for start in range(0, rows, block):
        scores = torch.matmul(q[:, start:start + block].float(), keys)  # [B, block, H, T]
        scores = torch.nn.functional.relu(scores) * self.softmax_scale
        out[:, start:start + block] = (scores * weights[:, start:start + block].unsqueeze(-1)).sum(dim=2)
    return out


FORMS = {"Qwen4ExpTextQSAIndexer": qsa_indexer_forward, "DeepseekV4IndexerScorer": indexer_scorer_forward}
# Each form's version in the reference's identity: stored scores of an earlier version are scored again. 2: selects
# exactly as transformers' loop, ties included (1 broke ties among blocks scoring 0 its own way).
FORM_VERSIONS = {"Qwen4ExpTextQSAIndexer": 2, "DeepseekV4IndexerScorer": 1}


def exact_forms(model):
    """Bind FORMS to `model`'s modules whose class forward is the source they were checked against (CHECKED).
    Returns the class names bound."""
    bound = set()
    for module in model.modules():
        name = type(module).__name__
        if name in FORMS and source_hash(type(module).forward) == CHECKED[name]:
            module.forward = types.MethodType(FORMS[name], module)
            bound.add(name)
    return sorted(bound)
