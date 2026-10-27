"""Attention the reference registers with transformers: eager attention computed on blocks of query rows, with and
without sinks, so a long text never holds its length x length score matrix."""
from ..limits.schemes import SchemeError

CHUNKED = "crit_chunked"
# Chunked attention for attention with sinks: the modeling code's eager attention in blocks of query rows, with
# eager's additive masks (chunked_sink_attention).
CHUNKED_SINKS = "crit_chunked_sinks"
QUERY_BLOCK = 256


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                      **kwargs):
    """Eager causal attention computed QUERY_BLOCK query rows at a time: FP32 scores, softmax over each whole
    row, never a full length x length matrix. Registered with transformers' AttentionInterface as CHUNKED (and
    sdpa's mask rules for its masks); causality and any sliding window come from positions, and a supplied mask
    (boolean, or additive as eager's) further restricts each block of rows. Query heads share their key/value head
    by broadcasting (transformers' repeat_kv without the copy), and a sliding-window layer reads only the columns
    its block can see."""
    import torch
    batch, heads, length, _ = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.view(batch, kv_heads, groups, length, query.shape[-1])
    key, value = key[:, :, None], value[:, :, None]
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    offset = key.shape[3] - length
    columns = torch.arange(key.shape[3], device=query.device)
    out = torch.empty(batch, kv_heads, groups, length, value.shape[-1], dtype=query.dtype, device=query.device)
    for start in range(0, length, QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, length)
        rows = torch.arange(start + offset, stop + offset, device=query.device)[:, None]
        width = stop + offset
        first = max(0, start + offset - sliding_window + 1) if sliding_window else 0
        scores = torch.matmul(query[:, :, :, start:stop], key[:, :, :, first:width].transpose(-2, -1)).mul_(scaling)
        allowed = columns[None, first:width] <= rows
        if sliding_window:
            allowed &= columns[None, first:width] > rows - sliding_window
        if attention_mask is not None:
            block = attention_mask[..., start:stop, first:width]
            block = (block.view(batch, kv_heads, groups, *block.shape[-2:]) if block.shape[1] == heads
                     else block[:, :, None])
            if block.dtype == torch.bool:
                allowed = allowed & block
            else:
                scores += block
        weights = torch.softmax(scores.masked_fill_(~allowed, float("-inf")), dim=-1, dtype=torch.float32)
        del scores
        out[:, :, :, start:stop] = torch.matmul(weights.to(query.dtype), value[:, :, :, first:width])
    return out.view(batch, heads, length, -1).transpose(1, 2).contiguous(), None


def chunked_sink_attention(module, query, key, value, attention_mask, **kwargs):
    """Attention with sinks as the modeling code's own eager attention computes it (its module's
    eager_attention_forward), called on QUERY_BLOCK query rows at a time with those rows of the mask. Registered as
    CHUNKED_SINKS with eager's additive masks: sinks have no sdpa form, so such modeling code is written for eager's
    masks, and may extend them (compressed entries appended to the key/value axis after the sliding window, and
    their visibility, a block bias, to the mask). Visibility so comes from the mask alone, never from
    positions; every block reads every key column, and each row is the one eager computes, without the length x
    length score matrix."""
    import sys
    import torch
    eager = getattr(sys.modules[type(module).__module__], "eager_attention_forward", None)
    if eager is None:
        raise SchemeError(f"{type(module).__name__}: attention with sinks but no eager_attention_forward in "
                          f"{type(module).__module__}; the reference cannot run its long texts in blocks")
    pieces = []
    for start in range(0, query.shape[2], QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, query.shape[2])
        rows = None if attention_mask is None else attention_mask[..., start:stop, :]
        pieces.append(eager(module, query[:, :, start:stop], key, value, rows, **kwargs)[0])
    return torch.cat(pieces, dim=1), None


def _sink_attention(model):
    """Whether `model`'s attention has sinks (a learned logit per head in every row's softmax, a `sinks`
    parameter): chunked attention then runs as CHUNKED_SINKS."""
    import torch
    return any(isinstance(getattr(module, "sinks", None), torch.nn.Parameter) for module in model.modules())
