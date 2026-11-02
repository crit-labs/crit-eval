"""The ceiling of serving several requests at once (protocol concurrency.ceiling): one decode step of B rows, one per
request, reads every weight read on every pass once for all of them, the routed experts its rows touch, and each row's
KV and fixed state, and computes every row's matmuls, logits and attention. Its floor is the larger of

    memory   (resident bytes + expert bytes x E(1 - (1 - k/E)^B)/E + B x (KV per token x context + fixed state)) / bandwidth
    compute  B x sum over stored formats of active matmul FLOPs per token / that format's best rate
             + B x context x attention FLOPs per query-key pair / the faster of the BF16 and FP8 rates

where a layer of E experts routing k per token has E(1 - (1 - k/E)^B) of them touched by B independent rows, and a
row's active matmul FLOPs are the dense matrices', the output head's and k/E of the experts'. The aggregate ceiling is
B tokens per step over that floor. Counted from the weight headers and config, as crit_eval.limits.ceiling counts a
single pass; engines that speculate decode more than a token per row per step, which this bound leaves out, as the
fixed decode denominator does."""
import math

from .ceiling import matmul_flops, model_config, weight_files
from .context import weight_split
from .schemes import quantization_of
from .tensors import tensor_role


def batch_terms(model, kv_bytes_per_element):
    """What a batched decode step reads and computes, from the stored model: {"resident_bytes" (read once per step,
    draft layers left out), "expert_bytes", "experts", "experts_per_token", "dense_flops" and "expert_flops" ({stored
    format: FLOPs per token}, the output head among the dense), "kv_bytes_per_token" (at kv_bytes_per_element),
    "fixed_state_bytes" (per sequence), "attention_flops_per_pair"}; None without weight headers."""
    config = model_config(model)
    split = weight_split(model, config)
    shards, drafts = weight_files(model)
    if split is None or not shards:
        return None
    dense, experts, head = {}, {}, 0
    quantization = quantization_of(config)
    for shard in shards:
        tensors = {tensor.name: tensor for tensor in shard.tensors}
        kinds = {name: tensor_role(name, shard.path.name, drafts).kind for name in tensors}
        for into, subset in ((dense, {n: t for n, t in tensors.items() if kinds[n] != "expert"}),
                             (experts, {n: t for n, t in tensors.items() if kinds[n] == "expert"})):
            flops, _ = matmul_flops(subset, shard.path.name, drafts, quantization)
            for fmt, value in flops.items():
                into[fmt] = into.get(fmt, 0) + value
        # The output head: every row of a decode step produces logits (matmul_flops leaves it out for prefill).
        head += sum(2 * math.prod(t.shape) for n, t in tensors.items() if kinds[n] == "head" and len(t.shape) == 2)
    if head:
        # Priced at the BF16 rate, as engines run the head.
        dense["bf16"] = dense.get("bf16", 0) + head
    shape = config.attention_shape()
    kv = attention = 0
    if shape:
        full, heads, head_dim, kv_heads = shape
        kv = 2 * kv_heads * head_dim * full * kv_bytes_per_element
        attention = 4 * heads * head_dim * full
    moe = config.moe
    return {"resident_bytes": split["resident"] - split["draft"], "expert_bytes": split["experts"],
            "experts": (moe.experts if moe else 0) or 0, "experts_per_token": (moe.experts_per_token if moe else 0) or 0,
            "dense_flops": dense, "expert_flops": experts, "kv_bytes_per_token": kv,
            "fixed_state_bytes": config.fixed_state_bytes(kv_bytes_per_element), "attention_flops_per_pair": attention}


def touched(experts, per_token, rows):
    """The share of a layer's experts that `rows` independent rows routing per_token each touch."""
    if not experts or not per_token:
        return 0.0
    return 1.0 - (1.0 - min(1.0, per_token / experts)) ** rows


def step_floor(terms, rows, context, bandwidth_gbs, tflops):
    """The floor of one decode step of `rows` rows at `context` tokens each: {"rows", "memory_ms", "compute_ms",
    "step_ms", "bound", "ceiling_tps" (rows over the step), "expert_share" (experts touched)}, or (None, missing)."""
    missing = [name for name, value in (("model weight headers", terms), ("device bandwidth (--device)", bandwidth_gbs),
                                        ("device GEMM rates (--device)", tflops)) if not value]
    if missing:
        return None, missing
    share = touched(terms["experts"], terms["experts_per_token"], rows)
    read = (terms["resident_bytes"] + terms["expert_bytes"] * share
            + rows * (terms["kv_bytes_per_token"] * context + terms["fixed_state_bytes"]))
    memory = read / (bandwidth_gbs * 1e9) * 1000
    active = min(1.0, terms["experts_per_token"] / terms["experts"]) if terms["experts"] else 0.0
    compute, unpriced = 0.0, []
    for fmt in sorted(set(terms["dense_flops"]) | set(terms["expert_flops"])):
        flops = terms["dense_flops"].get(fmt, 0) + terms["expert_flops"].get(fmt, 0) * active
        rate = (tflops or {}).get(fmt)
        if not rate:
            unpriced.append(f"{fmt} GEMM rate")
            continue
        compute += rows * flops / (rate * 1e12) * 1000
    attention_rate = max((tflops or {}).get("bf16") or 0, (tflops or {}).get("fp8") or 0)
    if terms["attention_flops_per_pair"] and attention_rate:
        compute += rows * context * terms["attention_flops_per_pair"] / (attention_rate * 1e12) * 1000
    if unpriced:
        return None, unpriced
    step = max(memory, compute)
    return {"rows": rows, "memory_ms": memory, "compute_ms": compute, "step_ms": step,
            "bound": "memory" if memory >= compute else "compute", "ceiling_tps": rows / step * 1000 if step else None,
            "expert_share": share if terms["experts"] else None}, []
