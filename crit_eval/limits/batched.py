"""The ceiling of serving several requests at once (protocol concurrency.ceiling): one decode step of B rows, one per
request, reads every weight read on every pass once for all of them, the routed experts its rows touch, and each row's
KV and fixed state, and computes every row's matmuls, logits and attention. Its floor is the largest of

    memory   (device-resident bytes + touched experts held on the device + B x (KV per token x context + fixed state))
             / bandwidth
    host     the host bytes (resident weights off the device + touched experts read from host memory) on the faster of
             two paths: copied over the host link (bytes / the link's rate) or computed on the CPU (the longer of bytes /
             host memory's read rate and B rows' arithmetic on their share of them / the CPU's rate)
    compute  B x sum over stored formats of active matmul FLOPs per token / that format's best rate
             + B x context x attention FLOPs per query-key pair / the faster of the BF16 and FP8 rates

where a layer of E experts routing k per token has E(1 - (1 - k/E)^B) of them touched by B independent rows, and a
row's active matmul FLOPs are the dense matrices', the output head's and k/E of the experts'. For a model whose weights
the device cannot all hold (the run's memory tiers, crit_eval.limits.tiers), the touched experts the device's capacity
cannot hold are read from host memory every step, and so are those the engine's expert cache misses at its measured
hit rate (tiers.expert_misses): each row brings its own misses, so an engine whose misses come from host memory serves
little more at B rows than at one. Without a measured hit rate that part is unmeasured and the floor holds the capacity
bound alone, every other routed expert read from the device: an optimistic floor.
The aggregate ceiling is B tokens per step over that floor; the speedup the cost model expects at B rows is the
ceiling at B over the ceiling at one row. Counted from the weight headers and config, as crit_eval.limits.ceiling counts
a single pass; engines that speculate decode more than a token per row per step, which this bound leaves out, as the
fixed decode denominator does."""
import math

from .ceiling import matmul_flops, model_config, weight_files
from .context import weight_split
from .schemes import quantization_of
from .tensors import tensor_role
from .tiers import UNMEASURED, expert_misses


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


def step_floor(terms, rows, context, bandwidth_gbs, tflops, host=None):
    """The floor of one decode step of `rows` rows at `context` tokens each: {"rows", "memory_ms", "compute_ms",
    "step_ms", "bound", "ceiling_tps" (rows over the step), "expert_share" (experts touched)}, or (None, missing).
    host: what a model whose weights the device cannot all hold reads from host memory ({"tiers": the run's memory
    tiers (crit_eval.limits.tiers.derive, with their link and host memory rates and the CPU's), "hit_rate": the expert
    cache's hit rate, measured or declared by host_bytes_per_pass, or None, "hit_rate_source"}); the step then records
    "host_ms" on its "host_path" (link or cpu; "host_paths_ms" both), "host_bytes", "hit_rate", "hit_rate_source" and,
    without a hit rate, "unmeasured" (the misses beyond the capacity bound)."""
    missing = [name for name, value in (("model weight headers", terms), ("device bandwidth (--device)", bandwidth_gbs),
                                        ("device GEMM rates (--device)", tflops)) if not value]
    if missing:
        return None, missing
    share = touched(terms["experts"], terms["experts_per_token"], rows)
    active = min(1.0, terms["experts_per_token"] / terms["experts"]) if terms["experts"] else 0.0
    tiers = host["tiers"] if host else None
    if tiers:
        # The placement's resident weights off the device, at most those a step reads: its resident weights hold the
        # draft layers, which a decode step leaves out (batch_terms).
        resident_off = min(int(tiers["resident_off_device_bytes"]), int(terms["resident_bytes"]))
        missed, miss_share, source = expert_misses({**tiers, "expert_bytes": terms["expert_bytes"]}, share,
                                                   host["hit_rate"], host["hit_rate_source"])
    else:
        resident_off, missed, miss_share, source = 0, 0.0, 0.0, None
    from_host = resident_off + missed
    read = (terms["resident_bytes"] - resident_off + terms["expert_bytes"] * share - missed
            + rows * (terms["kv_bytes_per_token"] * context + terms["fixed_state_bytes"]))
    memory = read / (bandwidth_gbs * 1e9) * 1000
    paths = {}
    if from_host:
        rates = tiers["rates_gbs"]
        if rates["link"]:
            paths["link"] = from_host / (rates["link"] * 1e9) * 1000
        if rates["memory"]:
            cpu = from_host / (rates["memory"] * 1e9) * 1000
            if tiers["flops_per_weight_byte"] and tiers["cpu_gflops"]:
                # Each row's arithmetic on its share of the host bytes: the resident ones off the device and its own
                # experts' misses, at the step's miss share.
                per_row = resident_off + terms["expert_bytes"] * active * miss_share
                cpu = max(cpu, rows * per_row * tiers["flops_per_weight_byte"] / (tiers["cpu_gflops"] * 1e9) * 1000)
            paths["cpu"] = cpu
        if not paths:
            return None, ["host link and host memory rates (re-run crit eval probe)"]
    path = min(paths, key=paths.get) if paths else None
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
    parts = {"memory": memory, "compute": compute, **({path: paths[path]} if path else {})}
    bound = max(parts, key=parts.get)
    step = parts[bound]
    out = {"rows": rows, "memory_ms": memory, "compute_ms": compute, "step_ms": step, "bound": bound,
           "ceiling_tps": rows / step * 1000 if step else None, "expert_share": share if terms["experts"] else None}
    if tiers:
        out.update(host_ms=paths[path] if path else 0.0, host_path=path, host_paths_ms=paths,
                   host_bytes=int(from_host), hit_rate=None if source == "unmeasured" else 1.0 - miss_share,
                   hit_rate_source=source)
        if source == "unmeasured":
            out["unmeasured"] = UNMEASURED
    return out, []
