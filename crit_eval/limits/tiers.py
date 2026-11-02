"""Memory tiers (protocol headroom.tiers): the bytes a decode pass and a prompt read from host memory rather than the
device's, each priced at the tier it is read from, as MoE-CAP's tiered cost model and its sparse memory bandwidth
utilization (S-MBU) price them.

The tiers come from the model files' placement on the device at the required context
(crit_eval.limits.context.placement), so no floor rests on the engine's own report: the resident weights the device
cannot hold (dense_device_host) are read from host memory on every pass, and the routed experts by the share a token
activates (experts_per_token of the experts, S-MBU's count), the device cache missing them at its hit rate: the
engine's own (expert_hits over expert_reads in its decode telemetry) where it reports one, else the share of the experts
the device holds (even routing, the pessimistic case). An engine that declares host_bytes_per_pass keeps its decode pass
priced by its declaration (headroom.host_pass).

A host byte goes by the faster of two paths. Copied over the host link, the device computes with it; computed on the
CPU, host memory is read at its read rate and the CPU's arithmetic runs at its probed rate. A decode pass is bound by
bytes on either path, so it takes the faster rate; a prompt touches each off-device expert at least once (E(1 - (1 -
k/E)^n) of a layer's E experts for n tokens at k per token) and does n tokens' arithmetic, so the CPU path also pays
that arithmetic and long prompts go over the link. Experts the placement leaves on disk are priced as host memory: the
probe measures no disk, so their floor is a lower bound.
"""


def derive(where, ceiling=None):
    """The tiers a placement (context.placement) implies, or None when every weight it reads is on the device or Eval
    cannot place the model. ceiling: the model's arithmetic (matmul FLOPs per token over its weight bytes, for the CPU
    path's arithmetic)."""
    if not where:
        return None
    b = where["bytes"]
    resident_off = int(b.get("resident_off_device") or 0)
    experts, off = int(b.get("experts") or 0), int(b.get("experts_off_device") or 0)
    if not resident_off and not off:
        return None
    active = int(where.get("active_expert_bytes_per_token") or 0)
    flops = sum(((ceiling or {}).get("matmul_flops_per_token") or {}).values())
    weights = (ceiling or {}).get("weight_bytes")
    rates = where.get("rates_gbs") or {}
    return {"source": "placement", "scenario": where["scenario"], "tokens": where.get("tokens"),
            "resident_off_device_bytes": resident_off, "expert_bytes": experts, "experts_off_device_bytes": off,
            "active_expert_bytes_per_token": active,
            "active_share": active / experts if experts else 0.0,
            # Even routing: a token's experts are on the device as often as the device holds experts.
            "even_hit_rate": (experts - off) / experts if experts else None,
            "flops_per_weight_byte": flops / weights if flops and weights else None,
            "rates_gbs": {"link": rates.get("link"), "memory": rates.get("host_memory_read")},
            "cpu_gflops": where.get("cpu_gflops")}


def hit_rate(samples):
    """The device expert cache's hit rate over `samples` (their expert_hits over expert_reads, which the engine reports
    for its decode), or None when they report none."""
    reads = hits = 0
    for sample in samples or []:
        if isinstance(sample.get("expert_reads"), int) and isinstance(sample.get("expert_hits"), int):
            reads += sample["expert_reads"]
            hits += sample["expert_hits"]
    return hits / reads if reads else None


def missed(tiers, rate=None):
    """(the miss rate a token's experts see, where it came from): the engine's hit rate, else even routing's."""
    if rate is not None:
        return 1.0 - min(1.0, max(0.0, rate)), "engine"
    even = tiers.get("even_hit_rate")
    return (1.0 - even if even is not None else 1.0), "even routing"


def decode_host(tiers, rate=None):
    """A decode pass's host bytes in host_pass's form ({"link", "cpu", "link_gbs", "memory_gbs"}) on the faster path:
    the resident bytes off the device plus the active experts the cache misses, at `rate` (the engine's hit rate;
    None: even routing). None without tiers."""
    if not tiers:
        return None
    miss, source = missed(tiers, rate)
    host = tiers["resident_off_device_bytes"] + tiers["active_expert_bytes_per_token"] * miss
    link, memory = tiers["rates_gbs"].get("link"), tiers["rates_gbs"].get("memory")
    over_link = bool(link) and (not memory or link > memory)
    return {"link": host if over_link else 0, "cpu": 0 if over_link else host, "link_gbs": link, "memory_gbs": memory,
            "source": tiers["source"], "hit_rate": 1.0 - miss, "hit_rate_source": source}


def prompt_host(tiers, prompt_tokens, rate=None):
    """The floor of a prompt's host bytes in ms, with its parts, or None without tiers: the resident bytes off the device
    and each off-device expert the prompt's tokens touch at least once (missed at `rate`'s miss share), on the faster
    of the link (bytes over its rate) and the CPU (bytes over host memory's read rate, and the tokens' arithmetic on
    those bytes over the CPU's rate)."""
    if not tiers:
        return None
    n = max(1, int(prompt_tokens))
    miss, source = missed(tiers, rate)
    touched = 1.0 - (1.0 - min(1.0, tiers["active_share"])) ** n if tiers["active_share"] else 0.0
    host = tiers["resident_off_device_bytes"] + tiers["expert_bytes"] * touched * miss
    per_token = tiers["resident_off_device_bytes"] + tiers["active_expert_bytes_per_token"] * miss
    link, memory = tiers["rates_gbs"].get("link"), tiers["rates_gbs"].get("memory")
    paths = {}
    if link:
        paths["link"] = host / (link * 1e9) * 1000
    if memory:
        cpu = host / (memory * 1e9) * 1000
        if tiers.get("flops_per_weight_byte") and tiers.get("cpu_gflops"):
            cpu = max(cpu, n * per_token * tiers["flops_per_weight_byte"] / (tiers["cpu_gflops"] * 1e9) * 1000)
        paths["cpu"] = cpu
    if not paths:
        return None
    path = min(paths, key=paths.get)
    return {"host_bytes": int(host), "host_ms": paths[path], "path": path, "paths_ms": paths,
            "hit_rate": 1.0 - miss, "hit_rate_source": source}
