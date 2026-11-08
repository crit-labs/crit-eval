"""Memory tiers (protocol headroom.tiers): the bytes a decode pass and a prompt read from host memory rather than the
device's, each priced at the tier it is read from, as MoE-CAP's tiered cost model and its sparse memory bandwidth
utilization (S-MBU) price them.

The tiers come from the model files' placement on the device at the required context
(crit_eval.limits.context.placement): the resident weights the device cannot hold (dense_device_host, or those the
engine's measured footprint at the requirement cannot hold beside its KV cache) are read from host memory on every
pass, and the routed experts by the share a token activates (experts_per_token of the experts, S-MBU's count).
Which of those the device's expert cache holds is the engine's: no rate is assumed. Reads that touch more distinct
experts than the device can hold (a long prompt, many batched rows) miss at least the excess, a bound the placement
alone gives (the device capacity bound); beyond it, misses are priced at the engine's own hit rate (expert_hits over
expert_reads in its decode telemetry, or its telemetry()'s expert cache counters) where it reports one, and are
otherwise unmeasured: the floor holds the bound alone and says so. An engine that declares host_bytes_per_pass keeps
its decode pass priced by its declaration (headroom.host_pass).

A host byte goes by the faster of two paths. Copied over the host link, the device computes with it; computed on the
CPU, host memory is read at its read rate and the CPU's arithmetic runs at its probed rate. A decode pass is bound by
bytes on either path, so it takes the faster rate; a prompt touches each off-device expert at least once (E(1 - (1 -
k/E)^n) of a layer's E experts for n tokens at k per token) and does n tokens' arithmetic, so the CPU path also pays
that arithmetic and long prompts go over the link. Experts the placement leaves on disk are priced as host memory: the
probe measures no disk, so their floor is a lower bound.
"""

# Why a floor's expert misses are unmeasured: the engine reports no hit rate, and none is assumed.
UNMEASURED = ("the expert cache's hit rate: the engine reports no expert_reads and expert_hits (per request, or its "
              "telemetry()'s expert cache reads and hits), so the misses its routed experts bring from host memory "
              "beyond what the device's capacity forces are left out of the floor")


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
            # The share of the experts' bytes the device can hold beside the KV cache: what bounds its misses from
            # below. Not a hit rate: which experts the cache holds is the engine's (protocol headroom.tiers).
            "device_expert_share": (experts - off) / experts if experts else None,
            "flops_per_weight_byte": flops / weights if flops and weights else None,
            "rates_gbs": {"link": rates.get("link"), "memory": rates.get("host_memory_read")},
            "cpu_gflops": where.get("cpu_gflops"),
            # The run's measured expert cache hit rate and its source, which headroom.summarize sets once the decode
            # suite has run; none before.
            "engine_hit_rate": None, "engine_hit_rate_source": "unmeasured"}


def hit_rate(samples):
    """The device expert cache's hit rate over `samples` (their expert_hits over expert_reads, which the engine reports
    for its decode), or None when they report none."""
    reads = hits = 0
    for sample in samples or []:
        if isinstance(sample.get("expert_reads"), int) and isinstance(sample.get("expert_hits"), int):
            reads += sample["expert_reads"]
            hits += sample["expert_hits"]
    return hits / reads if reads else None


def telemetry_counts(value):
    """(reads, hits) of an engine's telemetry() (crit's Core: {"expert_cache": {"reads", "hits", ...}, ...}; crit
    serve's /health engine_telemetry), or None when it reports no expert cache counters."""
    cache = (value or {}).get("expert_cache") if isinstance(value, dict) else None
    if not isinstance(cache, dict):
        return None
    reads, hits = cache.get("reads"), cache.get("hits")
    if type(reads) is not int or type(hits) is not int or not 0 <= hits <= reads:
        return None
    return reads, hits


def expert_misses(tiers, share, rate=None, source="engine"):
    """(bytes of routed experts read from host memory when reads touch `share` of every layer's experts, the share of
    the touched bytes they are, where the figure comes from). At least the touched bytes the device cannot hold (its
    capacity, the placement's experts on the device): "device capacity"; at the engine's hit rate `rate` its misses
    where they are more (`source`, "engine" for its requests' expert_reads and expert_hits); without a rate the bound
    alone, "unmeasured". No misses ("on device") when the device holds every expert."""
    experts, off = int(tiers.get("expert_bytes") or 0), int(tiers.get("experts_off_device_bytes") or 0)
    touched = experts * min(1.0, max(0.0, share))
    if not off or not touched:
        return 0.0, 0.0, "on device"
    bound = max(0.0, touched - (experts - off))
    if rate is None:
        return bound, bound / touched, "unmeasured"
    priced = touched * (1.0 - min(1.0, max(0.0, rate)))
    return (priced, priced / touched, source) if priced > bound else (bound, bound / touched, "device capacity")


def decode_host(tiers, rate=None, source="engine"):
    """A decode pass's host bytes in host_pass's form ({"link", "cpu", "link_gbs", "memory_gbs"}) on the faster path:
    the resident bytes off the device plus the active experts the cache misses (expert_misses at `rate`, the engine's
    measured hit rate, from `source`; None: unmeasured, the term left out and named). None without tiers."""
    if not tiers:
        return None
    experts, miss, found = expert_misses(tiers, tiers.get("active_share") or 0.0, rate, source)
    host = tiers["resident_off_device_bytes"] + experts
    link, memory = tiers["rates_gbs"].get("link"), tiers["rates_gbs"].get("memory")
    over_link = bool(link) and (not memory or link > memory)
    out = {"link": host if over_link else 0, "cpu": 0 if over_link else host, "link_gbs": link, "memory_gbs": memory,
           "source": tiers["source"], "hit_rate": None if found == "unmeasured" else 1.0 - miss,
           "hit_rate_source": found}
    if found == "unmeasured":
        out["unmeasured"] = UNMEASURED
    return out


def prompt_host(tiers, prompt_tokens, rate=None, source="engine"):
    """The floor of a prompt's host bytes in ms, with its parts, or None without tiers: the resident bytes off the device
    and the off-device experts the prompt's tokens touch (expert_misses: at least those beyond the device's capacity,
    at `rate`'s miss share where more), on the faster of the link (bytes over its rate) and the CPU (bytes over host
    memory's read rate, and the tokens' arithmetic on those bytes over the CPU's rate)."""
    if not tiers:
        return None
    n = max(1, int(prompt_tokens))
    share = 1.0 - (1.0 - min(1.0, tiers["active_share"])) ** n if tiers["active_share"] else 0.0
    experts, miss, found = expert_misses(tiers, share, rate, source)
    host = tiers["resident_off_device_bytes"] + experts
    # Each token's arithmetic on the CPU path covers its own experts' bytes off the device, at the prompt's miss share.
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
            "miss_share": miss, "hit_rate": None if found == "unmeasured" else 1.0 - miss, "hit_rate_source": found,
            **({"unmeasured": UNMEASURED} if found == "unmeasured" else {})}
