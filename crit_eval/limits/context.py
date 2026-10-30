"""The longest request an engine must hold: `crit eval context --model DIR --device device.json`.

    max context = min(requested, model limit, device limit)      tokens of prompt plus output

The requested context is the user's (settings.toml context_tokens, default 200,000; `--requested-context`): the
longest request they want served. The model and the device may allow less, and then bound it; the result says
which bound and why.

The model limit is the number of positions the checkpoint declares in its config: `max_position_embeddings`
(or another length key), or with YaRN rope scaling in the config, its original positions times the factor.

The device limit is the KV cache that fits in the device's free memory next to everything that has to
stay there. Weights read on every pass stay in device memory: all of them except gathered embedding
rows (unless the output head is tied to them), the vision tower and mixture-of-experts experts. Experts
are read sparsely, so they can move to host memory or disk; at the maximum context they get only the
device memory the KV cache leaves. Draft layers count as resident. State that does not grow with the
context (linear-attention recurrence, sliding-window KV) is counted once. The KV cache is counted at
the protocol's KV precision (FP8 by default; the quality gate judges it), and a reserve covers
activations, workspaces and graphs. The run's VRAM headroom stays free:

    device limit = (free memory - headroom - reserve - resident weights - fixed state) / KV bytes per token

rounded down to the protocol's granularity. The KV cache never leaves device memory: decode reads all
of it on every pass, so a context beyond the device limit is a cliff, not a slope.

Below the maximum, an engine sized for it must be at least as fast: crit-eval measures the engine built for
the maximum, a curve of long-context points up to it, and the speed cases again at a lower context.
"""
import re

from .ceiling import model_config, weight_files
from .tensors import tensor_role

GIB = 1 << 30
DRAFT_HEAD = re.compile(r"(^|\.)head\.weight$")   # a draft layer's own output head (shared_head.head)


def weight_split(model, config):
    """Stored bytes by where they have to live: `resident` (read on every pass, kept in device memory),
    `experts` (read sparsely, may move to host memory or disk), `draft` (part of resident), `embedding`
    (gathered rows, host memory is enough) and `vision` (not loaded by a text engine). None without
    safetensors files or a GGUF. config: the model's crit_core.model.ModelConfig."""
    shards, drafts = weight_files(model, drafts=True)
    if not shards:
        return None
    split = {"resident": 0, "experts": 0, "draft": 0, "embedding": 0, "vision": 0}
    tied = bool(config.tie_word_embeddings)
    heads = 0
    for shard in shards:
        for tensor in shard.tensors:
            role, size = tensor_role(tensor.name, shard.path.name, drafts), tensor.nbytes
            if role.kind == "other_modality":
                split["vision"] += size
            elif role.kind == "embedding":
                split["embedding"] += size
            elif role.kind == "expert":
                split["experts"] += size
            else:
                split["resident"] += size
                heads += role.kind == "head"
                if role.draft:
                    split["draft"] += size
    if tied and not heads:
        # The output head is the embedding table: every pass reads all of it.
        split["resident"] += split["embedding"]
        split["embedding"] = 0
    return split


def pass_bytes(model, kv_bytes_per_element):
    """What one decode pass of one token reads from the stored model, for an engine that reports no bytes_per_pass:
    the weights read on every pass but draft layers, the routed experts' share (experts_per_token of the experts,
    taken as equally sized) and the fixed state (linear-attention and Mamba recurrence, sliding-window KV at
    kv_bytes_per_element). None without weight headers."""
    config = model_config(model)
    split = weight_split(model, config)
    if split is None:
        return None
    moe = config.moe
    experts = (split["experts"] * min(1.0, moe.experts_per_token / moe.experts)
               if split["experts"] and moe and moe.experts and moe.experts_per_token else 0)
    return int(split["resident"] - split["draft"] + experts + config.fixed_state_bytes(kv_bytes_per_element))


def draft_step_bytes(model):
    """What one draft step of the checkpoint's own draft (MTP) layers reads, for an engine that drafts and reports no
    draft_bytes_per_step: those layers' weights (their routed experts' share as in pass_bytes) and the output head
    their logits go through, the draft layers' own head where they have one, else the model's (the embedding table
    when the head is tied to it). None without draft layers or weight headers."""
    config = model_config(model)
    shards, drafts = weight_files(model, drafts=True)
    layers = experts = draft_head = head = embedding = 0
    for shard in shards:
        for tensor in shard.tensors:
            role, size = tensor_role(tensor.name, shard.path.name, drafts), tensor.nbytes
            if role.draft:
                if role.kind == "expert":
                    experts += size
                elif role.kind == "head" or DRAFT_HEAD.search(tensor.name):
                    draft_head += size
                elif role.kind == "layer":
                    layers += size   # a draft embedding table is gathered by rows, never read whole
            elif role.kind == "head":
                head += size
            elif role.kind == "embedding":
                embedding += size
    if not layers and not experts:
        return None
    moe = config.moe
    if experts and moe and moe.experts and moe.experts_per_token:
        experts *= min(1.0, moe.experts_per_token / moe.experts)
    return int(layers + experts + (draft_head or head or (embedding if config.tie_word_embeddings else 0)))


def companion_drafter_bytes(drafter, model):
    """What a companion drafter (a published draft model in its own folder: EAGLE-3, DFlash, DSpark) costs beside
    MODEL, from the weight headers: `weight_bytes`, every stored tensor but an embedding table (gathered by rows,
    host memory is enough), the device memory it holds as stored; and `pass_bytes`, what one drafter pass reads:
    those weights plus the output head its logits go through, its own where it stores one, else MODEL's (the
    embedding table when MODEL ties its head to it). None without weight headers."""
    shards, _ = weight_files(drafter)
    if not shards:
        return None
    weights = own_head = 0
    for shard in shards:
        for tensor in shard.tensors:
            role = tensor_role(tensor.name, shard.path.name)
            if role.kind in ("embedding", "other_modality"):
                continue
            weights += tensor.nbytes
            own_head += tensor.nbytes if role.kind == "head" else 0
    head = 0
    if not own_head:
        target, _ = weight_files(model)
        roles = [(tensor_role(t.name, s.path.name).kind, t.nbytes) for s in target for t in s.tensors]
        head = sum(size for kind, size in roles if kind == "head")
        if not head and model_config(model).tie_word_embeddings:
            head = sum(size for kind, size in roles if kind == "embedding")
    return {"weight_bytes": int(weights), "pass_bytes": int(weights + head)}


def max_context(model, device, vram_headroom_bytes, settings, requested=None, drafters=()):
    """The required context of an engine for MODEL on DEVICE (a `crit eval probe` result) with the facts
    behind it. `tokens` is None when an input is missing (`missing`) or the weights read on every pass do
    not fit (`reason`). settings: protocol.json's `context` section. requested: the user's wanted context; the
    model's positions and the device's memory may cap it (`bound`, and `note` says by how much). drafters: companion
    drafter folders the engine may load beside the model; their stored weights (companion_drafter_bytes) are kept
    out of the KV cache, so a candidate that speculates with one still fits the required context."""
    config = model_config(model)
    limit, source = config.position_limit()
    element = float(settings["kv_bytes_per_element"])
    granularity = int(settings.get("granularity", 1))
    elements = config.kv_elements_per_token()
    kv = int(round(elements * element)) if elements else None
    out = {"tokens": None, "bound": None, "requested": int(requested) if requested else None, "note": None,
           "model_limit": limit, "model_limit_source": source, "device_limit": None,
           "kv_precision": settings.get("kv_precision"), "kv_bytes_per_token": kv, "missing": [], "reason": None}
    if limit is None:
        out["missing"].append("the model's positions (max_position_embeddings in config.json)")
    if elements is None:
        out["missing"].append("the model's attention layers (config.json)")
    try:
        split, problem = weight_split(model, config), None
    except (OSError, ValueError, KeyError) as error:
        split, problem = None, str(error)
    free = (device or {}).get("memory_free_bytes")
    if split is None:
        out["missing"].append(f"the model's weight headers ({problem})" if problem else
                              "the model's weight headers (safetensors files or a GGUF)")
    if not free:
        out["missing"].append("free device memory (re-run crit eval probe)")
    companions = []
    for path in drafters or ():
        found = companion_drafter_bytes(path, model)
        if found is None:
            out["missing"].append(f"the companion drafter's weight headers ({path})")
        else:
            companions.append({"path": str(path), **found})
    if out["missing"]:
        return out
    fixed = config.fixed_state_bytes(element)
    reserve = int(float(settings["reserve_gib"]) * GIB)
    drafter_bytes = sum(row["weight_bytes"] for row in companions)
    budget = free - int(vram_headroom_bytes) - reserve - split["resident"] - fixed - drafter_bytes
    out["basis"] = {"device": (device or {}).get("device"), "memory_free_bytes": free,
                    "memory_total_bytes": (device or {}).get("memory_total_bytes"),
                    "vram_headroom_bytes": int(vram_headroom_bytes), "reserve_bytes": reserve,
                    "resident_weight_bytes": split["resident"], "draft_weight_bytes": split["draft"],
                    "expert_bytes": split["experts"], "embedding_bytes": split["embedding"],
                    "fixed_state_bytes": fixed, "kv_cache_bytes_available": max(0, budget)}
    if companions:
        out["basis"] |= {"companion_drafter_bytes": drafter_bytes, "companion_drafters": companions}
    if budget <= 0:
        out["device_limit"] = 0
        out["reason"] = (f"the weights read on every pass ({split['resident'] / 1e9:.1f} GB)"
                         + (f", the companion drafters ({drafter_bytes / 1e9:.1f} GB)" if drafter_bytes else "")
                         + f" and fixed state do not fit the device's {free / 1e9:.1f} GB free memory with {vram_headroom_bytes / 1e9:.1f} GB headroom "
                         f"and a {reserve / 1e9:.1f} GB reserve")
        return out
    if kv:
        out["device_limit"] = budget // kv // granularity * granularity
        tokens, bound = min((limit, "model"), (out["device_limit"], "device"))
    else:
        tokens, bound = limit, "model"  # no cache grows with the context
    if requested and int(requested) <= tokens:
        tokens, bound = int(requested), "requested"
    elif requested:
        out["note"] = (f"the requested {int(requested):,} tokens exceed the "
                       + (f"model's {limit:,} positions ({source})" if bound == "model" else
                          f"{out['device_limit']:,} tokens whose KV cache fits the device")
                       + f": the required context is {tokens:,}")
    out.update(tokens=tokens, bound=bound)
    # Experts get the device memory the KV cache leaves at the maximum context.
    out["basis"]["expert_bytes_in_device_memory_at_max"] = max(0, min(split["experts"], budget - tokens * (kv or 0)))
    return out


# Where an engine's weights can live, by what fits: the scenario a build starts from (crit recon records it). Lookup
# tables (gathered embedding and n-gram rows) are a modifier on any scenario: a pass reads a few rows of them, so
# they live wherever room is left, down to the disk.
SCENARIOS = {
    "fits_device": (1, "every weight read on a pass fits device memory next to the KV cache at the required context"),
    "sparse_device_host": (2, "routed experts beyond device memory fit host memory"),
    "sparse_device_host_disk": (3, "routed experts beyond device memory exceed host memory: the rest stays on disk"),
    "dense_device_host": (5, "the weights read on every pass do not fit device memory; a pass streams or computes the "
                             "rest from host memory"),
    "beyond_host": (None, "the weights read on every pass exceed device and host memory together"),
}


def link_gbs(device):
    """The device's host-to-device rate in GB/s: its pinned copies at the largest size the probe timed, else None."""
    curve = ((device or {}).get("host_to_device_gbs") or {}).get("pinned") or {}
    return curve[max(curve, key=int)] if curve else None


def placement(model, device, vram_headroom_bytes, ram_headroom_bytes, settings, tokens, drafters=()):
    """The placement scenario of MODEL on DEVICE (a `crit eval probe` result) at `tokens` of required context: which of
    SCENARIOS applies, the bytes each memory tier holds, where the lookup tables go, and the bound one plain decode
    pass (one token, no speculation, routing spread evenly over the experts so a device cache holds its share of them)
    has in each expert source: on the device, misses copied over the host link, misses computed on the CPU from host
    memory. None when Eval cannot size the model (max_context's `missing`)."""
    value = max_context(model, device, vram_headroom_bytes, settings, tokens, drafters)
    basis = value.get("basis")
    if not basis:
        return None
    config = model_config(model)
    resident, experts, tables = basis["resident_weight_bytes"], basis["expert_bytes"], basis["embedding_bytes"]
    room = basis["kv_cache_bytes_available"] - (value.get("kv_bytes_per_token") or 0) * value["tokens"] if value["tokens"] else 0
    total, available = (device or {}).get("host_memory_total_bytes"), (device or {}).get("host_memory_available_bytes")
    host = max(0, (total or available or 0) - int(ram_headroom_bytes)) if (total or available) else None
    on_device = max(0, min(experts, room))
    off_device = experts - on_device
    if not value["tokens"]:
        name = "dense_device_host" if host is None or resident + experts <= host else "beyond_host"
    elif not off_device:
        name = "fits_device"
    else:
        name = "sparse_device_host" if host is None or off_device <= host else "sparse_device_host_disk"
    left = max(0, room - on_device)
    host_left = None if host is None else max(0, host - (off_device if value["tokens"] else resident + experts))
    tier = ("device" if tables <= left else "host" if host_left is None or tables <= host_left else "disk") if tables else None
    moe = config.moe
    share = min(1.0, moe.experts_per_token / moe.experts) if experts and moe and moe.experts and moe.experts_per_token else 0
    active, missed = experts * share, (experts * share * off_device / experts) if experts else 0
    bandwidth, link, memory = (device or {}).get("bandwidth_gbs"), link_gbs(device), ((device or {}).get("host_memory_gbs") or {}).get("read")
    floors = {}
    if bandwidth and value["tokens"]:
        own = (resident - basis["draft_weight_bytes"] + active - missed) / (bandwidth * 1e9) * 1000
        floors["device"] = (resident - basis["draft_weight_bytes"] + active) / (bandwidth * 1e9) * 1000
        if missed and link:
            floors["link"] = max(own, missed / (link * 1e9) * 1000)
        if missed and memory:
            floors["cpu"] = max(own, missed / (memory * 1e9) * 1000)
    return {"scenario": name, "number": SCENARIOS[name][0], "meaning": SCENARIOS[name][1], "tokens": int(tokens),
            "bytes": {"resident": resident, "experts": experts, "experts_on_device": on_device,
                      "experts_off_device": off_device, "lookup_tables": tables, "host_memory_for_weights": host},
            "lookup_tables": {"bytes": tables, "tier": tier} if tables else None,
            "active_expert_bytes_per_token": int(active), "missed_expert_bytes_per_token": int(missed),
            "rates_gbs": {"device": bandwidth, "link": link, "host_memory_read": memory},
            "plain_pass_floor_ms": floors,
            "plain_pass_ceiling_tps": {k: round(1000 / v, 1) for k, v in floors.items() if v}}


def render_placement(value):
    if not value:
        return "placement unknown: Eval cannot size the model on this device"
    b, gb = value["bytes"], 1e9
    number = f" ({value['number']})" if value["number"] else ""
    lines = [f"placement {value['scenario']}{number}: {value['meaning']}",
             f"  resident weights {b['resident'] / gb:.1f} GB; experts {b['experts'] / gb:.1f} GB, "
             f"{b['experts_on_device'] / gb:.1f} GB on the device, {b['experts_off_device'] / gb:.1f} GB off it"
             + (f"; host memory for weights {b['host_memory_for_weights'] / gb:.1f} GB" if b["host_memory_for_weights"] is not None else "")]
    if value["lookup_tables"]:
        lines.append(f"  lookup tables {value['lookup_tables']['bytes'] / gb:.1f} GB (gathered rows) in {value['lookup_tables']['tier']} memory"
                     if value["lookup_tables"]["tier"] != "disk" else
                     f"  lookup tables {value['lookup_tables']['bytes'] / gb:.1f} GB (gathered rows) on disk: read rows on demand, no read-ahead")
    if value["missed_expert_bytes_per_token"]:
        lines.append(f"  experts per token {value['active_expert_bytes_per_token'] / gb:.2f} GB, "
                     f"{value['missed_expert_bytes_per_token'] / gb:.2f} GB of them off the device at even routing")
    names = {"device": "all on the device", "link": "misses copied over the host link", "cpu": "misses computed on the CPU"}
    for key, tps in value["plain_pass_ceiling_tps"].items():
        lines.append(f"  plain pass, {names[key]}: {value['plain_pass_floor_ms'][key]:.2f} ms floor, {tps:,.1f} tok/s")
    return "\n".join(lines)


def lower_context(required, longest_case_tokens, settings):
    """The lower context the speed cases are measured again at, or None when no lower context still holds
    every speed case."""
    granularity = int(settings.get("granularity", 1))
    floor = -(-longest_case_tokens // granularity) * granularity
    lower = max(int(required * float(settings["lower_fraction"])) // granularity * granularity, floor)
    return lower if lower < required else None


def curve_points(required, settings, every=None):
    """Context lengths the long-context curve measures: powers of two from `curve_from` below the
    requirement, then the requirement itself. With settings' curve_points "ends" (a profile's curve), only the first and the
    requirement; every=True gives all of them regardless (the points a profile's curve interpolates). With "first"
    (a staged run that is not certified, protocol stages), only the first, the requirement itself when no point lies
    below it: a longer requirement is the certification's."""
    points, point = [], int(settings["curve_from"])
    while point < required:
        points.append(point)
        point *= 2
    if settings.get("curve_points") == "first" and not every:
        return points[:1] or [required]
    if settings.get("curve_points") == "ends" and not every:
        points = points[:1]
    return points + [required]


def interpolated_curve(points, required, settings):
    """{point: row} for the curve points a profile's "ends" curve skips, from its two measured ends: time to first
    token as a*n + c*n^2 through both (prefill work grows with the prompt and attention with its square) and decode
    milliseconds per token linear in the context (each step reads the weights and a KV cache that grows with it).
    Empty when either end was not measured."""
    measured = {int(p["context"]): p for p in points.values() if p.get("status") == "measured"}
    every = curve_points(required, settings, every=True)
    first, last = every[0], every[-1]
    if first == last or first not in measured or last not in measured or not all(
            measured[p].get(k) for p in (first, last) for k in ("ttft_ms", "decode_tps")):
        return {}
    low, high = measured[first], measured[last]
    n0, n1 = low["prompt_tokens"], high["prompt_tokens"]
    t0, t1 = low["ttft_ms"] / n0, high["ttft_ms"] / n1    # ttft / n = a + c * n
    c = (t1 - t0) / (n1 - n0)
    a = t0 - c * n0
    d0, d1 = 1000 / low["decode_tps"], 1000 / high["decode_tps"]
    out = {}
    for point in every[1:-1]:
        prompt = point - int(settings["output_tokens"])
        share = (point - first) / (last - first)
        decode_ms = d0 + share * (d1 - d0)
        out[str(point)] = {"status": "interpolated", "context": point, "prompt_tokens": prompt,
                           "ttft_ms": prompt * (a + c * prompt), "decode_tps": 1000 / decode_ms,
                           "from": [first, last]}
    return out


def retrieval_points(required, rule):
    """Context lengths the retrieval check fills: its fixed points below the requirement, then the requirement
    itself (near capacity: prompt plus output equal it). rule: protocol.json's context.retrieval."""
    return sorted({int(p) for p in rule["points"] if int(p) < required} | {int(required)})


def retrieval_prompt(pieces, length, positions, tail=None):
    """A retrieval prompt of exactly `length` tokens: the prepared head, the filler (repeated as needed) with
    one fact at each of `positions` (fractions of the filler), and the tail with the question (`tail`: another
    request after the same head). None when `length` cannot hold the head, tail and facts with filler between them.
    pieces: prepared["retrieval"]."""
    head, facts = pieces["head_ids"], pieces["facts"]
    tail = pieces["tail_ids"] if tail is None else tail
    plain = length - len(head) - len(tail) - sum(len(f["ids"]) for f in facts)
    if plain < len(facts) + 1:
        return None
    filler = pieces["filler_ids"]
    filler = (list(filler) * -(-plain // len(filler)))[:plain]
    out, last = list(head), 0
    for fraction, fact in zip(positions, facts):
        cut = max(last, min(plain, int(plain * float(fraction))))
        out += filler[last:cut] + list(fact["ids"])
        last = cut
    return out + filler[last:] + list(tail)


def curve_prompt(pieces, length, positions):
    """A long-context curve prompt of exactly `length` tokens: the retrieval prompt (retrieval_prompt) whose request
    asks for the facts and then for a summary of the archive (curve_tail_ids), so its reply is long enough to time
    the decode and is judged for retrieval too. None when `length` cannot hold it. pieces: prepared["retrieval"]."""
    return retrieval_prompt(pieces, length, positions, pieces["curve_tail_ids"])


def token_text(tokens, strings, stop_ids=()):
    """The text of `tokens` up to the first stop id, joined from the prepared per-token strings; a token with
    no ASCII string (or outside the table) becomes U+FFFD, so it can never complete an ASCII answer."""
    stops, out = set(stop_ids), []
    for token in tokens:
        if token in stops:
            break
        text = strings[token] if 0 <= token < len(strings) else None
        out.append(text if text is not None else "�")
    return "".join(out)


def render(value):
    lines = []
    if value["tokens"]:
        lines.append(f"max context {value['tokens']:,} tokens ({value['bound']} bound)")
        if value.get("requested"):
            lines.append(f"  requested    {value['requested']:,} tokens (settings.toml context_tokens)")
        if value.get("note"):
            lines.append(f"  note: {value['note']}")
    else:
        lines.append("max context unknown" + (f": {value['reason']}" if value["reason"] else ""))
    if value["model_limit"]:
        lines.append(f"  model limit  {value['model_limit']:,} tokens ({value['model_limit_source']})")
    basis = value.get("basis")
    if basis:
        lines.append(f"  device limit {value['device_limit']:,} tokens: {basis['kv_cache_bytes_available'] / 1e9:.2f} GB for the KV "
                     f"cache at {value['kv_bytes_per_token']:,} bytes per token ({value['kv_precision']})"
                     if value["device_limit"] is not None and value["kv_bytes_per_token"] else
                     "  device limit none: no cache grows with the context")
        lines.append(f"    {basis['memory_free_bytes'] / 1e9:.2f} GB free on {basis['device']} - {basis['vram_headroom_bytes'] / 1e9:.2f} "
                     f"headroom - {basis['reserve_bytes'] / 1e9:.2f} reserve - {basis['resident_weight_bytes'] / 1e9:.2f} weights read "
                     f"every pass - {basis['fixed_state_bytes'] / 1e9:.3f} fixed state"
                     + (f" - {basis['companion_drafter_bytes'] / 1e9:.2f} companion drafters"
                        if basis.get("companion_drafter_bytes") else ""))
        if basis["expert_bytes"]:
            lines.append(f"    experts {basis['expert_bytes'] / 1e9:.2f} GB, "
                         f"{basis.get('expert_bytes_in_device_memory_at_max', 0) / 1e9:.2f} GB of them in device memory at the max")
    if value["missing"]:
        lines.append("missing: " + "; ".join(value["missing"]))
    return "\n".join(lines)

