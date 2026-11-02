"""Several GPUs for one engine (a `crit eval probe --devices` result, protocol context.devices): the members' memory
pooled for the context math, and a decode pass's floor with the weights split across them.

A device file of one GPU has no "devices"; one of several keeps the first member's figures at its top and lists every
member's under "devices", with "links": each ordered pair's measured copy rate. The floors here are physical: they
price the bytes and the hand-overs a split cannot avoid, not a particular engine's schedule.
"""


def members(device):
    """The GPUs of a device file: its "devices" when it lists two or more, else the file itself (one GPU)."""
    listed = (device or {}).get("devices") or []
    return listed if len(listed) >= 2 else ([device] if device else [])


def indices(device):
    """The members' device indices (nvidia-smi's), or None for a single GPU."""
    found = members(device)
    return [int(m.get("index", i)) for i, m in enumerate(found)] if len(found) >= 2 else None


def pooled(device, vram_headroom_bytes, reserve_bytes):
    """The device file with its members' memory together, for max_context: the free memory of all members less the
    headroom and reserve each of them keeps (max_context takes one member's off itself), and their total. A single GPU's
    file is returned as it is."""
    found = members(device)
    if len(found) < 2:
        return device
    keep = (len(found) - 1) * (int(vram_headroom_bytes) + int(reserve_bytes))
    free = [m.get("memory_free_bytes") for m in found]
    total = [m.get("memory_total_bytes") for m in found]
    return {**device,
            "memory_free_bytes": max(0, sum(free) - keep) if all(isinstance(v, int) for v in free) else None,
            "memory_total_bytes": sum(total) if all(isinstance(v, int) for v in total) else None,
            "pooled_members": len(found)}


def link_gbs(device, a, b):
    """The measured copy rate from member index a to b in GB/s, or None."""
    for link in (device or {}).get("links") or []:
        if (link.get("from"), link.get("to")) == (a, b):
            return link.get("gbs") or None
    return None


def split_floor(bytes_per_pass, device, layers, hidden, activation_bytes=2):
    """One decode pass of one token with `bytes_per_pass` split across the members, in ms, by the faster of the two
    splits every multi-GPU engine is one of (or a mix of):

    tensor: every member reads its share of every layer, sized to its bandwidth, so the bytes take bytes over the
        bandwidths' sum; each layer's two partial outputs (attention, MLP) are summed across members, an all-reduce of
        `hidden` activations that a ring moves 2 (N - 1) / N of over its slowest link, twice per layer;
    pipeline: members hold consecutive layers, sized to their bandwidth, and a token runs through them in turn, so the
        bytes take N x bytes over the bandwidths' sum, plus N - 1 hand-overs of one token's activations.

    Returns {"members", "tensor_ms", "pipeline_ms", "floor_ms", "split", "bandwidth_gbs" (bytes over the floor: the
    bandwidth a single device would need for it), "link_gbs", "missing"}; None for one GPU."""
    found = members(device)
    if len(found) < 2:
        return None
    n = len(found)
    bandwidth = [m.get("bandwidth_gbs") for m in found]
    ids = [int(m.get("index", i)) for i, m in enumerate(found)]
    rates = [link_gbs(device, a, b) for a in ids for b in ids if a != b]
    missing = [name for name, ok in (("every member's bandwidth (crit eval probe --devices)", all(bandwidth)),
                                     ("the links between members (crit eval probe --devices)", rates and all(rates)),
                                     ("the model's layers and hidden size", layers and hidden),
                                     ("the bytes per pass", bytes_per_pass)) if not ok]
    if missing:
        return {"members": n, "missing": missing}
    slowest = min(rates)
    reads = bytes_per_pass / (sum(bandwidth) * 1e9) * 1000
    vector = hidden * activation_bytes
    all_reduce = 2 * (n - 1) / n * vector / (slowest * 1e9) * 1000
    tensor = reads + 2 * layers * all_reduce
    pipeline = n * reads + (n - 1) * vector / (slowest * 1e9) * 1000
    floor, split = min((tensor, "tensor"), (pipeline, "pipeline"))
    return {"members": n, "tensor_ms": tensor, "pipeline_ms": pipeline, "floor_ms": floor, "split": split,
            "bandwidth_gbs": bytes_per_pass / (floor / 1000) / 1e9, "link_gbs": slowest, "missing": []}


def combined(device, bytes_per_pass, layers, hidden):
    """The device file priced as one device for the floors: the bandwidth of the faster split (split_floor) and each
    format's GEMM and matrix-instruction rate summed over the members that measured it (a split's prompt arithmetic
    runs on all of them at once), with the split under "device_set". A single GPU's file is returned as it is."""
    found = members(device)
    if len(found) < 2:
        return device
    split = split_floor(bytes_per_pass, device, layers, hidden)
    out = {**device, "device_set": {"members": [{"index": m.get("index"), "device": m.get("device")} for m in found],
                                    "split": split}}
    if split and not split["missing"]:
        out["bandwidth_gbs"] = split["bandwidth_gbs"]
    for key in ("tflops", "mma_tflops"):
        rates = [m.get(key) or {} for m in found]
        formats = set.intersection(*(set(r) for r in rates)) if rates else set()
        if formats:
            out[key] = {f: sum(r[f] for r in rates) for f in formats}
    return out
