"""Timed requests and what a run reads from their timings: one generation, plain, sampled or replayed, timed by
phase; runs disturbed by other processes; summaries and rates."""
import hashlib
import json

from ..limits import headroom
from ..reports import metrics
from .chat import ChatEngine
from .journal import clock   # the round's clock: a resumed round's journaled requests keep their times


def _timed(call, prompt, output_tokens, expected=None, keep=()):
    deliveries = []

    def on_tokens(tokens):
        deliveries.append((clock(), list(tokens)))
    started = clock()
    result = call(on_tokens)
    tokens = list(result["tokens"])
    # An engine behind a chat API streams text: the server counts its tokens, and its reply may end early.
    served = result.get("served")
    streamed = [t for _, chunk in deliveries for t in chunk]
    if served is None and streamed != tokens:
        raise RuntimeError("tokens passed to on_tokens differ from the returned tokens")
    count = len(tokens) if served is None else int(served["output_tokens"])
    if count > output_tokens or count < output_tokens and served is None:
        raise RuntimeError(f"engine returned {count} tokens, expected {output_tokens} (stop tokens must be disabled)")
    if expected is not None and tokens != list(expected):
        diverge = next(i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b)
        raise RuntimeError(f"replay emitted a token different from the pinned continuation at position {diverge}")
    first = next(t for t, chunk in deliveries if chunk)
    prompt_tokens = len(prompt) if served is None else int(served["prompt_tokens"])
    sample = metrics.speed_sample(prompt_tokens, count, started, first, deliveries[-1][0], result.get("rounds"))
    sample["chunk_gaps"] = chunk_gaps([t for t, chunk in deliveries if chunk])
    if served is not None:
        sample["served"] = served
    sample["engine_timing"] = result.get("timing")
    sample["accepted"] = result.get("accepted")
    if isinstance(result.get("drafted"), list):
        sample["drafted"] = result["drafted"]
    # The decode's routed expert reads and those the device cache served: the hit rate the tiers price (headroom.tiers).
    if type(result.get("expert_reads")) is int and type(result.get("expert_hits")) is int             and 0 <= result["expert_hits"] <= result["expert_reads"]:
        sample["expert_reads"], sample["expert_hits"] = result["expert_reads"], result["expert_hits"]
    # Optional engine-measured times: the prefill and the first token from the engine's own clock (what the time to
    # first token holds beyond them is the server's, the tokenizer's, transfer and scheduling), and the decode's time
    # in verify passes and in draft steps (ms, whole request).
    for key in ("prefill_ms", "first_token_ms", "verify_ms", "draft_ms"):
        value = result.get(key)
        sample[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    sample.update({key: result[key] for key in keep if key in result})
    return tokens, sample


def chunk_gaps(arrivals):
    """The gaps between the streamed chunks of one request after its first token (protocol speed.stalls), from the
    host times the non-empty chunks arrived: their count and the longest and p99 gap in ms. A speculative engine sends
    several tokens per chunk, so these are chunk gaps, not token gaps. None with fewer than two chunks after the first
    token's: an engine that delivers its reply in one piece shows no stall."""
    gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    if len(gaps) < 2:
        return None
    return {"chunks": len(arrivals), "max_ms": max(gaps), "p99_ms": metrics.percentile(gaps, 0.99)}


def stalls(samples, round_ms):
    """A speed case's stalls over its measured samples (chunk_gaps): the longest gap and the largest p99 gap, in ms
    and over the case's median round_ms; None when no sample streamed enough chunks."""
    found = [s["chunk_gaps"] for s in samples if not s.get("warmup") and s.get("chunk_gaps")]
    if not found:
        return None
    longest, p99 = max(g["max_ms"] for g in found), max(g["p99_ms"] for g in found)
    return {"max_gap_ms": longest, "p99_gap_ms": p99, "chunks": min(g["chunks"] for g in found),
            "max_gap_rounds": longest / round_ms if round_ms else None,
            "p99_gap_rounds": p99 / round_ms if round_ms else None}


def timed_generate(engine, prompt, output_tokens, prompt_cache=False, logprobs=None):
    """prompt_cache: the request is judged on its text alone, so an engine behind a server may serve it from its
    prompt cache. logprobs: ask a served engine for that many alternatives per reply token; the sample then carries
    the reply in reply()'s shape as "reply" (or "reply_unmapped": why the server's tokens map to no ids)."""
    served = isinstance(engine, ChatEngine)
    options = {**({"prompt_cache": True} if prompt_cache and served else {}),
               **({"logprobs": logprobs} if logprobs and served else {})}
    # tool_calls: the calls a server parsed from its reply (protocol speed.tool_calls).
    return _timed(lambda on_tokens: engine.generate(list(prompt), output_tokens, on_tokens, **options), prompt,
                  output_tokens, keep=("tool_calls",) + (("reply", "reply_unmapped") if logprobs else ()))


def timed_plain(engine, prompt, output_tokens):
    """Plain one-token greedy decoding on a speculative engine that declares it (info "plain")."""
    return _timed(lambda on_tokens: engine.generate_plain(list(prompt), output_tokens, on_tokens), prompt, output_tokens)


def timed_sampled(engine, prompt, output_tokens, sampling, logprobs=None):
    """A sampled request (protocol sampled_speed): the engine's generate_sampled with the request's sampling fields
    (prepare.sampled_request); with logprobs, asking for that many alternatives of every emitted token. Returns
    (tokens, sample, the reported log-probabilities or None)."""
    replied = {}

    def call(on_tokens):
        extra = {"logprobs": logprobs} if logprobs else {}
        replied["result"] = engine.generate_sampled(list(prompt), output_tokens, on_tokens, dict(sampling), **extra)
        return replied["result"]
    tokens, sample = _timed(call, prompt, output_tokens)
    return tokens, sample, replied["result"].get("logprobs") if logprobs else None


def timed_replay(engine, prompt, continuation):
    """The engine's speculative loop on a fixed continuation: drafts are accepted while they match it and
    each round's correction is its next token, so every engine decodes the same text."""
    continuation = list(continuation)
    return _timed(lambda on_tokens: engine.replay(list(prompt), list(continuation), on_tokens), prompt,
                  len(continuation), expected=continuation)


class Disturbed(Exception):
    """A speed case's repetitions disagree far beyond any quiet engine's run-to-run spread: something else
    is using the device. The round stops instead of measuring the rest on a disturbed device."""


DISTURBED_MIN_REQUEST_MS = 100  # shorter requests vary with timer and scheduler jitter, not with the device
# Other processes' CPU load below which a disturbed case points at the engine itself rather than at the machine.
QUIET_HOST_CORES = 2.0


def check_disturbed(case_id, samples, limit, other_cores=None, key="decode_tps"):
    """Raise Disturbed when the measured `key` (decode tok/s, or round_ms for sampled repetitions, whose texts and
    so their tokens per round differ; at the median drafts per round where the rounds' widths vary,
    width_adjusted_rounds) of one case spreads (max - min over the median) by more than `limit`. Quiet
    rounds recorded at most about 10%; a disturbed one 37%. other_cores: CPU cores other processes kept busy during
    the case (HostLoad), named in the message."""
    measured = [s for s in samples if not s.get("warmup")]
    if not limit or not measured or metrics.median([s["request_ms"] for s in measured]) < DISTURBED_MIN_REQUEST_MS:
        return
    adjusted = width_adjusted_rounds(measured) if key == "round_ms" else None
    if key == "decode_tps" and all(s.get("reply_sha256") for s in measured) and len(replies(measured)) > 1:
        # Repetitions that decoded different replies differ in speed by their content, not by a disturbance: only
        # repetitions of the same reply are compared (protocol speed.output_varies).
        spread = max((headroom.spread([s[key] for s in group if s.get(key)]) or 0.0
                      for group in replies(measured).values() if len(group) > 1), default=None)
    else:
        spread = headroom.spread(adjusted if adjusted else [s[key] for s in measured if s.get(key)])
    if spread is not None and spread > limit:
        load = f"; other processes kept {other_cores:.1f} CPU cores busy" if other_cores is not None else ""
        what = ("decode" if key == "decode_tps" else
                "the time per target pass at the drafts each round made" if adjusted else "the time per target pass")
        if other_cores is not None and other_cores < QUIET_HOST_CORES:
            cause = ("the host was quiet, so the engine itself is the likely cause: reads of file-backed weights or "
                     "tables that miss the page cache (read-ahead, page faults), host threads, or expert misses that "
                     "change between requests. Time the case's repetitions inside the engine before measuring again")
        else:
            cause = ("the device was disturbed, or the engine varies on its own (cold reads of file-backed weights or "
                     "tables, host threads). Measure again when nothing else uses the device; if it varies again on a "
                     "quiet host, look inside the engine")
        raise Disturbed(f"speed {case_id}: {what} varied {spread:.0%} between repetitions (limit {limit:.0%}){load}; "
                        + cause + _memory_note())


def replies(samples):
    """The samples grouped by the reply they decoded (their `reply_sha256`, which the speed part records for every
    greedy request)."""
    groups = {}
    for sample in samples:
        groups.setdefault(sample.get("reply_sha256"), []).append(sample)
    return groups


def reply_sha256(tokens):
    """A greedy reply's identity: the SHA-256 of its token ids."""
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()


def width_adjusted_rounds(samples):
    """For an engine whose rounds draft a varying number of tokens (drafted telemetry: an adaptive budget that falls back
    to plain steps, or a width that follows acceptance), each sample's round_ms at the median drafts per round: its
    round_ms less slope x (its mean drafts per round - the median), the slope fitted over the samples (least squares,
    at least 0). A seed's mix of wide rounds and plain steps changes its time per round as its text does, which a
    disturbance check must not read as the device's. None with fewer than three samples, a sample without drafted
    counts, or the same mean drafts per round in every sample (round_ms already compares like with like)."""
    rows = [(sum(s["drafted"]) / len(s["drafted"]), s["round_ms"]) for s in samples
            if isinstance(s.get("drafted"), list) and s["drafted"] and s.get("round_ms")]
    if len(rows) < 3 or len(rows) < len(samples):
        return None
    widths = [w for w, _ in rows]
    if max(widths) - min(widths) < 1e-9:
        return None
    mean_w, mean_r = sum(widths) / len(rows), sum(r for _, r in rows) / len(rows)
    slope = max(0.0, sum((w - mean_w) * (r - mean_r) for w, r in rows) / sum((w - mean_w) ** 2 for w in widths))
    middle = metrics.median(widths)
    return [r - slope * (w - middle) for w, r in rows]


LOW_DEVICE_MEMORY = 1 << 30  # below it a full device may page its memory to the host (WDDM) or stall allocations


def _memory_note():
    """What a Disturbed message adds about the device's memory: its free bytes as nvidia-smi sees them, and when they
    are below LOW_DEVICE_MEMORY, that the engine's own memory may be the disturbance. Under Windows' WDDM a device
    past its memory pages to host RAM instead of failing, and the requests slow unevenly as on a shared device."""
    from .host import device_memory
    try:
        found = device_memory()
    except Exception:   # a probe never fails the run
        found = None
    if not found or found.get("free_bytes") is None:
        return ""
    free, total = found["free_bytes"], found.get("total_bytes")
    note = f" (device memory: {free / 1e9:.2f}" + (f" of {total / 1e9:.1f}" if total else "") + " GB free"
    if free < LOW_DEVICE_MEMORY:
        return note + (": nearly full, so the engine's own memory may be the disturbance; a device past its memory "
                       "pages to host RAM under Windows' WDDM and slows unevenly. Check its peak memory first)")
    return note + ")"


SUMMARY_KEYS = ("decode_tps", "prefill_tps", "ttft_ms", "request_ms", "tokens_per_round", "round_ms")


def _summary(samples):
    return {key: metrics.median([s[key] for s in samples if s[key] is not None]) for key in SUMMARY_KEYS}


def rate(value, digits):
    """A speed for the log; a reply streamed in one piece has no decode time, and one before any time no prefill."""
    return "unmeasured" if value is None else f"{value:.{digits}f} tok/s"


def decode_text(info, continuations):
    """Which text the decode KPIs are measured on: the pinned continuation when the data directory has
    one and the adapter can replay it, else each engine's own greedy output."""
    if continuations and info.get("replay") and info.get("speculative"):
        return {"source": "pinned", "sha256": continuations["sha256"], "pinned_by": continuations.get("engine")}
    reason = ("no pinned continuation in the data directory (crit eval pin)" if not continuations
              else "the adapter does not declare replay" if not info.get("replay") else "ordinary decoding")
    return {"source": "generated", "reason": reason}


def settling(samples, sample, rule, key="decode_tps"):
    """Whether `sample`, the request that would be a case's first measured repetition, is still warming up
    (protocol speed.steady_state): its `key` (decode tok/s, or round_ms for sampled repetitions) and the request's
    before it spread by more than rule["tolerance"], and the case's warmups are not yet done: fewer than
    rule["max_warmup_runs"], or fewer than rule["max_warmup_tokens"] tokens decoded in them,
    and in either case fewer than rule["max_warmup_requests"]. Such a request stands as one more warmup. A case's
    first request, a request after a measured one, and requests too short to time stand as measured."""
    if not rule or not samples or any(not s.get("warmup") for s in samples):
        return False
    warm = [s for s in samples if s.get("warmup")]
    tokens = sum(s.get("output_tokens") or 0 for s in warm)
    if len(warm) >= int(rule.get("max_warmup_requests", rule["max_warmup_runs"])):
        return False
    if len(warm) >= int(rule["max_warmup_runs"]) and tokens >= int(rule.get("max_warmup_tokens", 0)):
        return False
    before = samples[-1]
    if min(before.get("request_ms") or 0, sample.get("request_ms") or 0) < DISTURBED_MIN_REQUEST_MS:
        return False
    found = headroom.spread([before.get(key), sample.get(key)])
    return found is not None and found > float(rule["tolerance"])


def steady_state(samples, rule, added, key="decode_tps"):
    """A case's warmup as protocol speed.steady_state recorded it: its warmups, how many of them the drift added,
    whether its first measured repetition agreed with the request before it (None without one), and the drift: the
    first measured repetition's `key` over the case's first request's, less one (None when either lacks it)."""
    warm = [s for s in samples if s.get("warmup")]
    first = next((i for i, s in enumerate(samples) if not s.get("warmup")), None)
    steady, drift = None, None
    if rule and warm and first is not None and first > 0:
        found = headroom.spread([samples[first - 1].get(key), samples[first].get(key)])
        steady = None if found is None else found <= float(rule["tolerance"])
        if samples[0].get(key) and samples[first].get(key):
            drift = round(samples[first][key] / samples[0][key] - 1, 4)
    return {"warmup_runs": len(warm), "added": added, "steady": steady, "drift": drift, "measure": key}


def disagree(samples, outputs, tolerance):
    """Whether a case's measured repetitions disagree (protocol speed.extra_repetition): decode tok/s or time to
    first token spread by more than `tolerance`, or the outputs differ. With one measured repetition the warmup
    before it is compared in its place."""
    measured = [s for s in samples if not s.get("warmup")]
    if len(measured) == 1:
        measured = [s for s in samples if s.get("warmup")][-1:] + measured
    spreads = [headroom.spread([s[key] for s in measured if s.get(key)]) for key in ("decode_tps", "ttft_ms")]
    return any(spread is not None and spread > tolerance for spread in spreads) or any(o != outputs[0] for o in outputs)
