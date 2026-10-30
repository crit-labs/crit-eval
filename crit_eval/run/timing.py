"""Timed requests and what a run reads from their timings: one generation, plain, sampled or replayed, timed by
phase; runs disturbed by other processes; summaries and rates."""
import time

from ..limits import headroom
from ..reports import metrics
from .chat import ChatEngine


def _timed(call, prompt, output_tokens, expected=None, keep=()):
    deliveries = []

    def on_tokens(tokens):
        deliveries.append((time.perf_counter(), list(tokens)))
    started = time.perf_counter()
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
    if served is not None:
        sample["served"] = served
    sample["engine_timing"] = result.get("timing")
    sample["accepted"] = result.get("accepted")
    if isinstance(result.get("drafted"), list):
        sample["drafted"] = result["drafted"]
    # Optional engine-measured times: the prefill and the first token from the engine's own clock (what the time to
    # first token holds beyond them is the server's, the tokenizer's, transfer and scheduling), and the decode's time
    # in verify passes and in draft steps (ms, whole request).
    for key in ("prefill_ms", "first_token_ms", "verify_ms", "draft_ms"):
        value = result.get(key)
        sample[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    sample.update({key: result[key] for key in keep if key in result})
    return tokens, sample


def timed_generate(engine, prompt, output_tokens, prompt_cache=False, logprobs=None):
    """prompt_cache: the request is judged on its text alone, so an engine behind a server may serve it from its
    prompt cache. logprobs: ask a served engine for that many alternatives per reply token; the sample then carries
    the reply in reply()'s shape as "reply" (or "reply_unmapped": why the server's tokens map to no ids)."""
    served = isinstance(engine, ChatEngine)
    options = {**({"prompt_cache": True} if prompt_cache and served else {}),
               **({"logprobs": logprobs} if logprobs and served else {})}
    return _timed(lambda on_tokens: engine.generate(list(prompt), output_tokens, on_tokens, **options), prompt,
                  output_tokens, keep=("reply", "reply_unmapped") if logprobs else ())


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


def disagree(samples, outputs, tolerance):
    """Whether a case's measured repetitions disagree (protocol speed.extra_repetition): decode tok/s or time to
    first token spread by more than `tolerance`, or the outputs differ."""
    measured = [s for s in samples if not s.get("warmup")]
    spreads = [headroom.spread([s[key] for s in measured if s.get(key)]) for key in ("decode_tps", "ttft_ms")]
    return any(spread is not None and spread > tolerance for spread in spreads) or any(o != outputs[0] for o in outputs)
