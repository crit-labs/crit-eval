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


def check_disturbed(case_id, samples, limit, other_cores=None, key="decode_tps"):
    """Raise Disturbed when the measured `key` (decode tok/s, or round_ms for sampled repetitions, whose texts and
    so their tokens per round differ) of one case spreads (max - min over the median) by more than `limit`. Quiet
    rounds recorded at most about 10%; a disturbed one 37%. other_cores: CPU cores other processes kept busy during
    the case (HostLoad), named in the message."""
    measured = [s for s in samples if not s.get("warmup")]
    if not limit or not measured or metrics.median([s["request_ms"] for s in measured]) < DISTURBED_MIN_REQUEST_MS:
        return
    spread = headroom.spread([s[key] for s in measured if s.get(key)])
    if spread is not None and spread > limit:
        load = f"; other processes kept {other_cores:.1f} CPU cores busy" if other_cores is not None else ""
        what = "decode" if key == "decode_tps" else "the time per target pass"
        raise Disturbed(f"speed {case_id}: {what} varied {spread:.0%} between repetitions (limit {limit:.0%}){load}; "
                        "the device was disturbed. Measure again when nothing else uses it")


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
