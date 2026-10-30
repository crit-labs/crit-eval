"""The context parts of a run: the required context, the engine's speed across it, retrieval of facts placed in
long prompts, and the check that a lower context is not faster."""
from ..limits import headroom
from .deadline import Hung
from ..limits.context import (curve_points, curve_prompt, lower_context, max_context, retrieval_points,
                              retrieval_prompt, token_text)
from .adapters import open_engine
from .speed import run_speed
from .timing import Disturbed, _summary, timed_generate


def context_requirement(model, device, context_tokens, vram_headroom_bytes, settings, requested=None, basis=None):
    """The context every engine is built and measured for: the given one (a run's kv_tokens, with the run's basis:
    requested, bound, limits) or crit-eval's max context for the model on the device, capped at the requested context.
    `tokens` is None when neither is known."""
    if context_tokens:
        return {"tokens": int(context_tokens), "source": "given", "missing": [], "reason": None,
                **{key: (basis or {})[key] for key in ("requested", "bound", "note", "model_limit", "device_limit")
                   if (basis or {}).get(key) is not None}}
    return {**max_context(model, device, vram_headroom_bytes, settings, requested), "source": "computed"}


CONTEXT_TEXT = {"source": "generated", "reason": "each point answers the retrieval request in the engine's own words"}


def judged_retrieval(tokens, prepared, prompt, point):
    """A reply's retrieval verdict (protocol context.retrieval): every fact's answer in its text up to the first stop
    token, decoded with the prepared token strings."""
    pieces = prepared["retrieval"]
    text = token_text(tokens, pieces["token_text"], prepared["stop_ids"])
    found = {fact["id"]: fact["answer"] in text for fact in pieces["facts"]}
    return {"status": "measured", "context": point, "prompt_tokens": len(prompt), "found": found,
            "passed": all(found.values()), "text": text[:2000], "tokens": tokens}


def run_context(engine, info, prepared, settings, required, log, reply_logprobs=None):
    """One greedy request per curve point, each filling the context exactly (prompt plus output equal the point).
    The prompt is the retrieval check's archive, filler text with the retrieval facts in it, and a request for the
    facts and then a summary of the archive (curve_prompt): the point times that reply, the engine's own, and judges
    retrieval on it, so a long prompt is prefilled once for both. A request the engine cannot serve is recorded as
    failed. With reply_logprobs, the first point's request asks a served engine for that many alternatives per reply
    token, so its reply serves the honesty check's long reply (protocol quality.honesty long_reply). Returns (points,
    the decode text, the retrieval verdict of every point the curve served or failed, the long reply: {"point",
    "prompt_ids", "reply"} or {"point", "reason"} when there is none)."""
    pieces, rule = prepared["retrieval"], settings["retrieval"]
    n = int(settings["output_tokens"])
    points, asked = {}, {}
    curve = curve_points(required, settings)
    long_reply = {"point": curve[0], "reason": "the curve's first point was not measured"}
    for point in curve:
        if point > info["max_context"]:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
            log(f"context {point:,}: skipped (max_context {info['max_context']:,})")
            continue
        prompt = curve_prompt(pieces, point - n, rule["positions"])
        if prompt is None:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"no room for the curve's prompt before {n} output tokens"}
            continue
        samples = []
        try:
            for _ in range(int(settings.get("repetitions", 1))):
                if point == curve[0] and reply_logprobs:
                    try:
                        tokens, sample = timed_generate(engine, prompt, n, logprobs=reply_logprobs)
                    except Hung:
                        raise
                    except RuntimeError as error:   # a server that cannot stream log-probabilities: the point stands
                        log(f"context {point:,}: the request with log-probabilities failed ({error}); asking without")
                        long_reply = {"point": point, "reason": f"the reply with log-probabilities failed: {error}"}
                        tokens, sample = timed_generate(engine, prompt, n)
                    else:
                        reply, unmapped = sample.pop("reply", None), sample.pop("reply_unmapped", None)
                        long_reply = ({"point": point, "prompt_ids": list(prompt), "reply": reply} if reply else
                                      {"point": point, "reason": unmapped or "the engine returned no log-probabilities"})
                else:
                    tokens, sample = timed_generate(engine, prompt, n)
                samples.append(sample)
        except Hung:
            raise
        except Exception as error:  # an engine that cannot hold the context fails here, however it fails
            error = f"{type(error).__name__}: {error}"[:2000]
            points[str(point)] = {"status": "failed", "context": point, "error": error}
            asked[point] = {"status": "failed", "context": point, "error": error}
            log(f"context {point:,}: failed ({error})")
            continue
        summary = _summary(samples)
        points[str(point)] = {"status": "measured", "context": point, "prompt_tokens": len(prompt), "output_tokens": n,
                              **summary, "samples": samples}
        asked[point] = {**judged_retrieval(tokens, prepared, prompt, point), "ttft_ms": summary["ttft_ms"]}
        log(f"context {point:,}: decode {summary['decode_tps'] or 0:.1f} tok/s, first token after {summary['ttft_ms']:,.0f} ms, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}; retrieval "
            + ", ".join(f"{k} {'found' if v else 'MISSING'}" for k, v in asked[point]["found"].items()))
    return points, dict(CONTEXT_TEXT), asked, long_reply


def retrieval_point(engine, info, prepared, rule, prompt, point, log):
    """One retrieval request (protocol context.retrieval): passes when the output up to its first stop token
    contains every fact's answer verbatim; one the engine cannot serve fails. It is judged on its text alone, so it
    may be served from a prompt cache."""
    pieces, n = prepared["retrieval"], int(rule["output_tokens"])
    if point > info["max_context"]:
        log(f"retrieval {point:,}: not served (max_context {info['max_context']:,})")
        return {"status": "failed", "context": point,
                "error": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
    try:
        tokens, sample = timed_generate(engine, prompt, n, prompt_cache=True)
    except Hung:
        raise
    except Exception as error:  # an engine that cannot hold the context fails here, however it fails
        log(f"retrieval {point:,}: failed ({type(error).__name__}: {error})")
        return {"status": "failed", "context": point, "error": f"{type(error).__name__}: {error}"[:2000]}
    found = judged_retrieval(tokens, prepared, prompt, point)
    log(f"retrieval {point:,}: " + ", ".join(f"{k} {'found' if v else 'MISSING'}" for k, v in found["found"].items()))
    return {**found, "ttft_ms": sample["ttft_ms"], "cached_tokens": (sample.get("served") or {}).get("cached_tokens")}


def run_retrieval(engine, info, prepared, settings, required, log, from_curve=None):
    """Facts near the start, middle and end of filler text, asked for at the end (protocol context.retrieval):
    from_curve holds the verdicts of the points the long-context curve asked (run_context); every other point below
    the required context gets a request of its own, and the required one fails when the curve did not reach it."""
    from_curve = from_curve or {}
    pieces, rule = prepared.get("retrieval"), settings["retrieval"]
    if not pieces:
        log("retrieval: not in the prepared data (prepare it again with crit eval prepare into a new data directory)")
        return {"status": "missing", "reason": "the prepared data has no retrieval prompts; prepare it again"}
    n, points = int(rule["output_tokens"]), {}
    for point in sorted(set(retrieval_points(required, rule)) | set(from_curve)):   # every curve point too
        if point in from_curve:
            points[str(point)] = from_curve[point]
            continue
        if point == required:
            points[str(point)] = {"status": "failed", "context": point,
                                  "error": "the long-context curve did not reach the required context"}
            continue
        prompt = retrieval_prompt(pieces, point - n, rule["positions"])
        if prompt is None:
            points[str(point)] = {"status": "skipped", "context": point, "reason": "too short to hold the facts"}
            continue
        points[str(point)] = retrieval_point(engine, info, prepared, rule, prompt, point, log)
    judged_points = [p for p in points.values() if p["status"] != "skipped"]
    return {"status": "measured", "points": points,
            "passed": bool(judged_points) and all(p.get("passed") for p in judged_points)}


def _spread(case, key):
    return headroom.spread([s[key] for s in case.get("samples", []) if not s.get("warmup") and s.get(key)]) or 0.0


def compare_lower(at_max, at_lower, lower, settings):
    """Each speed case at a lower context against the same case at the required context. A case is slower
    when its decode falls, or its first token comes later, by more than the tolerance: the protocol's, or the
    case's own run-to-run spread in either measurement when that is larger."""
    cases, slower = {}, {}
    for case_id, high in at_max["cases"].items():
        low = at_lower["cases"].get(case_id) or {}
        if high.get("status") != "measured" or low.get("status") != "measured":
            continue
        row = {"decode_tps": low["decode_tps"], "decode_tps_at_required": high["decode_tps"],
               "ttft_ms": low["ttft_ms"], "ttft_ms_at_required": high["ttft_ms"],
               "decode_tolerance": max(float(settings["tolerance"]), _spread(high, "decode_tps"), _spread(low, "decode_tps")),
               "ttft_tolerance": max(float(settings["tolerance"]), _spread(high, "ttft_ms"), _spread(low, "ttft_ms")),
               "host_other_cpu_cores": low.get("host_other_cpu_cores"),
               "host_other_cpu_cores_at_required": high.get("host_other_cpu_cores")}
        row["decode_change"] = (low["decode_tps"] / high["decode_tps"] - 1
                                if low.get("decode_tps") and high.get("decode_tps") else None)
        row["ttft_change"] = low["ttft_ms"] / high["ttft_ms"] - 1 if low.get("ttft_ms") and high.get("ttft_ms") else None
        # The engine's own first-token time at both contexts, when it reports one: a change outside the engine
        # (HTTP, templating, host launches) reads differently from a slower prefill.
        parts, parts_at_required = headroom.ttft_parts(low), headroom.ttft_parts(high)
        if "outside_engine_ms" in parts and "outside_engine_ms" in parts_at_required:
            engine, engine_at_required = (p.get("engine_first_token_ms", p.get("engine_prefill_ms"))
                                          for p in (parts, parts_at_required))
            row.update(engine_ttft_ms=engine, engine_ttft_ms_at_required=engine_at_required,
                       outside_engine_ms=parts["outside_engine_ms"],
                       outside_engine_ms_at_required=parts_at_required["outside_engine_ms"])
        kinds = []
        if row["decode_change"] is not None and row["decode_change"] < -row["decode_tolerance"]:
            kinds.append("decode")
        if row["ttft_change"] is not None and row["ttft_change"] > row["ttft_tolerance"]:
            kinds.append("ttft")
        if kinds:
            slower[case_id] = kinds
        cases[case_id] = row
    unmatched = [cid for cid, c in at_max["cases"].items() if c.get("status") == "measured" and cid not in cases]
    return {"context": lower, "cases": cases, "slower": slower, "unmeasured": unmatched,
            "decode_score": at_lower.get("decode_score"), "decode_score_at_required": at_max.get("decode_score"),
            "ttft_score_ms": at_lower.get("ttft_score_ms"), "ttft_score_ms_at_required": at_max.get("ttft_score_ms"),
            "decode_text": at_lower.get("decode_text"),
            "passed": bool(cases) and not slower and not unmatched}


def run_lower_context(module, model, config, prepared, protocol, required, warmups, repetitions, log,
                      continuations=None, disturbed_spread=None, exact_prompts=False):
    """The speed cases on an engine opened at a lower context. It runs just before the engine built for the
    required context, so the two speed runs sit side by side and ahead of the long-context curve's heat.
    Lowering the context frees device memory and must never make a case slower (compare_lower)."""
    settings = protocol["context"]
    longest = max(len(case["prompt_ids"]) + case["output_tokens"] for case in prepared["speed"])
    lower = lower_context(required, longest, settings)
    if lower is None:
        return {"context": None, "passed": None,
                "reason": f"the required {required:,} tokens leave no lower context that holds the longest speed case ({longest:,})"}
    log(f"lower context: opening the engine at {lower:,} tokens")
    try:
        engine, info = open_engine(module, model, config, speculative=True, context_tokens=lower,
                                   exact_prompts=exact_prompts)
        try:
            measured = run_speed(engine, info, prepared, protocol, warmups, repetitions,
                                 lambda message: log("lower context: " + message), continuations, disturbed_spread)
        finally:
            engine.close()
    except (Disturbed, Hung):
        raise
    except RuntimeError as error:
        if "rendered the prompt as" in str(error):
            raise   # the server's template, not the lower context: the round cannot measure quality
        log(f"lower context: failed ({type(error).__name__}: {error})")
        return {"context": lower, "passed": False, "error": f"{type(error).__name__}: {error}"[:2000]}
    except Exception as error:  # an engine that cannot run at a lower context fails the gate, however it fails
        log(f"lower context: failed ({type(error).__name__}: {error})")
        return {"context": lower, "passed": False, "error": f"{type(error).__name__}: {error}"[:2000]}
    return {"context": lower, "engine_max_context": info["max_context"], "speed": measured}
