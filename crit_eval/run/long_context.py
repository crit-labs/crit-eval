"""The context parts of a run: the required context, the engine's speed across it, retrieval of facts placed in
long prompts, and the check that a lower context is not faster."""
from ..limits import headroom
from .deadline import Hung
from ..limits.context import curve_points, lower_context, max_context, retrieval_points, token_text
from .adapters import close_engine, open_engine
from .jitter import curve_point
from .requests import case_request, curve_case, curve_prompt_at, retrieval_case, send
from .speed import cannot_sample, run_speed
from .timing import Disturbed, _summary
from ..quality.prepare import sampled_request

# A curve point's decode fields: the sampled request's (decode speed is sampled decode); the greedy request's stay
# beside them as a comparison.
DECODE_KEYS = ("decode_tps", "tokens_per_round", "round_ms")


def context_requirement(model, device, context_tokens, vram_headroom_bytes, settings, requested=None, basis=None,
                        spec=None):
    """The context every engine is built and measured for: the given one (a run's kv_tokens, with the run's basis:
    requested, bound, limits) or crit-eval's max context for the model on the device, capped at the requested context
    (sized by crit's EngineSpec of the engine where given: limits.context.max_context's spec). `tokens` is None when
    neither is known."""
    if context_tokens:
        return {"tokens": int(context_tokens), "source": "given", "missing": [], "reason": None,
                **{key: (basis or {})[key] for key in ("requested", "bound", "note", "model_limit", "device_limit")
                   if (basis or {}).get(key) is not None}}
    return {**max_context(model, device, vram_headroom_bytes, settings, requested, spec=spec), "source": "computed"}


CONTEXT_TEXT = {"source": "generated", "reason": "each point answers the retrieval request in the engine's own words"}


def thinking_tokens(engine, rule):
    """The reasoning budget a retrieval request carries (context.retrieval.thinking_tokens) on an engine whose package
    ends reasoning at one (chat-thinking-budget-v1), else None."""
    return int(rule.get("thinking_tokens") or 0) or None if getattr(engine, "thinking_budget", False) else None


def judged_retrieval(tokens, prepared, prompt, point, sample=None):
    """A reply's retrieval verdict (protocol context.retrieval): every fact's answer in its text up to the first stop
    token, decoded with the prepared token strings; a reply that reasoned is judged on its answer (the sample's
    answer_tokens: the reply without its reasoning)."""
    pieces = prepared["retrieval"]
    answer = (sample or {}).get("answer_tokens")
    text = token_text(answer if answer is not None else tokens, pieces["token_text"], prepared["stop_ids"])
    found = {fact["id"]: fact["answer"] in text for fact in pieces["facts"]}
    return {"status": "measured", "context": point, "prompt_tokens": len(prompt), "found": found,
            "passed": all(found.values()), "text": text[:2000], "tokens": tokens,
            **({"judged": "answer"} if answer is not None else {})}


def run_context(engine, info, prepared, settings, required, log, reply_logprobs=None):
    """One greedy request per curve point, each filling the context exactly (prompt plus output equal the point),
    then one sampled request on the same prompt at the prepared sampling (seed 1), which an engine behind a server
    may serve from its prompt cache. The prompt is the retrieval check's archive, filler text with the retrieval facts
    in it, and a request for the facts and then a summary of the archive (curve_prompt): the greedy reply, the
    engine's own, times the point's first token and has retrieval judged on it, so a long prompt is prefilled once
    for both; the sampled reply times its decode (decode speed is sampled decode: DECODE_KEYS), and the greedy
    decode stays beside it as `greedy`, a comparison only. An engine that cannot sample has the point's decode
    unmeasured (`sampled_missing`). A request the engine cannot serve is recorded as
    failed. With reply_logprobs, the first point's request asks a served engine for that many alternatives per reply
    token, so its reply serves the honesty check's long reply (protocol quality.honesty long_reply). With settings'
    jitter ({"seed", "fraction"}, protocol jitter) a point below the requirement fills its drawn share of the point
    instead (filled_context), the requirement itself exactly. Returns (points,
    the decode text, the retrieval verdict of every point the curve served or failed, the long reply: {"point",
    "prompt_ids", "reply"} or {"point", "reason"} when there is none)."""
    pieces, rule = prepared["retrieval"], settings["retrieval"]
    n = int(settings["output_tokens"])
    jitter = settings.get("jitter")
    points, asked = {}, {}
    curve = curve_points(required, settings)
    long_reply = {"point": curve[0] if curve else None, "reason": "the curve's first point was not measured"}
    unsampled = cannot_sample(engine, prepared)
    for point in curve:
        if point > info["max_context"]:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
            log(f"context {point:,}: skipped (max_context {info['max_context']:,})")
            continue
        filled = curve_point(point, required, jitter)
        # The point's own request (protocol requests.cases), its prompt filling the drawn share of the point
        # (protocol requests.overrides.context).
        case = curve_case(pieces, settings, point)
        prompt = curve_prompt_at(pieces, settings, filled) if case is not None else None

        def point_request(**overrides):
            return case_request(case, "context", None, prompt_ids=prompt, **overrides)
        if prompt is None:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": f"no room for the curve's prompt before {n} output tokens"}
            continue
        samples, judged = [], None
        try:
            for _ in range(int(settings.get("repetitions", 1))):
                if point == curve[0] and reply_logprobs:
                    try:
                        tokens, sample, _ = send(engine, point_request(logprobs=reply_logprobs))
                    except Hung:
                        raise
                    except RuntimeError as error:   # a server that cannot stream log-probabilities: the point stands
                        log(f"context {point:,}: the request with log-probabilities failed ({error}); asking without")
                        long_reply = {"point": point, "reason": f"the reply with log-probabilities failed: {error}"}
                        tokens, sample, _ = send(engine, point_request())
                    else:
                        budget = thinking_tokens(engine, rule)
                        if budget:   # the request with log-probabilities takes no reasoning budget: one that does
                            judged = send(engine, point_request(prompt_cache=True, thinking_tokens=budget))[:2]
                        reply, unmapped = sample.pop("reply", None), sample.pop("reply_unmapped", None)
                        # scored after the prompt the server rendered (ChatEngine.served_ids), not the frame's ids
                        long_reply = ({"point": point, "prompt_ids": reply.pop("prompt_ids", None) or list(prompt),
                                       "reply": reply} if reply else
                                      {"point": point, "reason": unmapped or "the engine returned no log-probabilities"})
                else:
                    tokens, sample, _ = send(engine, point_request(thinking_tokens=thinking_tokens(engine, rule)))
                samples.append(sample)
            sampled, missing = None, unsampled
            if unsampled is None:
                try:
                    _, sampled, _ = send(engine, point_request(sampling=sampled_request(prepared["sampling"], 1),
                                                       prompt_cache=True))
                except Hung:
                    raise
                except Exception as error:   # the greedy request stands: its first token and retrieval
                    missing = f"the sampled request failed: {type(error).__name__}: {error}"[:2000]
                    log(f"context {point:,}: {missing}")
        except Hung:
            raise
        except Exception as error:  # an engine that cannot hold the context fails here, however it fails
            error = f"{type(error).__name__}: {error}"[:2000]
            points[str(point)] = {"status": "failed", "context": point, "error": error}
            asked[point] = {"status": "failed", "context": point, "error": error}
            log(f"context {point:,}: failed ({error})")
            continue
        summary = _summary(samples)
        greedy = {key: summary[key] for key in DECODE_KEYS}
        summary.update({key: (sampled or {}).get(key) for key in DECODE_KEYS})
        points[str(point)] = {"status": "measured", "context": point, "filled_context": filled,
                              "prompt_tokens": len(prompt), "output_tokens": n, **summary,
                              "greedy": {**greedy, "comparison_only": "greedy decode: comparison only, never a speed"},
                              **({"sampled_sample": sampled} if sampled else {"sampled_missing": missing}),
                              "samples": samples}
        retrieval = (judged_retrieval(judged[0], prepared, prompt, point, judged[1]) if judged else
                     judged_retrieval(tokens, prepared, prompt, point, samples[-1]))
        asked[point] = {**retrieval, "ttft_ms": summary["ttft_ms"],
                        **({"judged_request": "its own, with the reasoning budget"} if judged else {})}
        log(f"context {point:,}" + (f" (filled {filled:,})" if filled != point else "")
            + (f": sampled decode {summary['decode_tps']:.1f} tok/s" if summary["decode_tps"] else
               ": sampled decode unmeasured")
            + f", first token after {summary['ttft_ms']:,.0f} ms, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}; retrieval "
            + ", ".join(f"{k} {'found' if v else 'MISSING'}" for k, v in asked[point]["found"].items()))
    return points, dict(CONTEXT_TEXT), asked, long_reply


def retrieval_point(engine, info, prepared, rule, case, point, log):
    """One retrieval request (protocol context.retrieval), the point's case (retrieval_case): passes when the output
    up to its first stop token contains every fact's answer verbatim; one the engine cannot serve fails. It is judged
    on its text alone, so it may be served from a prompt cache."""
    n = int(rule["output_tokens"])
    if point > info["max_context"]:
        log(f"retrieval {point:,}: not served (max_context {info['max_context']:,})")
        return {"status": "failed", "context": point,
                "error": f"the engine's max_context {info['max_context']:,} is below {point:,}"}
    budget = thinking_tokens(engine, rule)   # its output_tokens follow the reasoning budget
    # The point's own request, allowed a prompt cache and a reasoning budget (protocol requests.overrides.retrieval).
    request = case_request(case, "retrieval", None, output_tokens=n + (budget or 0), prompt_cache=True,
                           thinking_tokens=budget)
    prompt = request["prompt_ids"]
    try:
        tokens, sample, _ = send(engine, request)
    except Hung:
        raise
    except Exception as error:  # an engine that cannot hold the context fails here, however it fails
        log(f"retrieval {point:,}: failed ({type(error).__name__}: {error})")
        return {"status": "failed", "context": point, "error": f"{type(error).__name__}: {error}"[:2000]}
    found = judged_retrieval(tokens, prepared, prompt, point, sample)
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
    points = {}
    for point in sorted(set(retrieval_points(required, rule)) | set(from_curve)):   # every curve point too
        if point in from_curve:
            points[str(point)] = from_curve[point]
            continue
        if point == required and settings.get("requirement") is False:
            points[str(point)] = {"status": "skipped", "context": point,
                                  "reason": "the required context is served by the certification only (protocol stages)"}
            continue
        if point == required:
            points[str(point)] = {"status": "failed", "context": point,
                                  "error": "the long-context curve did not reach the required context"}
            continue
        case = retrieval_case(pieces, rule, point)
        if case is None:
            points[str(point)] = {"status": "skipped", "context": point, "reason": "too short to hold the facts"}
            continue
        points[str(point)] = retrieval_point(engine, info, prepared, rule, case, point, log)
    judged_points = [p for p in points.values() if p["status"] != "skipped"]
    return {"status": "measured", "points": points,
            "passed": bool(judged_points) and all(p.get("passed") for p in judged_points)}


def _spread(case, key):
    return headroom.spread([s[key] for s in case.get("samples", []) if not s.get("warmup") and s.get(key)]) or 0.0


def compare_lower(at_max, at_lower, lower, settings):
    """Each speed case at a lower context against the same case at the required context. A case is slower
    when its decode falls, or its first token comes later, by more than the tolerance: the protocol's, or the
    case's own run-to-run spread in either measurement when that is larger. Where the engine reports its own
    first-token time at both contexts, the first token is judged on it (`ttft_judged` "engine"): the time outside
    the engine (HTTP, the server's templating and tokenizer, a reopened host's cold caches) is the same server at
    both contexts and says nothing about what the lower context did to the engine; `ttft_change` keeps the whole."""
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
        judged = row["ttft_change"]
        row["ttft_judged"] = "whole"
        if row.get("engine_ttft_ms") and row.get("engine_ttft_ms_at_required"):
            row["engine_ttft_change"] = judged = row["engine_ttft_ms"] / row["engine_ttft_ms_at_required"] - 1
            row["ttft_judged"] = "engine"
        kinds = []
        if row["decode_change"] is not None and row["decode_change"] < -row["decode_tolerance"]:
            kinds.append("decode")
        if judged is not None and judged > row["ttft_tolerance"]:
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


def run_required_again(module, model, config, prepared, protocol, required, warmups, repetitions, log,
                       continuations=None, disturbed_spread=None, exact_prompts=False):
    """The speed cases on an engine opened at the required context again, right after the lower context's, the same
    way (warm-ups, samples): the comparison a staged Eval's lower-context gate makes, under the device state the
    lower context ran in (after the long-context curve). None where it cannot run (the first measurement stands)."""
    log(f"lower context: opening the engine at the required {required:,} tokens again, to compare under the same "
        "conditions")
    try:
        engine, info = open_engine(module, model, config, speculative=True, context_tokens=required,
                                   exact_prompts=exact_prompts)
        closed = None
        try:
            found = run_speed(engine, info, prepared, protocol, warmups, repetitions,
                              lambda message: log("required again: " + message), continuations, disturbed_spread)
        finally:
            closed = close_engine(engine)
        if closed is not None:
            found = {**found, "close": closed} if isinstance(found, dict) else found
            if closed["returned"] is False:
                log(f"required again: the engine's close did not return its device memory: {closed['message']}")
        return found
    except (Disturbed, Hung):
        raise
    except Exception as error:   # noqa: BLE001 - the comparison with the first measurement stands, and says why
        log(f"required again: failed ({type(error).__name__}: {error})")
        return None


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
        closed = None
        try:
            measured = run_speed(engine, info, prepared, protocol, warmups, repetitions,
                                 lambda message: log("lower context: " + message), continuations, disturbed_spread)
        finally:
            closed = close_engine(engine)
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
    found = {"context": lower, "engine_max_context": info["max_context"], "speed": measured}
    if closed is not None:   # the engine is opened again next: its close must have returned its memory
        found["close"] = closed
        if closed["returned"] is False:
            log(f"lower context: the engine's close did not return its device memory: {closed['message']}")
            found.update(passed=False, error="its close did not return its device memory, so the next engine opens "
                                             f"beside the leak: {closed['message']}"[:2000])
    return found
