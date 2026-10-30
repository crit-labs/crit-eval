"""The stages of a run without a profile (protocol stages): what each stage sends, which stage fits the budget at the
rates the engine measured, and the long-context facts stage 1 gives without a long prompt (the memory fit and the
decode creep).

Stage 1 runs on every engine; after it the run goes on to the highest stage whose remaining requests fit what is
left of the budget. Each stage is a prefix of the next, so a run stopped by the budget still holds every stage
before it in full. A certified run (the candidate that will ship) goes to stage 3 whatever the budget and adds one
request at the requirement itself.
"""
from ..limits.context import curve_points, retrieval_points
from .deadline import RequestDeadline
from .estimate import DECODE_TERMS, SCORE_ROW_SECONDS, fit


def held_cases(prepared, info):
    """The speed cases the engine holds, in the prepared order."""
    return [c for c in prepared["speed"] if len(c["prompt_ids"]) + c["output_tokens"] <= info["max_context"]]


def short_cases(prepared, info, rule):
    """The held speed cases whose prompts are under rule long_prompt_tokens: the ones stage 2 repeats."""
    return [c for c in held_cases(prepared, info) if len(c["prompt_ids"]) < int(rule["long_prompt_tokens"])]


def shortest_case(prepared, info):
    """The held speed case with the shortest prompt (the request history's case in a staged run), as a list."""
    held = held_cases(prepared, info)
    return [min(held, key=lambda c: len(c["prompt_ids"]))] if held else []


def exactness_cases(prepared, info, rule):
    """The rule exactness_cases held speed cases with the shortest prompts, in the prepared order."""
    held = held_cases(prepared, info)
    keep = sorted(held, key=lambda c: len(c["prompt_ids"]))[:int(rule["exactness_cases"])]
    return [c for c in held if any(c is k for k in keep)]


def quality_sets(prepared, limit=None, above=False):
    """The quality sets whose longest text (prompt plus continuation) fits `limit` tokens, or with above those that
    do not; every set without a limit."""
    need = {}
    for t in prepared.get("quality") or []:
        need[t["set"]] = max(need.get(t["set"], 0), len(t["prompt_ids"]) + len(t["continuation_ids"]))
    if limit is None:
        return list(need)
    return [name for name, n in need.items() if (n > limit) == above]


def with_quality(prepared, sets):
    """The prepared data with only the quality texts of `sets`."""
    return {**prepared, "quality": [t for t in prepared.get("quality") or [] if t["set"] in sets]}


def with_speed(prepared, cases):
    """The prepared data with only the speed cases `cases`."""
    return {**prepared, "speed": list(cases)}


def quality_rows(prepared, protocol, info, sets):
    """The quality part's requests for `sets` (quality_checks.run_quality): each text the engine holds scored, and the
    honesty check's reply and its rescore for the texts of its sets."""
    honesty = protocol["quality"]["honesty"]
    rows, reply = [], int(honesty["reply_tokens"])
    for t in with_quality(prepared, sets)["quality"]:
        if len(t["prompt_ids"]) + len(t["continuation_ids"]) > info["max_context"]:
            continue
        rows.append((len(t["prompt_ids"]), len(t["continuation_ids"]),
                     len(t["prompt_ids"]) + len(t["continuation_ids"]) - 1, False))
        if t["set"] in honesty["sets"]:
            rows += [(len(t["prompt_ids"]), reply, 0, False), (len(t["prompt_ids"]), reply, reply, False)]
    return rows


def staged_settings(settings, rule, certify):
    """Context settings for a staged run: the curve at its first point (and the requirement when certified) and
    retrieval at its points below the curve's first, the curve's own replies judged at its points."""
    first = int(settings["curve_from"])
    return {**settings, "curve_points": "ends" if certify else "first", "measure_lower": False,
            "retrieval": {**settings["retrieval"], "points": [p for p in settings["retrieval"]["points"] if p < first]},
            "requirement": bool(certify)}


def stage_requests(prepared, protocol, required, info, settings, sampled=True, exactness=True, heldout=False,
                   lower=True, certify=False):
    """{stage: {timing part: [(prompt tokens, output tokens, scored rows, one token per pass)]}}: the requests each
    stage adds over the one before it, and under "certify" the request at the requirement. settings: the staged
    context settings (staged_settings). sampled, exactness, heldout and lower: whether those parts can run on this
    engine (an engine without sampling has no sampled suite, a served one no lower context, a run without a
    baseline no held-out prompts)."""
    rule = protocol["stages"]
    held = held_cases(prepared, info)
    each = [(len(c["prompt_ids"]), c["output_tokens"], 0, False) for c in held]
    out = {1: {}, 2: {}, 3: {}, "certify": {}}

    def quality(sets):
        return quality_rows(prepared, protocol, info, sets)

    short = int(rule["short_quality_tokens"])
    history = [(len(c["prompt_ids"]), c["output_tokens"], 0, False) for c in shortest_case(prepared, info)]
    out[1] = {"speed": each, "quality": quality(quality_sets(prepared, short)), "history": history,
              "history_again": history}
    repeated = [(len(c["prompt_ids"]), c["output_tokens"], 0, False) for c in short_cases(prepared, info, rule)]
    out[2]["speed"] = repeated * int(rule["repetitions"])
    turns, n = prepared.get("followup") or [], int(protocol["followup"]["output_tokens"])
    if turns and len(turns[-1]["prompt_ids"]) + n <= info["max_context"]:
        out[2]["followup"] = [(len(t["prompt_ids"]), n, 0, False) for t in turns]
        out[3]["followup_again"] = list(out[2]["followup"])
    out[2]["quality"] = quality(quality_sets(prepared, short, above=True))
    pieces = prepared.get("retrieval")
    if required and pieces:
        n = int(settings["output_tokens"])
        first = curve_points(required, settings)[0]
        if first <= info["max_context"]:
            out[2]["context"] = [(first - n, n, 0, False)]
            if out[1]["quality"]:
                out[2]["quality"].append((first - n, n, n, False))   # the honesty check's long reply, scored again
        m = int(settings["retrieval"]["output_tokens"])
        out[2]["retrieval"] = [(p - m, m, 0, False) for p in retrieval_points(required, settings["retrieval"])
                               if p < first and p <= info["max_context"]]
        if certify and first < required <= info["max_context"]:
            out["certify"]["context"] = [(required - n, n, 0, False)]
    if sampled and prepared.get("sampling") is not None:
        out[3]["sampled_speed"] = each * (int(rule["sampled_repetitions"]) + 1)
    if exactness and info.get("speculative"):
        out[3]["exactness"] = [(len(c["prompt_ids"]), c["output_tokens"], 0, True)
                               for c in exactness_cases(prepared, info, rule)]
    if heldout:
        out[3]["heldout"] = [(len(p["prompt_ids"]), p["output_tokens"], 0, False) for p in prepared.get("heldout") or []
                             if len(p["prompt_ids"]) + p["output_tokens"] <= info["max_context"]
                             for _ in range(int(protocol["heldout"]["repetitions"]))]
    if lower:
        out[3]["lower_context"] = list(each)
    return {stage: {part: rows for part, rows in parts.items() if rows} for stage, parts in out.items()}


def price(parts, found, tokens_per_round=None, row_seconds=SCORE_ROW_SECONDS):
    """{timing part: expected seconds} of requests (stage_requests' parts of one stage) at the fitted rates."""
    return {name: sum(found.request(n, m, (tokens_per_round or 1.0) if plain else 1.0) + rows * row_seconds
                      for n, m, rows, plain in reqs) for name, reqs in parts.items()}


def observations(engine, result, protocol):
    """What the engine served so far, for the rates: a served engine's RequestDeadline, else one built from the
    speed cases' samples."""
    seen = getattr(engine, "deadline", None)
    if seen is not None:
        return seen
    seen = RequestDeadline(protocol["requests"]["timeout"])
    for case in ((result.get("speed") or {}).get("cases") or {}).values():
        for sample in case.get("samples", []):
            if case.get("status") == "measured":
                seen.observe(case["prompt_tokens"], sample["ttft_ms"] / 1000, sample["output_tokens"],
                             (sample["request_ms"] - sample["ttft_ms"]) / 1000)
    return seen


def choose(planned, found, elapsed, budget, tokens_per_round=None, load_seconds=0.0, reserved=0.0):
    """(stage, {stage: expected seconds it adds}) the highest stage whose added requests fit the budget after
    `elapsed` seconds, `reserved` seconds kept for what every run ends with (the request history again); stage 3's
    lower context opens an engine of its own, load_seconds more. Stage 1 when no rate is known."""
    if found is None:
        return 1, {}
    costs = {}
    for stage in (2, 3):
        costs[stage] = sum(price(planned.get(stage) or {}, found, tokens_per_round).values())
        if stage == 3 and "lower_context" in (planned.get(3) or {}):
            costs[stage] += load_seconds
    left = budget - elapsed - reserved
    if costs[2] + costs[3] <= left:
        return 3, costs
    return (2 if costs[2] <= left else 1), costs


def creep(result, required):
    """protocol stages 1 context.creep: decode seconds per output token fitted as p + q * k across the contexts the
    run measured (prompt plus output of every greedy speed case and curve point), and the decode it projects at the
    requirement against the shortest context's. None without two distinct contexts."""
    points = []
    rows = [c for c in ((result.get("speed") or {}).get("cases") or {}).values() if c.get("status") == "measured"]
    rows += [p for p in ((result.get("context") or {}).get("points") or {}).values() if p.get("status") == "measured"]
    for row in rows:
        for sample in row.get("samples", []):
            if sample.get("warmup") or not sample.get("output_tokens") or sample["output_tokens"] < 2:
                continue
            seconds = (sample["request_ms"] - sample["ttft_ms"]) / 1000 / (sample["output_tokens"] - 1)
            if seconds > 0:
                points.append((row["prompt_tokens"] + sample["output_tokens"], seconds))
    if len({k for k, _ in points}) < 2:
        return None
    p, q = fit(points, DECODE_TERMS, powers=2)
    shortest, longest = min(k for k, _ in points), max(k for k, _ in points)
    at = lambda k: 1.0 / (p + q * k) if p + q * k > 0 else None   # noqa: E731
    projected, base = at(required), at(shortest)
    return {"seconds_per_token": p, "seconds_per_token_per_context_token": q, "measured_up_to": longest,
            "contexts": len({k for k, _ in points}), "decode_tps_at_shortest": base,
            "decode_tps_at_requirement": projected,
            "slowdown_at_requirement": (1 - projected / base) if projected and base else None,
            "extrapolated": required > longest}


def memory_fit(after_load, info, ceiling, required):
    """protocol stages 1 context.fit: the device memory free after the engine opened at the requirement against the
    KV cache the requirement needs. An engine that allocates its cache when it opens already holds it; one that
    allocates as the context fills needs it free. Reported, not judged: certification serves the requirement."""
    free = (after_load or {}).get("free_bytes")
    per_token = info.get("kv_bytes_per_token") or (ceiling or {}).get("kv_bytes_per_token")
    source = "engine" if info.get("kv_bytes_per_token") else "model" if per_token else None
    need = per_token * required if per_token and required else None
    return {"free_after_load_bytes": free, "kv_bytes_per_token": per_token, "kv_bytes_per_token_source": source,
            "kv_bytes_at_requirement": need,
            "fits_if_allocated_late": None if free is None or need is None else free >= need}
