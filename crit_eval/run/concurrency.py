"""The concurrency part of a run (protocol concurrency): how a served engine that runs several requests at once does
it, judged by the concurrency gate (crit_eval.reports.status). It runs after every other part on the engine, so no
single-stream figure or other gate sees it.

A level of N clients is one closed loop: the clients start together and each sends its next request when its previous
reply has ended, requests_per_client in all, over a pool of distinct prompts (the short speed cases, then the held-out
prompts), each the case's own request at its own output tokens (crit_eval.run.requests; ChatEngine.cohort). Each level
runs twice: greedy, whose replies the gate compares and
whose requests in flight and replies streaming at once say whether the requests ran together, then sampled at the
prepared sampling (each prompt at its own seed), which every speed figure is taken from: decode speed is sampled
decode, and the greedy aggregate stays beside it as `greedy`, a comparison only. Per level: the aggregate completion
tokens over the level's wall interval (crit_eval.run.outcomes.validate_cohort), the requests in flight at once at the
peak and the replies streaming at once (whether the requests ran together), each request's time to first token and
decode speed, each speed-case reply against that case's serial greedy reply to the same request (in quality mode speed,
each difference judged as protocol forks judges one), the level's aggregate against the batched ceiling (crit_eval.limits.batched), and its
speedup over the 1-client level beside the speedup the cost model expects (protocol concurrency.speedup; reported, not
judged)."""
from ..limits.batched import drafted_floor, step_floor
from ..limits.context import token_text
from ..limits.tiers import hit_rate
from ..reports import metrics
from .estimate import duration
from .forks import ForkJudge
from .outcomes import RequestRecord, StreamOutcome, validate_cohort
from .requests import case_request, request_sha256, send_cohort
from .speed import cannot_sample
from ..quality.prepare import sampled_request

# A level's speed figures: the sampled run's (summarize's keys).
SPEED_KEYS = ("wall_seconds", "completion_tokens", "aggregate_tps", "ttft_ms_p50", "ttft_ms_p95", "decode_tps_p50",
              "expert_hit_rate", "ceiling", "ceiling_missing", "fraction_of_ceiling", "expected_at",
              "expected_at_missing")
# What says a run's requests ran together and qualified (summarize's keys).
RUN_KEYS = ("qualified", "errors", "peak_concurrency", "streaming_peak", "ran_together")


def sampled_level(gate, drawn, reason):
    """A level from its greedy run (`gate`: the consistency gate's replies and whether they ran together) and its
    sampled run (`drawn`, None with `reason` where the engine cannot sample): every speed figure the sampled run's,
    the greedy aggregate and per-request decode kept as `greedy`, comparison only."""
    level = {key: value for key, value in gate.items() if key not in SPEED_KEYS}
    level["greedy"] = {"aggregate_tps": gate["aggregate_tps"], "decode_tps_p50": gate["decode_tps_p50"],
                       "comparison_only": "greedy decode: comparison only, never a speed"}
    if drawn is None:
        # The ceilings price the model, not a measurement: the greedy run's stand.
        level.update({key: gate.get(key) for key in SPEED_KEYS if key.startswith(("ceiling", "expected_at"))})
        level.update({key: None for key in SPEED_KEYS if not key.startswith(("ceiling", "expected_at"))},
                     sampled_missing=reason)
        return level
    level.update({key: drawn.get(key) for key in SPEED_KEYS})
    level["qualified"] = gate["qualified"] and drawn["qualified"]
    level["errors"] = (gate["errors"] + [f"sampled: {e}" for e in drawn["errors"]])[:20]
    level["sampled"] = {key: drawn[key] for key in RUN_KEYS}
    return level


def pool(prepared, protocol, max_context):
    """The prompts a level's clients send (protocol concurrency.pool): {"id", "prompt_ids", "case", "request"} rows,
    request the case's own greedy request (crit_eval.run.requests.case_request) and case naming the speed case whose
    serial reply a reply is compared with (None for a held-out prompt)."""
    rule = protocol["concurrency"]
    entries = [(c["id"], c, c["id"]) for c in prepared["speed"] if len(c["prompt_ids"]) <= rule["max_prompt_tokens"]]
    entries += [("heldout/" + p["id"], p, None) for p in prepared.get("heldout", [])]
    rows = [{"id": pid, "prompt_ids": list(entry["prompt_ids"]), "case": case, "entry": entry,
             "request": case_request(entry, "concurrency", protocol)} for pid, entry, case in entries]
    return [row for row in rows if len(row["prompt_ids"]) + row["request"]["output_tokens"] <= max_context]


def levels(protocol, capacity, required, prompts):
    """The client counts measured (protocol concurrency.clients_rule): at most the server's capacity, and with clients x
    (the longest prompt with its output) at most the required context."""
    rule = protocol["concurrency"]
    longest = max(len(p["prompt_ids"]) + p["request"]["output_tokens"] for p in prompts)
    return [n for n in rule["clients"] if n <= capacity and n * longest <= required]


def percentile(values, share):
    """The nearest-rank percentile of `values` (None without any)."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(-(-share * len(values) // 1)) - 1))]


def same_reply(tokens, serial, strings):
    """(whether a reply equals the serial reply as far as both go, the first token where they differ). A reply cut
    shorter may encode its last word into other tokens than the longer reply did: the same text is no difference."""
    shorter = min(len(tokens), len(serial))
    first = next((i for i in range(shorter) if tokens[i] != serial[i]), None)
    if first is None:
        return True, None
    if strings:
        a, b = token_text(tokens, strings), token_text(serial, strings)
        if a.startswith(b) or b.startswith(a):
            return True, None
    return False, first


def streaming_peak(rows):
    """The most replies streaming at once (protocol concurrency.together): requests whose first and last reply tokens
    arrived over overlapping intervals, a finish before a start at the same instant. Clients see every request they
    sent in flight, queued at the server or not; a server that runs its requests one after another streams one reply
    at a time. 0 when no reply streamed over an interval (each came in one chunk): no overlap shows either way."""
    events = []
    for row in rows:
        first, last = row["first"], row["last"]
        if first is not None and last is not None and last > first:
            events += [(first, 1), (last, -1)]
    current = peak = 0
    for _, change in sorted(events):
        current += change
        peak = max(peak, current)
    return peak


def together(clients, peak):
    """Whether a level's requests ran together (protocol concurrency.together): True with two or more replies streaming
    at once, False with one at a time, None when none streamed over an interval (each in one chunk) or for one
    client, which has nothing to run beside."""
    if clients < 2 or peak == 0:
        return None
    return peak >= 2


def summarize(found, clients, prompts_of, serial, strings, ceiling, mean_context, judge=None):
    """One level's figures from ChatEngine.cohort's rows. ceiling(rows, context[, hit rate, its source]): the batched
    floor (None: unpriced), at the run's single-stream hit rate without a rate. A level whose requests report
    expert_reads and expert_hits ("level requests") has its ceiling, and so its fraction_of_ceiling, priced at its
    own hit rate; its expectation (expected_at, expectations) stays at the run's, so every level's expected speedup
    compares like with like. judge(prompt_ids, serial, reply): a differing reply's fork (protocol forks), recorded
    with the difference."""
    rows = found["requests"]
    expected = [f"{c}:{i}" for c, prompts in enumerate(prompts_of) for i in range(len(prompts))]
    records = [RequestRecord(f"{row['client']}:{row['index']}", row["started"], row["finished"],
                             StreamOutcome(not row["errors"] and row["completion_tokens"] is not None,
                                           row["completion_tokens"], row["finish_reason"], tuple(row["errors"])))
               for row in rows]
    end = max([found["seconds"]] + [row["finished"] for row in rows])
    outcome = validate_cohort(expected, records, wall_started=0.0, wall_finished=end, minimum_peak=clients)
    ttft = [(row["first"] - row["started"]) * 1000 for row in rows if row["first"] is not None]
    decode = [(row["completion_tokens"] - 1) / (row["last"] - row["first"]) for row in rows
              if row["completion_tokens"] and row["completion_tokens"] > 1 and row["last"] and row["first"] is not None
              and row["last"] > row["first"]]
    compared = {}
    for row in rows:
        case = prompts_of[row["client"]][row["index"]]["case"]
        if case is None or serial.get(case) is None:
            continue
        match, first = same_reply(row["tokens"], serial[case], strings)
        compared[f"{row['client']}:{row['index']}"] = {"case": case, "match": match, "first_difference": first}
        if not match and judge is not None:
            prompt = prompts_of[row["client"]][row["index"]]["prompt_ids"]
            compared[f"{row['client']}:{row['index']}"]["fork"] = judge(prompt, serial[case], row["tokens"])
    matched = sum(1 for c in compared.values() if c["match"])
    aggregate = outcome.completion_tokens_per_second
    peak = streaming_peak(rows)
    own = hit_rate(rows)   # the level's own expert cache hits, where its requests report them
    if ceiling:
        expected = ceiling(clients, mean_context)
        floor = ceiling(clients, mean_context, own, "level requests") if own is not None else expected
    else:
        expected = floor = (None, ["model weight headers"])
    return {
        "clients": clients, "requests": len(rows), "qualified": outcome.qualified, "errors": list(outcome.errors)[:20],
        "wall_seconds": round(outcome.wall_seconds, 6), "completion_tokens": outcome.successful_completion_tokens,
        # Completion tokens of every request over the level's wall interval; None when the level did not qualify.
        "aggregate_tps": aggregate, "peak_concurrency": outcome.peak_concurrency,
        # The replies streaming at once at the peak: two or more when the requests ran together (protocol
        # concurrency.together), one when the server ran them one after another, none when each came in one chunk.
        "streaming_peak": peak, "ran_together": together(clients, peak),
        "ttft_ms_p50": percentile(ttft, .5), "ttft_ms_p95": percentile(ttft, .95),
        "decode_tps_p50": percentile(decode, .5),
        **({"expert_hit_rate": own} if own is not None else {}),
        # Each speed-case reply against the case's serial greedy reply (the concurrency gate).
        "consistency": {"compared": len(compared), "matched": matched,
                        "fraction": matched / len(compared) if compared else None,
                        "differences": {key: c for key, c in compared.items() if not c["match"]}},
        "ceiling": floor[0], "ceiling_missing": floor[1] or None,
        "fraction_of_ceiling": (aggregate / floor[0]["ceiling_tps"]
                                if aggregate and floor[0] and floor[0]["ceiling_tps"] else None),
        # The ceiling at the run's single-stream hit rate, which the expected speedup compares (expectations).
        "expected_at": ({k: expected[0].get(k) for k in ("ceiling_tps", "bound", "hit_rate", "hit_rate_source",
                                                          "unmeasured")} if expected[0] else None),
        "expected_at_missing": expected[1] or None,
    }


def expectations(found):
    """Each level's speedup, its aggregate over the 1-client level's, beside the speedup the cost model expects for the
    engine's placement, the batched ceiling at the level over the ceiling at one row, both at the run's single-stream
    hit rate (expected_at), and the fraction of it the level reached (protocol concurrency.speedup; reported, not
    judged), in place on `found` (run_concurrency's levels, the 1-client level first). A ceiling whose expert misses
    are unmeasured (no hit rate, measured or declared, for an engine with experts off the device) gives no
    expectation, and expected_missing says why (None where there is one)."""
    one = found["1"]
    base, base_floor = one["aggregate_tps"], one["expected_at"]
    for level in found.values():
        floor = level["expected_at"]
        level["speedup"] = level["aggregate_tps"] / base if level["aggregate_tps"] and base else None
        why = None
        if not floor or not base_floor or not floor["ceiling_tps"] or not base_floor["ceiling_tps"]:
            lacking = level["expected_at_missing"] or one["expected_at_missing"] or ["the 1-client level's ceiling"]
            why = "no batched ceiling to compare: " + ", ".join(lacking)
        elif floor["unmeasured"] or base_floor["unmeasured"]:
            why = floor["unmeasured"] or base_floor["unmeasured"]
        level["expected_speedup"] = None if why else floor["ceiling_tps"] / base_floor["ceiling_tps"]
        level["fraction_of_expected"] = (level["speedup"] / level["expected_speedup"]
                                         if level["speedup"] and level["expected_speedup"] else None)
        level["expected_missing"] = why


def pool_tokens_per_round(result, prompts):
    """The tokens per round the greedy speed suite measured on the pool's speed cases (geometric mean), which a drafting
    engine's batched round emits per row (drafted_floor): the level's requests are those prompts, greedy. None without
    a measured case of the pool."""
    cases = (result.get("speed") or {}).get("cases") or {}
    found = [cases[p["case"]]["tokens_per_round"] for p in prompts if p["case"] in cases
             and cases[p["case"]].get("status") == "measured" and cases[p["case"]].get("tokens_per_round")]
    return metrics.geometric_mean(found) if found else None


def drafted_levels(found, terms, bandwidth_gbs, tflops, host, width, tokens_per_round, draft_bytes):
    """Each level's ceiling with its drafts counted (drafted_floor), in place on `found`'s levels as "drafted_ceiling"
    with its fraction ("fraction_of_drafted"), priced as each level's own ceiling is: at its own hit rate where its
    requests report one. The limit of the concurrency goal (crit_eval.reports.status): the one-token ceiling stays the
    speedup's, whose ratio speculation leaves alone. Without what prices it, "drafted_missing" says why."""
    for clients, level in found["levels"].items():
        rate = level.get("expert_hit_rate")
        priced = ({**host, "hit_rate": rate, "hit_rate_source": "level requests"} if host and rate is not None else host)
        floor, missing = drafted_floor(terms, int(clients), found["mean_context"], bandwidth_gbs, tflops, priced,
                                       width, tokens_per_round, draft_bytes)
        level["drafted_ceiling"] = ({k: floor.get(k) for k in ("ceiling_tps", "bound", "step_ms", "draft_ms", "round_ms",
                                                               "tokens_per_round", "width", "basis", "unmeasured")}
                                    if floor else None)
        level["drafted_missing"] = missing or None
        level["fraction_of_drafted"] = (level["aggregate_tps"] / floor["ceiling_tps"]
                                        if floor and floor.get("ceiling_tps") and level["aggregate_tps"] else None)


def run_concurrency(engine, info, prepared, protocol, result, required, log, fits=None, ceiling_inputs=None,
                    forks=False, draft_bytes=None):
    """The concurrency part on an engine that runs a cohort (a served engine, or Core's package adapter in process):
    result["concurrency"]. fits(name, seconds): whether a level expected to take `seconds` still fits a staged run's
    budget (None: no budget). ceiling_inputs(): (crit_eval.limits.batched batch_terms of the model, the device
    bandwidth, its best rates, host) for each level's ceiling, read only when a level runs; host: what the model reads
    from host memory where the device cannot hold every weight ({"tiers", "hit_rate", "hit_rate_source"} at the run's
    single-stream hit rate, batched.step_floor), None where it can. forks (quality mode speed): each reply that
    differs from its serial one has its fork judged on the engine (protocol forks). draft_bytes: the bytes one draft
    step reads where the adapter declares none (the model's draft layers), which a drafting engine's level ceilings
    with its drafts counted take (drafted_levels)."""
    rule = protocol["concurrency"]
    if not callable(getattr(engine, "cohort", None)) or not callable(getattr(engine, "capacity", None)):
        return {"status": "unmeasured", "reason": "measured on an engine that runs requests together: a served one "
                                                  "(crit eval run --url, crit run eval) or a package (--package)"}
    capacity = engine.capacity()
    if not capacity or capacity < 2:
        reason = (f"the server runs {capacity} request at once" if capacity else
                  "the server reports no max_concurrent_requests (crit serve does)")
        log(f"concurrency: unmeasured ({reason})")
        return {"status": "unmeasured", "reason": reason, "max_concurrent_requests": capacity}
    context = int(required or info["max_context"])
    prompts = pool(prepared, protocol, min(context, int(info["max_context"])))
    planned = levels(protocol, capacity, context, prompts) if prompts else []
    if not planned:
        reason = "no prompt of the pool fits the context with its output" if not prompts else \
            "no level fits the server's capacity and the context"
        log(f"concurrency: unmeasured ({reason})")
        return {"status": "unmeasured", "reason": reason, "max_concurrent_requests": capacity}
    per = int(rule["requests_per_client"])
    unsampled = cannot_sample(engine, prepared)
    seed_of = {p["id"]: i + 1 for i, p in enumerate(prompts)}   # each prompt at its own seed, at every level
    # A batched reply is compared with a serial reply to the same request only (protocol requests.cases): the speed
    # part's, which records the request its tokens reply to, where that is the level's request, else the request run
    # alone first, a cohort of one client (crit eval run --only concurrency, a result from before the record).
    speed = (result.get("speed") or {}).get("cases") or {}
    serial, missing = {}, []
    for p in prompts:
        if p["case"] is None:
            continue
        found = speed.get(p["case"]) or {}
        if (found.get("status") == "measured" and found.get("tokens")
                and found.get("request_sha256") == request_sha256(p["request"])):
            serial[p["case"]] = found["tokens"]
        else:
            missing.append(p)
    serial_source = "speed part"
    if missing:
        for p in missing:
            alone = send_cohort(engine, [[p["request"]]])
            rows = alone.get("requests") or []
            if rows and not rows[0].get("errors") and rows[0].get("tokens"):
                serial[p["case"]] = rows[0]["tokens"]
        serial_source = "speed part and one-client runs" if len(missing) < len(
            [p for p in prompts if p["case"] is not None]) else "one-client runs"
        log(f"concurrency: {len(missing)} serial replies run alone first (no speed part reply to the same request)")
    strings = (prepared.get("retrieval") or {}).get("token_text")
    outputs = {p["id"]: p["request"]["output_tokens"] for p in prompts}
    mean_context = sum(len(p["prompt_ids"]) + outputs[p["id"]] / 2 for p in prompts) / len(prompts)
    terms, bandwidth_gbs, tflops, host = ceiling_inputs() if ceiling_inputs else (None, None, None, None)

    def ceiling(rows, at, rate=None, source=None):
        """The batched floor of `rows` rows, at `rate` (a level's own measured hit rate) where given, else the run's."""
        priced = {**host, "hit_rate": rate, "hit_rate_source": source} if host and rate is not None else host
        return step_floor(terms, rows, at, bandwidth_gbs, tflops, priced)
    judge = ForkJudge(engine, protocol).judge if forks else None
    out = {"status": "measured", "max_concurrent_requests": capacity, "planned": planned, "requests_per_client": per,
           "serial_source": serial_source,
           # The ceiling counts one token per row per step: speculation lifts an aggregate above it.
           "ceiling_basis": "one token per row per step",
           # The run's single-stream hit rate the expectations are priced at, for a model with weights off the device.
           **({"hit_rate": host["hit_rate"], "hit_rate_source": host["hit_rate_source"]} if host else {}),
           # Each pool prompt's output tokens: its case's own (protocol requests.cases).
           "output_tokens": outputs, "pool": [p["id"] for p in prompts], "mean_context": mean_context,
           "sampling": prepared.get("sampling") if unsampled is None else None,
           **({"sampled_missing": unsampled} if unsampled else {}), "levels": {},
           "skipped": {}}
    rate = None   # the best aggregate measured so far, which prices the next level
    for clients in planned:
        prompts_of = [[prompts[(c + r) % len(prompts)] for r in range(per)] for c in range(clients)]
        # Two runs a level where the engine samples: greedy for the gate, sampled for the speed.
        tokens = sum(outputs[p["id"]] for row in prompts_of for p in row)
        seconds = (tokens / rate * (1 if unsampled else 2) if rate else None)
        if fits is not None and not fits(f"concurrency@{clients}", seconds):
            out["skipped"][str(clients)] = f"over budget: about {duration(seconds)}"
            break
        found = send_cohort(engine, [[p["request"] for p in row] for row in prompts_of])
        gate = summarize(found, clients, prompts_of, serial, strings, ceiling if terms else None, mean_context, judge)
        drawn = None
        if unsampled is None:
            # The same requests, sampled: each prompt at its own seed (protocol requests.overrides.concurrency).
            drawn = summarize(send_cohort(engine, [[case_request(p["entry"], "concurrency", protocol,
                                                                 sampling=sampled_request(prepared["sampling"],
                                                                                          seed_of[p["id"]]))
                                                    for p in row] for row in prompts_of]),
                              clients, prompts_of, {}, strings, ceiling if terms else None, mean_context)
        level = sampled_level(gate, drawn, unsampled)
        out["levels"][str(clients)] = level
        # The next level's time is priced at the fastest aggregate yet, the greedy one where nothing sampled.
        rate = max(rate or 0.0, level["aggregate_tps"] or gate["aggregate_tps"] or 0.0) or rate
        consistent = level["consistency"]
        log(f"concurrency {clients}: "
            + (f"{level['aggregate_tps']:.1f} tok/s sampled aggregate" if level["aggregate_tps"] else
               f"sampled aggregate unmeasured ({level['sampled_missing']})" if level.get("sampled_missing") else
               "not qualified (" + "; ".join(level["errors"][:2]) + ")")
            + f", peak {level['peak_concurrency']} in flight, {level['streaming_peak']} streaming at once"
            + (f", first token p50 {level['ttft_ms_p50']:,.0f} ms" if level["ttft_ms_p50"] is not None else "")
            + (f", decode p50 {level['decode_tps_p50']:.1f} tok/s per request" if level["decode_tps_p50"] else "")
            + (f", {consistent['matched']}/{consistent['compared']} replies equal the serial ones"
               if consistent["compared"] else "")
            + (f", {level['fraction_of_ceiling']:.0%} of its ceiling" if level["fraction_of_ceiling"] else ""))
    expectations(out["levels"])
    if terms:
        # A drafting engine that states no width has no drafted ceiling (None): drafted_floor says what is missing.
        width = (int(info["draft_width"]) if info.get("draft_width") else None) if info.get("speculative") else 0
        drafted_levels(out, terms, bandwidth_gbs, tflops, host, width, pool_tokens_per_round(result, prompts) if width
                       else None, info.get("draft_bytes_per_step") or draft_bytes)
    for clients, level in out["levels"].items():
        if int(clients) >= 2 and level["speedup"]:
            log(f"concurrency {clients}: {level['speedup']:.2f}x one client's aggregate"
                + (f", {level['expected_speedup']:.2f}x expected from the cost model "
                   f"({level['fraction_of_expected']:.0%} of it)" if level["expected_speedup"] else
                   f"; no expectation ({level['expected_missing']})"))
    return out
