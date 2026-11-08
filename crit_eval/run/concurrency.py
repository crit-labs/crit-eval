"""The concurrency part of a run (protocol concurrency): how a served engine that runs several requests at once does
it, judged by the concurrency gate (crit_eval.reports.status). It runs after every other part on the engine, so no
single-stream figure or other gate sees it.

A level of N clients is one closed loop: the clients start together and each sends its next request when its previous
reply has ended, requests_per_client in all, over a pool of distinct prompts (the short speed cases, then the held-out
prompts), greedy and output_tokens long (ChatEngine.cohort). Per level: the aggregate completion tokens over the level's
wall interval (crit_eval.run.outcomes.validate_cohort), the requests in flight at once at the peak and the replies
streaming at once (whether the requests ran together), each request's time to first token and decode speed, each
speed-case reply against that case's serial greedy reply (in quality mode speed, each difference judged as
protocol forks judges one), the level's aggregate against the batched ceiling (crit_eval.limits.batched), and its
speedup over the 1-client level beside the speedup the cost model expects (protocol concurrency.speedup; reported, not
judged)."""
from ..limits.batched import step_floor
from ..limits.context import token_text
from ..limits.tiers import hit_rate
from .estimate import duration
from .forks import ForkJudge
from .outcomes import RequestRecord, StreamOutcome, validate_cohort


def pool(prepared, protocol, max_context):
    """The prompts a level's clients send (protocol concurrency.pool): {"id", "prompt_ids", "case"} rows, case naming the
    speed case whose serial reply a reply is compared with (None for a held-out prompt)."""
    rule = protocol["concurrency"]
    rows = [{"id": c["id"], "prompt_ids": c["prompt_ids"], "case": c["id"]} for c in prepared["speed"]
            if len(c["prompt_ids"]) <= rule["max_prompt_tokens"]]
    rows += [{"id": "heldout/" + p["id"], "prompt_ids": p["prompt_ids"], "case": None} for p in prepared.get("heldout", [])]
    return [row for row in rows if len(row["prompt_ids"]) + rule["output_tokens"] <= max_context]


def levels(protocol, capacity, required, prompts):
    """The client counts measured (protocol concurrency.clients_rule): at most the server's capacity, and with clients x
    (the longest prompt + output) at most the required context."""
    rule = protocol["concurrency"]
    longest = max(len(p["prompt_ids"]) for p in prompts) + rule["output_tokens"]
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


def run_concurrency(engine, info, prepared, protocol, result, required, log, fits=None, ceiling_inputs=None,
                    forks=False):
    """The concurrency part on an engine that runs a cohort (a served engine, or Core's package adapter in process):
    result["concurrency"]. fits(name, seconds): whether a level expected to take `seconds` still fits a staged run's
    budget (None: no budget). ceiling_inputs(): (crit_eval.limits.batched batch_terms of the model, the device
    bandwidth, its best rates, host) for each level's ceiling, read only when a level runs; host: what the model reads
    from host memory where the device cannot hold every weight ({"tiers", "hit_rate", "hit_rate_source"} at the run's
    single-stream hit rate, batched.step_floor), None where it can. forks (quality mode speed): each reply that
    differs from its serial one has its fork judged on the engine (protocol forks)."""
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
    per, output = int(rule["requests_per_client"]), int(rule["output_tokens"])
    serial = {cid: case.get("tokens") for cid, case in ((result.get("speed") or {}).get("cases") or {}).items()
              if case.get("status") == "measured" and case.get("tokens")}
    serial_source = "speed part"
    missing = [p for p in prompts if p["case"] is not None and serial.get(p["case"]) is None]
    if missing:
        # Without the speed part's replies (crit eval run --only concurrency) each speed case of the pool is run once
        # alone, a cohort of one client, so every level still has serial replies to compare with.
        for p in missing:
            alone = engine.cohort([[p["prompt_ids"]]], output)
            rows = alone.get("requests") or []
            if rows and not rows[0].get("errors") and rows[0].get("tokens"):
                serial[p["case"]] = rows[0]["tokens"]
        serial_source = "speed part and one-client runs" if len(missing) < len(
            [p for p in prompts if p["case"] is not None]) else "one-client runs"
        log(f"concurrency: {len(missing)} serial replies run alone first (no speed part to compare with)")
    strings = (prepared.get("retrieval") or {}).get("token_text")
    mean_context = sum(len(p["prompt_ids"]) for p in prompts) / len(prompts) + output / 2
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
           "output_tokens": output, "pool": [p["id"] for p in prompts], "mean_context": mean_context, "levels": {},
           "skipped": {}}
    rate = None   # the best aggregate measured so far, which prices the next level
    for clients in planned:
        prompts_of = [[prompts[(c + r) % len(prompts)] for r in range(per)] for c in range(clients)]
        seconds = (clients * per * output / rate if rate else None)
        if fits is not None and not fits(f"concurrency@{clients}", seconds):
            out["skipped"][str(clients)] = f"over budget: about {duration(seconds)}"
            break
        found = engine.cohort([[p["prompt_ids"] for p in row] for row in prompts_of], output)
        level = summarize(found, clients, prompts_of, serial, strings, ceiling if terms else None, mean_context, judge)
        out["levels"][str(clients)] = level
        rate = max(rate or 0.0, level["aggregate_tps"] or 0.0) or rate
        consistent = level["consistency"]
        log(f"concurrency {clients}: "
            + (f"{level['aggregate_tps']:.1f} tok/s aggregate" if level["aggregate_tps"] else "not qualified ("
               + "; ".join(level["errors"][:2]) + ")")
            + f", peak {level['peak_concurrency']} in flight, {level['streaming_peak']} streaming at once"
            + (f", first token p50 {level['ttft_ms_p50']:,.0f} ms" if level["ttft_ms_p50"] is not None else "")
            + (f", decode p50 {level['decode_tps_p50']:.1f} tok/s per request" if level["decode_tps_p50"] else "")
            + (f", {consistent['matched']}/{consistent['compared']} replies equal the serial ones"
               if consistent["compared"] else "")
            + (f", {level['fraction_of_ceiling']:.0%} of its ceiling" if level["fraction_of_ceiling"] else ""))
    expectations(out["levels"])
    for clients, level in out["levels"].items():
        if int(clients) >= 2 and level["speedup"]:
            log(f"concurrency {clients}: {level['speedup']:.2f}x one client's aggregate"
                + (f", {level['expected_speedup']:.2f}x expected from the cost model "
                   f"({level['fraction_of_expected']:.0%} of it)" if level["expected_speedup"] else
                   f"; no expectation ({level['expected_missing']})"))
    return out
