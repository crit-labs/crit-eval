"""The concurrency part of a run (protocol concurrency): how a served engine that runs several requests at once does
it. Reported, not judged; it runs after every other part on the engine, so no single-stream figure or gate sees it.

A level of N clients is one closed loop: the clients start together and each sends its next request when its previous
reply has ended, requests_per_client in all, over a pool of distinct prompts (the short speed cases, then the held-out
prompts), greedy and output_tokens long (ChatEngine.cohort). Per level: the aggregate completion tokens over the level's
wall interval (crit_eval.run.outcomes.validate_cohort), the requests in flight at once at the peak, each request's
time to first token and decode speed, each speed-case reply against that case's serial greedy reply, and the level's
aggregate against the batched ceiling (crit_eval.limits.batched)."""
from ..limits.batched import step_floor
from ..limits.context import token_text
from .estimate import duration
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


def summarize(found, clients, prompts_of, serial, strings, ceiling, mean_context):
    """One level's figures from ChatEngine.cohort's rows."""
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
    matched = sum(1 for c in compared.values() if c["match"])
    aggregate = outcome.completion_tokens_per_second
    floor = ceiling(clients, mean_context) if ceiling else (None, ["model weight headers"])
    return {
        "clients": clients, "requests": len(rows), "qualified": outcome.qualified, "errors": list(outcome.errors)[:20],
        "wall_seconds": round(outcome.wall_seconds, 6), "completion_tokens": outcome.successful_completion_tokens,
        # Completion tokens of every request over the level's wall interval; None when the level did not qualify.
        "aggregate_tps": aggregate, "peak_concurrency": outcome.peak_concurrency,
        "ttft_ms_p50": percentile(ttft, .5), "ttft_ms_p95": percentile(ttft, .95),
        "decode_tps_p50": percentile(decode, .5),
        # Each speed-case reply against the case's serial greedy reply (reported, not judged).
        "consistency": {"compared": len(compared), "matched": matched,
                        "fraction": matched / len(compared) if compared else None,
                        "differences": {key: c for key, c in compared.items() if not c["match"]}},
        "ceiling": floor[0], "ceiling_missing": floor[1] or None,
        "fraction_of_ceiling": (aggregate / floor[0]["ceiling_tps"]
                                if aggregate and floor[0] and floor[0].get("ceiling_tps") else None),
    }


def run_concurrency(engine, info, prepared, protocol, result, required, log, fits=None, ceiling_inputs=None):
    """The concurrency part on a served engine: result["concurrency"]. fits(name, seconds): whether a level expected to
    take `seconds` still fits a staged run's budget (None: no budget). ceiling_inputs(): (crit_eval.limits.batched
    batch_terms of the model, the device bandwidth, its best rates) for each level's ceiling, read only when a level
    runs."""
    rule = protocol["concurrency"]
    if not info.get("served") or not callable(getattr(engine, "cohort", None)):
        return {"status": "unmeasured", "reason": "measured on a served engine only (crit eval run --url, crit run eval)"}
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
              if case.get("status") == "measured"}
    strings = (prepared.get("retrieval") or {}).get("token_text")
    mean_context = sum(len(p["prompt_ids"]) for p in prompts) / len(prompts) + output / 2
    terms, bandwidth_gbs, tflops = ceiling_inputs() if ceiling_inputs else (None, None, None)
    ceiling = (lambda rows, at: step_floor(terms, rows, at, bandwidth_gbs, tflops)) if terms else None
    out = {"status": "measured", "max_concurrent_requests": capacity, "planned": planned, "requests_per_client": per,
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
        level = summarize(found, clients, prompts_of, serial, strings, ceiling, mean_context)
        out["levels"][str(clients)] = level
        rate = max(rate or 0.0, level["aggregate_tps"] or 0.0) or rate
        consistent = level["consistency"]
        log(f"concurrency {clients}: "
            + (f"{level['aggregate_tps']:.1f} tok/s aggregate" if level["aggregate_tps"] else "not qualified ("
               + "; ".join(level["errors"][:2]) + ")")
            + f", peak {level['peak_concurrency']} in flight"
            + (f", first token p50 {level['ttft_ms_p50']:,.0f} ms" if level["ttft_ms_p50"] is not None else "")
            + (f", decode p50 {level['decode_tps_p50']:.1f} tok/s per request" if level["decode_tps_p50"] else "")
            + (f", {consistent['matched']}/{consistent['compared']} replies equal the serial ones"
               if consistent["compared"] else "")
            + (f", {level['fraction_of_ceiling']:.0%} of its ceiling" if level["fraction_of_ceiling"] else ""))
    return out
