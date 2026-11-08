"""The speed parts of a run: greedy and sampled speed, the sampling check, held-out prompts, a wider draft and
exactness against plain greedy."""
from .jitter import heldout_prompt
from .hostload import HostLoad
from ..limits import headroom
from ..reports import metrics
from .pin import CONTINUATIONS
from ..quality.prepare import sampled_request, unapplied
from .adapters import open_engine
from .tool_calls import case_tool_calls
from .timing import (_summary, check_disturbed, decode_text, disagree, rate, replies, reply_sha256, settling, stalls,
                     steady_state, timed_generate, timed_plain, timed_replay, timed_sampled)


def run_speed(engine, info, prepared, protocol, warmups, repetitions, log, continuations=None, disturbed_spread=None,
              after_first=None, admit=None, warmed=None):
    """Speed cases. With a pinned continuation and a replaying adapter, every speed KPI comes from replays
    of the pinned text (identical for every engine) and two free generations record the engine's own
    output for repeatability and exactness; otherwise the KPIs come from free generation. After the case's
    repetitions, one more when they disagree (protocol speed.extra_repetition); a single repetition is compared
    with the warmup before it, and may get two more. Before them, a request whose decode differs from the one before it
    by more than speed.steady_state's tolerance is one more warmup, up to its max_warmup_runs: an engine that adapts
    to a case (a device cache filling) is measured at the speed it settles to, and the case records its warmups and
    drift (steady_state).
    disturbed_spread: stop the round (Disturbed) when a case's decode speeds spread by more than this.
    after_first: called with the first measured case (the run's time estimate). admit: called with each case after
    the first and the cases measured so far; the reason it returns (a staged run's budget) skips the case.
    warmed: case id -> the case as an earlier call measured it (a staged run's stage 1), whose requests stand as
    this call's warmups and whose output counts for the repeatable gate."""
    tolerance = protocol["speed"]["extra_repetition"]["tolerance"]
    steady = protocol["speed"].get("steady_state")
    cases = {}
    stop_ids = set(prepared["stop_ids"])
    text = decode_text(info, continuations)
    pinned = (continuations or {}).get("cases", {}) if text["source"] == "pinned" else {}
    for case in prepared["speed"]:
        prompt, n = case["prompt_ids"], case["output_tokens"]
        if len(prompt) + n > info["max_context"]:
            cases[case["id"]] = {"status": "skipped", "reason": f"needs {len(prompt) + n} tokens of context"}
            log(f"speed {case['id']}: skipped (context)")
            continue
        refused = admit(case, cases) if admit is not None and cases else None
        if refused:
            cases[case["id"]] = {"status": "skipped", "reason": refused, "over_budget": True}
            log(f"speed {case['id']}: skipped ({refused})")
            continue
        before = (warmed or {}).get(case["id"]) or {}
        samples = [{**s, "warmup": True} for s in before.get("samples", [])]
        outputs = [before["tokens"]] if before.get("tokens") is not None else []
        own = len(outputs)   # outputs before this call's: the warmup's, compared but not the case's tokens
        load = HostLoad().start()
        continuation = pinned.get(case["id"])
        if pinned and (continuation is None or len(continuation) != n):
            raise SystemExit(f"{CONTINUATIONS} has no {n}-token continuation for speed case {case['id']}")
        # A single repetition is compared with its warmup: when they disagree it gets a second, and a third when
        # those two disagree in turn (protocol speed.extra_repetition).
        extra = 2 if repetitions == 1 and (warmups or samples) else 1
        # A request whose decode is still moving away from the one before it (a cache adapting to the case) stands as
        # one more warmup, up to the protocol's bound (protocol speed.steady_state).
        warm, added = warmups, 0
        # The case's first request (protocol speed.fresh_decode): with a pinned continuation the first free generation,
        # which runs before the replays; a staged run's is its first stage's.
        first = before.get("first_request")
        if continuation is not None:
            for _ in range(2):
                tokens, sample = timed_generate(engine, prompt, n)
                outputs.append(tokens)
                first = first or first_request(sample)
        i = 0
        while i < warm + repetitions + extra:
            if i >= warm + repetitions and not disagree(samples, outputs, tolerance):
                break
            if continuation is not None:
                _, sample = timed_replay(engine, prompt, continuation)
            else:
                tokens, sample = timed_generate(engine, prompt, n)
                # Which reply this request decoded: repetitions of different replies are not a disturbance
                # (protocol speed.output_varies).
                sample["reply_sha256"] = reply_sha256(tokens)
            sample["warmup"] = i < warm
            if not sample["warmup"] and settling(samples, sample, steady):
                sample["warmup"], warm, added = True, warm + 1, added + 1
            samples.append(sample)
            if continuation is None:
                outputs.append(tokens)   # the warmup's too: the repeatable gate compares every output
            i += 1
        if continuation is None:
            own += warm
        first = first or first_request(samples[0] if samples else {})
        other = load.cores()
        check_disturbed(case["id"], samples, disturbed_spread, other)
        summary = _summary([s for s in samples if not s["warmup"]])
        tokens = outputs[own] if continuation is None else outputs[-2]
        crossed = [i for i, t in enumerate(tokens) if t in stop_ids]
        cases[case["id"]] = {
            "status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n, **summary,
            "repeatable": all(o == outputs[0] for o in outputs),
            # How many outputs the repeatable gate compares: a case decoded once repeats nothing.
            "outputs": len(outputs),
            # The distinct replies in order when they differ, for the fork judge (protocol forks).
            **({"variants": distinct(outputs)} if any(o != outputs[0] for o in outputs) else {}),
            # Measured repetitions that decoded different replies, and how far their decode spread (protocol
            # speed.output_varies; the repeatable gate judges the outputs).
            "output_varies": output_varies(samples),
            "stop_crossed": bool(crossed), "first_stop_index": crossed[0] if crossed else None,
            "tokens": tokens, "samples": samples,
            # The prompt's first request, before any warmup or replay (protocol speed.fresh_decode).
            "first_request": first,
            # The warmups the case took to reach steady speed and how far its speed moved meanwhile.
            "steady_state": steady_state(samples, steady, added),
            # The longest and p99 gaps between streamed chunks after the first token (protocol speed.stalls).
            "stalls": stalls(samples, summary["round_ms"]),
            # The calls an agent case's output makes, valid, invalid or truncated (protocol speed.tool_calls).
            "tool_calls": case_tool_calls(case, tokens, next(s for s in samples if not s["warmup"]), prepared),
            # CPU cores other processes kept busy during the case: engines computing on the host slow down with them.
            "host_other_cpu_cores": None if other is None else round(other, 2),
        }
        log(f"speed {case['id']}: decode {rate(summary['decode_tps'], 1)}"
            + (" (pinned continuation)" if continuation is not None else "")
            + (f" after {added} more warmup(s) while it moved {cases[case['id']]['steady_state']['drift'] or 0:+.0%}"
               if added else "")
            + f", prefill {rate(summary['prefill_tps'], 0)}, tokens/round {summary['tokens_per_round'] or 0:.2f}, "
            f"round {summary['round_ms'] or 0:.2f} ms"
            + (f", other processes {other:.1f} CPU cores" if other is not None and other >= 0.5 else ""))
        if after_first is not None:
            after_first(cases[case["id"]])
            after_first = None
    return speed_summary(cases, protocol, warmups, repetitions, text)


def distinct(outputs):
    """The distinct outputs, first occurrence first."""
    seen, out = set(), []
    for o in outputs:
        if tuple(o) not in seen:
            seen.add(tuple(o))
            out.append(list(o))
    return out


def output_varies(samples):
    """protocol speed.output_varies: {replies, spread} when a case's measured repetitions decoded more than one reply
    (their decode tok/s spread over all of them), else None."""
    measured = [s for s in samples if not s.get("warmup") and s.get("reply_sha256")]
    groups = replies(measured)
    if len(groups) < 2:
        return None
    return {"replies": len(groups), "spread": headroom.spread([s["decode_tps"] for s in measured if s.get("decode_tps")])}


def first_request(sample):
    """The timings of a case's first request that protocol speed.fresh_decode reports."""
    keys = ("decode_tps", "request_ms", "ttft_ms", "output_tokens", "tokens_per_round", "round_ms")
    return {key: sample.get(key) for key in keys} if sample.get("decode_tps") else None


def fresh_decode(measured):
    """protocol speed.fresh_decode: the geometric mean of each measured case's first request's decode tok/s
    (fresh_decode_score), None unless every case has one, and repeat_speedup, the settled decode score over it: how
    much faster the engine decodes a prompt it has just served than a new one. A case's first request is its
    first_request (the pinned path's first free generation), else its first sample."""
    first = [((c.get("first_request") or (c.get("samples") or [{}])[0]) or {}).get("decode_tps") for c in measured]
    fresh = metrics.geometric_mean(first) if first and all(first) else None
    settled = metrics.geometric_mean([c["decode_tps"] for c in measured]) if measured else None
    return {"fresh_decode_score": fresh,
            "repeat_speedup": round(settled / fresh, 3) if fresh and settled else None}


def speed_summary(cases, protocol, warmups, repetitions, text):
    """The speed part's scores over its cases (run_speed's result): a staged run's cases measured in different
    stages are summarized together again."""
    measured = [c for c in cases.values() if c["status"] == "measured"]
    prefill = [cases[i]["prefill_tps"] for i in protocol["speed"].get("prefill_cases", [])
               if cases.get(i, {}).get("status") == "measured" and cases[i].get("prefill_tps")]
    shortest = min((c for c in measured if c.get("ttft_ms")), key=lambda c: c["prompt_tokens"], default=None)
    return {
        "warmup_runs": warmups, "repetitions": repetitions, "decode_text": text,
        # The first token of the shortest prompt: what a request costs before its prompt does (protocol stages 1).
        "ttft_floor_ms": shortest["ttft_ms"] if shortest else None,
        "ttft_floor_case": next((i for i, c in cases.items() if c is shortest), None),
        "decode_score": metrics.geometric_mean([c["decode_tps"] for c in measured]),
        # Each case's first request, before its warmups: a repeated prompt can decode faster (protocol
        # speed.fresh_decode; reported, not judged).
        **fresh_decode(measured),
        "decode_se": metrics.score_error([[s["decode_tps"] for s in c["samples"] if not s["warmup"]] for c in measured]),
        "prefill_score": metrics.geometric_mean(prefill)
        if prefill and len(prefill) == len(protocol["speed"].get("prefill_cases", [])) else None,
        "ttft_score_ms": metrics.geometric_mean([c["ttft_ms"] for c in measured if c.get("ttft_ms")]),
        "complete": len(measured) == len(cases),
        "cases": cases,
    }


def run_sampled_speed(engine, info, prepared, protocol, warmups, repetitions, log, disturbed_spread=None,
                      checked=None, adaptive=False):
    """The speed cases again at the model's own sampling (protocol sampled_speed), each repetition with its own
    fixed seed: `repetitions` of them (None: one per protocol seed), with no extra repetition, since different seeds
    decode different texts. A disturbed device shows in the time per target pass, which the text does not change.
    The sampled decode score is the headline decode from crit-eval-v7 on. checked: a dict that gets each case's
    warmup reply with its log-probabilities (protocol sampling_check top_logprobs), the sampling check's replies;
    a warmup the engine cannot serve with them runs without, and the check sends its own. adaptive (protocol
    sampled_speed.adaptive, on request): after its repetitions, a case with one repetition (no error yet) or whose
    relative standard error exceeds target_se gets one more repetition at the next seed until its error is within
    target_se or it has max_repetitions. Repetitions past the protocol's seeds take the next seeds (2, 3, ... after
    1). Every case reports its standard error (None measured at one seed) and the repetitions target_se would take."""
    rule = protocol["sampled_speed"]
    settings = prepared.get("sampling")
    if not hasattr(engine, "generate_sampled"):
        log("sampled speed: unsupported (the engine has no generate_sampled)")
        return {"status": "unsupported", "reason": "the engine cannot sample (no generate_sampled)"}
    if settings is None:
        return {"status": "unsupported", "reason": "the prepared data names no sampling (prepare it again)"}
    # An engine that names the sampling fields it applies (sampling_fields) leaves out the others, which only
    # neutral values may be.
    fields = getattr(engine, "sampling_fields", None)
    missing = unapplied(sampled_request(settings, 0), fields) if fields is not None else []
    if missing:
        log(f"sampled speed: unsupported (the engine does not apply {', '.join(missing)})")
        return {"status": "unsupported", "reason": f"the engine does not apply the sampling's {', '.join(missing)}"}
    listed = list(rule["seeds"]["repetitions"])
    repetitions = len(listed) if repetitions is None else repetitions

    def seed_at(j):
        """The j-th measured repetition's seed: the protocol's, then the next ones."""
        return listed[j] if j < len(listed) else max(listed) + 1 + j - len(listed)
    steady = rule.get("steady_state") and protocol["speed"].get("steady_state")

    def full_length(sample):
        """Whether a reply decoded to its full length, the only ones whose time per pass shows the device."""
        return (sample["output_tokens"] == n and bool(sample.get("round_ms"))
                and (sample.get("served") or {}).get("finish_reason", "length") == "length")
    more = rule.get("adaptive") or {}
    target, cap = float(more.get("target_se", 0.03)), int(more.get("max_repetitions", repetitions))
    cases = {}
    for case in prepared["speed"]:
        prompt, n = case["prompt_ids"], case["output_tokens"]
        if len(prompt) + n > info["max_context"]:
            cases[case["id"]] = {"status": "skipped", "reason": f"needs {len(prompt) + n} tokens of context"}
            continue
        samples = []
        load = HostLoad().start()
        # A first measured repetition whose time per target pass still moves away from the request before it stands
        # as one more warmup, and its seed is asked again (protocol sampled_speed.steady_state).
        warm, added, i = warmups, 0, 0
        while i < warm + repetitions:
            request = sampled_request(settings, rule["seeds"]["warmup"] if i < warmups else seed_at(i - warm))
            if i == 0 and i < warmups and checked is not None:
                reported = None
                try:   # the warmup's reply, its timing unused, is the sampling check's
                    _, sample, reported = timed_sampled(engine, prompt, n, request,
                                                        int(protocol["sampling_check"]["top_logprobs"]))
                except RuntimeError as error:
                    log(f"sampled speed {case['id']}: the warmup with log-probabilities failed ({error}); "
                        "warming up without them")
                    _, sample, _ = timed_sampled(engine, prompt, n, request)
                if reported:
                    checked[case["id"]] = {"logprobs": reported, "seed": request.get("seed")}
            else:
                _, sample, _ = timed_sampled(engine, prompt, n, request)
            sample["warmup"] = i < warm
            if (not sample["warmup"] and full_length(sample) and settling(samples, sample, steady, key="round_ms")
                    and full_length(samples[-1])):
                sample["warmup"], warm, added = True, warm + 1, added + 1
            samples.append(sample)
            i += 1
        speeds = lambda: [s["decode_tps"] for s in samples if not s["warmup"] and s.get("decode_tps")]  # noqa: E731
        extra = 0
        while adaptive and repetitions + extra < cap and (len(speeds()) < 2
                                                          or (metrics.case_error(speeds())[0] or 0.0) > target):
            # The next seed after those measured (2, 3, ...): one more text, until the case's error reaches the target.
            seed = seed_at(repetitions + extra)
            _, sample, _ = timed_sampled(engine, prompt, n, sampled_request(settings, seed))
            sample.update(warmup=False, seed=seed)
            samples.append(sample)
            extra += 1
        error, deviation = metrics.case_error(speeds())
        other = load.cores()
        # Only replies decoded to their full length show the device: one a server ended early, at an end token or a
        # tool call, has its time per pass from a few tokens and from how the server streamed the call.
        full = [s for s in samples if not s["warmup"] and full_length(s)]
        if len(full) >= 2:
            check_disturbed(case["id"], full, disturbed_spread, other, key="round_ms")
        summary = _summary([s for s in samples if not s["warmup"]])
        cases[case["id"]] = {"status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n,
                             **summary, "samples": samples,
                             "steady_state": steady_state(samples, steady, added, key="round_ms"),
                             "host_other_cpu_cores": None if other is None else round(other, 2),
                             # The case's relative standard error over its measured texts, the repetitions it took,
                             # and how many target_se would take at its spread.
                             "decode_se": error, "repetitions": repetitions + extra,
                             "repetitions_for_target": metrics.repetitions_for(deviation, target)}
        log(f"sampled speed {case['id']}: decode {rate(summary['decode_tps'], 1)}, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}"
            + (f", standard error {error:.1%} over {repetitions + extra} repetitions" if error is not None else ""))
    measured = [c for c in cases.values() if c["status"] == "measured"]
    rounds = [c["tokens_per_round"] for c in measured if c.get("tokens_per_round")]
    return {"status": "measured", "sampling": settings, "sampling_source": prepared.get("sampling_source"),
            "warmup_runs": warmups, "repetitions": repetitions,
            "adaptive": {"target_se": target, "max_repetitions": cap} if adaptive else None, "target_se": target,
            "decode_score": metrics.geometric_mean([c["decode_tps"] for c in measured]),
            **fresh_decode(measured),
            "decode_se": metrics.score_error([[s["decode_tps"] for s in c["samples"] if not s["warmup"]]
                                              for c in measured]),
            "tokens_per_round": metrics.geometric_mean(rounds) if rounds else None,
            "complete": len(measured) == len(cases), "cases": cases}


def run_sampling_check(engine, info, prepared, protocol, log, checked=None):
    """protocol sampling_check: each speed case's sampled reply with log-probabilities, the sampled suite's warmup
    reply where it has one (checked, from run_sampled_speed), else one more reply, and whether its tokens follow the
    processed distribution (metrics.sampling_calibration)."""
    rule = protocol["sampling_check"]
    settings = prepared.get("sampling")
    if not hasattr(engine, "generate_sampled") or settings is None:
        return {"status": "unsupported", "reason": "the engine cannot sample"}
    top = int(rule["top_logprobs"])
    rows, cases, sources = [], {}, set()
    for case in prepared["speed"]:
        prompt, n = case["prompt_ids"], case["output_tokens"]
        if len(prompt) + n > info["max_context"]:
            continue
        if case["id"] in (checked or {}):
            reported = checked[case["id"]]["logprobs"]
            sources.add("warmup")
        else:
            try:
                result = engine.generate_sampled(list(prompt), n, lambda tokens: None,
                                                 sampled_request(settings, rule["seed"]), logprobs=top)
            except RuntimeError as error:
                log(f"sampling check: unsupported ({error})")
                return {"status": "unsupported", "reason": f"a sampled reply with log-probabilities failed: {error}"}
            reported = result.get("logprobs")
            sources.add("own request")
        if not reported:
            log("sampling check: unsupported (the engine reports no log-probabilities for sampled tokens)")
            return {"status": "unsupported", "reason": "the engine reports no log-probabilities for sampled tokens"}
        # A token missing from its alternatives is one of the unreported rest (metrics.sampling_calibration).
        case_rows = [{"top": [lp for _, lp in r["top"]],
                      "chosen": next((i for i, (t, _) in enumerate(r["top"]) if t == r["token"]), None)}
                     for r in reported]
        rows += case_rows
        cases[case["id"]] = metrics.sampling_calibration(case_rows, settings, rule["seed"], prepared.get("valid_vocab"),
                                                         float(rule["tail_tolerance"]))
    found = metrics.sampling_calibration(rows, settings, rule["seed"], prepared.get("valid_vocab"),
                                         float(rule["tail_tolerance"]))
    enough = found["tokens"] >= int(rule["min_tokens"])
    passed = (found["outside"] == 0 and abs(found["z"]) <= float(rule["z_max"])) if enough else None
    log(f"sampling check: {found['tokens']} tokens judged, {found['unresolved']} unresolved, z "
        f"{found['z'] if found['z'] is None else round(found['z'], 2)}, {found['outside']} outside the processed "
        "distribution" + ("" if enough else f" (fewer than {rule['min_tokens']} judged: unmeasured)"))
    return {"status": "measured", "sampling": settings, "sampling_source": prepared.get("sampling_source"),
            **found, "z_max": float(rule["z_max"]), "min_tokens": int(rule["min_tokens"]), "passed": passed,
            "replies": " and ".join(sorted(sources)), "cases": cases}


def run_heldout(engine, info, prepared, protocol, log, jitter=None):
    """The held-out prompts (protocol "heldout"): each decodes its own greedy output on the engine built for the
    required context, cut to the run's drawn length (`jitter` {"seed", "fraction"}, protocol jitter; None: whole).
    Data prepared before the held-out set has none; it is reported as missing."""
    settings = protocol["heldout"]
    if "heldout" not in prepared:
        log("held-out: not in the prepared data (prepare it again with crit eval prepare into a new data directory)")
        return {"status": "missing", "reason": "the prepared data has no held-out prompts; prepare it again"}
    prompts = {}
    for prompt in prepared["heldout"]:
        ids, drawn = heldout_prompt(prepared, prompt, jitter)
        n = prompt["output_tokens"]
        if len(ids) + n > info["max_context"]:
            prompts[prompt["id"]] = {"status": "skipped", "reason": f"needs {len(ids) + n} tokens of context"}
            continue
        samples = []
        for _ in range(int(settings["repetitions"])):
            _, sample = timed_generate(engine, ids, n)
            samples.append(sample)
        summary = _summary(samples)
        prompts[prompt["id"]] = {"status": "measured", "language": prompt["language"], "kind": prompt["kind"],
                                 "prompt_tokens": len(ids), "output_tokens": n,
                                 **({"nominal_tokens": drawn["nominal_tokens"]} if drawn else {}),
                                 **({"jitter_skipped": drawn["reason"]} if drawn and drawn.get("reason") else {}),
                                 **summary, "samples": samples}
        log(f"held-out {prompt['id']} ({prompt['language']}): decode {rate(summary['decode_tps'], 1)}, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}")
    measured = [p for p in prompts.values() if p["status"] == "measured"]
    return {"status": "measured", "prompts": prompts,
            "decode_score": metrics.geometric_mean([p["decode_tps"] for p in measured]),
            "tokens_per_round": metrics.geometric_mean([p["tokens_per_round"] for p in measured if p.get("tokens_per_round")]),
            "complete": len(measured) == len(prompts)}


def run_wider_draft(module, model, config, prepared, speed, wider, log, context_tokens=None):
    """protocol headroom.wider_draft: the drafts accepted per round at a wider draft width than the engine is built
    with, on an engine the adapter's wider_draft ({"width": W, "config": overrides}) opens: one greedy request per
    measured speed case, its acceptance only (its speed is not the engine's), so the headroom projects draft widths
    up to W instead of stopping at the built one."""
    width, overrides = int(wider["width"]), dict(wider.get("config") or {})
    engine, info = open_engine(module, model, {**config, **overrides}, speculative=True, context_tokens=context_tokens)
    cases = {}
    try:
        for case in prepared["speed"]:
            if (speed["cases"].get(case["id"]) or {}).get("status") != "measured":
                continue
            _, sample = timed_generate(engine, case["prompt_ids"], case["output_tokens"])
            if sample.get("accepted"):
                cases[case["id"]] = {"accepted": sample["accepted"], "tokens_per_round": sample["tokens_per_round"]}
    finally:
        engine.close()
    log(f"wider draft: acceptance at width {width} on " + (", ".join(cases) or "no case (no accepted counts)"))
    return {"width": width, "engine_width": info.get("draft_width"), "config": overrides, "cases": cases}


def run_exactness(module, model, config, prepared, speed, log, mode="reported", context_tokens=None, engine=None,
                  info=None):
    """Plain greedy output of every measured speed case against the speculative output. With `engine` (a
    speculative engine whose plain_on_engine holds) the plain requests run on it; otherwise on a second
    engine opened with speculative false."""
    shared = engine is not None
    if not shared:
        engine, info = open_engine(module, model, config, speculative=False, context_tokens=context_tokens)
    generate = timed_plain if shared else timed_generate
    try:
        cases, samples, steps = {}, [], {}
        for case in prepared["speed"]:
            measured = speed["cases"].get(case["id"], {})
            if measured.get("status") != "measured":
                continue
            tokens, sample = generate(engine, case["prompt_ids"], case["output_tokens"])
            samples.append(sample)
            steps[case["id"]] = headroom.step_ms_from([sample])
            diverge = next((i for i, (a, b) in enumerate(zip(tokens, measured["tokens"])) if a != b), None)
            agreeing = len(tokens) if diverge is None else diverge
            cases[case["id"]] = {"identical": diverge is None, "first_difference": diverge,
                                 "agreeing_prefix": agreeing / max(len(tokens), 1)}
            log(f"exactness {case['id']}: {'identical' if diverge is None else f'differs at token {diverge}'}")
        return {"mode": mode, "baseline": info["configuration"], "engine": "shared" if shared else "separate",
                "cases": cases,
                "identical": all(c["identical"] for c in cases.values()),
                "passed": mode == "reported" or all(c["identical"] for c in cases.values()),
                "step_ms": headroom.step_ms_from(samples), "steps_ms": steps}
    finally:
        if not shared:
            engine.close()


def drafts(speed):
    """Whether any measured speed case decoded more than one token per target pass. The first token comes from the
    prefill's logits, so a plain engine counts one round per later token (n tokens over n - 1 rounds, n / (n - 1)
    tokens per round); only fewer rounds than that mean accepted drafts."""
    return any(s.get("rounds") is not None and s["rounds"] < s["output_tokens"] - 1
               for c in (speed or {}).get("cases", {}).values() if c.get("status") == "measured"
               for s in c.get("samples", []) if not s.get("warmup"))
