"""Run one engine through the suite: `crit eval run --adapter FILE --model DIR --data DIR --out FILE`."""
import datetime
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import CONTRACT, EVAL_DIR, __version__, load_json, portable_path, write_json
from .hostload import HostLoad
from . import fixed, headroom, host, metrics, status
from .ceiling import best_rates, model_ceiling
from .chat import ChatEngine, Server
from .deadline import Hung, RequestDeadline
from .estimate import estimate, summary as estimate_summary
from .context import (GIB, curve_points, curve_prompt, draft_step_bytes, interpolated_curve, lower_context,
                      max_context, pass_bytes, retrieval_points, retrieval_prompt, token_text)
from .pin import CONTINUATIONS, load_continuations
from .prepare import PREPARED, check_prepared, check_template, model_description, sampled_request, unapplied
from .reference import REFERENCE_VERSION


def reference_origin(data, store=fixed.STORE):
    """Where the data's reference scores came from: the reference that computed them (its version, transformers,
    torch and device, from reference/index.json, or the quality mode's store) and the package they were fetched from
    (crit_eval.packages), or package None for scores computed on this machine. None without stored scores."""
    from .packages import PACKAGE
    index = Path(data) / store / "index.json"
    if not index.is_file():
        return None
    identity = load_json(index).get("identity", {})
    package = load_json(Path(data) / PACKAGE) if (Path(data) / PACKAGE).is_file() else None
    if package and package.get("index_sha256") != hashlib.sha256(index.read_bytes()).hexdigest():
        package = None   # scored again since it was fetched
    origin = {key: identity.get(key) for key in ("reference", "transformers", "torch", "device", "forms",
                                                "long_attention")}
    return {**origin, "package": package and {"name": package["name"], "source": package["source"]}}


def load_adapter(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"crit_eval_adapter_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    # A retained package is inventoried byte for byte: importing its adapter (and the modules it imports) must
    # not write __pycache__ into it.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not hasattr(module, "create_eval_engine"):
        raise SystemExit(f"{path} does not export create_eval_engine(context); see docs/ADAPTER.md")
    return module


def open_engine(module, model, config, speculative, context_tokens=None, exact_prompts=False):
    """context_tokens: the context (prompt plus output) the engine is built for, when one is required. exact_prompts:
    an engine behind a server must render every frozen prompt to its prepared ids from the first request on, as the
    quality honesty check needs (crit_eval.chat), so a wrong template fails at once instead of after the speed parts."""
    context = {"contract": CONTRACT, "model": str(model), "config": dict(config), "speculative": speculative}
    if context_tokens:
        context["context_tokens"] = int(context_tokens)
    engine = module.create_eval_engine(context)
    if exact_prompts and hasattr(engine, "exact_prompts"):
        engine.exact_prompts = True
    info = engine.info()
    for key in ("name", "configuration", "max_context"):
        if key not in info:
            raise SystemExit(f"engine.info() is missing {key!r}")
    return engine, info


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


def run_speed(engine, info, prepared, protocol, warmups, repetitions, log, continuations=None, disturbed_spread=None,
              after_first=None):
    """Speed cases. With a pinned continuation and a replaying adapter, every speed KPI comes from replays
    of the pinned text (identical for every engine) and two free generations record the engine's own
    output for repeatability and exactness; otherwise the KPIs come from free generation. After the case's
    repetitions, one more when they disagree (protocol speed.extra_repetition).
    disturbed_spread: stop the round (Disturbed) when a case's decode speeds spread by more than this.
    after_first: called with the first measured case (the run's time estimate)."""
    tolerance = protocol["speed"]["extra_repetition"]["tolerance"]
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
        samples, outputs = [], []
        load = HostLoad().start()
        continuation = pinned.get(case["id"])
        if pinned and (continuation is None or len(continuation) != n):
            raise SystemExit(f"{CONTINUATIONS} has no {n}-token continuation for speed case {case['id']}")
        if continuation is not None:
            for _ in range(2):
                tokens, _ = timed_generate(engine, prompt, n)
                outputs.append(tokens)
            for i in range(warmups + repetitions + 1):
                if i == warmups + repetitions and not disagree(samples, outputs, tolerance):
                    break
                _, sample = timed_replay(engine, prompt, continuation)
                sample["warmup"] = i < warmups
                samples.append(sample)
        else:
            for i in range(warmups + repetitions + 1):
                if i == warmups + repetitions and not disagree(samples, outputs, tolerance):
                    break
                tokens, sample = timed_generate(engine, prompt, n)
                sample["warmup"] = i < warmups
                samples.append(sample)
                if i >= warmups:
                    outputs.append(tokens)
        other = load.cores()
        check_disturbed(case["id"], samples, disturbed_spread, other)
        summary = _summary([s for s in samples if not s["warmup"]])
        crossed = [i for i, t in enumerate(outputs[0]) if t in stop_ids]
        cases[case["id"]] = {
            "status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n, **summary,
            "repeatable": all(o == outputs[0] for o in outputs),
            "stop_crossed": bool(crossed), "first_stop_index": crossed[0] if crossed else None,
            "tokens": outputs[0], "samples": samples,
            # CPU cores other processes kept busy during the case: engines computing on the host slow down with them.
            "host_other_cpu_cores": None if other is None else round(other, 2),
        }
        log(f"speed {case['id']}: decode {rate(summary['decode_tps'], 1)}"
            + (" (pinned continuation)" if continuation is not None else "")
            + f", prefill {rate(summary['prefill_tps'], 0)}, tokens/round {summary['tokens_per_round'] or 0:.2f}, "
            f"round {summary['round_ms'] or 0:.2f} ms"
            + (f", other processes {other:.1f} CPU cores" if other is not None and other >= 0.5 else ""))
        if after_first is not None:
            after_first(cases[case["id"]])
            after_first = None
    measured = [c for c in cases.values() if c["status"] == "measured"]
    prefill = [cases[i]["prefill_tps"] for i in protocol["speed"].get("prefill_cases", [])
               if cases.get(i, {}).get("status") == "measured" and cases[i].get("prefill_tps")]
    return {
        "warmup_runs": warmups, "repetitions": repetitions, "decode_text": text,
        "decode_score": metrics.geometric_mean([c["decode_tps"] for c in measured]),
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
    sampled_speed.adaptive, the default): after its repetitions, a case whose relative standard
    error exceeds target_se gets one more repetition at the next seed until it does not or it has max_repetitions.
    Every case reports its standard error and the repetitions target_se would take."""
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
    repetitions = len(rule["seeds"]["repetitions"]) if repetitions is None else repetitions
    seeds = [rule["seeds"]["warmup"]] * warmups + list(rule["seeds"]["repetitions"])
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
        for i in range(warmups + repetitions):
            request = sampled_request(settings, seeds[min(i, len(seeds) - 1)])
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
            sample["warmup"] = i < warmups
            samples.append(sample)
        speeds = lambda: [s["decode_tps"] for s in samples if not s["warmup"] and s.get("decode_tps")]  # noqa: E731
        extra = 0
        while adaptive and repetitions + extra < cap and (metrics.case_error(speeds())[0] or 0.0) > target:
            # The next seed after the protocol's (4, 5, ...): one more text, until the case's error reaches the target.
            seed = max(rule["seeds"]["repetitions"]) + 1 + extra
            _, sample, _ = timed_sampled(engine, prompt, n, sampled_request(settings, seed))
            sample.update(warmup=False, seed=seed)
            samples.append(sample)
            extra += 1
        error, deviation = metrics.case_error(speeds())
        other = load.cores()
        # Only replies decoded to their full length show the device: one a server ended early, at an end token or a
        # tool call, has its time per pass from a few tokens and from how the server streamed the call.
        full = [s for s in samples if not s["warmup"] and s["output_tokens"] == n and s.get("round_ms")
                and (s.get("served") or {}).get("finish_reason", "length") == "length"]
        if len(full) >= 2:
            check_disturbed(case["id"], full, disturbed_spread, other, key="round_ms")
        summary = _summary([s for s in samples if not s["warmup"]])
        cases[case["id"]] = {"status": "measured", "kind": case["kind"], "prompt_tokens": len(prompt), "output_tokens": n,
                             **summary, "samples": samples,
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


def run_heldout(engine, info, prepared, protocol, log):
    """The held-out prompts (protocol "heldout"): each decodes its own greedy output on the engine built for the
    required context. Data prepared before the held-out set has none; it is reported as missing."""
    settings = protocol["heldout"]
    if "heldout" not in prepared:
        log("held-out: not in the prepared data (prepare it again with crit eval prepare into a new data directory)")
        return {"status": "missing", "reason": "the prepared data has no held-out prompts; prepare it again"}
    prompts = {}
    for prompt in prepared["heldout"]:
        ids, n = prompt["prompt_ids"], prompt["output_tokens"]
        if len(ids) + n > info["max_context"]:
            prompts[prompt["id"]] = {"status": "skipped", "reason": f"needs {len(ids) + n} tokens of context"}
            continue
        samples = []
        for _ in range(int(settings["repetitions"])):
            _, sample = timed_generate(engine, ids, n)
            samples.append(sample)
        summary = _summary(samples)
        prompts[prompt["id"]] = {"status": "measured", "language": prompt["language"], "kind": prompt["kind"],
                                 "prompt_tokens": len(ids), "output_tokens": n, **summary, "samples": samples}
        log(f"held-out {prompt['id']} ({prompt['language']}): decode {rate(summary['decode_tps'], 1)}, "
            f"tokens/round {summary['tokens_per_round'] or 0:.2f}")
    measured = [p for p in prompts.values() if p["status"] == "measured"]
    return {"status": "measured", "prompts": prompts,
            "decode_score": metrics.geometric_mean([p["decode_tps"] for p in measured]),
            "tokens_per_round": metrics.geometric_mean([p["tokens_per_round"] for p in measured if p.get("tokens_per_round")]),
            "complete": len(measured) == len(prompts)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def native_identity(info):
    """sha256 and size of each native library the adapter declares it loaded (`native_libraries`), each path as
    portable_path stores it."""
    out = []
    for path in info.get("native_libraries") or []:
        entry = {"path": portable_path(path)}
        try:
            entry.update(sha256=file_sha256(path), bytes=Path(path).stat().st_size)
        except OSError as error:
            entry["error"] = error.strerror or type(error).__name__   # str(error) would name the path
        out.append(entry)
    return out


QUALITY_KEYS = ("ppl_ratio_max", "kld_mean_max", "kld_p99_max")
EXACTNESS_MODES = ("reported", "required")


def quality_thresholds(protocol, budget=None):
    """The protocol's drift budget, with a run's own limits (partial is fine) laid over it."""
    budget = budget or {}
    unknown = set(budget) - set(QUALITY_KEYS)
    if unknown:
        raise SystemExit(f"--quality-budget keys are {', '.join(QUALITY_KEYS)}; got {', '.join(sorted(unknown))}")
    for key, value in budget.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < (1.0 if key == "ppl_ratio_max" else 0.0):
            raise SystemExit(f"--quality-budget {key} must be a number of at least {1.0 if key == 'ppl_ratio_max' else 0.0}")
    return {**protocol["quality"]["thresholds"], **{k: float(v) for k, v in budget.items()}}


def _engine_rows(rows):
    return [{"logprob": row["logprob"], "top": row["top"]} for row in rows]


def _phase_timing(timings):
    """Sums of request timings (crit_eval.chat): requests, seconds, the engine's own prefill and decode seconds, the
    rest of the request's seconds (transfer and the server's own work), response bytes and parse seconds."""
    engine = [t for t in timings if t.get("engine_prefill_ms") is not None]
    out = {"requests": len(timings), "seconds": sum(t["seconds"] for t in timings),
           "response_bytes": sum(t["response_bytes"] for t in timings),
           "parse_seconds": sum(t["parse_seconds"] for t in timings)}
    if engine:
        out["engine_prefill_seconds"] = sum(t["engine_prefill_ms"] or 0.0 for t in engine) / 1000
        out["engine_decode_seconds"] = sum(t["engine_decode_ms"] or 0.0 for t in engine) / 1000
        out["outside_engine_seconds"] = sum(t["seconds"] for t in engine) - out["engine_prefill_seconds"] - \
            out["engine_decode_seconds"]
    return out


def set_limits(thresholds, spread, factor):
    """A set's drift limits (runner.quality_thresholds) raised to `factor` times the reference's own spread on it
    (protocol quality.spread): kld_mean_max and kld_p99_max to factor times the spread's, ppl_ratio_max to 1 plus
    factor times the spread ratio's distance from 1. Returns (limits, the keys raised)."""
    if not spread:
        return dict(thresholds), []
    limits, raised = dict(thresholds), []
    for key, name in (("kld_mean_max", "kld_mean"), ("kld_p99_max", "kld_p99")):
        if key in limits and spread.get(name) is not None and factor * spread[name] > limits[key]:
            limits[key], raised = factor * spread[name], raised + [key]
    if "ppl_ratio_max" in limits and spread.get("ppl_ratio") is not None:
        from_spread = 1.0 + factor * abs(spread["ppl_ratio"] - 1.0)
        if from_spread > limits["ppl_ratio_max"]:
            limits["ppl_ratio_max"], raised = from_spread, raised + ["ppl_ratio_max"]
    return limits, raised


def run_quality(engine, info, prepared, protocol, log, reference, thresholds, spread=None):
    """protocol quality: every quality text scored by the engine through its serving paths (an engine's score(prompt,
    continuation, top_k): an engine behind crit serve, crit_eval.chat) and judged against the reference's stored
    scores of the same text (reference: crit_eval.fixed.load, None when the data has none). Each set is judged on
    both paths the engine computes it with: its prompt positions from the prefill (SET/prompt: the arithmetic every
    reply's first token comes from, and the KV and state the decode reads) and the positions it serves (SET/served:
    the continuation, its first token from the prefill's last position, the rest from decode); both per bucket.
    Then the honesty check (run_honesty). A set whose texts do not fit the engine's max_context is skipped. Sets
    are scored in the prepared order and judged as each completes; a set far over the budget (clearly_failed) stops
    the scoring, and the result says so (stopped_early). spread: the reference's own spread per set and path
    (fixed.load_spread), which raises a set's limits where it exceeds the budget (set_limits)."""
    rule = protocol["quality"]
    # A served engine scores through the paths it serves with; an in-process adapter's score() is development evidence:
    # reported, never a gate result (protocol quality.development).
    out = {"top_k": rule["top_k"], "continuation_tokens": rule["continuation_tokens"],
           "evidence": "served" if info.get("served") else "development"}
    if not callable(getattr(engine, "score", None)):
        log("quality: not measured (the adapter's engine has no score(); served engines score through crit serve's "
            "chat-score-v1)")
        return {**out, "status": "unsupported",
                "reason": "quality is measured through crit serve's score call: serve the engine and run crit eval "
                          "run --url (an adapter with score() gives development evidence)"}
    if out["evidence"] == "development":
        log("quality: development evidence (the adapter's own score(), not the served paths): reported, not a gate "
            "result")
    if reference is None:
        log("quality: not measured (no reference scores for this data)")
        return {**out, "status": "unsupported",
                "reason": "the data holds no reference scores for this quality mode: run crit eval reference "
                          "--model --data (with the run's --quality-mode) first"}
    started, skipped, parts, labels, timings = time.time(), {}, {}, {}, []
    sets, buckets, judged, stopped = {}, {}, 0.0, None
    order = list(dict.fromkeys(text["set"] for text in prepared["quality"]))
    for set_name in order:
        if stopped is not None:
            skipped[set_name] = stopped
            continue
        texts = [text for text in prepared["quality"] if text["set"] == set_name]
        need = max(len(t["prompt_ids"]) + len(t["continuation_ids"]) for t in texts)
        if need > info["max_context"]:
            skipped[set_name] = f"needs {need:,} tokens of context"
            continue
        for text in texts:
            prompt, continuation = text["prompt_ids"], text["continuation_ids"]
            scored = engine.score(prompt, continuation, rule["top_k"])
            if scored.get("timing"):
                timings.append({"id": text["id"], "positions": len(prompt) + len(continuation) - 1, **scored["timing"]})
            rows = reference[text["id"]]
            if len(rows) != len(prompt) + len(continuation) - 1:
                raise RuntimeError(f"{text['id']}: the reference scores {len(rows)} positions, the text has "
                                   f"{len(prompt) + len(continuation) - 1}")
            engine_rows = _engine_rows(scored["prompt"] + scored["continuation"])
            last = len(prompt) - 1   # rows before it predict prompt tokens; from it on, the served continuation
            for part, pick, bucket in (("prompt", slice(0, last), lambda i: metrics.log2_bucket(i + 1, 512)),
                                       ("served", slice(last, None), lambda i: metrics.log2_bucket(i - last + 1, 32))):
                reference_rows, found, ids = parts.setdefault(f"{set_name}/{part}", ([], [], []))
                reference_rows += rows[pick]
                found += engine_rows[pick]
                ids.append(text["id"])
                labels.setdefault(f"{set_name}/{part}", []).extend(bucket(i) for i in range(len(rows))[pick])
        judging = time.perf_counter()
        for name in (f"{set_name}/prompt", f"{set_name}/served"):
            reference_rows, found, ids = parts.get(name, ([], [], []))
            if not found:
                continue   # a prompt of a single token has no prompt positions
            row = metrics.text_quality(reference_rows, found)
            limits, raised = set_limits(thresholds, (spread or {}).get(name), float(rule["spread"]["factor"])
                                        if rule.get("spread") else 0.0)
            failures = metrics.judge(row, limits)
            # The share's dependence on the positions the set holds (protocol quality.position_spread): reported.
            noise = rule.get("position_spread") or {}
            spread_found = metrics.budget_spread(reference_rows, found, limits, int(noise.get("block", 64)),
                                                 int(noise.get("draws", 200)), int(noise.get("seed", 0))) if noise else None
            sets[name] = {**row, "texts": ids, "passed": not failures, "failures": failures,
                          "budget_used": metrics.budget_used(row, limits), "limits": limits,
                          **({"budget_spread": spread_found} if spread_found else {}),
                          **({"reference_spread": spread[name], "raised": raised} if raised else {})}
            if raised:
                log(f"quality {name}: judged against the reference's own spread, "
                    + ", ".join(f"{key[:-4]} <= {limits[key]:.3g}" for key in raised))
            buckets[name] = {label: {key: value[key] for key in ("targets", "ppl_ratio", "kld_mean", "kld_p99",
                                                                  "top1_agreement")}
                             for label, value in metrics.bucketed(reference_rows, found, labels[name]).items()}
            log(f"quality {name}: KLD mean {row['kld_mean']:.2e}, p99 {row['kld_p99']:.2e}, PPL x{row['ppl_ratio']:.4f}, "
                f"top-1 {row['top1_agreement']:.3f} over {row['targets']:,} positions"
                + ("" if not failures else " - " + "; ".join(failures)))
        judged += time.perf_counter() - judging
        stopped = clearly_failed(set_name, sets, thresholds, rule.get("fail_fast"))
        if stopped is not None:
            log(f"quality: {stopped}; the remaining sets are not scored")
    for name, reason in skipped.items():
        log(f"quality {name}: skipped ({reason})")
    if not parts:
        return {**out, "status": "skipped", "skipped": skipped,
                "reason": f"no quality text fits the engine's {info['max_context']:,}-token context"}
    honesty = run_honesty(engine, info, prepared, protocol, log, skipped)
    passed = all(s["passed"] for s in sets.values()) and honesty["passed"] is not False
    timing = {"scores": _phase_timing(timings), "metrics_seconds": judged, "texts": timings,
              **({"honesty": honesty.pop("timing")} if "timing" in honesty else {})}
    return {**out, "status": "scored", "sets": sets, "buckets": buckets, "honesty": honesty,
            "skipped": skipped, "thresholds": thresholds, "passed": passed, "seconds": time.time() - started,
            "timing": timing, **({"stopped_early": stopped} if stopped else {})}


def run_attribution(module, model, config, prepared, protocol, log, reference, thresholds, spread, quality, variants,
                    context_tokens=None):
    """protocol quality.attribution: the quality texts scored again on an engine opened with each of the adapter's
    exact_variants ({name: config overrides that switch one matrix group or arithmetic choice back to exact}, or
    {name: {"config": overrides, "description": text}}), one engine at a time; per variant and set, the budget it
    uses and the share of the full engine's that the variant's exactness takes away. Shares need not add up: the
    terms interact."""
    rule = protocol["quality"].get("attribution") or {}
    if quality.get("status") != "scored":
        return {"status": "unmeasured", "reason": "quality was not scored"}
    if not variants:
        return {"status": "unmeasured", "reason": "the adapter declares no exact_variants (info): configuration "
                                                   "overrides that switch one matrix group or arithmetic choice back to exact"}
    base = {name: s["budget_used"] for name, s in quality["sets"].items()}
    out = {"status": "measured", "base": base, "variants": {}}
    for name, value in variants.items():
        overrides = value.get("config", value) if isinstance(value, dict) else {}
        described = value.get("description") if isinstance(value, dict) and "config" in value else None
        engine, info = open_engine(module, model, {**config, **overrides}, speculative=True, context_tokens=context_tokens)
        try:
            scored = run_quality(engine, info, prepared, protocol, lambda line: log(f"attribution {name}: {line}"),
                                 reference, thresholds, spread)
        finally:
            engine.close()
        sets = {set_name: s["budget_used"] for set_name, s in (scored.get("sets") or {}).items()}
        # The share of each set's budget the variant takes away: what this group's (or choice's) arithmetic spends.
        shares = {set_name: (base[set_name] - used) for set_name, used in sets.items()
                  if set_name in base and base[set_name] is not None and used is not None}
        out["variants"][name] = {"config": overrides, "description": described, "budget_used": sets,
                                 "share": shares, "status": scored.get("status"),
                                 **({"stopped_early": scored["stopped_early"]} if scored.get("stopped_early") else {})}
        worst = max(base, key=lambda s: -1 if base[s] is None else base[s])
        log(f"attribution {name}: {worst} uses {sets.get(worst, float('nan')):.1%} of the budget with it exact "
            f"(full engine {base[worst]:.1%})")
    out["rule"] = rule.get("rule")
    return out


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


def clearly_failed(set_name, sets, thresholds, rule):
    """protocol quality.fail_fast: the reason the run stops scoring after `set_name` when either of its paths drifts
    kld_mean_factor times the budget's mean KLD, else None. A set so far over the budget fails whatever follows;
    the remaining texts, the long-context curve and the checks after it would only cost the author time."""
    if not rule or not thresholds.get("kld_mean_max"):
        return None
    limit = float(rule["kld_mean_factor"]) * float(thresholds["kld_mean_max"])
    for path in ("prompt", "served"):
        row = sets.get(f"{set_name}/{path}")
        if row and row["kld_mean"] > limit:
            return (f"{set_name}/{path} drifts a mean KLD of {row['kld_mean']:.3g}, over {rule['kld_mean_factor']}x "
                    f"the budget's {thresholds['kld_mean_max']}")
    return None


def run_honesty(engine, info, prepared, protocol, log, skipped, extra=(), texts=True):
    """protocol quality.honesty: the engine's greedy reply over the chat API to every text of the honesty sets (an
    engine's reply(prompt, max_new_tokens, top_k)), then the same reply tokens scored as the continuation: the served
    and the scored log-probabilities must agree within tolerance, top-k alternatives included, and the top-k lists
    hold the same tokens but those within tolerance of the k-th. extra: (id, prompt ids, served reply) of replies
    already served with their log-probabilities (the long-context curve's first point), judged the same way, their
    tokens scored once; texts False judges only those."""
    rule, top_k = protocol["quality"]["honesty"], protocol["quality"]["top_k"]
    tolerance = rule["tolerance"]
    if not callable(getattr(engine, "reply", None)):
        return {"passed": None, "reason": "the engine replies over no chat API", "tolerance": tolerance}
    replies, failures, worst, timing = [], [], 0.0, {"replies": [], "scores": []}
    asked = [(text["id"], text["prompt_ids"], None) for text in prepared["quality"]
             if texts and text["set"] in rule["sets"] and text["set"] not in skipped] + list(extra)
    for text_id, prompt_ids, served in asked:
        if served is None:
            served = engine.reply(prompt_ids, rule["reply_tokens"], top_k)
        # Only the reply's rows are compared: the prompt is prefilled as for the reply, its positions not scored.
        rescored = engine.score(prompt_ids, served["tokens"], top_k, prompt_rows=False)
        scored = rescored["continuation"]
        for part, value in (("replies", served), ("scores", rescored)):
            if value.get("timing"):
                timing[part].append(value["timing"])
        diff, lists = 0.0, []
        for position, (logprob, top, row) in enumerate(zip(served["logprobs"], served["top"], scored)):
            diff = max(diff, abs(logprob - row["logprob"]))
            mine, theirs = dict(top), dict(row["top"])
            diff = max([diff] + [abs(mine[t] - theirs[t]) for t in mine.keys() & theirs.keys()])
            # A token in only one list must sit at that list's edge: within tolerance of its k-th value.
            for found, other in ((mine, theirs), (theirs, mine)):
                if len(found) < top_k:
                    continue   # alternatives the server could not name by bytes were left out
                edge = min(found.values())
                if any(value > edge + tolerance for t, value in found.items() if t not in other):
                    lists.append(position)
                    break
        worst = max(worst, diff)
        replies.append({"id": text_id, "tokens": len(served["tokens"]), "max_abs_logprob_diff": diff,
                        "top_k_differs_at": lists[:8]})
        if diff > tolerance:
            failures.append(f"{text_id}: scored log-probabilities differ from the served ones by {diff:.2e} "
                            f"(tolerance {tolerance})")
        if lists:
            failures.append(f"{text_id}: the scored and served top-{top_k} lists differ at {len(lists)} tokens")
    log(f"quality honesty: {len(replies)} served replies scored again, max |difference| {worst:.2e}"
        + ("" if not failures else " - " + "; ".join(failures)))
    return {"passed": not failures if replies else None, "tolerance": tolerance, "max_abs_logprob_diff": worst,
            "replies": replies, "failures": failures,
            "timing": {part: _phase_timing(values) for part, values in timing.items()}}


def merge_honesty(quality, more):
    """Fold the honesty verdict `more` (run_honesty of extra replies, timing included) into result part `quality`:
    its replies, failures, worst difference and timing join the first check's, and the part's pass is judged again."""
    honesty = quality.get("honesty") or {}
    if more.get("passed") is None and not more.get("replies"):
        return quality
    merged = {**honesty, "replies": honesty.get("replies", []) + more["replies"],
              "failures": honesty.get("failures", []) + more["failures"],
              "max_abs_logprob_diff": max(honesty.get("max_abs_logprob_diff", 0.0), more["max_abs_logprob_diff"])}
    merged["passed"] = not merged["failures"]
    timing = dict(quality.get("timing") or {})
    if "timing" in more:
        before = timing.get("honesty") or {}
        timing["honesty"] = {part: {key: (before.get(part) or {}).get(key, 0) + value.get(key, 0)
                                    for key in set(before.get(part) or {}) | set(value)}
                             for part, value in more["timing"].items()}
    passed = all(s["passed"] for s in quality.get("sets", {}).values()) and merged["passed"] is not False
    return {**quality, "honesty": merged, "timing": timing, "passed": passed}


def plain_on_engine(engine, info):
    """Whether the speculative engine itself decodes plain one-token greedy (info "plain", generate_plain),
    so exactness needs no second engine."""
    return bool(info.get("speculative") and info.get("plain") and callable(getattr(engine, "generate_plain", None)))


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


def run_followup(engine, info, prepared, protocol, log):
    """protocol followup: the conversation's turns in order, each allowed a prompt cache, with time to first token,
    the prompt tokens the engine reports as cached and the prefill rate of the rest."""
    turns, n = prepared.get("followup"), int(protocol["followup"]["output_tokens"])
    if not turns:
        return {"status": "missing", "reason": "the prepared data has no follow-up turns; prepare it again"}
    if len(turns[-1]["prompt_ids"]) + n > info["max_context"]:
        reason = f"its last turn needs {len(turns[-1]['prompt_ids']) + n:,} tokens of context"
        log(f"followup: skipped ({reason})")
        return {"status": "skipped", "reason": reason}
    rows = []
    for turn in turns:
        tokens, sample = timed_generate(engine, turn["prompt_ids"], n, prompt_cache=True)
        served = sample.get("served") or {}
        prompt = int(served.get("prompt_tokens") or len(turn["prompt_ids"]))
        cached = served.get("cached_tokens")
        fresh = prompt - (cached or 0)
        rows.append({"turn": turn["turn"], "prompt_tokens": prompt, "cached_tokens": cached, "ttft_ms": sample["ttft_ms"],
                     "prefill_tps": fresh * 1000 / sample["ttft_ms"] if sample["ttft_ms"] else None, "tokens": tokens})
        log(f"followup turn {turn['turn']}: {prompt:,} prompt tokens"
            + (f", {cached:,} cached" if cached is not None else "")
            + f", first token after {sample['ttft_ms']:,.0f} ms")
    later = [r for r in rows if r["turn"] > 1]
    return {"status": "measured", "turns": rows,
            "ttft_ms": metrics.geometric_mean([r["ttft_ms"] for r in later if r["ttft_ms"]]),
            "prefill_tps": metrics.geometric_mean([r["prefill_tps"] for r in later if r["prefill_tps"]])}


def compare_followup(first, again):
    """Each turn's output when the turns ran again after every other part against its first output."""
    turns = {}
    for before, after in zip(first["turns"], again["turns"]):
        diverge = next((i for i, (a, b) in enumerate(zip(before["tokens"], after["tokens"])) if a != b), None)
        if diverge is None and len(before["tokens"]) != len(after["tokens"]):
            diverge = min(len(before["tokens"]), len(after["tokens"]))
        turns[str(before["turn"])] = {"identical": diverge is None, "first_difference": diverge,
                                      "cached_tokens": after["cached_tokens"]}
    return {"turns": turns, "passed": bool(turns) and all(t["identical"] for t in turns.values())}


def drafts(speed):
    """Whether any measured speed case decoded more than one token per target pass. The first token comes from the
    prefill's logits, so a plain engine counts one round per later token (n tokens over n - 1 rounds, n / (n - 1)
    tokens per round); only fewer rounds than that mean accepted drafts."""
    return any(s.get("rounds") is not None and s["rounds"] < s["output_tokens"] - 1
               for c in (speed or {}).get("cases", {}).values() if c.get("status") == "measured"
               for s in c.get("samples", []) if not s.get("warmup"))


def history_cases(prepared, info):
    """The speed cases the request-history check repeats (protocol history): those with the shortest and the
    longest prompt the engine holds."""
    held = [c for c in prepared["speed"] if len(c["prompt_ids"]) + c["output_tokens"] <= info["max_context"]]
    if not held:
        return []
    ends = (min(held, key=lambda c: len(c["prompt_ids"])), max(held, key=lambda c: len(c["prompt_ids"])))
    return [c for c in held if any(c is end for end in ends)]


def history_outputs(engine, cases):
    return {case["id"]: timed_generate(engine, case["prompt_ids"], case["output_tokens"])[0] for case in cases}


def compare_history(fresh, after, served):
    """Each case's output on the freshly opened engine against its output after everything `served` names."""
    cases = {}
    for case_id, first in fresh.items():
        later = after.get(case_id) or []
        diverge = next((i for i, (a, b) in enumerate(zip(first, later)) if a != b), None)
        if diverge is None and len(first) != len(later):
            diverge = min(len(first), len(later))
        cases[case_id] = {"identical": diverge is None, "first_difference": diverge, "fresh": first, "after": later}
    return {"cases": cases, "served_between": served,
            "passed": bool(cases) and all(c["identical"] for c in cases.values())}


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


PARTS = ("speed", "quality", "exactness", "context", "heldout", "history", "followup")


class PartTimes:
    """Wall seconds each part of a run took, in the order the parts ran (result timing.parts), and the run's total
    from its start to the result (timing.total_seconds). A part that runs again after everything else (the
    follow-up turns, the request history) is an entry of its own."""

    def __init__(self):
        self.started = time.perf_counter()
        self.parts = {}

    @contextmanager
    def part(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.parts[name] = round(self.parts.get(name, 0.0) + time.perf_counter() - started, 3)

    def record(self):
        return {"parts": dict(self.parts), "total_seconds": round(time.perf_counter() - self.started, 3)}

    def summary(self):
        return ", ".join(f"{name} {seconds:.0f} s" for name, seconds in self.parts.items())


def run(adapter, model, data, out, config=None, parts=PARTS, warmups=None,
        repetitions=None, label=None, log=print, bandwidth_gbs=None, device=None, baseline=None,
        quality_budget=None, exactness=None, context_tokens=None, vram_headroom_bytes=None, context_reserve_bytes=None,
        abort_disturbed=True, url=None, served_model=None, requested_context=None, context_basis=None, profile=None,
        chat_template=None, attribute_drift=False, adaptive_sampled=True, quality_mode="preserve"):
    """adapter: the eval-v1 adapter file, or an object that opens engines as an adapter module does
    (create_eval_engine), its `server` naming them in the result (crit run eval serves a candidate per engine); with
    url instead, the engine behind that OpenAI-compatible server (crit_eval.chat), served_model naming its model.
    device: `crit eval probe` result (bandwidth_gbs, tflops, free memory); bandwidth_gbs overrides its bandwidth.
    baseline: an earlier result to show KPI changes against.
    quality_budget: the run's drift limits over the protocol's (ppl_ratio_max, kld_mean_max, kld_p99_max). The
    `quality` part scores the quality texts through the engine's serving paths and judges every set's prefill and
    decode by these limits against the reference scores `crit eval reference` stored under the data.
    exactness: "reported" (default) or "required" (any difference from plain greedy fails the gate).
    context_tokens: the context every engine is built for (a run's kv_tokens); without it, crit-eval's max context
    for the model on the device, with vram_headroom_bytes (default 1 GiB) left free, capped at requested_context
    (the user's wanted context); context_basis: the run's record of how its kv_tokens came about. The `context` part
    measures the long-context curve and the retrieval check up to it and the speed cases again at a lower context.
    The `heldout` part decodes the held-out prompts; with a baseline that measured them, the generality gate judges
    them. The `history` part repeats protocol history's cases right after the engine opens and after every other
    part ran on it.
    abort_disturbed: stop the round when a speed case's repetitions show a disturbed device (protocol
    speed.disturbed_spread); the result then holds what was measured and `aborted`, and no status.
    profile: a protocol profile's name (protocol profiles), fixing the context, the curve and the lower-context
    gate as it says; the result records it.
    chat_template: the source of the --chat-template file the data must have been prepared with; the data's own
    template (crit_eval.prepare.check_template) otherwise.
    attribute_drift: after quality, score it again on an engine per declared exact variant (run_attribution).
    adaptive_sampled: repeat sampled cases past the protocol's seeds until their error reaches the protocol's target
    (run_sampled_speed), the protocol's rule; False (crit eval run --no-adaptive-sampled) keeps
    the seeds' repetitions alone, an override the result records (sampled_speed.adaptive null), as --repetitions is.
    A fixed repetition count (--repetitions, a profile's sampled_repetitions) is never extended.
    quality_mode: "preserve" judges quality against the reference with exact activations, "as_quantized" against the
    checkpoint run as quantized (its own store of reference scores, crit eval reference --quality-mode as_quantized);
    the result records it in requirements and its status prices prefill against that mode's arithmetic."""
    times = PartTimes()
    if quality_mode not in fixed.QUALITY_MODES:
        raise SystemExit(f"quality mode must be one of {', '.join(fixed.QUALITY_MODES)}")
    store = fixed.STORES[quality_mode]
    protocol = load_json(EVAL_DIR / "protocol.json")
    thresholds = quality_thresholds(protocol, quality_budget)
    exactness = exactness or protocol["exactness"].get("default", "required")
    if exactness not in EXACTNESS_MODES:
        raise SystemExit(f"exactness must be one of {', '.join(EXACTNESS_MODES)}")
    prepared = load_json(Path(data) / PREPARED)
    check_prepared(prepared, protocol, data)
    check_template(prepared, model, data, chat_template)
    device = device or {}
    bandwidth_gbs = bandwidth_gbs or device.get("bandwidth_gbs")
    settings = protocol["context"]
    if context_reserve_bytes is not None:  # the machine's own reserve for workspaces and speed techniques
        settings = {**settings, "reserve_gib": context_reserve_bytes / GIB}
    if profile is not None:
        rules = {k: v for k, v in (protocol.get("profiles") or {}).items() if isinstance(v, dict)}
        if profile not in rules:
            raise SystemExit(f"crit-eval has no profile {profile!r} (protocol profiles: {', '.join(rules) or 'none'})")
        if context_tokens and int(context_tokens) != rules[profile]["context_tokens"]:
            raise SystemExit(f"the {profile} profile measures at {rules[profile]['context_tokens']:,} tokens of "
                             f"context, not {int(context_tokens):,}")
        context_tokens = rules[profile]["context_tokens"]
        settings = {**settings, "curve_points": rules[profile]["curve"], "measure_lower": rules[profile]["lower_context"],
                    "retrieval": {**settings["retrieval"], "points": rules[profile]["retrieval_points"]}}
        if rules[profile].get("heldout") is False and "heldout" in parts:
            parts = tuple(part for part in parts if part != "heldout")
            settings["heldout_skipped"] = f"the {profile} profile measures no held-out prompts: their gate needs a baseline"
        profile_sampled = rules[profile].get("sampled_repetitions")
    else:
        profile_sampled = None
    requirement = context_requirement(model, device, context_tokens,
                                      GIB if vram_headroom_bytes is None else vram_headroom_bytes, settings,
                                      requested_context, context_basis)
    required = requirement["tokens"]
    ceiling = prepared.get("ceiling")
    if ceiling is None:
        try:
            ceiling = model_ceiling(model)
        except (OSError, ValueError, KeyError):
            ceiling = None
    continuations = load_continuations(data, prepared)
    reference_identity = {"model_sha256": prepared["model"]["sha256"], "reference": REFERENCE_VERSION,
                          **({"quality_mode": quality_mode} if quality_mode != "preserve" else {}),
                          **fixed.identity(fixed.texts(prepared), protocol["quality"]["top_k"], prepared["valid_vocab"])}
    reference_scores = fixed.load(data, reference_identity, store) if "quality" in parts else None
    reference_spread = (fixed.load_spread(data, {**reference_identity,
                                                 "spread_perturbation": protocol["quality"]["spread"]["perturbation"]},
                                          store)
                        if "quality" in parts and protocol["quality"].get("spread") else None)
    opener = None if url or isinstance(adapter, (str, Path)) else adapter
    module = Server(url, prepared, served_model) if url else opener or load_adapter(adapter)
    result = {
        "contract": CONTRACT, "protocol": protocol["protocol"], "profile": profile, "label": label,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "crit_eval_version": __version__,
        "model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
        "model": model_description(model),
        "reference": reference_origin(data, store) if "quality" in parts else None,
        "host": host.describe(),
        # The adapter by its checkout-relative path (no user's home in results) and its bytes.
        "adapter": None if url or opener else portable_path(adapter),
        "adapter_sha256": None if url or opener else file_sha256(adapter),
        "server": url or getattr(opener, "server", None),
        "device": {k: device.get(k) for k in ("device", "compute_capability", "cuda_runtime", "bandwidth_gbs", "tflops",
                                               "mma_tflops", "host_to_device_gbs", "device_to_host_gbs", "host_threads",
                                               "cpu_isa",
                                               "host_memory_gbs", "cpu_gflops", "memory_total_bytes",
                                               "memory_free_bytes", "host_memory_total_bytes", "measured_at")}
                   if device else None,
        "bandwidth_gbs": bandwidth_gbs,
        "requirements": {"quality_budget": thresholds, "exactness": exactness, "quality_mode": quality_mode},
        "context": requirement,
        # The model's arithmetic per token and weight bytes (crit_eval.ceiling), which the floors are priced on.
        "ceiling": ceiling,
    }
    log("host: " + host.summary(result["host"]))
    if required:
        log(f"context: {required:,} tokens ({requirement['source']}"
            + (f", {requirement['bound']} bound" if requirement.get("bound") else "") + ")"
            + (f"; {requirement['note']}" if requirement.get("note") else ""))
    else:
        log("context: unknown (" + "; ".join(requirement["missing"] or [requirement["reason"] or "no input"])
            + "); engines use their own")
    warmups = protocol["speed"]["warmup_runs"] if warmups is None else warmups
    sampled_repetitions = repetitions if repetitions is not None else profile_sampled   # None: one per protocol seed
    repetitions = protocol["speed"]["repetitions"] if repetitions is None else repetitions
    disturbed = protocol["speed"].get("disturbed_spread") if abort_disturbed else None
    try:
        return _measure(result, module, model, config or {}, prepared, protocol, parts, warmups, repetitions, log,
                        continuations, required, settings, exactness, disturbed, bandwidth_gbs,
                        device, ceiling, baseline, out, reference_scores, sampled_repetitions, times,
                        reference_spread, attribute_drift, adaptive_sampled)
    except Disturbed as error:
        log(f"aborted: {error}")
        result["aborted"] = {"reason": "disturbed", "detail": str(error)}
        result["timing"] = times.record()
        result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out, result)
        return result


def _measure(result, module, model, config, prepared, protocol, parts, warmups, repetitions, log, continuations,
             required, settings, exactness, disturbed, bandwidth_gbs, device, ceiling, baseline,
             out, reference_scores=None, sampled_repetitions=None, times=None, reference_spread=None,
             attribute_drift=False, adaptive_sampled=True):
    times = times or PartTimes()
    lower = None
    if "context" in parts and "speed" in parts and required and settings.get("measure_lower") is False:
        lower = {"context": None, "passed": None,
                 "reason": f"the {result['profile']} profile loads the engine once, at the required context"}
    elif "context" in parts and "speed" in parts and required and isinstance(module, Server):
        lower = {"context": None, "passed": None, "reason": "a served engine cannot be opened at a lower context"}
    elif "context" in parts and "speed" in parts and required:
        with times.part("lower_context"):
            lower = run_lower_context(module, model, config, prepared, protocol, required, warmups, repetitions, log,
                                      continuations, disturbed, exact_prompts="quality" in parts)
    started = time.perf_counter()
    with times.part("load"):
        engine, info = open_engine(module, model, config, speculative=True, context_tokens=required,
                                   exact_prompts="quality" in parts)
    result["load_ms"] = (time.perf_counter() - started) * 1000
    result["engine"] = info if not info.get("native_libraries") else         {**info, "native_libraries": [portable_path(p) for p in info["native_libraries"]]}
    result["context"]["engine_max_context"] = info["max_context"]
    # protocol context.device_memory: what the device keeps free with this engine built for the required context.
    memory = {"after_load": host.device_memory()}
    if info.get("native_libraries"):
        result["native_libraries"] = native_identity(info)
    log(f"loaded {info['name']} in {result['load_ms'] / 1000:.1f} s")
    try:
        fresh = None
        if "history" in parts:
            # Before any other request: what this engine answers fresh, repeated once everything else has run.
            cases = history_cases(prepared, info)
            with times.part("history"):
                fresh = history_outputs(engine, cases)
            log("history: fresh outputs of " + (", ".join(fresh) or "no case (none fits the engine)"))
        if "followup" in parts:
            with times.part("followup"):
                result["followup"] = run_followup(engine, info, prepared, protocol, log)
        if "speed" in parts:
            def log_estimate(case):
                # The run's time estimate (crit_eval.estimate) from what this engine served so far: a served engine's
                # deadline holds every request (history, follow-up, this case); an adapter's engine keeps none, and
                # the case's samples stand in.
                seen = getattr(engine, "deadline", None)
                if seen is None:
                    seen = RequestDeadline(protocol["requests"]["timeout"])
                    for sample in case["samples"]:
                        seen.observe(case["prompt_tokens"], sample["ttft_ms"] / 1000, sample["output_tokens"],
                                     (sample["request_ms"] - sample["ttft_ms"]) / 1000)
                skip = [name for name, out in (
                    ("lower_context", "lower_context" not in times.parts),   # it ran before this engine, or never
                    ("sampled_speed", not hasattr(engine, "generate_sampled")),
                    ("quality", reference_scores is None or not callable(getattr(engine, "score", None)))) if out]
                found = estimate(prepared, protocol, parts, required, seen, settings, info, warmups, repetitions,
                                 sampled_repetitions, decode_text(info, continuations)["source"] == "pinned", skip,
                                 load_seconds=result["load_ms"] / 1000, tokens_per_round=case["tokens_per_round"],
                                 measured=times.parts)
                if found is not None:
                    result["estimate"] = found
                    log(estimate_summary(found))
            with times.part("speed"):
                result["speed"] = run_speed(engine, info, prepared, protocol, warmups, repetitions, log,
                                            continuations, disturbed, after_first=log_estimate)
            if info.get("served") and not info.get("speculative") and drafts(result["speed"]):
                # A server that publishes no engine facts but decodes more than a token per target pass speculates:
                # its exactness is unmeasured, not taken for a plain engine's.
                info.update(speculative=True, speculative_source="telemetry")
                log("engine: speculative (its telemetry reports more than one token per target pass)")
            checked = {}
            with times.part("sampled_speed"):
                result["sampled_speed"] = run_sampled_speed(engine, info, prepared, protocol, warmups,
                                                            sampled_repetitions, log, disturbed, checked,
                                                            adaptive=adaptive_sampled and sampled_repetitions is None)
            if result["sampled_speed"]["status"] == "measured":
                with times.part("sampling_check"):
                    result["sampling_check"] = run_sampling_check(engine, info, prepared, protocol, log, checked)
        if "heldout" in parts:
            with times.part("heldout"):
                result["heldout"] = run_heldout(engine, info, prepared, protocol, log)
        elif settings.get("heldout_skipped"):
            result["heldout"] = {"status": "skipped", "reason": settings["heldout_skipped"]}
        stopped = None
        if "quality" in parts:
            with times.part("quality"):
                result["quality"] = run_quality(engine, info, prepared, protocol, log, reference_scores,
                                                result["requirements"]["quality_budget"], reference_spread)
            stopped = result["quality"].get("stopped_early")
            if stopped:
                # protocol quality.fail_fast: nothing after quality can pass a run this far over the budget.
                result["stopped_early"] = {"after": "quality", "reason": stopped,
                                           "skipped": [part for part in ("context", "exactness", "followup_again",
                                                                         "history_again")]}
                log(f"stopped early: {stopped}; the long-context curve, exactness and the repeated requests are skipped")
        if "context" in parts and required and not stopped:
            scored = (result.get("quality") or {}).get("status") == "scored"
            with times.part("context"):
                result["context"]["points"], result["context"]["decode_text"], asked, long_reply = run_context(
                    engine, info, prepared, settings, required, log,
                    reply_logprobs=protocol["quality"]["top_k"] if scored else None)
            if scored:
                with times.part("quality"):
                    if long_reply.get("reply"):
                        more = run_honesty(engine, info, prepared, protocol, log, {}, texts=False,
                                           extra=[(f"curve@{long_reply['point']}", long_reply["prompt_ids"],
                                                   long_reply["reply"])])
                        result["quality"] = merge_honesty(result["quality"], more)
                    else:
                        result["quality"]["honesty"]["long_reply"] = {"status": "unmeasured", **long_reply}
                        log(f"quality honesty: no long reply ({long_reply['reason']})")
            if settings.get("curve_points") == "ends":
                result["context"]["interpolated"] = interpolated_curve(result["context"]["points"], required,
                                                                       settings)
                for key, row in result["context"]["interpolated"].items():
                    log(f"context {int(key):,}: interpolated decode {row['decode_tps']:.1f} tok/s, first token "
                        f"after {row['ttft_ms']:,.0f} ms")
            memory["after_curve"] = host.device_memory()
            with times.part("retrieval"):
                result["context"]["retrieval"] = run_retrieval(engine, info, prepared, settings, required, log,
                                                               asked)
        if "exactness" in parts and "speed" in result and plain_on_engine(engine, info) and not stopped:
            # The engine decodes plain greedy itself: no second engine to load.
            with times.part("exactness"):
                result["exactness"] = run_exactness(module, model, config, prepared, result["speed"], log,
                                                    exactness, engine=engine, info=info)
        if result.get("followup", {}).get("status") == "measured" and not stopped:
            with times.part("followup_again"):
                again = run_followup(engine, info, prepared, protocol, lambda message: log("again: " + message))
            result["followup"]["again"] = compare_followup(result["followup"], again)
        if fresh is not None and not stopped:
            served = [part for part in ("followup", "speed", "heldout", "quality", "context", "exactness") if part in result
                      and (part != "followup" or result["followup"].get("status") == "measured")
                      and (part != "context" or "points" in result["context"])
                      and (part != "quality" or result["quality"].get("sets"))
                      and (part != "exactness" or result["exactness"].get("engine") == "shared")]
            with times.part("history_again"):
                repeated = history_outputs(engine, cases)
            result["history"] = compare_history(fresh, repeated, served)
            between = ", ".join(served) or "nothing else"
            for case_id, row in result["history"]["cases"].items():
                log(f"history {case_id}: " + (f"identical after {between}" if row["identical"] else
                                              f"differs at token {row['first_difference']} after {between}"))
    finally:
        engine.close()
    measured_memory = {k: v for k, v in memory.items() if v and v.get("free_bytes") is not None}
    if measured_memory:
        # The least the device kept free while this engine held the required context (after its load, and after the
        # curve filled the context up to it).
        least = min(measured_memory.items(), key=lambda item: item[1]["free_bytes"])
        result["context"]["device_memory"] = {**measured_memory, "free_bytes": least[1]["free_bytes"],
                                              "total_bytes": least[1]["total_bytes"], "least_at": least[0]}
        log(f"device memory: {least[1]['free_bytes'] / 1e9:.2f} of {least[1]['total_bytes'] / 1e9:.1f} GB free "
            f"({least[0].replace('_', ' ')})")
    if attribute_drift and "quality" in result:
        # protocol quality.attribution: engines with one group or choice exact, one after another, after this one.
        with times.part("attribution"):
            if info.get("served"):
                result["quality"]["attribution"] = {"status": "unmeasured",
                                                    "reason": "a served engine cannot be opened with other arithmetic"}
            else:
                result["quality"]["attribution"] = run_attribution(
                    module, model, config, prepared, protocol, log, reference_scores,
                    result["requirements"]["quality_budget"], reference_spread, result["quality"],
                    info.get("exact_variants"), required)
    wider = info.get("wider_draft")
    if (wider and "speed" in result and info.get("speculative") and not info.get("served")
            and int(wider.get("width") or 0) > int(info.get("draft_width") or 0)):
        with times.part("wider_draft"):
            result["speed"]["wider_draft"] = run_wider_draft(module, model, config, prepared, result["speed"], wider,
                                                             log, required)
    if "exactness" in parts and "speed" in result and info.get("speculative") and "exactness" not in result \
            and not result.get("stopped_early"):
        if info.get("served"):
            # Another engine opened for a server would be the same speculative one: comparing it with itself proves
            # nothing.
            reason = "the served package declares no plain decoding (chat-plain-v1)"
            log(f"exactness: unmeasured ({reason})")
            result["exactness"] = {"mode": exactness, "status": "unmeasured", "reason": reason, "cases": {}}
        else:
            with times.part("exactness"):
                result["exactness"] = run_exactness(module, model, config, prepared, result["speed"], log,
                                                    exactness, context_tokens=required)
    if lower is not None and "speed" in lower:
        result["context"]["lower"] = {**compare_lower(result["speed"], lower["speed"], lower["context"], settings),
                                      "engine_max_context": lower["engine_max_context"]}
        for case_id, kinds in result["context"]["lower"]["slower"].items():
            log(f"lower context: {case_id} is slower at {lower['context']:,} tokens ({', '.join(kinds)})")
    elif lower is not None:
        result["context"]["lower"] = lower
    if "speed" in result:
        if info.get("speculative"):
            step_ms = result.get("exactness", {}).get("step_ms")
            steps = result.get("exactness", {}).get("steps_ms")
        else:
            measured = {cid: c for cid, c in result["speed"]["cases"].items() if c["status"] == "measured"}
            step_ms = headroom.step_ms_from([s for c in measured.values() for s in c["samples"] if not s["warmup"]])
            steps = {cid: headroom.step_ms_from([s for s in c["samples"] if not s["warmup"]]) for cid, c in measured.items()}
        decode_name, decode_speed = headroom.decode_suite(result)
        facts, source = dict(info), "engine" if info.get("bytes_per_pass") else None
        if not facts.get("bytes_per_pass"):
            try:
                facts["bytes_per_pass"] = pass_bytes(model, float(settings["kv_bytes_per_element"]))
            except (OSError, ValueError, KeyError):
                facts["bytes_per_pass"] = None
            source = "model files" if facts["bytes_per_pass"] else None
        draft_source = "engine" if info.get("draft_bytes_per_step") else None
        if facts.get("speculative") and not facts.get("draft_bytes_per_step"):
            # A drafting engine that reports none: the checkpoint's own draft layers and the head they read.
            try:
                facts["draft_bytes_per_step"] = draft_step_bytes(model)
            except (OSError, ValueError, KeyError):
                facts["draft_bytes_per_step"] = None
            draft_source = "model files" if facts["draft_bytes_per_step"] else None
        # Each format's best probed rate, cuBLASLt's GEMM or the raw tensor cores' (crit_eval.ceiling.best_rates).
        rates, rate_sources = best_rates(device)
        result["headroom"] = headroom.summarize(facts, result["speed"], step_ms, bandwidth_gbs, protocol["headroom"],
                                                ceiling=ceiling, tflops=rates or None, steps=steps,
                                                prefill_cases=protocol["speed"].get("prefill_cases", []),
                                                context=result["context"], context_settings=settings,
                                                decode_speed=decode_speed, decode_name=decode_name)
        # Bytes per pass from the engine, or from the stored model when it reports none (crit_eval.context.pass_bytes).
        result["headroom"]["bytes_per_pass_source"] = source
        result["headroom"]["draft_bytes_per_step_source"] = draft_source
        # Prefill and TTFT priced again with the ceilings by prompt activation format, judged by this result's quality
        # and the baseline's (crit_eval.headroom.price_prompts); the rates' sources recorded.
        result["headroom"] = headroom.price_prompts(result, rates, rate_sources,
                                                    protocol["speed"].get("prefill_cases", []),
                                                    [baseline] if baseline else [])
        for warning in result["headroom"]["warnings"]:
            log("headroom: " + warning)
    result["timing"] = times.record()
    log(f"timing: {times.summary()}; total {result['timing']['total_seconds']:.0f} s")
    result["status"] = status.build(result, protocol, baseline)
    result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out, result)
    return result
