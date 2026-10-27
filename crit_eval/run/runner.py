"""Run one engine through the suite: `crit eval run --adapter FILE --model DIR --data DIR --out FILE`."""
import datetime
import time
from contextlib import contextmanager
from pathlib import Path

from .. import CONTRACT, EVAL_DIR, __version__, load_json, portable_path, write_json
from ..quality import fixed
from ..limits import headroom
from . import host
from ..reports import status
from ..limits.ceiling import best_rates, model_ceiling
from .chat import Server
from .deadline import RequestDeadline
from .estimate import estimate, summary as estimate_summary
from ..limits.context import GIB, draft_step_bytes, interpolated_curve, pass_bytes
from .pin import load_continuations
from ..quality.prepare import PREPARED, check_prepared, check_template, model_description
from ..quality.reference import REFERENCE_VERSION
from .adapters import file_sha256, load_adapter, native_identity, open_engine, plain_on_engine
from .history import compare_followup, compare_history, history_cases, history_outputs, run_followup
from .long_context import compare_lower, context_requirement, run_context, run_lower_context, run_retrieval
from .quality_checks import (merge_honesty, quality_thresholds, reference_origin, run_attribution, run_honesty,
                             run_quality, stored_spread)
from .speed import drafts, run_exactness, run_heldout, run_sampled_speed, run_sampling_check, run_speed, run_wider_draft
from .timing import Disturbed, decode_text

EXACTNESS_MODES = ("reported", "required")


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
    url instead, the engine behind that OpenAI-compatible server (crit_eval.run.chat), served_model naming its model.
    device: `crit eval probe` result (bandwidth_gbs, tflops, free memory); bandwidth_gbs overrides its bandwidth.
    baseline: an earlier result to show KPI changes against.
    quality_budget: the run's drift limits over the protocol's for its quality mode (ppl_ratio_max, kld_mean_max,
    kld_p99_max; quality_thresholds). The
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
    template (crit_eval.quality.prepare.check_template) otherwise.
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
    thresholds = quality_thresholds(protocol, quality_budget, quality_mode)
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
    reference_spread = stored_spread(data, prepared, protocol, reference_identity, store) if "quality" in parts else None
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
        # The model's arithmetic per token and weight bytes (crit_eval.limits.ceiling), which the floors are priced on.
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
                # The run's time estimate (crit_eval.run.estimate) from what this engine served so far: a served engine's
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
        # Each format's best probed rate, cuBLASLt's GEMM or the raw tensor cores' (crit_eval.limits.ceiling.best_rates).
        rates, rate_sources = best_rates(device)
        result["headroom"] = headroom.summarize(facts, result["speed"], step_ms, bandwidth_gbs, protocol["headroom"],
                                                ceiling=ceiling, tflops=rates or None, steps=steps,
                                                prefill_cases=protocol["speed"].get("prefill_cases", []),
                                                context=result["context"], context_settings=settings,
                                                decode_speed=decode_speed, decode_name=decode_name)
        # Bytes per pass from the engine, or from the stored model when it reports none (crit_eval.limits.context.pass_bytes).
        result["headroom"]["bytes_per_pass_source"] = source
        result["headroom"]["draft_bytes_per_step_source"] = draft_source
        # Prefill and TTFT priced again with the ceilings by prompt activation format, judged by this result's quality
        # and the baseline's (crit_eval.limits.headroom.price_prompts); the rates' sources recorded.
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
