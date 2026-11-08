"""Run one engine through the suite: `crit eval run --adapter FILE --model DIR --data DIR --out FILE`."""
import datetime
import time
from contextlib import contextmanager
from pathlib import Path

from .. import CONTRACT, EVAL_DIR, __version__, load_json, machine_settings, portable_path, write_json
from ..quality import fixed
from ..limits import headroom
from . import host, journal
from ..reports import status
from ..limits.batched import batch_terms
from ..limits.ceiling import best_rates, model_ceiling, model_config
from .chat import Server
from .concurrency import run_concurrency
from .deadline import Hung, RequestDeadline
from .estimate import estimate, summary as estimate_summary
from ..limits.context import GIB, draft_step_bytes, interpolated_curve, link_gbs, pass_bytes, placement, weight_split
from ..limits.tiers import derive as derive_tiers, telemetry_counts
from ..limits import devices as device_set
from .pin import load_continuations
from ..quality.prepare import PREPARED, check_prepared, check_template, model_description
from ..quality.reference import REFERENCE_VERSION
from .adapters import file_sha256, load_adapter, native_identity, open_engine, plain_on_engine
from .forks import judge_forks
from .history import compare_followup, compare_history, history_cases, history_outputs, run_followup
from .jitter import new_seed, record as jitter_record
from .tracks import closed as closed_track, engine_loads, mode_budgets
from .long_context import compare_lower, context_requirement, run_context, run_lower_context, run_retrieval
from .quality_checks import (merge_honesty, merge_quality, quality_thresholds, reference_origin, run_attribution,
                             run_honesty, run_quality, stored_spread)
from .speed import (drafts, run_exactness, run_heldout, run_sampled_speed, run_sampling_check, run_speed,
                    run_wider_draft, speed_summary)
from .stages import (choose, creep, exactness_cases, memory_fit, observations, price, quality_rows, quality_sets,
                     short_cases, shortest_case, stage_requests, staged_settings, with_quality, with_speed)
from .estimate import duration, rates
from .timing import Disturbed, decode_text

EXACTNESS_MODES = ("reported", "required")


PARTS = ("speed", "quality", "exactness", "context", "heldout", "history", "followup", "concurrency")


class PartTimes:
    """Wall seconds each part of a run took, in the order the parts ran (result timing.parts), and the run's total
    from its start to the result (timing.total_seconds), on the round's clock (crit_eval.run.journal.clock: a
    resumed round's requests answered from its journal count their recorded seconds). A part that runs again after
    everything else (the follow-up turns, the request history) is an entry of its own. Each part's start is a line
    of the round's journal."""

    def __init__(self):
        self.started = journal.clock()
        self.parts = {}

    @contextmanager
    def part(self, name):
        written = journal.active()
        if written is not None:
            written.mark(name)
        started = journal.clock()
        try:
            yield
        finally:
            self.parts[name] = round(self.parts.get(name, 0.0) + journal.clock() - started, 3)

    def record(self):
        return {"parts": dict(self.parts), "total_seconds": round(journal.clock() - self.started, 3)}

    def summary(self):
        return ", ".join(f"{name} {seconds:.0f} s" for name, seconds in self.parts.items())


def run(adapter, model, data, out, config=None, parts=PARTS, warmups=None,
        repetitions=None, label=None, log=print, bandwidth_gbs=None, device=None, baseline=None,
        quality_budget=None, exactness=None, context_tokens=None, vram_headroom_bytes=None, context_reserve_bytes=None,
        abort_disturbed=True, url=None, served_model=None, requested_context=None, context_basis=None, profile=None,
        chat_template=None, attribute_drift=False, adaptive_sampled=False, quality_mode=fixed.DEFAULT_QUALITY_MODE,
        stage=None, certify=False, budget_seconds=None, journaled=True, resume=None, identity=None, jitter_seed=None):
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
    (run_sampled_speed; crit eval run --adaptive-sampled), recorded in the result (sampled_speed.adaptive); off by
    default, the seeds' repetitions alone (sampled_speed.adaptive null). A fixed repetition count (--repetitions, a
    profile's sampled_repetitions) is never extended.
    quality_mode (protocol quality.modes): how much output drift from the checkpoint the engine may trade for speed,
    never how the checkpoint is stored; both modes judge against the same reference scores (the checkpoint's stored
    weights, exact activations). \"speed\" (the default) judges by its own budget (protocol
    quality.mode_thresholds), output differences that part at near-ties allowed (protocol forks) and the honesty
    check bounded (quality.honesty.modes); "preserve" by the protocol's thresholds. The result records it in
    requirements and its status prices prefill against the arithmetic that mode admits.
    stage: a run without a profile measures in stages (protocol stages): None goes as far as budget_seconds (default
    the protocol's) allows at the engine's measured rates, 1, 2 or 3 at most that far, "full" every part at the
    protocol's repetitions as before stages (warmups and repetitions apply to it alone). certify: the candidate that
    will ship, stage 3 whatever the budget, one request at the requirement itself, which the capacity gate needs, and
    the speed cases on an engine opened at the lower context, which the lower-context gate needs.
    journaled: write the round's journal beside `out` (crit_eval.run.journal: R.json's is R.journal.jsonl), every
    engine opened and request served one synced line as it completes; False writes none. resume: an earlier round's
    journal to go on from: its requests are answered again at their recorded times while this round makes the same
    ones, and an engine loads only for the requests left. identity: what else (a caller's package, attempt owner)
    makes two rounds the same measurement, beside the data, protocol, adapter and options this run records; a
    resumed journal's must be the same. A round that stops (interrupted, failed, a disturbed device, a hung engine)
    writes what it measured to `out`, marked aborted, and its journal's end.
    jitter_seed: the seed that draws the held-out prompts' and the curve's lengths (protocol jitter; crit eval run
    --jitter-seed): a new one per run by default, the resumed journal's when resuming; the result records it and
    every length it drew."""
    times = PartTimes()
    if quality_mode not in fixed.QUALITY_MODES:
        raise SystemExit(f"quality mode must be one of {', '.join(fixed.QUALITY_MODES)}")
    protocol = load_json(EVAL_DIR / "protocol.json")
    thresholds = quality_thresholds(protocol, quality_budget, mode=quality_mode)
    exactness = exactness or protocol["exactness"].get("default", "required")
    if exactness not in EXACTNESS_MODES:
        raise SystemExit(f"exactness must be one of {', '.join(EXACTNESS_MODES)}")
    prepared = load_json(Path(data) / PREPARED)
    check_prepared(prepared, protocol, data)
    check_template(prepared, model, data, chat_template)
    device = device or {}
    if device_set.indices(device):
        # Several GPUs (crit eval probe --devices): priced as one device by the faster split of the weights across
        # them (protocol context.devices), the members' memory pooled by max_context.
        try:
            config_ = model_config(model)
            stored_ = pass_bytes(model, float(protocol["context"]["kv_bytes_per_element"]))
            device = device_set.combined(device, stored_, config_.num_hidden_layers, config_.hidden_size)
        except (OSError, ValueError, KeyError):
            device = device_set.combined(device, None, None, None)
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
    staged = None
    if certify and (profile is not None or stage == "full"):
        raise SystemExit("--certify is for a staged run: a profile or --stage full measures the requirement already")
    if profile is None and stage != "full":
        if stage not in (None, 1, 2, 3):
            raise SystemExit("stage must be 1, 2, 3 or full")
        rule = protocol["stages"]
        staged = {"stage": 3 if certify else (stage or 3), "certify": bool(certify),
                  "budget_seconds": None if certify else float(budget_seconds or rule["budget_seconds"])}
        settings = staged_settings(settings, rule, certify)
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
                          **fixed.identity(fixed.texts(prepared), protocol["quality"]["top_k"], prepared["valid_vocab"])}
    reference_scores = fixed.load(data, reference_identity) if "quality" in parts else None
    reference_spread = stored_spread(data, prepared, protocol, reference_identity) if "quality" in parts else None
    opener = None if url or isinstance(adapter, (str, Path)) else adapter
    module = Server(url, prepared, served_model) if url else opener or load_adapter(adapter)
    result = {
        "contract": CONTRACT, "protocol": protocol["protocol"], "profile": profile, "label": label,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "crit_eval_version": __version__,
        "model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
        "model": model_description(model),
        "reference": reference_origin(data) if "quality" in parts else None,
        "host": host.describe(),
        # The adapter by its checkout-relative path (no user's home in results) and its bytes.
        "adapter": None if url or opener else portable_path(adapter),
        "adapter_sha256": None if url or opener else file_sha256(adapter),
        "server": url or getattr(opener, "server", None),
        "device": {k: device.get(k) for k in ("device", "compute_capability", "cuda_runtime", "bandwidth_gbs", "tflops",
                                               "mma_tflops", "host_to_device_gbs", "device_to_host_gbs", "host_threads",
                                               "cpu_isa", "host_expert_gbs", "host_expert_isa",
                                               "host_memory_gbs", "cpu_gflops", "memory_total_bytes",
                                               "memory_free_bytes", "host_memory_total_bytes", "measured_at")}
                   | {k: device[k] for k in ("backend", "arch", "runtime_version", "fp8_encoding", "unsupported_formats",
                                           "device_set") if k in device}
                   if device else None,
        "bandwidth_gbs": bandwidth_gbs,
        "requirements": {"quality_budget": thresholds, "exactness": exactness, "quality_mode": quality_mode},
        "stages": None,
        "context": requirement,
        # The model's arithmetic per token and weight bytes (crit_eval.limits.ceiling), which the floors are priced on.
        "ceiling": ceiling,
    }
    log("host: " + host.summary(result["host"]))
    from ..limits.probe import outdated
    stale = outdated(device)
    if stale:
        result["device"]["outdated"] = stale
        log("device: the device file is older than this probe (" + ", ".join(stale) + "): floors that need what it "
            "lacks read as missing; re-run crit eval probe")
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
    if resume is not None and not journaled:
        raise SystemExit("resuming a round journals it: drop --no-journal")
    if jitter_seed is None and resume is not None:   # a resumed round draws the lengths its journal's round drew
        earlier = journal.read(resume)
        jitter_seed = ((earlier[0].get("identity") or {}).get("jitter_seed") if earlier else None)
    jitter = {"seed": int(new_seed() if jitter_seed is None else jitter_seed),
              "fraction": list(protocol["jitter"]["fraction"])}
    settings = {**settings, "jitter": jitter,
                # The device memory left free, which the headroom's placement of the weights keeps too (headroom.tiers).
                "vram_headroom_bytes": GIB if vram_headroom_bytes is None else int(vram_headroom_bytes)}
    result["jitter"] = jitter_record(jitter)
    # What makes two rounds the same measurement: a journal answers only a round with this identity.
    round_identity = {"protocol": protocol["protocol"], "crit_eval_version": __version__,
                      "model_sha256": result["model_sha256"], "suite_sha256": result["suite_sha256"],
                      "adapter_sha256": result["adapter_sha256"], "server": result["server"],
                      "config": config or {}, "parts": sorted(parts), "warmups": warmups, "repetitions": repetitions,
                      "sampled_repetitions": sampled_repetitions, "profile": profile, "stages": staged,
                      "context_tokens": required, "requirements": result["requirements"],
                      "abort_disturbed": abort_disturbed, "adaptive_sampled": adaptive_sampled,
                      "attribute_drift": attribute_drift, "device": (device or {}).get("device"),
                      "jitter_seed": jitter["seed"], **({"caller": identity} if identity else {})}
    written = journal.Journal(journal.path_for(out), round_identity, resume, log) if journaled else None
    try:
        with written or _nothing():
            times = PartTimes() if written is not None else times   # the round's clock starts with its journal
            try:
                return _measure(result, module, model, config or {}, prepared, protocol, parts, warmups, repetitions,
                                log, continuations, required, settings, exactness, disturbed, bandwidth_gbs,
                                device, ceiling, baseline, out, reference_scores, sampled_repetitions, times,
                                reference_spread, attribute_drift, adaptive_sampled, staged)
            except (Disturbed, Hung) as error:
                # What the run measured until then stays in the result, marked aborted.
                log(f"aborted: {error}")
                _stopped(result, times, out, written, "disturbed" if isinstance(error, Disturbed) else "hung",
                         str(error))
                return result
            except BaseException as error:
                # Interrupted or failed: what it measured stays too, and the journal says how it ended.
                reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                log(f"aborted: {reason}" + (f": {type(error).__name__}: {error}" if reason == "failed" else ""))
                _stopped(result, times, out, written, reason, f"{type(error).__name__}: {error}"[:2000])
                raise
    finally:
        if written is not None and written.ended is None:
            written.end("failed", "the round ended before its result")


@contextmanager
def _nothing():
    yield None


class Counters:
    """The engine's own expert cache counters (its telemetry(), crit_eval.limits.tiers.telemetry_counts) over the
    requests of each speed suite: result telemetry {suite: {"reads", "hits", "hit_rate"}}, the hit rate the memory
    tiers price the expert misses at where the requests report no expert reads (protocol headroom.tiers). A window is
    the counters' change from before a suite's first request to after its last: it covers every request the suite
    sent (warmups and the prompts' prefills included) and, on a server, whatever else the engine served meanwhile; a
    served engine's counters are those its last status report carried (crit serve refreshes /health after each
    request), which may lag the last request. A window is recorded only when its suite completes, and dropped when
    the counters went backwards (the engine opened again). An engine without telemetry(), or one whose telemetry
    names no expert cache, records none."""

    def __init__(self, engine):
        self.engine, self.windows = engine, {}

    def read(self):
        if not callable(getattr(self.engine, "telemetry", None)):
            return None
        try:
            return telemetry_counts(self.engine.telemetry())
        except Exception:   # noqa: BLE001 - counters are a figure, never a reason to stop the round
            return None

    @contextmanager
    def window(self, name):
        before = self.read()
        yield   # a suite that raises records no window: its counters cover requests it did not finish
        after = self.read() if before is not None else None
        if after is not None and after[0] >= before[0] and after[1] >= before[1]:
            row = self.windows.setdefault(name, {"reads": 0, "hits": 0})
            row["reads"] += after[0] - before[0]
            row["hits"] += after[1] - before[1]

    def record(self):
        return {name: {**row, "hit_rate": row["hits"] / row["reads"] if row["reads"] else None}
                for name, row in self.windows.items()}


def placement_tiers(model, device, settings, required, info, ceiling, loaded_free):
    """The model's memory tiers on the device at the required context (protocol headroom.tiers,
    crit_eval.limits.tiers.derive), or None when every weight a pass reads is on the device or Eval cannot place the
    model. loaded_free: the device memory free with the engine opened at the requirement, whose footprint (free memory
    before its load less after it) says which resident weights fit beside its KV cache."""
    before = (device or {}).get("memory_free_bytes")
    held = (before - loaded_free if isinstance(loaded_free, int) and isinstance(before, int) and before >= loaded_free
            else None)
    try:
        where = placement(model, device, settings.get("vram_headroom_bytes", GIB),
                          float(machine_settings().get("ram_headroom_gib", 2.0)) * GIB, settings, required,
                          held_bytes=held, kv_bytes_per_token=info.get("kv_bytes_per_token")) \
            if device and required else None
    except (OSError, ValueError, KeyError, TypeError):
        where = None
    return derive_tiers(where, ceiling)


def declared_hit_rate(info, tiers):
    """The expert cache hit rate an engine's host_bytes_per_pass implies (its declared host bytes beyond the resident
    weights off the device, over the experts a token activates), or None without a declaration or tiers."""
    found = info.get("host_bytes_per_pass")
    active = (tiers or {}).get("active_expert_bytes_per_token")
    if not isinstance(found, dict) or not active or not (found.get("link") or found.get("cpu")):
        return None
    misses = (found.get("link") or 0) + (found.get("cpu") or 0) - (tiers.get("resident_off_device_bytes") or 0)
    return 1.0 - min(1.0, max(0.0, misses / active))


def _stopped(result, times, out, written, reason, detail):
    """Write what a round that stopped measured, marked aborted, and end its journal with the reason."""
    result["aborted"] = {"reason": reason, "detail": detail}
    result["timing"] = times.record()
    result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if written is not None:
        written.end(reason, detail)
        result["journal"] = written.summary(portable_path)
    write_json(out, result)


def _measure(result, module, model, config, prepared, protocol, parts, warmups, repetitions, log, continuations,
             required, settings, exactness, disturbed, bandwidth_gbs, device, ceiling, baseline,
             out, reference_scores=None, sampled_repetitions=None, times=None, reference_spread=None,
             attribute_drift=False, adaptive_sampled=False, staged=None):
    times = times or PartTimes()
    lower = None
    # A staged run measures the lower context in its stage 3, after its engine closed.
    if staged is None and "context" in parts and "speed" in parts and required:
        if settings.get("measure_lower") is False:
            lower = {"context": None, "passed": None,
                     "reason": f"the {result['profile']} profile loads the engine once, at the required context"}
        elif isinstance(module, Server):
            lower = {"context": None, "passed": None, "reason": "a served engine cannot be opened at a lower context"}
        else:
            with times.part("lower_context"):
                lower = run_lower_context(module, model, config, prepared, protocol, required, warmups, repetitions,
                                          log, continuations, disturbed, exact_prompts="quality" in parts)
    started = journal.clock()
    with times.part("load"):
        engine, info = open_engine(module, model, config, speculative=True, context_tokens=required,
                                   exact_prompts="quality" in parts)
    result["load_ms"] = (journal.clock() - started) * 1000
    # A served engine's load count, again when its run ends: the closed track's one load (protocol tracks).
    result["engine_loads"] = {"first": engine_loads(engine), "last": None}
    result["engine"] = info if not info.get("native_libraries") else         {**info, "native_libraries": [portable_path(p) for p in info["native_libraries"]]}
    result["context"]["engine_max_context"] = info["max_context"]
    # protocol context.device_memory: what the device keeps free with this engine built for the required context.
    gpus = device_set.indices(device)   # every GPU of a device set, else the engine's one
    memory = {"after_load": host.with_engine_reading(host.device_memory(indices=gpus), engine)}
    if info.get("native_libraries"):
        result["native_libraries"] = native_identity(info)
    log(f"loaded {info['name']} in {result['load_ms'] / 1000:.1f} s")

    counters = Counters(engine)

    def ceiling_inputs():
        """The batched ceiling's inputs (crit_eval.limits.batched): the model's step terms, bandwidth and rates, and
        for a model whose weights the device cannot all hold its memory tiers with the expert cache's measured hit rate
        (the decode suite's requests, else the engine's counters, else its host_bytes_per_pass; unmeasured without)
        and the probed link rate."""
        try:
            terms = batch_terms(model, float(settings["kv_bytes_per_element"]))
        except (OSError, ValueError, KeyError):
            terms = None
        tiers = placement_tiers(model, device, settings, required, info, ceiling,
                                (memory.get("after_load") or {}).get("free_bytes"))
        rate, source = headroom.measured_hit_rate({**result, "telemetry": counters.record()})
        if rate is None and declared_hit_rate(info, tiers) is not None:
            rate, source = declared_hit_rate(info, tiers), "host_bytes_per_pass"
        host = {"tiers": tiers, "hit_rate": rate, "hit_rate_source": source} if tiers else None
        return terms, bandwidth_gbs, best_rates(device)[0] or None, host

    def concurrency(fits=None):
        """protocol concurrency: after every other part on this engine, the request history's closing request
        included, so no single-stream figure or gate sees it."""
        if "concurrency" in parts and not result.get("stopped_early"):
            with times.part("concurrency"):
                result["concurrency"] = run_concurrency(engine, info, prepared, protocol, result, required, log, fits,
                                                        ceiling_inputs,
                                                        forks=headroom.quality_mode(result) == "speed")
    try:
        if staged is not None:
            plan = _staged(result, engine, info, module, prepared, protocol, parts, log, continuations, required,
                           settings, exactness, disturbed, reference_scores, reference_spread,
                           memory, times, baseline, staged, concurrency, counters)
        else:
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
                with times.part("speed"), counters.window("speed"):
                    result["speed"] = run_speed(engine, info, prepared, protocol, warmups, repetitions, log,
                                                continuations, disturbed, after_first=log_estimate)
                if info.get("served") and not info.get("speculative") and drafts(result["speed"]):
                    # A server that publishes no engine facts but decodes more than a token per target pass speculates:
                    # its exactness is unmeasured, not taken for a plain engine's.
                    info.update(speculative=True, speculative_source="telemetry")
                    log("engine: speculative (its telemetry reports more than one token per target pass)")
                checked = {}
                with times.part("sampled_speed"), counters.window("sampled_speed"):
                    result["sampled_speed"] = run_sampled_speed(engine, info, prepared, protocol, warmups,
                                                                sampled_repetitions, log, disturbed, checked,
                                                                adaptive=adaptive_sampled and sampled_repetitions is None)
                if result["sampled_speed"]["status"] == "measured":
                    with times.part("sampling_check"):
                        result["sampling_check"] = run_sampling_check(engine, info, prepared, protocol, log, checked)
            if "heldout" in parts:
                with times.part("heldout"):
                    result["heldout"] = run_heldout(engine, info, prepared, protocol, log, settings["jitter"])
            elif settings.get("heldout_skipped"):
                result["heldout"] = {"status": "skipped", "reason": settings["heldout_skipped"]}
            stopped = None
            if "quality" in parts:
                with times.part("quality"):
                    result["quality"] = run_quality(engine, info, prepared, protocol, log, reference_scores,
                                                    result["requirements"]["quality_budget"], reference_spread,
                                                    mode=headroom.quality_mode(result))
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
                                                       long_reply["reply"])], mode=headroom.quality_mode(result))
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
                memory["after_curve"] = host.with_engine_reading(host.device_memory(indices=gpus), engine)
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
            if headroom.quality_mode(result) == "speed" and not stopped:
                # protocol forks: each recorded output difference judged on this engine before it closes
                with times.part("forks"):
                    judge_forks(result, engine, prepared, protocol, log)
            concurrency()
        # Only a round that measured to its end reads it: a request after a stop would be a line in the journal that a
        # round resuming it never makes.
        result["engine_loads"]["last"] = engine_loads(engine)
    finally:
        engine.close()
    if counters.windows:
        # The engine's own expert cache counters over each speed suite (protocol headroom.tiers).
        result["telemetry"] = counters.record()
    if staged is not None and plan["lower"]:
        if plan["fits"]("lower_context", plan["lower_seconds"]):
            with times.part("lower_context"):
                # One warm-up per case: the engine at the requirement measured its cases after the history and
                # other parts had warmed it, the lower context's is freshly opened.
                lower = run_lower_context(module, model, config, prepared, protocol, required, 1, 1, log,
                                          continuations, disturbed, exact_prompts="quality" in parts)
    elif staged is not None and "context" in parts and "speed" in parts and required:
        lower = {"context": None, "passed": None, "reason": plan["lower_reason"]}
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
                    info.get("exact_variants"), required, mode=headroom.quality_mode(result))
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
        element = float(settings["kv_bytes_per_element"])
        try:
            stored = pass_bytes(model, element)
        except (OSError, ValueError, KeyError):
            stored = None
        if not facts.get("bytes_per_pass"):
            facts["bytes_per_pass"] = stored
            source = "model files" if facts["bytes_per_pass"] else None
        # protocol headroom.fixed_decode: the model files' bytes per pass and KV per token at the protocol's KV
        # precision (the stored ceiling's is BF16), whatever the engine reports.
        kv = (ceiling or {}).get("kv_bytes_per_token") or 0
        model_pass = ({"bytes_per_pass": stored, "kv_bytes_per_token": kv / 2 * element}
                      if stored and ceiling else None)
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
        # The model's memory tiers on the device at the required context (protocol headroom.tiers): the weights its
        # placement leaves off the device, which an engine that declares no host bytes is priced by. The engine opened
        # at the requirement: its measured footprint says which resident weights fit beside its KV cache.
        loaded = ((result["context"].get("device_memory") or {}).get("after_load") or {}).get("free_bytes")
        tiers = placement_tiers(model, device, settings, required, info, ceiling, loaded)
        result["headroom"] = headroom.summarize(facts, result["speed"], step_ms, bandwidth_gbs, protocol["headroom"],
                                                ceiling=ceiling, tflops=rates or None, steps=steps,
                                                prefill_cases=protocol["speed"].get("prefill_cases", []),
                                                context=result["context"], context_settings=settings,
                                                decode_speed=decode_speed, decode_name=decode_name,
                                                host_rates={"link": link_gbs(device),
                                                            "memory": ((device or {}).get("host_memory_gbs") or {}).get("read")},
                                                model_pass=model_pass, tiers=tiers, telemetry=result.get("telemetry"))
        # An engine whose weights cannot all be on the device: priced by the placement's tiers when it declares no host
        # bytes, and asked for them only when Eval cannot place the model.
        try:
            split = weight_split(model, model_config(model))
        except (OSError, ValueError, KeyError):
            split = None
        undeclared = headroom.undeclared_host_bytes(
            info, split and split["resident"] + split["experts"], (device or {}).get("memory_total_bytes"))
        if undeclared and tiers:
            off = result["headroom"]["host_pass"] or {}
            rate = (f"at a {off['hit_rate']:.0%} expert cache hit rate ({off.get('hit_rate_source')})"
                    if off.get("hit_rate") is not None else
                    "with the expert cache's misses unmeasured (no expert_reads and expert_hits)")
            result["headroom"]["warnings"].append(
                f"pass floor priced by the placement's tiers ({tiers['scenario']}): {(off.get('link') or 0) + (off.get('cpu') or 0):,.0f} "
                f"bytes per pass from host memory {rate}; the engine's host_bytes_per_pass would price its own path")
            log("headroom: " + result["headroom"]["warnings"][-1])
        elif undeclared:
            result["headroom"]["missing"].append(undeclared)
            result["headroom"]["warnings"].append("pass floor priced as if every weight were on the device: "
                                                  + undeclared.split(": ", 1)[1])
            log("headroom: " + result["headroom"]["warnings"][-1])
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
    result["jitter"] = jitter_record(settings.get("jitter"), result.get("heldout"), result.get("context"))
    # protocol tracks: what Eval can check of the closed track (the card declares the rest).
    try:
        own = draft_step_bytes(model)
    except (OSError, ValueError, KeyError):
        own = None
    result["track"] = {"closed": closed_track(result, result.get("engine") or {}, own,
                                              float(protocol["tracks"]["draft_tolerance"]), mode_budgets(protocol))}
    result["timing"] = times.record()
    log(f"timing: {times.summary()}; total {result['timing']['total_seconds']:.0f} s")
    written = journal.active()
    if written is not None:
        written.end("completed")
        result["journal"] = written.summary(portable_path)
    result["status"] = status.build(result, protocol, baseline)
    result["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out, result)
    return result


def _staged(result, engine, info, module, prepared, protocol, parts, log, continuations, required, settings, exactness,
            disturbed, reference_scores, reference_spread, memory, times, baseline, staged,
            concurrency, counters):
    """The parts of a staged run on its engine (protocol stages): stage 1, then the highest stage that fits the
    budget at the rates the engine measured (with the certification's request at the requirement when certified),
    the request history again, and the concurrency part, each level while it fits (concurrency(fits)). counters: the
    engine's expert cache counters over each speed suite (Counters). Returns what the run does after the engine
    closed: {"lower": whether stage 3 measures the lower context, "lower_seconds", "lower_reason" when it does not,
    "fits": the budget check}."""
    rule, certify, budget = protocol["stages"], staged["certify"], staged["budget_seconds"]
    record = result["stages"] = {"budget_seconds": budget, "cap": staged["stage"], "certify": certify, "reached": 1,
                                 "skipped": {}, "estimate": None}
    result["context"]["certified"] = certify

    def elapsed():
        return journal.clock() - times.started

    def fits(name, seconds):
        """Whether a part expected to take `seconds` still fits the budget; one that does not is skipped and named."""
        if budget is None or seconds is None or elapsed() + seconds <= budget:
            return True
        record["skipped"][name] = (f"would take about {duration(seconds)} with {duration(max(0.0, budget - elapsed()))}"
                                   " of the budget left")
        log(f"stages: {name} skipped ({record['skipped'][name]})")
        return False

    if required:
        result["context"]["fit"] = memory_fit(memory.get("after_load"), info, required)
    sampled = "speed" in parts and hasattr(engine, "generate_sampled")
    shared = "exactness" in parts and "speed" in parts and plain_on_engine(engine, info)
    # The held-out prompts run in stage 1, so a candidate's first staged Eval measures them and a later candidate's
    # generality gate has a baseline that did (protocol stages 1).
    heldout = "heldout" in parts
    lower = "context" in parts and "speed" in parts and bool(required) and not isinstance(module, Server)
    # Which run part each timing part belongs to: a part the run leaves out costs nothing.
    owner = {"followup_again": "followup", "retrieval": "context", "sampled_speed": "speed",
             "lower_context": "context", "history_again": "history"}
    planned = {stage: {name: rows for name, rows in found.items() if owner.get(name, name) in parts}
               for stage, found in stage_requests(prepared, protocol, required, info, settings, sampled, shared,
                                                  heldout, lower, certify).items()}

    def expected(rows, measured=None):
        """Seconds requests `rows` are expected to take at the rates this engine measured so far (measured: the speed
        cases while the speed part runs), None before them."""
        seen = {"speed": {"cases": measured}} if measured is not None else result
        now = rates(observations(engine, seen, protocol))
        if now is None or not rows:
            return None if now is None else 0.0
        speed_rows = ((seen.get("speed") or {}).get("cases") or {}).values()
        rounds = sorted(c["tokens_per_round"] for c in speed_rows if c.get("tokens_per_round"))
        return sum(price({"rows": rows}, now, rounds[len(rounds) // 2] if rounds else None).values())

    def admit(case, measured):
        """Why stage 1 leaves out a speed case after the first: its expected time and stage 1's held-out prompts,
        quality and request history after it would pass the budget. None when it fits, or before the rates are
        known."""
        if budget is None:
            return None
        need = expected([(len(case["prompt_ids"]), case["output_tokens"], 0, False)], measured)
        after = expected(planned[1].get("heldout", []) + planned[1].get("quality", []) + planned[1].get("history_again", []),
                         measured)
        if need is None:
            return None
        left = budget - elapsed() - (after or 0.0)
        return None if need <= left else (f"over budget: about {duration(need)} with {duration(max(0.0, left))} left "
                                          "before stage 1's held-out prompts and quality")

    def log_estimate(case):
        """The run's first estimate, after its first speed case: stage 1 in all and what each later stage adds."""
        early = rates(observations(engine, {"speed": {"cases": {"first": case}}}, protocol))
        if early is None:
            return
        found = {stage: sum(price(planned.get(stage) or {}, early, case.get("tokens_per_round")).values())
                 for stage in (1, 2, 3, "certify")}
        record["first_estimate"] = {str(k): v for k, v in found.items()}
        log(f"estimate: stage 1 about {duration(elapsed() + found[1])} in all"
            + "".join(f"; {'the certification' if k == 'certify' else f'stage {k}'} would add about {duration(v)}"
                      for k, v in found.items() if k != 1 and v)
            + ("; certified, no budget" if certify else f"; budget {duration(budget)}"))

    fresh, cases = None, []
    if "history" in parts:
        # Before any other request: what this engine answers fresh to the shortest speed case, repeated at the end.
        cases = shortest_case(prepared, info)
        with times.part("history"):
            fresh = history_outputs(engine, cases)
        log("history: fresh outputs of " + (", ".join(fresh) or "no case (none fits the engine)"))
    text = decode_text(info, continuations)
    if "speed" in parts:
        with times.part("speed"), counters.window("speed"):
            result["speed"] = run_speed(engine, info, prepared, protocol, 0, 1, log, continuations, disturbed,
                                        after_first=log_estimate, admit=admit)
        if info.get("served") and not info.get("speculative") and drafts(result["speed"]):
            info.update(speculative=True, speculative_source="telemetry")
            log("engine: speculative (its telemetry reports more than one token per target pass)")
        if result["speed"].get("ttft_floor_ms"):
            log(f"speed: first token of the shortest prompt ({result['speed']['ttft_floor_case']}) after "
                f"{result['speed']['ttft_floor_ms']:,.0f} ms")
    if heldout:
        # protocol stages 1: the held-out prompts on every candidate, so the first staged Eval is a baseline that
        # measured them (protocol heldout.generality).
        with times.part("heldout"):
            result["heldout"] = run_heldout(engine, info, prepared, protocol, log, settings["jitter"])
    short = int(rule["short_quality_tokens"])
    later = quality_sets(prepared, short, above=True)
    stopped = None
    over = {}
    if "quality" in parts:
        # Stage 1's sets in order while they and the request history after them fit the budget.
        first, total = [], 0.0
        tail = expected(planned[1].get("history_again", [])) or 0.0
        for name in quality_sets(prepared, short):
            need = expected(quality_rows(prepared, protocol, info, [name])) if budget is not None else None
            if need is not None and elapsed() + total + need + tail > budget:
                over[name] = (f"over budget: about {duration(need)} with "
                              f"{duration(max(0.0, budget - elapsed() - total - tail))} left")
                log(f"quality {name}: skipped ({over[name]})")
                continue
            first.append(name)
            total += need or 0.0
        if first:
            with times.part("quality"):
                result["quality"] = run_quality(engine, info, with_quality(prepared, first), protocol, log,
                                                reference_scores, result["requirements"]["quality_budget"],
                                                reference_spread, mode=headroom.quality_mode(result))
            if result["quality"].get("status") == "scored":
                result["quality"]["skipped"].update({**over, **{name: "a stage 2 set" for name in later}})
        else:
            result["quality"] = {"status": "unsupported", "over_budget": True, "skipped": over,
                                 "reason": "no quality set fits what stage 1 left of the budget"}
        stopped = result["quality"].get("stopped_early")
    later = [*over, *later]   # a set stage 1 had no room for is scored with stage 2's

    found = rates(observations(engine, result, protocol))
    rounds = sorted(c["tokens_per_round"] for c in ((result.get("speed") or {}).get("cases") or {}).values()
                    if c.get("tokens_per_round"))
    per_round = rounds[len(rounds) // 2] if rounds else None
    load = result["load_ms"] / 1000

    def cost(stage_no, name):
        """Expected seconds of one timing part of a stage, None before the rates are known."""
        rows = (planned.get(stage_no) or {}).get(name)
        return sum(price({name: rows}, found, per_round).values()) if found and rows else (0.0 if found else None)

    if certify:
        stage = 3
        costs = {s: sum(price(planned.get(s) or {}, found, per_round).values()) for s in (2, 3, "certify")} if found else {}
    else:
        stage, costs = choose(planned, found, elapsed(), budget, per_round, load, cost(1, "history_again") or 0.0)
        if found is None and "speed" not in parts:
            # Rates come from the speed part: a run that leaves it out (crit eval run --only context) has none to
            # check a budget against, and runs the later stages' parts it asked for rather than none of them.
            stage = 3
            log("stages: no speed part, so no rates to budget the later stages by: the parts asked for run")
    stage = 1 if stopped else min(stage, staged["stage"])
    if stopped:
        # protocol quality.fail_fast: nothing after quality can pass a run this far over the budget.
        result["stopped_early"] = {"after": "quality", "reason": stopped, "skipped": ["stages 2 and 3"]}
    record.update(reached=stage, estimate={"elapsed_seconds": elapsed(),
                                           "stage_seconds": {str(k): v for k, v in costs.items()},
                                           "fit": found.record() if found else None})
    log(f"stages: stage 1 took {duration(elapsed())}"
        + "".join(f"; {'the certification' if k == 'certify' else f'stage {k}'} would add about {duration(v)}"
                  for k, v in costs.items())
        + ("; certified, no budget" if certify else f"; budget {duration(budget)}")
        + (f": going on to stage {stage}" if stage > 1 else ": stopping after stage 1")
        + (f" (quality stopped early: {stopped})" if stopped else ""))

    def not_run(stage_no):
        return {"status": "skipped", "reason": f"a stage {stage_no} part: this run stopped after stage {stage}"}

    if stage >= 2:
        repeat = short_cases(prepared, info, rule)
        if "speed" in parts and repeat and fits("speed", cost(2, "speed")):
            # The stage 1 request warmed the engine for the repetitions: it stays as their warmup, compared with the
            # first of them (protocol speed.extra_repetition).
            rows = dict(result["speed"]["cases"])
            with times.part("speed"), counters.window("speed"):
                again = run_speed(engine, info, with_speed(prepared, repeat), protocol, 0, int(rule["repetitions"]),
                                  log, continuations, disturbed, warmed=rows)
            rows.update({case_id: row for case_id, row in again["cases"].items() if row["status"] == "measured"})
            result["speed"] = speed_summary(rows, protocol, 0, int(rule["repetitions"]), text)
        if "followup" in parts and fits("followup", cost(2, "followup")):
            with times.part("followup"):
                result["followup"] = run_followup(engine, info, prepared, protocol, log)
        if later and (result.get("quality") or {}).get("status") == "scored" and fits("quality", cost(2, "quality")):
            with times.part("quality"):
                more = run_quality(engine, info, with_quality(prepared, later), protocol, log, reference_scores,
                                   result["requirements"]["quality_budget"], reference_spread,
                                   mode=headroom.quality_mode(result))
            result["quality"] = merge_quality(result["quality"], more)
            stopped = result["quality"].get("stopped_early")
        curve_seconds = (cost(2, "context") or 0.0) + (cost("certify", "context") or 0.0) if found else None
        if "context" in parts and required and not stopped and fits("context", curve_seconds):
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
                                                   long_reply["reply"])], mode=headroom.quality_mode(result))
                        result["quality"] = merge_honesty(result["quality"], more)
                    else:
                        result["quality"]["honesty"]["long_reply"] = {"status": "unmeasured", **long_reply}
            if certify:
                result["context"]["interpolated"] = interpolated_curve(result["context"]["points"], required,
                                                                       settings)
            memory["after_curve"] = host.with_engine_reading(
                host.device_memory(indices=(memory.get("after_load") or {}).get("indices")), engine)
            with times.part("retrieval"):
                result["context"]["retrieval"] = run_retrieval(engine, info, prepared, settings, required, log, asked)
    elif "followup" in parts:
        result["followup"] = not_run(2)
    if stage >= 3 and not stopped:
        if sampled and fits("sampled_speed", cost(3, "sampled_speed")):
            checked = {}
            with times.part("sampled_speed"), counters.window("sampled_speed"):
                result["sampled_speed"] = run_sampled_speed(engine, info, prepared, protocol, 1,
                                                            int(rule["sampled_repetitions"]), log, disturbed, checked)
            if result["sampled_speed"]["status"] == "measured":
                with times.part("sampling_check"):
                    result["sampling_check"] = run_sampling_check(engine, info, prepared, protocol, log, checked)
        if shared and "speed" in result and fits("exactness", cost(3, "exactness")):
            with times.part("exactness"):
                result["exactness"] = run_exactness(module, None, {}, with_speed(prepared, exactness_cases(
                    prepared, info, rule)), result["speed"], log, exactness, engine=engine, info=info)
        if (result.get("followup") or {}).get("status") == "measured" and fits("followup_again",
                                                                             cost(3, "followup_again")):
            with times.part("followup_again"):
                again = run_followup(engine, info, prepared, protocol, lambda message: log("again: " + message))
            result["followup"]["again"] = compare_followup(result["followup"], again)
    else:
        if sampled:
            result["sampled_speed"] = not_run(3)
    if info.get("speculative") and "exactness" in parts and "exactness" not in result and (stage < 3 or info.get("served")):
        result["exactness"] = {"mode": exactness, "status": "unmeasured", "cases": {},
                               "reason": not_run(3)["reason"] if stage < 3 else
                               "the served package declares no plain decoding (chat-plain-v1)"}
    if fresh is not None:
        between = [part for part in ("followup", "speed", "heldout", "quality", "context", "exactness")
                   if (result.get(part) or {}).get("status", "measured") not in ("skipped", "unmeasured", "missing")
                   and part in result
                   and (part != "context" or "points" in result["context"])
                   and (part != "quality" or result["quality"].get("sets"))
                   and (part != "exactness" or result["exactness"].get("engine") == "shared")]
        with times.part("history_again"):
            repeated = history_outputs(engine, cases)
        result["history"] = compare_history(fresh, repeated, between)
        for case_id, row in result["history"]["cases"].items():
            log(f"history {case_id}: " + (f"identical after {', '.join(between) or 'nothing else'}" if row["identical"]
                                          else f"differs at token {row['first_difference']}"))
    if headroom.quality_mode(result) == "speed" and not stopped:
        # protocol forks: each recorded output difference judged on this engine before it closes
        with times.part("forks"):
            judge_forks(result, engine, prepared, protocol, log)
    if not stopped:
        concurrency(fits)
    if required:
        result["context"]["creep"] = creep(result, required)
        projected = result["context"]["creep"]
        if projected and projected.get("decode_tps_at_requirement"):
            log(f"context: decode projected at {required:,} tokens {projected['decode_tps_at_requirement']:.1f} tok/s "
                f"(fitted up to {projected['measured_up_to']:,} tokens over {projected['contexts']} contexts)")
    # protocol stages certify: the lower-context gate opens a second engine, which only the candidate that will ship
    # pays for.
    return {"lower": lower and certify and stage >= 3 and not stopped,
            "lower_seconds": cost("certify", "lower_context") + load if found else None, "fits": fits,
            "lower_reason": ("a served engine cannot be opened at a lower context" if not lower else
                             not_run(3)["reason"] if stage < 3 or stopped else
                             "the lower-context gate runs on a certified run (crit eval run --certify)")}
