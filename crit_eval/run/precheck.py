"""The gate pre-check: in minutes, the gates a candidate most often fails a whole Eval round on, before that round
(`crit eval precheck`, and `crit run serve --check --data` for a crit candidate).

A certified round takes most of an hour and judges a gate once it has measured everything before it. The pre-check
asks only what those gates ask, each once: served and scored log-probabilities of the honesty texts and of one
long reply at the curve's first point (honesty), the request history around them (request_history, its forks judged
in quality mode speed), the device memory left after the load (resources), and, on an engine that can be opened again,
the shortest and longest speed cases at a lower context against the required one (lower_context). Its verdicts
are hints with the protocol's own rules, never a gate: a single sample stands in for Eval's repetitions, so the
lower-context comparison allows twice the protocol's tolerance. Each item is {name, state, detail}, state passed,
failed or unverified, as crit's serving checks report theirs."""
from pathlib import Path

from .. import EVAL_DIR, load_json
from ..limits.context import curve_points, curve_prompt, lower_context
from ..quality.fixed import DEFAULT_QUALITY_MODE, QUALITY_MODES
from ..quality.prepare import PREPARED, check_prepared, check_template
from . import host
from .adapters import load_adapter, open_engine
from .chat import Server
from .forks import ForkJudge
from .history import compare_history, history_cases, history_outputs
from .quality_checks import run_honesty
from .timing import timed_generate

# A single sample stands in for Eval's repetitions: the lower-context comparison allows this many times the
# protocol's tolerance.
LOWER_SLACK = 2.0


def _item(name, state, detail, **facts):
    return {"name": name, "state": state, "detail": detail, **facts}


def honesty_item(name, verdict):
    """An item from run_honesty's verdict."""
    if verdict.get("passed") is None:
        return _item(name, "unverified", verdict.get("reason") or "no reply was served with its log-probabilities")
    worst = verdict.get("max_abs_logprob_diff") or 0.0
    if verdict["passed"]:
        return _item(name, "passed", f"{len(verdict['replies'])} replies scored again, max |difference| {worst:.2e} "
                                     f"(tolerance {verdict['tolerance']})", max_abs_logprob_diff=worst)
    return _item(name, "failed", "; ".join(verdict["failures"])[:2000], max_abs_logprob_diff=worst)


def long_reply_item(engine, info, prepared, protocol, settings, required, mode, log):
    """The honesty check's long reply: one request at the curve's first point with log-probabilities, scored again
    after the prompt the server rendered. The point that failed most Eval rounds: long prompts cross attention and
    selection paths short ones never reach."""
    curve = curve_points(required, settings)
    if not curve or curve[0] > info["max_context"]:
        return _item("precheck_long_honesty", "unverified", "no curve point fits the engine's context"), None
    point, n = curve[0], int(settings["output_tokens"])
    prompt = curve_prompt(prepared["retrieval"], point - n, settings["retrieval"]["positions"])
    if prompt is None:
        return _item("precheck_long_honesty", "unverified", f"no room for the curve's prompt at {point:,}"), None
    if not callable(getattr(engine, "reply", None)):
        return _item("precheck_long_honesty", "unverified", "the engine replies over no chat API"), None
    tokens, sample = timed_generate(engine, prompt, n, logprobs=protocol["quality"]["top_k"])
    reply = sample.pop("reply", None)
    if not reply:
        return _item("precheck_long_honesty", "unverified",
                     sample.pop("reply_unmapped", None) or "the engine returned no log-probabilities"), point
    verdict = run_honesty(engine, info, prepared, protocol, log, {}, texts=False, mode=mode,
                          extra=[(f"curve@{point}", reply.pop("prompt_ids", None) or list(prompt), reply)])
    item = honesty_item("precheck_long_honesty", verdict)
    item["context"] = point
    return item, point


def history_item(engine, prepared, protocol, cases, fresh, served, mode):
    """The request history: the cases' outputs right after the engine opened against the same requests after the
    pre-check's other requests. In quality mode speed a difference passes where it parts at a near-tie (protocol forks)."""
    if not cases:
        return _item("precheck_history", "unverified", "no speed case fits the engine's context")
    verdict = compare_history(fresh, history_outputs(engine, cases), served)
    if verdict["passed"]:
        return _item("precheck_history", "passed", f"{len(cases)} requests answered alike after {', '.join(served)}")
    prompts = {case["id"]: case["prompt_ids"] for case in cases}
    judge, failed = ForkJudge(engine, protocol), []
    for case_id, row in verdict["cases"].items():
        if row["identical"]:
            continue
        fork = judge.judge(prompts[case_id], row["fresh"], row["after"]) if mode == "speed" else None
        if fork is not None and fork.get("near_tie"):
            continue
        gap = f", gap {fork['gap']:.2f} nats" if fork and fork.get("gap") is not None else ""
        failed.append(f"{case_id} differs at token {row['first_difference']}{gap}")
    if failed:
        return _item("precheck_history", "failed", "; ".join(failed) + f" (after {', '.join(served)})")
    return _item("precheck_history", "passed", "the replies part only at near-ties (protocol forks)")


def speed_samples(engine, cases):
    """{case id: (decode tok/s, time to first token ms)} of each case's second request (the first warms it)."""
    found = {}
    for case in cases:
        timed_generate(engine, case["prompt_ids"], case["output_tokens"])
        _, sample = timed_generate(engine, case["prompt_ids"], case["output_tokens"])
        found[case["id"]] = (sample.get("decode_tps"), sample.get("ttft_ms"))
    return found


def lower_item(at_required, at_lower, lower, tolerance):
    """Each case's decode and first token at the lower context against the required one, within LOWER_SLACK times
    the protocol's tolerance."""
    slower = []
    for case_id, (decode, ttft) in at_required.items():
        low_decode, low_ttft = at_lower.get(case_id, (None, None))
        if decode and low_decode and low_decode / decode - 1 < -tolerance * LOWER_SLACK:
            slower.append(f"{case_id} decode {decode:.1f} -> {low_decode:.1f} tok/s")
        if ttft and low_ttft and low_ttft / ttft - 1 > tolerance * LOWER_SLACK:
            slower.append(f"{case_id} first token {ttft:,.0f} -> {low_ttft:,.0f} ms")
    if slower:
        return _item("precheck_lower_context", "failed", f"at {lower:,} tokens: " + "; ".join(slower), context=lower)
    return _item("precheck_lower_context", "passed",
                 f"at {lower:,} tokens no case slower than {tolerance * LOWER_SLACK:.0%}", context=lower)


def memory_item(memory, headroom_bytes):
    if not memory or memory.get("free_bytes") is None:
        return _item("precheck_device_memory", "unverified", "nvidia-smi gave no device memory")
    free = memory["free_bytes"]
    state = "passed" if headroom_bytes is None or free >= headroom_bytes else "failed"
    return _item("precheck_device_memory", state, f"{free / 2**30:.2f} GiB free after the load"
                 + (f", the headroom is {headroom_bytes / 2**30:.2f} GiB" if headroom_bytes is not None else ""))


def run_precheck(adapter, model, data, context_tokens, quality_mode=DEFAULT_QUALITY_MODE, config=None, url=None,
                 served_model=None, chat_template=None, lower=True, vram_headroom_bytes=None, log=print):
    """The pre-check's items. adapter: an eval-v1 adapter file or an object that opens engines as an adapter module
    does (create_eval_engine); url instead: the engine behind that server, which cannot be opened again (no
    lower-context item). context_tokens: the context the engine is built for. quality_mode: the run's (protocol
    quality.modes; speed unless the run names preserve). lower: also open the engine at the lower context (it
    closes the first engine first)."""
    if quality_mode not in QUALITY_MODES:
        raise SystemExit(f"quality mode must be one of {', '.join(QUALITY_MODES)}")
    protocol = load_json(EVAL_DIR / "protocol.json")
    prepared = load_json(Path(data) / PREPARED)
    check_prepared(prepared, protocol, data)
    check_template(prepared, model, data, chat_template)
    settings = protocol["context"]
    module = Server(url, prepared, served_model) if url else (
        adapter if not isinstance(adapter, (str, Path)) else load_adapter(adapter))
    required, items = int(context_tokens), []
    engine, info = open_engine(module, model, config or {}, speculative=True, context_tokens=required, exact_prompts=True)
    cases = history_cases(prepared, info)
    try:
        items.append(memory_item(host.device_memory(), vram_headroom_bytes))
        fresh = history_outputs(engine, cases)
        items.append(honesty_item("precheck_honesty",
                                  run_honesty(engine, info, prepared, protocol, log, {}, mode=quality_mode)))
        long_item, point = long_reply_item(engine, info, prepared, protocol, settings, required, quality_mode, log)
        items.append(long_item)
        served = ["the honesty replies"] + ([f"a {point:,}-token prompt"] if point else [])
        items.append(history_item(engine, prepared, protocol, cases, fresh, served, quality_mode))
        at_required = speed_samples(engine, cases) if lower and not url else None
    finally:
        engine.close()
    if at_required is not None:
        longest = max(len(case["prompt_ids"]) + case["output_tokens"] for case in prepared["speed"])
        point = lower_context(required, longest, settings)
        if point is None:
            items.append(_item("precheck_lower_context", "unverified", "no lower context holds the longest speed case"))
        else:
            try:
                engine, _ = open_engine(module, model, config or {}, speculative=True, context_tokens=point,
                                        exact_prompts=True)
                try:
                    items.append(lower_item(at_required, speed_samples(engine, cases), point,
                                            float(settings["tolerance"])))
                finally:
                    engine.close()
            except Exception as error:   # an engine that cannot open again at a lower context fails the gate
                items.append(_item("precheck_lower_context", "failed",
                                   f"opening at {point:,} tokens: {type(error).__name__}: {error}"[:2000]))
    for item in items:
        log(f"precheck {item['name']}: {item['state']} ({item['detail']})")
    return items
