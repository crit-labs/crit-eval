"""The history parts of a run: follow-up turns run again after everything else, and greedy requests repeated on
the fresh engine and after what it served."""
from ..reports import metrics
from .timing import timed_generate


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
