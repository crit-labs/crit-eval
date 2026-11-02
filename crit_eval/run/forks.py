"""protocol forks: in quality mode as_quantized, greedy replies to the same request may part where the engine's
arithmetic depends on what it served before (an expert computed on the CPU in one request and on the device in the
next, a cache in another state, a prompt read from a prompt cache). Rounding can only flip a near-tie, so a
difference passes the repeatable, request_history and prompt_cache gates when the two replies part where the engine
itself ranks both tokens within `margin` nats of each other. A fork where one token clearly wins (a stale cache entry,
a race reading another request's rows) still fails.

The judge scores the shared prefix plus the first reply's token through the engine's scoring path (chat-score-v1)
once per fork, and reads both tokens' log-probabilities from that position's top_k alternatives.
"""


def first_difference(a, b):
    """The first position where token lists a and b differ (the shorter one's length when one extends the other),
    or None when they are equal."""
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if i is None and len(a) != len(b):
        i = min(len(a), len(b))
    return i


class ForkJudge:
    def __init__(self, engine, protocol):
        rule = protocol["forks"]
        self.engine, self.margin, self.top_k = engine, float(rule["margin"]), int(rule["top_k"])

    def judge(self, prompt_ids, a, b):
        """{"position", "tokens", "logprobs", "gap", "near_tie"} for where replies a and b first differ, or None when
        they are equal. near_tie is None (unjudged) when the engine cannot score."""
        i = first_difference(a, b)
        if i is None:
            return None
        if i >= len(a) or i >= len(b):
            return {"position": i, "near_tie": False, "reason": "one reply ends where the other goes on"}
        try:
            rows = self.engine.score(prompt_ids, list(a[:i + 1]), self.top_k, prompt_rows=False)["continuation"]
        except (AttributeError, RuntimeError, NotImplementedError) as exc:
            return {"position": i, "tokens": [a[i], b[i]], "near_tie": None, "reason": f"not scored: {exc}"}
        row = rows[i]
        top = {int(token): float(logprob) for token, logprob in row["top"]}
        first, second = float(row["logprob"]), top.get(int(b[i]))
        if second is None:
            return {"position": i, "tokens": [a[i], b[i]], "logprobs": [first, None], "gap": None, "near_tie": False,
                    "reason": f"the other reply's token is outside the engine's top {self.top_k} there"}
        gap = first - second
        return {"position": i, "tokens": [a[i], b[i]], "logprobs": [first, second], "gap": gap,
                "near_tie": abs(gap) <= self.margin}


def judge_forks(result, engine, prepared, protocol, log):
    """Judges every recorded difference of the speed repetitions, the request history and the repeated follow-up
    turns on `engine` (quality mode as_quantized), in place: speed cases get forks, history cases and follow-up turns
    a fork."""
    judge = ForkJudge(engine, protocol)
    prompts = {case["id"]: case["prompt_ids"] for case in prepared["speed"]}
    for case_id, case in ((result.get("speed") or {}).get("cases") or {}).items():
        variants = case.get("variants")
        if variants and case_id in prompts and "forks" not in case:
            case["forks"] = [judge.judge(prompts[case_id], variants[0], v) for v in variants[1:]]
            _log(log, f"speed {case_id}", case["forks"])
    for case_id, row in ((result.get("history") or {}).get("cases") or {}).items():
        if not row["identical"] and case_id in prompts and "fork" not in row:
            row["fork"] = judge.judge(prompts[case_id], row["fresh"], row["after"])
            _log(log, f"history {case_id}", [row["fork"]])
    turns = {str(t["turn"]): t["prompt_ids"] for t in prepared.get("followup") or []}
    for turn, row in (((result.get("followup") or {}).get("again") or {}).get("turns") or {}).items():
        if not row["identical"] and turn in turns and "fork" not in row:
            row["fork"] = judge.judge(turns[turn], row["first"], row["again"])
            _log(log, f"followup turn {turn}", [row["fork"]])


def near_ties(forks):
    """Whether every fork is a judged near-tie."""
    return bool(forks) and all(f is not None and f.get("near_tie") for f in forks)


def describe(fork):
    if fork is None:
        return "no difference"
    if fork.get("near_tie"):
        return f"a near-tie at token {fork['position']} ({abs(fork['gap']):.2f} nats apart)"
    if fork.get("gap") is not None:
        return f"a decisive fork at token {fork['position']} ({abs(fork['gap']):.2f} nats apart)"
    return f"at token {fork['position']}: {fork.get('reason', 'not judged')}"


def _log(log, what, forks):
    log(f"{what}: " + "; ".join(describe(f) for f in forks))
