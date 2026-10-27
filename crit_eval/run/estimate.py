"""How long a crit-eval run will take, projected from the first requests the engine served.

Every request a run sends is known before it starts: the prepared data fixes each prompt and output length (its
speed, heldout, quality, followup and retrieval pieces) and the protocol how often each goes (warmups, repetitions
and seeds, the curve and retrieval points). Only the engine's speed is unknown, and its first requests measure it as
crit_eval.run.deadline keeps them: (prompt tokens, seconds to the first token) and (context tokens, seconds per later
output token). Where deadline bounds a request from above, this fits the expected cost with the same shapes: a
prompt of n tokens prefills in t0 + a * n + c * n^2 seconds (a fixed cost per request, the work per token, attention
over every earlier token), and an output token at a context of k decodes in p + q * k (the weights and a KV cache
growing with the context). Each fit minimizes the squared relative errors with no coefficient negative and no more
coefficients than distinct lengths observed, so a single prompt length gives a per-token rate alone. A request of n
prompt and m output tokens is then expected to take prefill(n) + (m - 1) * step(n + m); a quality score request
also returns a row per scored position, the server's own work no reply shows (SCORE_ROW_SECONDS each), and the
exactness check decodes one token per target pass (step times the tokens per round the speed cases measured).

Left out: a speed case's third repetition (only when its first two disagree, which a quiet engine's rarely do),
prompt caches (follow-up turns and retrieval requests prefill in full) and replies ending early (each runs its
length), so a cached or early-stopping engine takes less.
"""
from ..limits.context import curve_points, lower_context, retrieval_points

# Seconds per scored position of crit serve's score call beyond the engine's own prefill and decode: computing,
# packing, sending and parsing its rows (a 27B model's packed rows took 0.35 ms each).
SCORE_ROW_SECONDS = 3.5e-4

# The parts of a run in the order they run (result timing.parts), and the timing parts each runner part
# (crit_eval.run.runner PARTS) runs.
ORDER = ("lower_context", "load", "history", "followup", "speed", "sampled_speed", "sampling_check", "heldout",
         "quality", "context", "retrieval", "exactness", "followup_again", "history_again")
RUNS = {"speed": ("speed", "sampled_speed", "sampling_check"), "context": ("lower_context", "context", "retrieval"),
        "history": ("history", "history_again"), "followup": ("followup", "followup_again")}

PREFILL_TERMS = ((1,), (0, 1), (1, 2), (0, 1, 2))   # powers of n: 0 fixed, 1 per token, 2 attention
DECODE_TERMS = ((0,), (0, 1))                       # powers of k: 0 per token, 1 per token and context token


def _least_squares(rows):
    """The coefficients x minimizing sum((row . x - 1)^2), or None when the normal equations are singular."""
    size = len(rows[0])
    matrix = [[sum(r[i] * r[j] for r in rows) for j in range(size)] + [sum(r[i] for r in rows)] for i in range(size)]
    tiny = 1e-12 * max(abs(matrix[i][i]) for i in range(size))
    for col in range(size):
        pivot = max(range(col, size), key=lambda i: abs(matrix[i][col]))
        if abs(matrix[pivot][col]) <= tiny:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for i in range(size):
            if i != col:
                factor = matrix[i][col] / matrix[col][col]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[col])]
    return [matrix[i][size] / matrix[i][i] for i in range(size)]


def fit(points, subsets, powers=3):
    """Coefficients of x^0 .. x^(powers - 1) through points [(x, y)], each point's error relative to its y, from the
    subset of powers with the least error whose coefficients are all non-negative and that has at most as many
    members as there are distinct x (fewer members among equal errors); None without a point."""
    points = [(float(x), float(y)) for x, y in points if x > 0 and y > 0]
    if not points:
        return None
    scale = max(x for x, _ in points)   # fitted on x / scale: n^2 at 262K tokens would swamp the normal equations
    best = None
    for subset in subsets:
        if len(subset) > len({x for x, _ in points}):
            continue
        rows = [[(x / scale) ** j / y for j in subset] for x, y in points]
        found = _least_squares(rows)
        if found is None or min(found) < 0:
            continue
        error = sum((sum(c * v for c, v in zip(found, row)) - 1) ** 2 for row in rows)
        key = (round(error, 9), len(subset))
        if best is None or key < best[0]:
            best = key, {j: c / scale ** j for j, c in zip(subset, found)}
    return [best[1].get(j, 0.0) for j in range(powers)] if best else None


class Rates:
    """The engine's expected costs: prefill seconds t0 + a * n + c * n^2 and decode seconds per token p + q * k."""

    def __init__(self, prefill, decode):
        (self.t0, self.a, self.c), (self.p, self.q) = prefill, decode

    def prefill(self, n):
        return self.t0 + self.a * n + self.c * n * n

    def step(self, context):
        return self.p + self.q * context

    def request(self, prompt_tokens, output_tokens, per_pass=1.0):
        """A request's seconds; per_pass: tokens per round of the decode step, for one token per target pass."""
        decode = max(0, output_tokens - 1) * self.step(prompt_tokens + output_tokens) * per_pass
        return self.prefill(prompt_tokens) + decode

    def record(self):
        return {"prefill": {"fixed_seconds": self.t0, "seconds_per_token": self.a, "seconds_per_token_squared": self.c},
                "decode": {"seconds_per_token": self.p, "seconds_per_token_per_context_token": self.q}}


def rates(observations):
    """Rates fitted from what a RequestDeadline observed (its prefills and decodes, or a dict of both lists); None
    before a request with a prefill and one with a decode. Only streamed replies time their prefill: a deadline also
    takes a score or honesty request's whole seconds as its prefill (an upper bound), which a fit would read as a far
    steeper prefill, so estimate before the quality part, or from the streamed requests alone."""
    seen = observations if isinstance(observations, dict) else vars(observations)
    prefill = fit(seen.get("prefills") or [], PREFILL_TERMS)
    decode = fit(seen.get("decodes") or [], DECODE_TERMS, powers=2)
    return Rates(prefill, decode) if prefill and decode else None


def _speed(prepared, max_context, each):
    """Each speed case that fits, `each` times: (prompt tokens, output tokens)."""
    return [(len(c["prompt_ids"]), c["output_tokens"]) for c in prepared["speed"]
            if len(c["prompt_ids"]) + c["output_tokens"] <= max_context for _ in range(each)]


def _history(prepared, max_context):
    """protocol history's cases: the speed cases with the shortest and the longest prompt the engine holds."""
    held = [c for c in prepared["speed"] if len(c["prompt_ids"]) + c["output_tokens"] <= max_context]
    if not held:
        return []
    ends = (min(held, key=lambda c: len(c["prompt_ids"])), max(held, key=lambda c: len(c["prompt_ids"])))
    return [(len(c["prompt_ids"]), c["output_tokens"]) for c in held if any(c is end for end in ends)]


def _holds(pieces, length, tail):
    """Whether a retrieval or curve prompt of `length` tokens has room for its facts (context.retrieval_prompt)."""
    facts = pieces["facts"]
    return length - len(pieces["head_ids"]) - len(tail) - sum(len(f["ids"]) for f in facts) >= len(facts) + 1


def _quality(prepared, protocol, max_context):
    """The quality part's requests (quality_checks.run_quality): every text of a set that fits scored, then the honesty
    replies and their rescores. Requests are (prompt, output tokens, scored rows)."""
    rule, texts = protocol["quality"]["honesty"], prepared.get("quality") or []
    need = {}
    for t in texts:
        need[t["set"]] = max(need.get(t["set"], 0), len(t["prompt_ids"]) + len(t["continuation_ids"]))
    held = [t for t in texts if need[t["set"]] <= max_context]
    out = [(len(t["prompt_ids"]), len(t["continuation_ids"]), len(t["prompt_ids"]) + len(t["continuation_ids"]) - 1)
           for t in held]
    reply = int(rule["reply_tokens"])
    for t in held:
        if t["set"] not in rule["sets"]:
            continue
        # The greedy reply, then its tokens scored as the continuation without the prompt's rows.
        out += [(len(t["prompt_ids"]), reply, 0), (len(t["prompt_ids"]), reply, reply)]
    return out


def requests(prepared, protocol, parts, required, settings=None, info=None, warmups=None, repetitions=None,
             sampled_repetitions=None, pinned=False, skip=()):
    """{timing part: [(prompt tokens, output tokens, scored rows, one token per pass)]} for every part of the run that
    sends requests, as crit_eval.run.runner sends them. parts: runner parts (crit_eval.run.runner PARTS, each standing for its
    timing parts: RUNS) or timing parts; skip: timing parts that will not run (lower_context for a server crit-eval did
    not start, sampled_speed for an engine that cannot sample, quality without reference scores). settings: the run's
    protocol context settings (profiles included); info: engine.info() (max_context, speculative, plain, served),
    without which the engine holds the required context and checks exactness on itself. warmups, repetitions and
    sampled_repetitions as crit_eval.run.runner.run takes them; pinned: the speed cases replay a pinned continuation."""
    settings = settings if settings is not None else protocol["context"]
    names = {name for part in parts for name in RUNS.get(part, (part,))} - set(skip)
    held = (info or {}).get("max_context") or required or float("inf")
    warmups = protocol["speed"]["warmup_runs"] if warmups is None else warmups
    each = (protocol["speed"]["repetitions"] if repetitions is None else repetitions) + warmups + (2 if pinned else 0)
    sampled = (len(protocol["sampled_speed"]["seeds"]["repetitions"]) if sampled_repetitions is None
               else sampled_repetitions) + warmups
    out = {}

    def add(name, found, plain=False):   # found: (prompt tokens, output tokens[, scored rows]); a part may add twice
        if name in names and found:
            out.setdefault(name, []).extend((r[0], r[1], r[2] if len(r) > 2 else 0, plain) for r in found)
    longest = max((len(c["prompt_ids"]) + c["output_tokens"] for c in prepared["speed"]), default=0)
    lower = lower_context(required, longest, settings) if required else None
    if "speed" in names and "context" in names and lower and settings.get("measure_lower") is not False:
        add("lower_context", _speed(prepared, lower, each))
    if "history" in names:
        add("history", _history(prepared, held))
        add("history_again", _history(prepared, held))
    turns, n = prepared.get("followup") or [], int(protocol["followup"]["output_tokens"])
    if turns and len(turns[-1]["prompt_ids"]) + n <= held:
        add("followup", [(len(t["prompt_ids"]), n) for t in turns])
        add("followup_again", [(len(t["prompt_ids"]), n) for t in turns])
    if "speed" in names:
        add("speed", _speed(prepared, held, each))
        if prepared.get("sampling") is not None and "sampled_speed" in names:
            add("sampled_speed", _speed(prepared, held, sampled))
            add("sampling_check", _speed(prepared, held, 1))
        if "exactness" in names and (info is None or info.get("speculative")) and (
                info is None or info.get("plain") or not info.get("served")):
            add("exactness", _speed(prepared, held, 1), plain=True)
    heldout = prepared.get("heldout") or []
    add("heldout", [(len(p["prompt_ids"]), p["output_tokens"]) for p in heldout
                    if len(p["prompt_ids"]) + p["output_tokens"] <= held
                    for _ in range(int(protocol["heldout"]["repetitions"]))])
    add("quality", _quality(prepared, protocol, held))
    pieces = prepared.get("retrieval")
    if required and pieces:
        n, curve = int(settings["output_tokens"]), []
        for point in curve_points(required, settings):
            if point <= held and _holds(pieces, point - n, pieces["curve_tail_ids"]):
                curve += [(point - n, n)] * int(settings.get("repetitions", 1))
        add("context", curve)
        if curve and "quality" in names and out.get("quality"):
            # The honesty check's long reply is the curve's first reply, scored once more (quality_checks.run_honesty).
            add("quality", [(curve[0][0], n, n)])
        asked = {p + m for p, m in curve}
        rule = settings["retrieval"]
        n = int(rule["output_tokens"])
        add("retrieval", [(point - n, n) for point in retrieval_points(required, rule)
                          if point not in asked and point != required and point <= held
                          and _holds(pieces, point - n, pieces["tail_ids"])])
    return out


def estimate(prepared, protocol, parts, required, observations, settings=None, info=None, warmups=None,
             repetitions=None, sampled_repetitions=None, pinned=False, skip=(), load_seconds=None,
             tokens_per_round=None, row_seconds=SCORE_ROW_SECONDS, measured=None):
    """The expected wall seconds of every part of the run that will run (requests: which), in the order they run, and
    their total: {"parts": {timing part: seconds}, "total_seconds", "requests": {timing part: count}, "fit"}; None
    before the observations hold a prefill and a decode. observations: the engine's RequestDeadline (chat.ChatEngine
    deadline) or {"prefills": [(prompt tokens, seconds)], "decodes": [(context tokens, seconds per token)]}.
    load_seconds: how long the engine took to open, counted for the load part and again for each engine opened
    besides (the lower context's, a separate exactness engine's); tokens_per_round: what the speculative decode
    measured, for the exactness check's plain decode (one token per pass); measured: timing parts that already ran
    (PartTimes.parts), which count as they took."""
    found = rates(observations)
    if found is None:
        return None
    measured, planned = dict(measured or {}), requests(prepared, protocol, parts, required, settings, info, warmups,
                                                       repetitions, sampled_repetitions, pinned, skip)
    separate = info is not None and not info.get("plain")   # exactness opens a second engine
    loads = {"load": load_seconds is not None, "lower_context": True, "exactness": separate}
    out = {}
    for name in ORDER:
        if name in measured:
            out[name] = float(measured[name])
        elif name == "load" and load_seconds is not None:
            out[name] = float(load_seconds)
        elif name in planned:
            out[name] = (sum(found.request(n, m, (tokens_per_round or 1.0) if plain else 1.0) + rows * row_seconds
                             for n, m, rows, plain in planned[name])
                         + (float(load_seconds or 0.0) if loads.get(name) else 0.0))
    return {"parts": out, "total_seconds": sum(out.values()),
            "requests": {name: len(planned[name]) for name in out if name in planned and name not in measured},
            "fit": found.record()}


def duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 7200 else f"{seconds / 3600:.1f} h"


def summary(found):
    """One line for the log: 'estimate: about 14 min (speed 2 min, quality 3 min, ...)'."""
    if not found:
        return "estimate: unknown (no request measured yet)"
    return (f"estimate: about {duration(found['total_seconds'])} ("
            + ", ".join(f"{name} {duration(seconds)}" for name, seconds in found["parts"].items()) + ")")
