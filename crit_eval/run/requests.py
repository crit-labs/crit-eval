"""One request per case (protocol requests.cases): every request Eval sends for a suite case is built here, from the
case's one definition and the overrides of it the sending part declares (protocol requests.overrides), and sent
through one function.

A case is a speed case, a held-out prompt or a follow-up turn as prepare froze it (its token ids, the chat messages
and tools a server is sent for them, its output_tokens), or a curve or retrieval point (curve_case, retrieval_case).
Its definition is that prompt and length asked greedily, stop tokens off, with no prompt cache, no log-probabilities,
no reasoning budget and free decoding. A part that asks a case otherwise names the fields it changes; a field its
protocol entry does not declare is refused, and the request records the fields it changed (`overrides`). Two replies
are compared only when their requests are the same (same_request): the same case, the same fields."""
import functools
import hashlib
import json

from .. import EVAL_DIR, load_json
from ..limits.context import curve_prompt, retrieval_prompt
from .timing import timed_generate, timed_plain, timed_replay, timed_sampled

# Every field of a request, at its definition's value where the case does not give one.
DEFAULTS = {"sampling": None, "logprobs": None, "prompt_cache": False, "thinking_tokens": None, "decode": "free",
            "continuation": None}
FIELDS = ("prompt_ids", "output_tokens") + tuple(DEFAULTS)
DECODES = ("free", "plain")

_observers = []   # callables each built request is passed to (observe): the requests test walks every part's


def case_key(case):
    """The case's identity among every case of the prepared data: a speed case's id, "heldout/<id>" for a held-out
    prompt, "followup/<turn>" for a follow-up turn, a point's own ("curve@<point>", "retrieval@<point>")."""
    if "turn" in case:
        return f"followup/{case['turn']}"
    if "language" in case:
        return f"heldout/{case['id']}"
    return case["id"]


def definition(case):
    """The case's one request: its prompt and output tokens, greedy, stop tokens off, no prompt cache, no
    log-probabilities, no reasoning budget, free decoding."""
    if case.get("output_tokens") is None:
        raise SystemExit(f"case {case_key(case)} has no output_tokens: the data was prepared before crit-eval-v0.9.0; "
                         "prepare it again into a new directory")
    return {"case": case_key(case), "prompt_ids": list(case["prompt_ids"]), "output_tokens": int(case["output_tokens"]),
            **DEFAULTS}


@functools.lru_cache(maxsize=1)
def _shipped():
    return load_json(EVAL_DIR / "protocol.json")


def case_request(case, part, protocol=None, **overrides):
    """The request `part` sends for `case`: its definition with `overrides`, each a field the part's protocol entry
    (requests.overrides[part]; protocol None: the shipped protocol's) declares. A field at its definition's value is
    no override. Raises ValueError for a field the part does not declare."""
    declared = (((protocol or _shipped()).get("requests") or {}).get("overrides") or {}).get(part)
    if declared is None:
        raise ValueError(f"part {part!r} is not one protocol requests.overrides names")
    base = definition(case)
    changed = {}
    for field, value in overrides.items():
        if field not in FIELDS:
            raise ValueError(f"{field!r} is no request field")
        if field == "prompt_ids":
            value = list(value)
        if value == base[field]:
            continue
        if field not in declared:
            raise ValueError(f"part {part!r} overrides {field} of case {base['case']}, which protocol "
                             f"requests.overrides.{part} does not declare")
        changed[field] = value
    if changed.get("decode", base["decode"]) not in DECODES:
        raise ValueError(f"decode {changed['decode']!r} is not one of {DECODES}")
    request = {**base, **changed, "part": part, "overrides": sorted(changed)}
    for observer in list(_observers):
        observer(request)
    return request


def same_request(a, b):
    """Whether two requests ask the same: the same case and every field alike (the part that sent them aside)."""
    return a is not None and b is not None and all(a.get(key) == b.get(key) for key in ("case",) + FIELDS)


def request_sha256(request):
    """A request's identity, stored beside its reply so a later part compares only with replies of its own request."""
    return hashlib.sha256(json.dumps({key: request.get(key) for key in ("case",) + FIELDS},
                                     sort_keys=True).encode()).hexdigest()


def send(engine, request):
    """The request on `engine`, timed: (tokens, sample, the reported log-probabilities of a sampled request asked
    for them, else None). Greedy free decoding by timed_generate, a sampling by timed_sampled, plain decoding by
    timed_plain, a continuation by timed_replay."""
    prompt, n = request["prompt_ids"], request["output_tokens"]
    if request["continuation"] is not None:
        tokens, sample = timed_replay(engine, prompt, request["continuation"])
        return tokens, sample, None
    if request["decode"] == "plain":
        tokens, sample = timed_plain(engine, prompt, n)
        return tokens, sample, None
    if request["sampling"] is not None:
        return timed_sampled(engine, prompt, n, request["sampling"], request["logprobs"], request["prompt_cache"])
    tokens, sample = timed_generate(engine, prompt, n, prompt_cache=request["prompt_cache"],
                                    logprobs=request["logprobs"], thinking_tokens=request["thinking_tokens"])
    return tokens, sample, None


def send_cohort(engine, requests):
    """One closed-loop level of requests (requests[c] client c's, in order) on an engine's cohort: each request's
    prompt, output tokens and sampling (greedy where it has none). The requests of a level are all greedy or all
    sampled."""
    sampled = {r["sampling"] is not None for row in requests for r in row}
    if len(sampled) > 1:
        raise ValueError("a cohort's requests are all greedy or all sampled")
    if any(r["decode"] != "free" or r["continuation"] is not None or r["logprobs"] or r["prompt_cache"]
           or r["thinking_tokens"] for row in requests for r in row):
        raise ValueError("a cohort sends free decoding without log-probabilities, prompt cache or reasoning budget")
    prompts = [[r["prompt_ids"] for r in row] for row in requests]
    lengths = {r["output_tokens"] for row in requests for r in row}
    # One length for every request is the cohort's int; per-request lengths are max_new_tokens[c][r].
    length = lengths.pop() if len(lengths) == 1 else [[r["output_tokens"] for r in row] for row in requests]
    if sampled == {True}:
        return engine.cohort(prompts, length, [[r["sampling"] for r in row] for row in requests])
    return engine.cohort(prompts, length)


def curve_case(pieces, settings, point):
    """The long-context curve's case at `point` (protocol context.curve): the retrieval archive with the curve's
    request filling the point with its output (prompt + context.output_tokens), or None when the point leaves no room
    for the prompt."""
    n = int(settings["output_tokens"])
    prompt = curve_prompt(pieces, int(point) - n, settings["retrieval"]["positions"])
    return None if prompt is None else {"id": f"curve@{int(point)}", "prompt_ids": prompt, "output_tokens": n}


def curve_prompt_at(pieces, settings, filled):
    """The curve's prompt for a point filled to `filled` tokens (a drawn share of the point, protocol jitter), the
    curve part's declared prompt override; None without room."""
    return curve_prompt(pieces, int(filled) - int(settings["output_tokens"]), settings["retrieval"]["positions"])


def retrieval_case(pieces, rule, point):
    """A retrieval point's case (protocol context.retrieval): the facts in filler text with the question, the prompt
    and context.retrieval.output_tokens filling the point; None when the point is too short to hold the facts."""
    n = int(rule["output_tokens"])
    prompt = retrieval_prompt(pieces, int(point) - n, rule["positions"])
    return None if prompt is None else {"id": f"retrieval@{int(point)}", "prompt_ids": prompt, "output_tokens": n}


class observe:
    """with observe() as seen: every request built inside the block, in order, in `seen`."""

    def __enter__(self):
        self.requests = []
        _observers.append(self.requests.append)
        return self.requests

    def __exit__(self, *exc):
        _observers.remove(self.requests.append)
        return False
