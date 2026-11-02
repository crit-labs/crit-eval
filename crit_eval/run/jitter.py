"""Prompt-length jitter (protocol jitter): the held-out prompts and the long-context curve's points below the required
context are cut to a length drawn per run from a fraction of their nominal length, so an engine tuned to the suite's
exact shapes (CUDA graphs captured at one token count, a tile or split choice for one prompt length, a prefill chunk
that divides it) shows it as a slower held-out score and curve. The speed suite stays fixed, so its scores keep
comparing run to run, and the curve's point at the required context stays exact: the capacity gate fills it.

A run's seed draws every length: each prompt's or point's fraction comes from the seed and its own name, so it does
not depend on the order the parts run in or on which of them a stage measures. The result records the seed and every
drawn length (`jitter`), and `crit eval run --jitter-seed` draws a recorded run's lengths again.
"""
import math
import random
import secrets


def new_seed():
    """A run's seed when none is given: 32 random bits."""
    return secrets.randbits(32)


def fraction(seed, kind, key, rule):
    """The fraction of its nominal length the prompt or point `key` of `kind` ("heldout", "curve") is cut to."""
    low, high = (float(v) for v in rule["fraction"])
    return random.Random(f"{int(seed)}/{kind}/{key}").uniform(low, high)


def drawn(seed, kind, key, nominal, rule, least=1):
    """The drawn length of a prompt or point of `nominal` tokens: its fraction of them, rounded up, at least `least`
    and at most `nominal`."""
    return max(int(least), min(int(nominal), math.ceil(int(nominal) * fraction(seed, kind, key, rule))))


def heldout_prompt(prepared, prompt, jitter):
    """A held-out prompt's ids at the run's drawn length, and what was drawn: the template's head and tail around its
    user text (prepared heldout_frame) with that text cut, in tokens, so the whole prompt is the drawn share of its
    nominal length. `jitter` {"seed", "fraction"} (None: the nominal prompt, nothing drawn). Data whose prompt does not
    split into the frame around its text (content_ids) keeps the prompt whole, with the reason."""
    ids = list(prompt["prompt_ids"])
    if jitter is None:
        return ids, None
    frame, content = prepared.get("heldout_frame"), prompt.get("content_ids")
    if not frame or content is None:
        return ids, {"nominal_tokens": len(ids), "prompt_tokens": len(ids),
                     "reason": "the prepared prompt does not split into the template around its text; measured whole"}
    head, tail = list(frame["head_ids"]), list(frame["tail_ids"])
    tokens = drawn(jitter["seed"], "heldout", prompt["id"], len(ids), jitter, len(head) + len(tail) + 1)
    cut = head + list(content)[:tokens - len(head) - len(tail)] + tail
    return cut, {"nominal_tokens": len(ids), "prompt_tokens": len(cut)}


def curve_point(point, required, jitter):
    """The context a curve point fills: its drawn share of the point below the required context, the requirement
    itself exactly (the capacity gate's request), and the point as it is without jitter."""
    if jitter is None or int(point) >= int(required):
        return int(point)
    return drawn(jitter["seed"], "curve", int(point), int(point), jitter)


def record(jitter, heldout=None, context=None):
    """The result's jitter: the seed, the fraction range and every length it drew ({id: tokens} of the held-out
    prompts, {point: context} of the curve)."""
    if jitter is None:
        return None
    held = {pid: row["prompt_tokens"] for pid, row in ((heldout or {}).get("prompts") or {}).items()
            if row.get("status") == "measured"}
    curve = {key: row["filled_context"] for key, row in ((context or {}).get("points") or {}).items()
             if row.get("filled_context") is not None}
    return {"seed": int(jitter["seed"]), "fraction": list(jitter["fraction"]), "heldout": held, "curve": curve}
