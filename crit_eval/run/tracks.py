"""Closed and open tracks (protocol tracks): which track a result may be submitted to, as far as Eval can check it.

The closed track fixes the choices that move scores for reasons other than engine work (MLPerf's closed division,
SPEC's base): either quality mode judged by that mode's protocol budget unchanged, one configuration for every
workload, and the checkpoint's own draft layers as the only drafter. Results compare within one quality mode. Eval
checks what a run shows: the quality mode it was judged in and that its budget is the mode's own (protocol
quality.mode_thresholds, else quality.thresholds; a --quality-budget that differs, looser or tighter, leaves the closed
track); that the engine it measured was loaded once for the run, a served engine's own load count (crit serve's
/health engine_loads) unchanged from its first request to its last; and that an engine that drafts reads no more per
draft step than the checkpoint's own draft layers and the head their logits go through
(crit_eval.limits.context.draft_step_bytes), which a re-encoded head can only shrink. How the engine encodes the
checkpoint's weights is not one of the closed track's choices: the quality budget judges it. The card declares the
rest, such as no per-case or per-prompt-length settings. The open track is everything the protocol allows. Reported,
not judged: no gate depends on it.
"""
from .. import EVAL_DIR, load_json
from ..quality.fixed import QUALITY_MODES, mode_of
from ..reports.metrics import mode_thresholds

CHECKS = ("quality_mode", "one_load", "own_drafter")


def engine_loads(engine):
    """A served engine's load count now (crit serve's /health), or None from an engine or server that reports none."""
    reader = getattr(engine, "loads", None)
    if not callable(reader):
        return None
    try:
        value = reader()
    except (AttributeError, OSError, RuntimeError, ValueError):
        return None
    return value if type(value) is int else None


def mode_budgets(protocol=None):
    """{quality mode: its protocol budget} (protocol quality.mode_thresholds, else quality.thresholds)."""
    rule = (protocol or load_json(EVAL_DIR / "protocol.json"))["quality"]
    return {mode: mode_thresholds(rule, mode) for mode in QUALITY_MODES}


def quality_check(result, budgets):
    """The closed track's quality_mode check: the result was judged in a quality mode (which every result states,
    fixed.mode_of) by that mode's protocol budget unchanged."""
    requirements = result.get("requirements") or {}
    mode, budget = mode_of(requirements), requirements.get("quality_budget")
    if budget is None:
        return {"state": "unmeasured", "detail": f"judged in {mode}; the result records no quality budget to compare "
                                                 "with the mode's"}
    if budget != budgets[mode]:
        return {"state": "fail", "detail": f"judged in {mode} by the budget {budget}, not the mode's protocol budget "
                                           f"{budgets[mode]}: the closed track judges each mode by its own budget"}
    return {"state": "pass", "detail": f"judged in {mode} by its protocol budget"}


def closed(result, info, checkpoint_draft_bytes, tolerance, budgets=None):
    """The closed track's checks on a result ({"qualifies", "checks": {name: {"state", "detail"}}}): qualifies is
    True when every check passes, False when one fails, None while one is unmeasured. info: the engine's info();
    checkpoint_draft_bytes: what one draft step of the checkpoint's own draft layers reads (None: it has none);
    tolerance: how far above it a declared draft step may read (rounding of a stored format); budgets: each quality
    mode's protocol budget (mode_budgets, the protocol's by default)."""
    checks = {"quality_mode": quality_check(result, mode_budgets() if budgets is None else budgets)}
    loads = result.get("engine_loads") or {}
    start, end = loads.get("first"), loads.get("last")
    if start is None or end is None:
        checks["one_load"] = {"state": "pass", "detail": "Eval loaded the engine once for the run" + (
            "; its server reports no load count" if info.get("served") else "")}
    elif end == start:
        checks["one_load"] = {"state": "pass", "detail": f"the server loaded no engine between the run's first request "
                                                        f"and its last ({end} loads)"}
    else:
        checks["one_load"] = {"state": "fail", "detail": f"the server's engine loads went from {start} to {end} during the "
                                                        "run: an engine loaded again (another configuration, a model "
                                                        "switch, a reload) serves some of its workloads"}
    if not info.get("speculative"):
        checks["own_drafter"] = {"state": "pass", "detail": "the engine does not draft"}
    elif checkpoint_draft_bytes is None:
        checks["own_drafter"] = {"state": "fail", "detail": "the engine drafts, and the checkpoint has no draft layers of "
                                                           "its own: its drafter comes from elsewhere"}
    elif not info.get("draft_bytes_per_step"):
        checks["own_drafter"] = {"state": "unmeasured", "detail": "the engine drafts but declares no draft_bytes_per_step "
                                                                 "to compare with the checkpoint's draft layers"}
    elif info["draft_bytes_per_step"] > checkpoint_draft_bytes * (1 + tolerance):
        checks["own_drafter"] = {"state": "fail", "detail": (
            f"a draft step reads {info['draft_bytes_per_step'] / 1e9:.2f} GB, more than the checkpoint's draft layers "
            f"and head ({checkpoint_draft_bytes / 1e9:.2f} GB): the drafter holds more than the checkpoint's own")}
    else:
        checks["own_drafter"] = {"state": "pass", "detail": (
            f"a draft step reads {info['draft_bytes_per_step'] / 1e9:.2f} GB, within the checkpoint's draft layers and "
            f"head ({checkpoint_draft_bytes / 1e9:.2f} GB)")}
    states = {c["state"] for c in checks.values()}
    qualifies = False if "fail" in states else None if "unmeasured" in states else True
    return {"qualifies": qualifies, "checks": checks}
