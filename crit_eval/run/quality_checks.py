"""The quality parts of a run: the engine's scores of the fixed texts against the reference's within the drift
budget, attribution of drift to the speculative path, and honesty of the served replies."""
import hashlib
import time
from pathlib import Path

from .. import load_json
from ..quality import fixed
from ..reports import metrics
from .adapters import open_engine


def reference_origin(data, store=fixed.STORE):
    """Where the data's reference scores came from: the reference that computed them (its version, transformers,
    torch and device, from reference/index.json, or the quality mode's store) and the package they were fetched from
    (crit_eval.quality.packages), or package None for scores computed on this machine. None without stored scores."""
    from ..quality.packages import PACKAGE
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


QUALITY_KEYS = ("ppl_ratio_max", "kld_mean_max", "kld_p99_max")


def quality_thresholds(protocol, budget=None, mode="preserve"):
    """The protocol's drift budget for quality mode `mode` (quality.mode_thresholds, else quality.thresholds), with a
    run's own limits (partial is fine) laid over it."""
    budget = budget or {}
    unknown = set(budget) - set(QUALITY_KEYS)
    if unknown:
        raise SystemExit(f"--quality-budget keys are {', '.join(QUALITY_KEYS)}; got {', '.join(sorted(unknown))}")
    for key, value in budget.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < (1.0 if key == "ppl_ratio_max" else 0.0):
            raise SystemExit(f"--quality-budget {key} must be a number of at least {1.0 if key == 'ppl_ratio_max' else 0.0}")
    return {**metrics.mode_thresholds(protocol["quality"], mode), **{k: float(v) for k, v in budget.items()}}


def _engine_rows(rows):
    return [{"logprob": row["logprob"], "top": row["top"]} for row in rows]


def _phase_timing(timings):
    """Sums of request timings (crit_eval.run.chat): requests, seconds, the engine's own prefill and decode seconds, the
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


def stored_spread(data, prepared, protocol, reference_identity, store=fixed.STORE):
    """The reference's own spread stored under `data` (fixed.load_spread) for the texts protocol quality.spread names
    (reference.store_spread), None without one: a spread stored for fewer texts (an earlier rule's) is not used until
    crit eval reference scores it again."""
    rule = protocol["quality"].get("spread")
    if not rule:
        return None
    texts = sorted(name for name, (ids, _) in fixed.texts(prepared).items() if len(ids) >= int(rule["from_tokens"]))
    return fixed.load_spread(data, {**reference_identity, "spread_perturbation": rule["perturbation"],
                                    "spread_texts": texts}, store)


def set_limits(thresholds, spread, factor, cap=None):
    """A set's drift limits (quality_thresholds) raised to `factor` times the reference's own spread on it
    (protocol quality.spread), or the as_quantized allowance (protocol quality.allowance): kld_mean_max and
    kld_p99_max to factor times the spread's, ppl_ratio_max to 1 plus factor times the spread ratio's distance from 1,
    each raise stopping at cap's value for that limit when cap names one. Returns (limits, the keys raised)."""
    if not spread:
        return dict(thresholds), []
    cap = cap or {}
    limits, raised = dict(thresholds), []
    for key, name in (("kld_mean_max", "kld_mean"), ("kld_p99_max", "kld_p99")):
        if key in limits and spread.get(name) is not None:
            value = min(factor * spread[name], cap.get(key, float("inf")))
            if value > limits[key]:
                limits[key], raised = value, raised + [key]
    if "ppl_ratio_max" in limits and spread.get("ppl_ratio") is not None:
        from_spread = min(1.0 + factor * abs(spread["ppl_ratio"] - 1.0), cap.get("ppl_ratio_max", float("inf")))
        if from_spread > limits["ppl_ratio_max"]:
            limits["ppl_ratio_max"], raised = from_spread, raised + ["ppl_ratio_max"]
    return limits, raised


def run_quality(engine, info, prepared, protocol, log, reference, thresholds, spread=None, allowance=None):
    """protocol quality: every quality text scored by the engine through its serving paths (an engine's score(prompt,
    continuation, top_k): an engine behind crit serve, crit_eval.run.chat) and judged against the reference's stored
    scores of the same text (reference: crit_eval.quality.fixed.load, None when the data has none). Each set is judged on
    both paths the engine computes it with: its prompt positions from the prefill (SET/prompt: the arithmetic every
    reply's first token comes from, and the KV and state the decode reads) and the positions it serves (SET/served:
    the continuation, its first token from the prefill's last position, the rest from decode); both per bucket.
    Then the honesty check (run_honesty). A set whose texts do not fit the engine's max_context is skipped. Sets
    are scored in the prepared order and judged as each completes; a set far over the budget (clearly_failed) stops
    the scoring, and the result says so (stopped_early). spread: the reference's own spread per set and path
    (fixed.load_spread), which raises a set's limits where it exceeds the budget (set_limits). allowance (quality mode
    as_quantized): how far the checkpoint run with its own activation quantization lands from the reference per set
    and path (fixed.load_allowance), which raises a set's limits to protocol quality.allowance.margin times it where
    that exceeds them, at most to quality.allowance.cap."""
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
                "reason": "the data holds no reference scores (or, for quality mode as_quantized, no allowance "
                          "measured with them): run crit eval reference --model --data first"}
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
            allowed = []
            if allowance is not None:
                # protocol quality.allowance: the checkpoint's own quantization drift, times the margin, capped.
                limits, allowed = set_limits(limits, allowance.get(name), float(rule["allowance"]["margin"]),
                                             rule["allowance"].get("cap"))
            if name.endswith("/prompt"):
                # protocol quality.prompt_perplexity: reported, not judged on the prompt positions.
                limits.pop("ppl_ratio_max", None)
                raised = [key for key in raised if key != "ppl_ratio_max"]
                allowed = [key for key in allowed if key != "ppl_ratio_max"]
            raised = [key for key in raised if key not in allowed]   # the larger raise names the limit
            failures = metrics.judge(row, limits)
            # The share's dependence on the positions the set holds (protocol quality.position_spread): reported.
            noise = rule.get("position_spread") or {}
            spread_found = metrics.budget_spread(reference_rows, found, limits, int(noise.get("block", 64)),
                                                 int(noise.get("draws", 200)), int(noise.get("seed", 0))) if noise else None
            sets[name] = {**row, "texts": ids, "passed": not failures, "failures": failures,
                          "budget_used": metrics.budget_used(row, limits), "limits": limits,
                          **({"budget_spread": spread_found} if spread_found else {}),
                          **({"reference_spread": spread[name], "raised": raised} if raised else {}),
                          **({"allowance": allowance[name], "allowance_raised": allowed} if allowed else {})}
            if raised:
                log(f"quality {name}: judged against the reference's own spread, "
                    + ", ".join(f"{key[:-4]} <= {limits[key]:.3g}" for key in raised))
            if allowed:
                log(f"quality {name}: judged with the checkpoint's own quantization drift as allowance, "
                    + ", ".join(f"{key[:-4]} <= {limits[key]:.3g}" for key in allowed))
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
                    context_tokens=None, allowance=None):
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
                                 reference, thresholds, spread, allowance)
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
