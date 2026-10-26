"""Where the GPU time goes, kernel by kernel: `crit eval kernels --adapter FILE --model DIR --data DIR --out FILE`.

Runs the speed cases in a child process with a CUPTI injection library (probe/kernel_trace.cpp), so every kernel
the engine launches is recorded, CUDA-graph nodes included, with no change to the engine. Each case first runs a
warmup on another prompt of the same length (the case's prompt rotated by half), so graphs and workspaces for that
length exist, then its prefill as a request of one output token on a fresh prompt: the case's prompt with its first
token changed to one no earlier request of the trace began with, so no prefix cache can serve any of it (cases share
prefixes, such as tool definitions, and an engine may cache several prompts). Then the whole request, whose kernels from the first delivered
token on are its decode. A one-token request runs no verify pass, so a speculative engine's verify and draft
kernels never count as prefill; kernels the adapter classes as `draft` are kept apart in any phase. For each
phase the report ranks kernels by total time and shows calls, average duration and launch width (blocks against the
device's SMs; a kernel with fewer blocks than SMs leaves part of the GPU idle). Time is exclusive: a kernel that
starts before the previous one ends (programmatic dependent launch, or another stream) is charged only from that
end, so a kernel waiting on its predecessor is not charged the wait; `span_ms` keeps the raw durations. When the adapter declares
`kernel_work` estimates, it also shows achieved TFLOPS and GB/s against the device ceilings from `crit eval probe`.
Linux only; needs a CUDA toolkit with CUPTI.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from . import CONTRACT, EVAL_DIR, load_json, portable_path, write_json
from .ceiling import best_rates
from .prepare import PREPARED

SOURCE = EVAL_DIR / "probe" / "kernel_trace.cpp"
TOP = 12


def find_toolkit(explicit_nvcc=None):
    """nvcc plus the CUPTI include and library directories of the same toolkit (not every toolkit ships CUPTI)."""
    roots = []
    if explicit_nvcc:
        roots.append(Path(explicit_nvcc).resolve().parents[1])
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root:
            roots.append(Path(root))
    roots += sorted(Path("/usr/local").glob("cuda-*"), reverse=True)
    which = shutil.which("nvcc")
    if which:
        roots.insert(0 if not explicit_nvcc else 1, Path(which).resolve().parents[1])
    for root in roots:
        nvcc = root / "bin" / "nvcc"
        for inc, lib in ((root / "targets/x86_64-linux/include", root / "targets/x86_64-linux/lib"),
                         (root / "extras/CUPTI/include", root / "extras/CUPTI/lib64"),
                         (root / "include", root / "lib64")):
            if nvcc.is_file() and (inc / "cupti.h").is_file() and any(lib.glob("libcupti.so*")):
                return {"nvcc": str(nvcc), "include": str(inc), "lib": str(lib)}
    raise SystemExit("no CUDA toolkit with CUPTI found (looked for include/cupti.h and libcupti.so); pass --nvcc")


def build(toolkit, directory):
    library = Path(directory) / "libcrit_kernel_trace.so"
    command = [toolkit["nvcc"], "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-o", str(library), str(SOURCE),
               "-I", toolkit["include"], "-L", toolkit["lib"], "-lcupti", "-lcuda",
               "-Xlinker", "-rpath", "-Xlinker", toolkit["lib"]]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("building the kernel tracer failed:\n" + done.stderr[-4000:])
    return library


# --- child: runs inside the traced process -------------------------------------------------------------------

def warmup_prompt(ids):
    """A prompt of the same length as `ids`: ids rotated by half (at least one)."""
    shift = max(1, len(ids) // 2)
    return ids[shift:] + ids[:shift]


def fresh_prompt(ids, began):
    """`ids` with its first token changed to one no prompt in `began` (the first tokens of every earlier request)
    began with, the first such of its own later tokens, else the next id after them: a prompt of the same length and
    nearly the same text whose every prefix is new to a prompt cache."""
    taken = set(began)
    choice = next((t for t in ids[1:] if t not in taken), None)
    if choice is None:
        choice = max(taken | set(ids)) + 1
    return [choice] + list(ids[1:])


def tracer_clock():
    """(now(), name): the clock phase marks are taken on. The tracer's own (CUPTI's timestamps, which its kernel records
    carry) when the injection library exports crit_ktrace_timestamp, else the host's realtime clock, which read_traces
    converts kernel times to through the trace's clock pairs."""
    import ctypes
    try:
        library = ctypes.CDLL(os.environ["CUDA_INJECTION64_PATH"])
        now = library.crit_ktrace_timestamp
    except (KeyError, OSError, AttributeError):
        return time.time_ns, "realtime"
    now.restype, now.argtypes = ctypes.c_ulonglong, []
    return now, "cupti"


def _timed(engine, ids, n, now=time.time_ns):
    """(start, first delivered token, end) ns on the clock `now` of one greedy request, and its result."""
    first = []

    def on_tokens(tokens):
        if tokens and not first:
            first.append(now())
    start = now()
    result = engine.generate(ids, n, on_tokens)
    end = now()
    return (start, first[0] if first else end, end), result


def _child(spec):
    from .runner import load_adapter, open_engine
    prepared = load_json(Path(spec["data"]) / PREPARED)
    module = load_adapter(spec["adapter"])
    engine, info = open_engine(module, spec["model"], spec["config"], speculative=True,
                               context_tokens=spec.get("context_tokens"))
    now, clock = tracer_clock()
    marks = {"info": {k: info.get(k) for k in ("name", "configuration", "speculative", "draft_width", "kernel_work")},
             "clock": clock, "cases": []}
    began = set()   # the first token of every request so far
    try:
        for case in prepared["speed"]:
            if spec["cases"] and case["id"] not in spec["cases"]:
                continue
            ids, n = list(case["prompt_ids"]), case["output_tokens"]
            # Warmup on a prompt of the same length: graph capture and lazy allocation for this length.
            warm = warmup_prompt(ids)
            engine.generate(warm, n, lambda tokens: None)
            began.update((warm[0], ids[0]))
            time.sleep(0.1)
            # The prefill: one output token on a prompt no prefix cache holds any of, so the engine computes every
            # position and no verify pass or draft round follows.
            fresh = fresh_prompt(ids, began)
            began.add(fresh[0])
            (prefill_start, prefill_first, prefill_end), _ = _timed(engine, fresh, 1, now)
            time.sleep(0.1)
            (start, first, end), result = _timed(engine, ids, n, now)
            marks["cases"].append({"id": case["id"], "prompt_tokens": len(ids), "output_tokens": len(result["tokens"]),
                                   "rounds": result.get("rounds"), "prefill_start": prefill_start,
                                   "prefill_first": prefill_first, "prefill_end": prefill_end,
                                   "start": start, "first": first, "end": end})
            time.sleep(0.2)  # keeps cases apart on the timeline
    finally:
        engine.close()
    write_json(Path(spec["dir"]) / "marks.json", marks)


# --- analysis -------------------------------------------------------------------------------------------------

def read_traces(directory, clock="realtime"):
    """Kernels from every traced process: (start_ns, end_ns, blocks, threads, name), on the host realtime clock, or as
    recorded with clock "cupti" (phase marks taken on the tracer's clock, tracer_clock)."""
    kernels, device = [], {"sms": None, "name": None}
    for path in sorted(Path(directory).glob("trace-*.txt")):
        clocks, rows = [], []
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("#clock "):
                cupti, real = map(int, line.split()[1:3])
                clocks.append(real - cupti)
            elif line.startswith("#device "):
                parts = line.split(" ", 2)
                if int(parts[1]) > 0:
                    device = {"sms": int(parts[1]), "name": parts[2] if len(parts) > 2 else None}
            elif line and not line.startswith("#"):
                parts = line.split(" ", 4)
                if len(parts) == 5:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[4]))
        if rows and (clocks or clock == "cupti"):
            offset = 0 if clock == "cupti" else round(statistics.mean(clocks))
            kernels += [(s + offset, e + offset, b, t, n) for s, e, b, t, n in rows]
    kernels.sort()
    return kernels, device


def _mangled(name):
    """Kernels with internal linkage carry a `__nv_static_<n>__<hash>_<file>_<n>_` prefix before the mangled name."""
    if name.startswith("__nv_static_"):
        at = name.find("__Z", 12)
        if at > 0:
            return name[at + 1:]
    return name


def demangle(names):
    names = sorted(set(names))
    tool = shutil.which("c++filt")
    if not tool or not names:
        return {n: n for n in names}
    done = subprocess.run([tool], input="\n".join(_mangled(n) for n in names), capture_output=True, text=True)
    out = done.stdout.split("\n") if done.returncode == 0 else names
    return {n: (d or n) for n, d in zip(names, out)}


_ANONYMOUS = "(anonymous namespace)"


def short_name(name, limit=90):
    """Function name with namespaces and template arguments, without the parameter list."""
    name = re.sub(r"^void ", "", name)
    depth, cut = 0, len(name)
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0 and not name.startswith(_ANONYMOUS, i):
            cut = i
            break
    name = name[:cut]
    return name if len(name) <= limit else name[:limit - 3] + "..."


def _work_rule(rules, name):
    for rule in rules or []:
        if rule.get("match") and rule["match"] in name:
            return rule
    return None


# The terms of the prefill floor a kernel's time joins (crit_eval.ceiling), and draft work kept apart from them.
TERMS = ("matmul", "attention", "recurrent", "other", "draft")
# Name heuristics for kernels no kernel_work rule classes, checked in order on the lower-cased demangled name.
NAME_TERMS = (("draft", ("draft", "mtp", "speculat", "eagle")),
              ("attention", ("attn", "attention", "flash", "fmha", "sdpa", "_kv", "kv_", "rope")),
              ("recurrent", ("delta", "scan", "recurrent", "ssm", "mamba", "gdn", "conv_", "conv1d", "linear_att")),
              ("matmul", ("gemm", "gemv", "matmul", "_mma", "mma_", "cutlass", "cublas", "xmma", "_dense", "linear")))


def classify(rules, name):
    """(term, weight format or None, "rules" or "names") of a kernel: the first kernel_work rule matching it with a
    `term` (one of TERMS) and its `format`, else NAME_TERMS on its name ("other" when none matches)."""
    rule = _work_rule([r for r in rules or [] if r.get("term") in TERMS], name)
    if rule:
        return rule["term"], rule.get("format"), "rules"
    lowered = name.lower()
    for term, needles in NAME_TERMS:
        if any(needle in lowered for needle in needles):
            return term, None, "names"
    return "other", None, "names"


def term_key(term, fmt):
    """A term's key in a case's terms: "matmul:nvfp4" for a format's matmul, else the term."""
    return f"{term}:{fmt}" if term == "matmul" and fmt else term


def exclusive(inside):
    """(kernel, exclusive ns) in start order: each kernel's time after every earlier kernel has ended. The
    charges sum to the time the GPU ran any kernel."""
    out, covered = [], None
    for k in sorted(inside, key=lambda k: (k[0], k[1])):
        start = k[0] if covered is None else max(k[0], covered)
        out.append((k, max(0, k[1] - start)))
        covered = k[1] if covered is None else max(covered, k[1])
    return out


def analyze(kernels, device, marks, device_ceilings=None):
    names = demangle(k[4] for k in kernels)
    rules = (marks.get("info") or {}).get("kernel_work")
    peaks = best_rates(device_ceilings)[0]   # each format's best probed rate, GEMM or raw tensor cores
    bandwidth = (device_ceilings or {}).get("bandwidth_gbs")
    phases = {p: {"wall_ms": 0.0, "kernel_ms": 0.0, "launches": 0, "by": {}} for p in ("prefill", "decode")}
    cases = {}
    for case in marks["cases"]:
        # A trace from before the one-token prefill request split one request at its first delivered token.
        prefill = (case["prefill_start"], case["prefill_end"]) if "prefill_start" in case else (case["start"], case["first"])
        spans = {"prefill": prefill, "decode": (case["first"], case["end"])}
        # The whole request's target passes after prefill and its tokens, which divide its decode kernels.
        cases[case["id"]] = {"rounds": case.get("rounds"), "output_tokens": case.get("output_tokens")}
        for phase, (lo, hi) in spans.items():
            inside = exclusive([k for k in kernels if lo <= k[0] < hi])
            wall = (hi - lo) / 1e6
            busy = sum(ns for _, ns in inside) / 1e6
            per, terms, sources = {}, {}, {"rules": 0.0, "names": 0.0}
            for k, ns in inside:
                name = names.get(k[4], k[4])
                per[name] = per.get(name, 0.0) + ns / 1e6
                term, fmt, source = classify(rules, name)
                terms[term_key(term, fmt)] = terms.get(term_key(term, fmt), 0.0) + ns / 1e6
                sources[source] += ns / 1e6
            heavy = sorted(per.items(), key=lambda x: -x[1])[:5]
            cases[case["id"]][phase] = {"wall_ms": wall, "kernel_ms": busy, "launches": len(inside),
                                        "top": [{"kernel": short_name(n), "ms": t, "share": t / busy if busy else 0.0}
                                                for n, t in heavy],
                                        # GPU time per prefill-floor term, and how much of it rules and names classed.
                                        "terms": terms, "classified_ms": sources}
            if phase == "prefill" and "prefill_first" in case:
                # The fresh request's time to its first token, and the GPU's busy time inside it: the rest is the
                # engine's host work, launches and synchronization (crit_eval.status: the TTFT breakdown).
                first_ms = (case["prefill_first"] - lo) / 1e6
                gpu = sum(ns for k, ns in inside if k[0] < case["prefill_first"]) / 1e6
                cases[case["id"]][phase].update(first_token_ms=first_ms, gpu_before_first_ms=min(gpu, first_ms))
            agg = phases[phase]
            agg["wall_ms"] += wall
            agg["kernel_ms"] += busy
            agg["launches"] += len(inside)
            tokens = case["prompt_tokens"] if phase == "prefill" else case["output_tokens"]
            for (start, end, blocks, threads, raw), ns in inside:
                name = names.get(raw, raw)
                row = agg["by"].setdefault(name, {"ms": 0.0, "span_ms": 0.0, "calls": 0, "blocks": [], "flops": 0.0,
                                                  "bytes": 0.0, "cases": set()})
                row["ms"] += ns / 1e6
                row["span_ms"] += (end - start) / 1e6
                row["calls"] += 1
                row["blocks"].append(blocks)
                rule = _work_rule([r for r in rules or [] if r.get("flops_per_token") or r.get("bytes_per_call")], name)
                if rule:
                    row["bytes"] += float(rule.get("bytes_per_call") or 0)
                    if case["id"] not in row["cases"] and phase == "prefill":
                        row["flops"] += float(rule.get("flops_per_token") or 0) * tokens
                    row["format"] = rule.get("format")
                row["cases"].add(case["id"])
    out = {}
    for phase, agg in phases.items():
        rows = []
        for name, row in sorted(agg["by"].items(), key=lambda x: -x[1]["ms"]):
            blocks = int(statistics.median(row["blocks"]))
            entry = {"kernel": short_name(name), "ms": row["ms"], "span_ms": row["span_ms"],
                     "share": row["ms"] / agg["kernel_ms"] if agg["kernel_ms"] else 0.0,
                     "calls": row["calls"], "avg_us": 1000 * row["ms"] / row["calls"], "blocks": blocks,
                     "underfills": bool(device.get("sms")) and blocks < device["sms"]}
            seconds = row["ms"] / 1000
            if row["flops"] and seconds:
                entry["tflops"] = row["flops"] / seconds / 1e12
                peak = peaks.get(row.get("format") or "")
                if peak:
                    entry["flops_of_peak"] = entry["tflops"] / peak
            if row["bytes"] and seconds:
                entry["gbs"] = row["bytes"] / seconds / 1e9
                if bandwidth:
                    entry["bandwidth_of_peak"] = entry["gbs"] / bandwidth
            rows.append(entry)
        out[phase] = {"wall_ms": agg["wall_ms"], "kernel_ms": agg["kernel_ms"], "launches": agg["launches"],
                      "idle_ms": max(0.0, agg["wall_ms"] - agg["kernel_ms"]), "kernels": rows}
    return {"device": device, "phases": out, "cases": cases,
            "prefill_request": ("one output token on a fresh prompt (the case's prompt with a first token no earlier "
                                "request began with)") if all("prefill_start" in c for c in marks["cases"])
            else "the request up to its first delivered token"}


def render(result, top=TOP):
    sms = result["device"].get("sms")
    lines = [f"kernels: {result['engine'].get('name')} on {result['device'].get('name')} ({sms} SMs)"]
    for phase, agg in result["phases"].items():
        lines.append(f"{phase}: {agg['wall_ms']:.1f} ms wall over {len(result['cases'])} cases, "
                     f"{agg['kernel_ms']:.1f} ms in {agg['launches']} kernel launches"
                     + (f", GPU idle {agg['idle_ms']:.1f} ms ({agg['idle_ms'] / agg['wall_ms']:.0%})" if agg["wall_ms"] else ""))
        for row in agg["kernels"][:top]:
            extra = []
            if row.get("tflops") is not None:
                extra.append(f"{row['tflops']:.0f} TFLOPS" + (f" ({row['flops_of_peak']:.0%} of peak)" if "flops_of_peak" in row else ""))
            if row.get("gbs") is not None:
                extra.append(f"{row['gbs']:.0f} GB/s" + (f" ({row['bandwidth_of_peak']:.0%} of bandwidth)" if "bandwidth_of_peak" in row else ""))
            if row["underfills"]:
                extra.append(f"only {row['blocks']} blocks for {sms} SMs")
            lines.append(f"  {row['ms']:9.1f} ms {row['share']:5.1%}  x{row['calls']:<6d} {row['avg_us']:8.1f} us  "
                         f"{row['kernel']}" + (f"  [{'; '.join(extra)}]" if extra else ""))
        overlapped = sum(r["span_ms"] - r["ms"] for r in agg["kernels"])
        if overlapped > 0.02 * agg["kernel_ms"]:
            lines.append(f"  ({overlapped:.1f} ms of kernel spans overlapped an earlier kernel, e.g. under programmatic "
                         "dependent launch, and are not charged; span_ms in the JSON keeps raw durations)")
    lines.append(f"prefill: {result.get('prefill_request', 'the request up to its first delivered token')}; GPU time "
                 "per prefill-floor term (kernel_work rules with a term, else names):")
    for case, phases in result["cases"].items():
        entry = phases.get("prefill") if isinstance(phases.get("prefill"), dict) else {}
        terms = entry.get("terms") or {}
        if terms:
            named = (entry.get("classified_ms") or {}).get("names", 0.0)
            lines.append(f"  {case:13s} " + ", ".join(f"{k} {v:.1f} ms" for k, v in sorted(terms.items(), key=lambda x: -x[1]))
                         + (f" ({named:.1f} ms classed by name)" if named else "")
                         + (f"; first token {entry['first_token_ms']:.1f} ms, GPU busy {entry['gpu_before_first_ms']:.1f} ms"
                            " of it" if entry.get("first_token_ms") is not None else ""))
    lines.append("by case, top kernels per phase:")
    for case, phases in result["cases"].items():
        for phase, entry in ((p, e) for p, e in phases.items() if p in ("prefill", "decode")):
            tops = ", ".join(f"{short_name(t['kernel'], 48)} {t['share']:.0%}" for t in entry.get("top", [])[:3])
            lines.append(f"  {case:13s} {phase:7s} {entry['wall_ms']:8.1f} ms  {tops}")
    return "\n".join(lines)


def trace(adapter, model, data, out, config=None, cases=None, nvcc=None, device=None, log=print, context_tokens=None,
          work=None):
    """work: kernel_work rules ahead of the adapter's own (crit eval kernels --work)."""
    if not sys.platform.startswith("linux"):
        raise SystemExit("crit eval kernels needs Linux (CUPTI injection); on Windows run it under WSL")
    toolkit = find_toolkit(nvcc)
    with tempfile.TemporaryDirectory() as directory:
        library = build(toolkit, directory)
        spec = {"adapter": str(Path(adapter).resolve()), "model": str(model), "data": str(Path(data).resolve()),
                "config": config or {}, "cases": list(cases or []), "dir": directory, "context_tokens": context_tokens}
        import crit_core
        # The child imports crit_eval and crit_core from where this process does.
        paths = [str(EVAL_DIR.parent), str(Path(crit_core.__file__).resolve().parents[1]), os.environ.get("PYTHONPATH")]
        env = dict(os.environ, CUDA_INJECTION64_PATH=str(library), CRIT_KTRACE_DIR=directory,
                   PYTHONPATH=os.pathsep.join(filter(None, paths)))
        log(f"tracing kernels with {library.name} ({toolkit['nvcc']})")
        done = subprocess.run([sys.executable, "-B", "-m", "crit_eval.kernels", "--child", json.dumps(spec)], env=env)
        if done.returncode != 0:
            raise SystemExit(f"traced engine process failed ({done.returncode})")
        marks = load_json(Path(directory) / "marks.json")
        if work:
            marks["info"]["kernel_work"] = list(work) + list(marks["info"].get("kernel_work") or [])
        kernels, device_info = read_traces(directory, marks.get("clock", "realtime"))
    if not kernels:
        raise SystemExit("no kernels were recorded: the engine may not run on CUDA, or CUPTI could not attach")
    result = {"contract": CONTRACT, "kind": "kernels", "engine": marks["info"], "toolkit": portable_path(toolkit["nvcc"]),
              **analyze(kernels, device_info, marks, device)}
    write_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", required=True)
    _child(json.loads(parser.parse_args().child))
