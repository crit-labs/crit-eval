"""`crit-eval` command line (`crit eval` inside crit). Also runs as `python -m crit_eval` from a checkout."""
import argparse
import json
from pathlib import Path
import sys

# runner.PARTS, kept here so parsing imports nothing else.
PARTS = ("speed", "quality", "exactness", "context", "heldout", "history", "followup")
CHAT_TEMPLATE_HELP = ("Chat template (jinja) to render the prompts with, for a model that carries none (default: the "
                      "one crit ships for its family, as for DeepSeek V4); it becomes part of the model's identity")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="crit eval", description="Run the common crit speed, quality and speculative-exactness suite and report the engine against its ceilings")
    subs = parser.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="Evaluate one engine (prepares the data first when missing)")
    engine = run.add_mutually_exclusive_group(required=True)
    engine.add_argument("--adapter", help="Python file exporting create_eval_engine(context)")
    engine.add_argument("--url", help="Base URL of an OpenAI-compatible server the engine runs behind, for example "
                                      "http://127.0.0.1:8000/v1 (needs --context: the context the server was started with)")
    run.add_argument("--served-model", help="With --url: the model to request (default: the one the server lists)")
    run.add_argument("--model", required=True, help="Model directory passed to the adapter")
    run.add_argument("--data", required=True, help="Prepared suite directory for this model")
    run.add_argument("--out", required=True, help="Result JSON path")
    run.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    run.add_argument("--only", choices=list(PARTS), action="append",
                     help="Run only these parts (repeatable); default is all. context: the long-context curve and "
                          "retrieval up to the required context and the speed cases again at a lower context; "
                          "history: the same requests on the fresh engine and after every other part")
    run.add_argument("--repetitions", type=int, help="Override the protocol's measured repetitions")
    run.add_argument("--warmups", type=int, help="Override the protocol's warmup runs")
    run.add_argument("--label", help="Name shown in comparisons")
    run.add_argument("--device", metavar="JSON", help="Device ceilings from `crit eval probe` (bandwidth and GEMM rates)")
    run.add_argument("--bandwidth", type=float, metavar="GBS",
                     help="Achievable device memory bandwidth in GB/s; overrides the --device value")
    run.add_argument("--baseline", metavar="RESULT", help="Earlier result of the same data to show KPI changes against")
    run.add_argument("--quality-budget", metavar="JSON",
                     help='Drift limits over the protocol defaults, e.g. \'{"ppl_ratio_max": 1.02}\'; '
                          'keys ppl_ratio_max, kld_mean_max, kld_p99_max')
    run.add_argument("--exactness", choices=["reported", "required"],
                     help="reported (default): agreement with plain greedy is shown and the quality budget judges "
                          "the output; required: any difference fails the gate")
    run.add_argument("--profile", choices=["hub"],
                     help="A protocol profile fixing what the run measures (protocol profiles). hub: crit-hub "
                          "submissions, 200,000 tokens of context, the curve at its two ends, one engine load")
    run.add_argument("--context", type=int, metavar="TOKENS",
                     help="Context (prompt plus output) every engine is built and measured for: the run's kv_tokens. "
                          "Default: crit-eval's max context for the model on the --device")
    run.add_argument("--requested-context", type=int, metavar="TOKENS", default=None,
                     help="The longest request wanted, which caps the max context crit-eval computes without --context "
                          "(default: settings.toml context_tokens, else 262,144)")
    run.add_argument("--context-reserve-gib", type=float, default=None,
                     help="Device memory kept out of the KV cache when crit-eval computes the max context: "
                          "activations, workspaces, graphs and speed techniques (default: settings.toml "
                          "context_reserve_gib, else the protocol's reserve_gib)")
    run.add_argument("--vram-headroom-gib", type=float, default=None,
                     help="Device memory left free when crit-eval computes the max context (default 1)")
    run.add_argument("--adaptive-sampled", action=argparse.BooleanOptionalAction, default=True,
                     help="Repeat each sampled speed case at further seeds until its relative standard error reaches "
                          "the protocol's target or its repetition cap (protocol sampled_speed.adaptive; the default). "
                          "--no-adaptive-sampled keeps the seeds' repetitions alone: an override like --repetitions, "
                          "recorded in the result")
    run.add_argument("--attribute-drift", action="store_true",
                     help="After quality, score it again on an engine opened with each of the adapter's exact_variants "
                          "(one matrix group or arithmetic choice exact) and report each one's share of the drift budget")
    run.add_argument("--keep-disturbed", action="store_true",
                     help="Finish the round even when a speed case's repetitions show a disturbed device (by default "
                          "the round stops there; protocol speed.disturbed_spread)")
    run.add_argument("--chat-template", metavar="FILE",
                     help=CHAT_TEMPLATE_HELP + "; data prepared already must have been prepared with it")

    context = subs.add_parser("context", help="The max context for a model on a device: the model's positions and "
                                              "the KV cache that fits next to its weights")
    context.add_argument("--model", required=True, help="Model directory")
    context.add_argument("--device", required=True, metavar="JSON", help="Device file from `crit eval probe`")
    context.add_argument("--vram-headroom-gib", type=float, default=None,
                         help="Device memory left free (default: settings.toml vram_headroom_gib, else 1)")
    context.add_argument("--context-reserve-gib", type=float, default=None,
                         help="Device memory kept out of the KV cache for workspaces and speed techniques "
                              "(default: settings.toml context_reserve_gib, else the protocol's reserve_gib)")
    context.add_argument("--requested-context", type=int, metavar="TOKENS", default=None,
                         help="The longest request wanted (default: settings.toml context_tokens, else 262,144)")
    context.add_argument("--json", action="store_true", help="Print the computation as JSON")

    probe = subs.add_parser("probe", help="Measure device bandwidth and GEMM rates per weight format (needs nvcc)")
    probe.add_argument("--out", required=True, help="Device JSON path for `run --device`")
    probe.add_argument("--nvcc", help="nvcc to build the probe with (default: PATH, CUDA_HOME, /usr/local/cuda*)")
    probe.add_argument("--repeats", type=int, default=3, help="Probe runs; each value is the median")

    status = subs.add_parser("status", help="Gates, goals, KPIs against their physical limits, gaps and missing inputs")
    status.add_argument("result")
    status.add_argument("--baseline", metavar="RESULT", help="Earlier result to show KPI changes against")
    status.add_argument("--json", action="store_true", help="Print the status as JSON")
    status.add_argument("--device", metavar="JSON", help="Price prefill and time to first token with this `crit eval probe` "
                                                       "result's rates instead of the result's own device")
    status.add_argument("--kernels", metavar="JSON", help="A kernel trace of the engine (crit eval kernels): its prefill GPU "
                                                         "time per floor term against the term's floor, with a gap each")
    status.add_argument("--data", metavar="DIR", help="The data directory the result was measured on: its model arithmetic "
                                                     "prices prefill for a result that does not record it")

    prepare = subs.add_parser("prepare", help="Freeze the suite as token ids for one model")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--sampling", metavar="NAME=VALUE,...",
                         help="The user's sampling for the sampled speed suite in place of the model's "
                              "generation_config.json, e.g. temperature=0.6,top_p=0.95,top_k=20 (temperature "
                              "required; top_p, top_k, min_p optional). Results record and label it as an override")
    prepare.add_argument("--chat-template", metavar="FILE", help=CHAT_TEMPLATE_HELP)

    reference = subs.add_parser("reference", help="Score the prepared quality texts once on the checkpoint they were "
                                                  "prepared from and store the scores under the data (torch + "
                                                  "transformers, GPU; stop any engine first)")
    reference.add_argument("--model", required=True, help="The checkpoint the data was prepared from")
    reference.add_argument("--data", required=True, help="Prepared suite directory")

    fetch = subs.add_parser("fetch", help="Fetch the checkpoint's reference package (prepared data and reference "
                                          "scores) into the data, so a run needs only the engine")
    fetch.add_argument("--model", required=True, help="The checkpoint the engine serves")
    fetch.add_argument("--data", required=True, help="Data directory to write")
    fetch.add_argument("--packages", help="Package source: a folder or an https base URL (default: settings.toml's "
                                          "eval_packages)")
    fetch.add_argument("--chat-template", metavar="FILE",
                       help=CHAT_TEMPLATE_HELP + "; the package was prepared with it")

    pack = subs.add_parser("pack", help="Add prepared data scored by the reference to a package folder")
    pack.add_argument("--data", required=True, help="Prepared suite directory holding reference scores")
    pack.add_argument("--packages", required=True, help="Package folder to add it to")

    compare = subs.add_parser("compare", help="Print a side-by-side table of result files")
    compare.add_argument("results", nargs="+")

    pin = subs.add_parser("pin", help="Record one engine's plain greedy output of every speed case as the data "
                                      "directory's pinned continuation (decode is then measured on that text)")
    pin.add_argument("--adapter", required=True, help="Python file exporting create_eval_engine(context)")
    pin.add_argument("--model", required=True, help="Model directory passed to the adapter")
    pin.add_argument("--data", required=True, help="Prepared data directory to pin the continuation in")
    pin.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    pin.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    ab = subs.add_parser("ab", help="Alternate two engines' speed runs in one session and report paired ratios")
    ab.add_argument("--adapter", required=True, help="Adapter of engine A")
    ab.add_argument("--config", default="{}", help="JSON configuration of engine A")
    ab.add_argument("--adapter-b", help="Adapter of engine B (default: engine A's adapter)")
    ab.add_argument("--config-b", default="{}", help="JSON configuration of engine B")
    ab.add_argument("--model", required=True)
    ab.add_argument("--data", required=True)
    ab.add_argument("--out", required=True, help="Result JSON path")
    ab.add_argument("--rounds", type=int, default=2, help="Rounds; each round runs both engines, order alternating")
    ab.add_argument("--repetitions", type=int, help="Measured repetitions per engine and round")
    ab.add_argument("--warmups", type=int, help="Warmup runs per engine and round")
    ab.add_argument("--context", type=int, metavar="TOKENS", help="Context both engines are built for (the run's kv_tokens)")

    kernels = subs.add_parser("kernels", help="Trace every kernel of the speed cases (CUPTI, Linux) and rank them "
                                              "per phase by time, launch width and share of peak")
    kernels.add_argument("--adapter", required=True, help="Python file exporting create_eval_engine(context)")
    kernels.add_argument("--model", required=True, help="Model directory passed to the adapter")
    kernels.add_argument("--data", required=True, help="Prepared suite directory for this model")
    kernels.add_argument("--out", required=True, help="Kernel report JSON path")
    kernels.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    kernels.add_argument("--case", action="append", help="Trace only these speed cases (repeatable); default all")
    kernels.add_argument("--device", metavar="JSON", help="Device ceilings from `crit eval probe` for share of peak")
    kernels.add_argument("--nvcc", help="nvcc of a CUDA toolkit that ships CUPTI")
    kernels.add_argument("--top", type=int, default=12, help="Kernels listed per phase")
    kernels.add_argument("--work", metavar="JSON", help="kernel_work rules ahead of the adapter's own, as JSON text or a "
                                                       "file: [{\"match\": NAME_PART, \"term\": matmul|attention|recurrent|"
                                                       "other|draft, \"format\": nvfp4|fp8|bf16, ...}]")
    kernels.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    args = parser.parse_args(argv)
    # An engine behind a server is the server's to load (crit serve holds the lock while it serves).
    if args.command in ENGINE_COMMANDS and not getattr(args, "url", None):
        from crit_core.machine import engine_lock
        with engine_lock(f"crit eval {args.command}"):
            return _command(args)
    return _command(args)


# Commands that load an engine or the reference model: one at a time per machine (crit_core.machine); run only
# with --adapter.
ENGINE_COMMANDS = {"run", "pin", "ab", "kernels", "reference"}


def _command(args):
    if args.command == "kernels":
        from . import load_json as load
        from .kernels import render as render_kernels, trace
        from .prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        work = None
        if args.work:   # JSON text, else a file holding it
            try:
                work = json.loads(args.work)
            except json.JSONDecodeError:
                work = json.loads(Path(args.work).read_text(encoding="utf-8"))
        result = trace(args.adapter, args.model, args.data, args.out, json.loads(args.config), args.case, args.nvcc,
                       load(args.device) if args.device else None, context_tokens=args.context, work=work)
        print(f"wrote {args.out}")
        print(render_kernels(result, args.top))
        return 0
    if args.command == "prepare":
        from .prepare import prepare as do_prepare
        try:
            prepared = do_prepare(args.model, args.data, sampling=args.sampling, chat_template=args.chat_template)
        except ValueError as error:
            raise SystemExit(f"crit eval prepare: {error}")
        print(f"prepared {len(prepared['speed'])} speed cases and {len(prepared['quality'])} quality texts in {args.data}")
        template = prepared["chat_template"]
        where = {"model": "the model's own", "crit": f"crit's {template.get('file')} (the model carries none)",
                 "file": f"{template.get('file')} (--chat-template)"}
        print(f"chat template: {where[template['origin']]}")
        print(f"sampled suite: {prepared['sampling']} ({prepared['sampling_source'].replace('_', ' ')})")
        print(f"quality: score them once on the checkpoint: crit eval reference --model {args.model} --data {args.data}")
        return 0
    if args.command == "reference":
        from .reference import score_data
        how = score_data(args.model, args.data, log=lambda line: print(line, flush=True))
        print(f"reference scores of {how['positions']:,} positions {how['method']} in {Path(args.data) / 'reference'}"
              + (f" ({how['seconds']:.0f} s, {how['bytes'] / 2**20:.1f} MiB)" if how["method"] == "scored" else "")
              + (f"; the reference's own spread on the long texts {how['spread']}" if how.get("spread") else ""))
        return 0
    if args.command == "fetch":
        from .packages import fetch as do_fetch, source
        from .prepare import read_template
        where = source(args.packages)
        if where is None:
            raise SystemExit("crit eval fetch: name a package source (--packages, or eval_packages in settings.toml)")
        try:
            fetched = do_fetch(args.model, args.data, where, chat_template=read_template(args.chat_template))
        except ValueError as error:   # a --chat-template for a model that carries its own
            raise SystemExit(f"crit eval fetch: {error}")
        if fetched is None:
            print(f"{where} has no package for {args.model} under this suite: prepare and score it here "
                  f"(crit eval prepare, then crit eval reference)")
            return 1
        return 0
    if args.command == "pack":
        from .packages import pack as do_pack
        entry = do_pack(args.data, args.packages)
        print(f"packed {entry['name']} ({entry['bytes'] / 2**20:.1f} MiB) into {args.packages}")
        return 0
    if args.command == "context":
        from . import load_json as load
        from .context import GIB, max_context, render as render_context
        from . import EVAL_DIR
        settings = context_settings(load(EVAL_DIR / "protocol.json")["context"], args)
        value = max_context(args.model, load(args.device), headroom_gib(args) * GIB, settings, requested(args))
        print(json.dumps(value, indent=1) if args.json else render_context(value))
        return 0 if value["tokens"] else 1
    if args.command == "compare":
        from .report import compare as do_compare
        print(do_compare(args.results))
        return 0
    if args.command == "pin":
        from .pin import CONTINUATIONS, pin as do_pin
        from .prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        pinned = do_pin(args.adapter, args.model, args.data, json.loads(args.config), context_tokens=args.context)
        print(f"pinned {len(pinned['cases'])} speed cases in {Path(args.data) / CONTINUATIONS} (sha256 {pinned['sha256'][:12]})")
        return 0
    if args.command == "ab":
        from .ab import render as render_ab, run_ab
        result = run_ab(args.adapter, json.loads(args.config), args.adapter_b or args.adapter, json.loads(args.config_b),
                        args.model, args.data, args.out, args.rounds, args.warmups, args.repetitions,
                        context_tokens=args.context)
        print(f"wrote {args.out}")
        print(render_ab(result))
        return 0
    if args.command == "probe":
        from .probe import probe as do_probe
        device = do_probe(args.out, args.nvcc, args.repeats)
        print(f"wrote {args.out}: {device['bandwidth_gbs']:.0f} GB/s, "
              + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in device["tflops"].items())
              + "; raw MMA " + (", ".join(f"{k} {v:.0f}" for k, v in (device.get("mma_tflops") or {}).items()) or "none")
              + (f", {device['memory_free_bytes'] / 1e9:.1f} of {device['memory_total_bytes'] / 1e9:.1f} GB free"
                 if device.get("memory_free_bytes") and device.get("memory_total_bytes") else ""))
        return 0
    if args.command == "status":
        from . import load_json as load
        from .status import render, status_of
        from . import EVAL_DIR
        protocol = load(EVAL_DIR / "protocol.json")
        current = status_of(args.result, protocol, args.baseline, args.device, args.data, args.kernels)
        print(json.dumps(current, indent=1) if args.json else render(current))
        return 0
    from .prepare import PREPARED, read_template
    from .runner import run as do_run
    parts = tuple(args.only or PARTS)
    data = Path(args.data)
    template = read_template(args.chat_template)
    if not (data / PREPARED).is_file():
        from .packages import fetch as do_fetch, source
        from .prepare import prepare as do_prepare
        try:
            # A published package brings the reference scores too; otherwise prepare here.
            if source() is None or do_fetch(args.model, data, source(), chat_template=template) is None:
                do_prepare(args.model, data, chat_template=args.chat_template)
        except ValueError as error:   # no chat template to render with, or one for a model that carries its own
            raise SystemExit(f"crit eval run: {error}")
    from . import load_json as load
    from .status import render
    device = load(args.device) if args.device else None
    baseline = load(args.baseline) if args.baseline else None
    result = do_run(args.adapter, args.model, data, args.out, json.loads(args.config), parts,
                    args.warmups, args.repetitions, args.label, bandwidth_gbs=args.bandwidth, device=device,
                    baseline=baseline, quality_budget=json.loads(args.quality_budget) if args.quality_budget else None,
                    exactness=args.exactness, context_tokens=args.context, requested_context=requested(args),
                    vram_headroom_bytes=round(headroom_gib(args) * (1 << 30)),
                    context_reserve_bytes=round(reserve_gib(args) * (1 << 30)) if reserve_gib(args) is not None else None,
                    abort_disturbed=not args.keep_disturbed, url=args.url, served_model=args.served_model,
                    profile=args.profile, chat_template=template, attribute_drift=args.attribute_drift,
                    adaptive_sampled=args.adaptive_sampled)
    print(f"wrote {args.out}")
    if result.get("aborted"):
        print(f"aborted: {result['aborted']['detail']}")
        return 3
    print(render(result["status"]))
    return 0


def headroom_gib(args):
    """--vram-headroom-gib, else settings.toml's vram_headroom_gib, else 1 GiB."""
    from . import machine_settings
    return args.vram_headroom_gib if args.vram_headroom_gib is not None else machine_settings().get("vram_headroom_gib", 1.0)


def reserve_gib(args):
    """--context-reserve-gib, else settings.toml's context_reserve_gib, else None (the protocol's reserve)."""
    from . import machine_settings
    return args.context_reserve_gib if args.context_reserve_gib is not None else machine_settings().get("context_reserve_gib")


def requested(args):
    """--requested-context, else settings.toml's context_tokens, else 262,144."""
    from . import requested_context
    return args.requested_context if args.requested_context is not None else requested_context()


def context_settings(settings, args):
    reserve = reserve_gib(args)
    return settings if reserve is None else {**settings, "reserve_gib": reserve}
