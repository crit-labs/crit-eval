"""`crit-eval` command line (`crit eval` inside crit). Also runs as `python -m crit_eval` from a checkout."""
import argparse
import json
from pathlib import Path
import sys

# runner.PARTS, kept here so parsing imports nothing else.
PARTS = ("speed", "quality", "exactness", "context", "heldout", "history", "followup", "concurrency")
# The quality modes and the default, defined once (crit_eval.quality.fixed imports nothing outside the standard library).
from .quality.fixed import DEFAULT_QUALITY_MODE, QUALITY_MODES  # noqa: E402
QUALITY_MODE_HELP = ("how much output drift from the checkpoint the engine may trade for speed, whatever format the "
                     "checkpoint is stored in; both modes judge against the same reference (its stored weights, exact "
                     f"activations). {DEFAULT_QUALITY_MODE} (the default): a budget eight times "
                     "preserve's that only keeps out a broken engine, output differences that part at near-ties "
                     "allowed; preserve: a budget that keeps the output close to the checkpoint's")
CHAT_TEMPLATE_HELP = ("Chat template (jinja) to render the prompts with, for a model that carries none (default: the "
                      "one crit ships for the model's family); it becomes part of the model's identity")


def json_option(text, option):
    """An option's JSON text, else the file holding it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        path = Path(text)
        if not path.is_file():
            raise SystemExit(f"{option} is neither JSON ({error}) nor a file: {text}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{option} file {path} is not JSON: {error}")


def core_adapter():
    """Core's eval-v1 adapter file (crit_core/eval_adapter.py), the one adapter of every crit engine package."""
    import crit_core.eval_adapter
    return crit_core.eval_adapter.__file__


def package_engine(package, text, option):
    """--package: Core's adapter and its configuration, the option's overrides with the package named (the adapter
    reads the package's own configuration.json under them)."""
    folder = Path(package).expanduser().resolve()
    if not (folder / "plugins").is_dir() or not (folder / "configuration.json").is_file():
        raise SystemExit(f"{folder} is no crit engine package: it holds plugins/ and configuration.json")
    return core_adapter(), json.dumps({**json_option(text, option), "package": str(folder)})


def run_folder(path):
    """The crit run folder (run.json beside candidates/) holding `path`, or None."""
    for folder in Path(path).expanduser().resolve().parents:
        if (folder / "run.json").is_file() and (folder / "candidates").is_dir():
            return folder
    return None


def refuse_run_adapter(adapter, option):
    """An --adapter file inside a crit run folder is refused: a crit engine is its plugins, measured through Core's
    loader with --package, and an adapter of the run's own would measure an engine crit serve never runs."""
    found = run_folder(adapter) if adapter else None
    if found is not None:
        raise SystemExit(f"{option} {adapter} is inside the crit run folder {found}: measure a crit engine package with "
                         "--package DIR, which opens it through Core's loader as crit serve does")


def adapter_config(adapter, text, option):
    """An adapter's configuration: the configuration.json packaged beside it, the one crit serve opens the candidate
    with, then the option's overrides. An adapter outside a package gets the overrides alone."""
    overrides = json_option(text, option)
    packaged = Path(adapter).parent / "configuration.json" if adapter else None
    if packaged is None or not packaged.is_file():
        return overrides
    print(f"{option}: {packaged} with {len(overrides)} override(s)", file=sys.stderr)
    return {**json_option(str(packaged), option), **overrides}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="crit eval", description="Run the common crit speed, quality and speculative-exactness suite and report the engine against its ceilings")
    subs = parser.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="Evaluate one engine (prepares the data first when missing)")
    engine = run.add_mutually_exclusive_group(required=True)
    engine.add_argument("--package", help="A crit engine package (plugins/ and configuration.json), which Core's "
                                          "eval-v1 adapter opens through Core's loader")
    engine.add_argument("--adapter", help="Python file exporting create_eval_engine(context)")
    engine.add_argument("--url", help="Base URL of an OpenAI-compatible server the engine runs behind, for example "
                                      "http://127.0.0.1:8000/v1 (needs --context: the context the server was started with)")
    run.add_argument("--served-model", help="With --url: the model to request (default: the one the server lists)")
    run.add_argument("--model", required=True, help="Model directory passed to the adapter")
    run.add_argument("--data", required=True, help="Prepared suite directory for this model")
    run.add_argument("--out", required=True, help="Result JSON path")
    run.add_argument("--config", default="{}", help="JSON configuration overrides for the adapter, text or a file, over the configuration.json packaged beside it")
    run.add_argument("--only", choices=list(PARTS), action="append",
                     help="Run only these parts (repeatable); default is all. context: the long-context curve and "
                          "retrieval up to the required context and the speed cases again at a lower context; "
                          "history: the same requests on the fresh engine and after every other part; "
                          "concurrency: closed-loop clients on a served engine that runs several requests at once")
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
    run.add_argument("--quality-mode", choices=QUALITY_MODES, default=DEFAULT_QUALITY_MODE,
                     help="The quality mode: " + QUALITY_MODE_HELP)
    run.add_argument("--exactness", choices=["reported", "required"],
                     help="reported (default): agreement with plain greedy is shown and the quality budget judges "
                          "the output; required: any difference fails the gate")
    run.add_argument("--profile", choices=["hub"],
                     help="A protocol profile fixing what the run measures (protocol profiles). hub: crit-hub "
                          "submissions, 200,000 tokens of context, the curve at its two ends, one engine load")
    run.add_argument("--stage", choices=["1", "2", "3", "full"],
                     help="How far a run without a profile measures (protocol stages): by default as far as the "
                          "budget allows at the engine's measured rates; 1, 2 or 3 at most that far; full every part "
                          "at the protocol's repetitions, as before stages (--warmups and --repetitions apply to it)")
    run.add_argument("--certify", action="store_true",
                     help="The candidate that will ship: stage 3 whatever the budget, then one request at the "
                          "required context itself, which the capacity gate needs")
    run.add_argument("--budget-minutes", type=float, metavar="MINUTES",
                     help="The staged run's wall-clock budget, loads included (default: the protocol's 45)")
    run.add_argument("--context", type=int, metavar="TOKENS",
                     help="Context (prompt plus output) every engine is built and measured for: the run's kv_tokens. "
                          "Default: crit-eval's max context for the model on the --device")
    run.add_argument("--requested-context", type=int, metavar="TOKENS", default=None,
                     help="The longest request wanted, which caps the max context crit-eval computes without --context "
                          "(default: settings.toml context_tokens, else 200,000)")
    run.add_argument("--context-reserve-gib", type=float, default=None,
                     help="Device memory kept out of the KV cache when crit-eval computes the max context: "
                          "activations, workspaces, graphs and speed techniques (default: settings.toml "
                          "context_reserve_gib, else the protocol's reserve_gib)")
    run.add_argument("--vram-headroom-gib", type=float, default=None,
                     help="Device memory left free when crit-eval computes the max context (default 1)")
    run.add_argument("--adaptive-sampled", action=argparse.BooleanOptionalAction, default=False,
                     help="Repeat each sampled speed case at further seeds until its relative standard error reaches "
                          "the protocol's target or its repetition cap (protocol sampled_speed.adaptive), recorded in "
                          "the result; off by default: the protocol's one seed")
    run.add_argument("--attribute-drift", action="store_true",
                     help="After quality, score it again on an engine opened with each of the adapter's exact_variants "
                          "(one matrix group or arithmetic choice exact) and report each one's share of the drift budget")
    run.add_argument("--keep-disturbed", action="store_true",
                     help="Finish the round even when a speed case's repetitions show a disturbed device (by default "
                          "the round stops there; protocol speed.disturbed_spread)")
    run.add_argument("--chat-template", metavar="FILE",
                     help=CHAT_TEMPLATE_HELP + "; data prepared already must have been prepared with it")
    run.add_argument("--resume", metavar="JOURNAL",
                     help="Go on from a round that stopped (interrupted, crashed, a disturbed device, a hung engine): "
                          "its journal (R.journal.jsonl beside its result R.json) answers the requests it served, "
                          "the engine loads only for the requests left; the round's options must be the same")
    run.add_argument("--jitter-seed", type=int, metavar="SEED",
                     help="Draw the held-out prompts' and the long-context curve's lengths from this seed, a recorded "
                          "result's jitter.seed, to measure at its lengths again (default: a new seed per run; a "
                          "resumed round's from its journal; protocol jitter)")
    run.add_argument("--no-journal", action="store_true",
                     help="Write no journal beside the result (by default every engine opened and request served is "
                          "one line of OUT's .journal.jsonl, synced as it completes)")

    precheck = subs.add_parser("precheck", help="In minutes, the gates a candidate most often fails a whole round on: "
                                                "honesty of short replies and of one long reply, the request history "
                                                "around them, device memory and (with an adapter) the lower context")
    engine = precheck.add_mutually_exclusive_group(required=True)
    engine.add_argument("--package", help="A crit engine package, opened through Core's loader")
    engine.add_argument("--adapter", help="Python file exporting create_eval_engine(context)")
    engine.add_argument("--url", help="Base URL of an OpenAI-compatible server the engine runs behind (no lower-context "
                                      "check: a server cannot be opened again)")
    precheck.add_argument("--served-model", help="With --url: the model to request")
    precheck.add_argument("--model", required=True, help="Model directory passed to the adapter")
    precheck.add_argument("--data", required=True, help="Prepared suite directory for this model")
    precheck.add_argument("--context", type=int, required=True, metavar="TOKENS",
                          help="Context the engine is built for (the run's kv_tokens)")
    precheck.add_argument("--config", default="{}", help="JSON configuration overrides for the adapter")
    precheck.add_argument("--quality-mode", choices=QUALITY_MODES, default=DEFAULT_QUALITY_MODE,
                          help=f"The run's quality mode (default {DEFAULT_QUALITY_MODE}): "
                               "speed bounds honesty as Eval does and lets history part at near-ties; preserve does neither")
    precheck.add_argument("--no-lower", action="store_true", help="Skip the lower-context reopen")
    precheck.add_argument("--vram-headroom-gib", type=float, default=None,
                          help="Device memory to find free after the load (default: settings.toml vram_headroom_gib, "
                               "else 1)")
    precheck.add_argument("--chat-template", metavar="FILE", help=CHAT_TEMPLATE_HELP)
    precheck.add_argument("--out", help="Write the items as JSON here")

    context = subs.add_parser("context", help="The max context for a model on a device: the model's positions and "
                                              "the KV cache that fits next to its weights")
    context.add_argument("--model", required=True, help="Model directory")
    context.add_argument("--device", required=True, metavar="JSON", help="Device file from `crit eval probe`")
    context.add_argument("--vram-headroom-gib", type=float, default=None,
                         help="Device memory left free (default: settings.toml vram_headroom_gib, else 1)")
    context.add_argument("--ram-headroom-gib", type=float, default=None,
                         help="Host memory left free, for the placement scenario (default: settings.toml "
                              "ram_headroom_gib, else 2)")
    context.add_argument("--context-reserve-gib", type=float, default=None,
                         help="Device memory kept out of the KV cache for workspaces and speed techniques "
                              "(default: settings.toml context_reserve_gib, else the protocol's reserve_gib)")
    context.add_argument("--requested-context", type=int, metavar="TOKENS", default=None,
                         help="The longest request wanted (default: settings.toml context_tokens, else 200,000)")
    context.add_argument("--drafter", action="append", default=[], metavar="DIR",
                         help="A companion drafter the engine may load beside the model; its stored weights are kept "
                              "out of the KV cache (repeatable)")
    context.add_argument("--json", action="store_true", help="Print the computation as JSON")

    probe = subs.add_parser("probe", help="Measure device bandwidth and GEMM rates per weight format (needs nvcc, "
                                          "or hipcc on ROCm)")
    probe.add_argument("--out", required=True, help="Device JSON path for `run --device`")
    probe.add_argument("--backend", choices=("cuda", "hip"), help="Device API to probe (default: cuda when nvcc is found, "
                                                                  "else hip when hipcc is)")
    probe.add_argument("--nvcc", help="nvcc to build the probe with (default: PATH, CUDA_HOME, /usr/local/cuda*)")
    probe.add_argument("--hipcc", help="hipcc to build the HIP probe with (default: PATH, ROCM_PATH, /opt/rocm*)")
    probe.add_argument("--repeats", type=int, default=3, help="Probe runs; each value is the median")
    probe.add_argument("--devices", metavar="all|N,M", type=lambda v: v if v == "all" else [int(x) for x in v.split(",")],
                       help="Probe several GPUs (CUDA): each one's figures and the copy rate between each pair, for "
                            "multi-GPU limits (default: device 0 alone)")

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
    reference.add_argument("--allowance", action="store_true",
                           help="Also score the texts with the activations quantized as the checkpoint declares and "
                                "store how far that lands from the scores (allowance.json): what the format itself "
                                "costs, reported, not judged; a second scoring pass")

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

    journaled = subs.add_parser("journal", help="How far a round got: its journal's engines, parts and requests, "
                                                "replayed and live, and how it ended")
    journaled.add_argument("journal", help="A round's journal (R.journal.jsonl beside its result R.json)")

    compare = subs.add_parser("compare", help="Print a side-by-side table of result files")
    compare.add_argument("results", nargs="+")

    pin = subs.add_parser("pin", help="Record one engine's plain greedy output of every speed case as the data "
                                      "directory's pinned continuation (decode is then measured on that text)")
    engine = pin.add_mutually_exclusive_group(required=True)
    engine.add_argument("--package", help="A crit engine package, opened through Core's loader")
    engine.add_argument("--adapter", help="Python file exporting create_eval_engine(context)")
    pin.add_argument("--model", required=True, help="Model directory passed to the adapter")
    pin.add_argument("--data", required=True, help="Prepared data directory to pin the continuation in")
    pin.add_argument("--config", default="{}", help="JSON configuration overrides for the adapter, text or a file, over the configuration.json packaged beside it")
    pin.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    ab = subs.add_parser("ab", help="Alternate two engines' speed runs in one session and report paired ratios")
    engine = ab.add_mutually_exclusive_group(required=True)
    engine.add_argument("--package", help="Engine A's crit package, opened through Core's loader")
    engine.add_argument("--adapter", help="Adapter of engine A")
    ab.add_argument("--config", default="{}", help="JSON configuration overrides of engine A (text or a file), over its packaged configuration.json")
    engine = ab.add_mutually_exclusive_group()
    engine.add_argument("--package-b", help="Engine B's crit package (default: engine A's engine)")
    engine.add_argument("--adapter-b", help="Adapter of engine B (default: engine A's adapter)")
    ab.add_argument("--config-b", default="{}", help="JSON configuration overrides of engine B (text or a file), over its packaged configuration.json")
    ab.add_argument("--model", required=True)
    ab.add_argument("--data", required=True)
    ab.add_argument("--out", required=True, help="Result JSON path")
    ab.add_argument("--rounds", type=int, default=2, help="Rounds; each round runs both engines, order alternating")
    ab.add_argument("--repetitions", type=int, help="Measured repetitions per engine and round")
    ab.add_argument("--warmups", type=int, help="Warmup runs per engine and round")
    ab.add_argument("--context", type=int, metavar="TOKENS", help="Context both engines are built for (the run's kv_tokens)")

    kernels = subs.add_parser("kernels", help="Trace every kernel of the speed cases (CUPTI, Linux) and rank them "
                                              "per phase by time, launch width and share of peak")
    engine = kernels.add_mutually_exclusive_group(required=True)
    engine.add_argument("--package", help="A crit engine package, opened through Core's loader")
    engine.add_argument("--adapter", help="Python file exporting create_eval_engine(context)")
    kernels.add_argument("--model", required=True, help="Model directory passed to the adapter")
    kernels.add_argument("--data", required=True, help="Prepared suite directory for this model")
    kernels.add_argument("--out", required=True, help="Kernel report JSON path")
    kernels.add_argument("--config", default="{}", help="JSON configuration overrides for the adapter, text or a file, over the configuration.json packaged beside it")
    kernels.add_argument("--case", action="append", help="Trace only these speed cases (repeatable); default all")
    kernels.add_argument("--device", metavar="JSON", help="Device ceilings from `crit eval probe` for share of peak")
    kernels.add_argument("--nvcc", help="nvcc of a CUDA toolkit that ships CUPTI")
    kernels.add_argument("--backend", choices=("cuda", "hip"), default="cuda",
                         help="Device API the engine runs on: cuda (CUPTI) or hip (rocprofiler-sdk)")
    kernels.add_argument("--hipcc", help="hipcc of a ROCm that ships rocprofiler-sdk (--backend hip)")
    kernels.add_argument("--top", type=int, default=12, help="Kernels listed per phase")
    kernels.add_argument("--work", metavar="JSON", help="kernel_work rules ahead of the adapter's own, as JSON text or a "
                                                       "file: [{\"match\": NAME_PART, \"term\": matmul|attention|recurrent|"
                                                       "other|draft, \"format\": nvfp4|fp8|bf16, ...}]")
    kernels.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    workbench = subs.add_parser("workbench", help="Check, time or profile a kernel on its workbench folder "
                                                  "(crit kernels new): reference, floor, yardstick")
    workbench.add_argument("action", choices=("check", "bench", "profile"))
    workbench.add_argument("folder", help="The workbench folder")
    workbench.add_argument("--python", help="The engine's Python (torch and the GPU); default this one")
    workbench.add_argument("--shape", type=int, help="One shape of kernel.json by index (profile: default the first)")
    workbench.add_argument("--iters", type=int, help="bench: timed calls per shape and column count")
    workbench.add_argument("--device", metavar="JSON", help="bench: a `crit eval probe` result, for the floor")
    workbench.add_argument("--out", help="check and bench: the result JSON (default FOLDER/check.json, bench.json)")
    workbench.add_argument("--ncu", help="profile: Nsight Compute's ncu, to add its sections")

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
    # A crit engine package is measured through Core's own adapter, never one of the package's.
    if args.command == "ab" and args.package and not (args.package_b or args.adapter_b):
        args.package_b = args.package   # engine B defaults to engine A
    refuse_run_adapter(None if getattr(args, "package", None) else getattr(args, "adapter", None), "--adapter")
    refuse_run_adapter(None if getattr(args, "package_b", None) else getattr(args, "adapter_b", None), "--adapter-b")
    if getattr(args, "package", None):
        args.adapter, args.config = package_engine(args.package, args.config, "--config")
    if getattr(args, "package_b", None):
        args.adapter_b, args.config_b = package_engine(args.package_b, args.config_b, "--config-b")
    if args.command == "workbench":
        from .workbench import profile, run_harness
        if args.action == "profile":
            result = profile(args.folder, args.python, args.ncu, args.shape or 0)
            print(json.dumps(result, indent=2))
            return 0
        options = {"iters": args.iters, "device": args.device, "out": args.out, "shape": args.shape}             if args.action == "bench" else {"out": args.out, "shape": args.shape}
        code, result = run_harness(args.folder, args.action, args.python, **options)
        if result is None:
            raise SystemExit(f"the workbench harness exited {code} without a result")
        return 0 if code == 0 else 1
    if args.command == "kernels":
        from . import load_json as load
        from .limits.kernels import render as render_kernels, trace
        from .quality.prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .quality.prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        work = json_option(args.work, "--work") if args.work else None
        result = trace(args.adapter, args.model, args.data, args.out, adapter_config(args.adapter, args.config, "--config"), args.case,
                       args.nvcc,
                       load(args.device) if args.device else None, context_tokens=args.context, work=work,
                       backend=args.backend, hipcc=args.hipcc)
        print(f"wrote {args.out}")
        print(render_kernels(result, args.top))
        return 0
    if args.command == "prepare":
        from .quality.prepare import prepare as do_prepare
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
        from .quality.reference import score_data
        from .quality.fixed import STORE
        how = score_data(args.model, args.data, log=lambda line: print(line, flush=True), allowance=args.allowance)
        print(f"reference scores of {how['positions']:,} positions {how['method']} in {Path(args.data) / STORE}"
              + (f" ({how['seconds']:.0f} s, {how['bytes'] / 2**20:.1f} MiB)" if how["method"] == "scored" else "")
              + (f"; the reference's own spread {how['spread']}" if how.get("spread") else "")
              + {"none": "; the checkpoint declares no quantized activations: no allowance to measure",
                 "unsupported": "; the allowance not measured (see above)"}.get(
                  how.get("allowance"), f"; the allowance (what the declared activation format costs, reported only) "
                                        f"{how['allowance']}" if how.get("allowance") else ""))
        return 0
    if args.command == "fetch":
        from .quality.packages import fetch as do_fetch, source
        from .quality.prepare import read_template
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
        from .quality.packages import pack as do_pack
        entry = do_pack(args.data, args.packages)
        print(f"packed {entry['name']} ({entry['bytes'] / 2**20:.1f} MiB) into {args.packages}")
        return 0
    if args.command == "context":
        from . import load_json as load
        from .limits.context import GIB, max_context, render as render_context
        from . import EVAL_DIR
        settings = context_settings(load(EVAL_DIR / "protocol.json")["context"], args)
        value = max_context(args.model, load(args.device), headroom_gib(args) * GIB, settings, requested(args),
                            drafters=args.drafter)
        from .limits.context import placement, render_placement
        from . import machine_settings
        ram = args.ram_headroom_gib if args.ram_headroom_gib is not None else machine_settings().get("ram_headroom_gib", 2.0)
        where = placement(args.model, load(args.device), headroom_gib(args) * GIB, ram * GIB, settings, requested(args),
                          drafters=args.drafter)
        print(json.dumps({**value, "placement": where}, indent=1) if args.json
              else render_context(value) + "\n" + render_placement(where))
        return 0 if value["tokens"] else 1
    if args.command == "journal":
        from .run.journal import describe
        print(describe(args.journal))
        return 0
    if args.command == "compare":
        from .reports.compare import compare as do_compare
        print(do_compare(args.results))
        return 0
    if args.command == "pin":
        from .run.pin import CONTINUATIONS, pin as do_pin
        from .quality.prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .quality.prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        pinned = do_pin(args.adapter, args.model, args.data, adapter_config(args.adapter, args.config, "--config"),
                        context_tokens=args.context)
        print(f"pinned {len(pinned['cases'])} speed cases in {Path(args.data) / CONTINUATIONS} (sha256 {pinned['sha256'][:12]})")
        return 0
    if args.command == "ab":
        from .run.ab import render as render_ab, run_ab
        result = run_ab(args.adapter, adapter_config(args.adapter, args.config, "--config"), args.adapter_b or args.adapter,
                        adapter_config(args.adapter_b or args.adapter, args.config_b, "--config-b"),
                        args.model, args.data, args.out, args.rounds, args.warmups, args.repetitions,
                        context_tokens=args.context)
        print(f"wrote {args.out}")
        print(render_ab(result))
        return 0
    if args.command == "probe":
        from .limits.probe import probe as do_probe
        device = do_probe(args.out, args.nvcc, args.repeats, backend=args.backend, hipcc=args.hipcc, devices=args.devices)
        print(f"wrote {args.out}: {device['bandwidth_gbs']:.0f} GB/s, "
              + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in device["tflops"].items())
              + "; raw MMA " + (", ".join(f"{k} {v:.0f}" for k, v in (device.get("mma_tflops") or {}).items()) or "none")
              + (f", {device['memory_free_bytes'] / 1e9:.1f} of {device['memory_total_bytes'] / 1e9:.1f} GB free"
                 if device.get("memory_free_bytes") and device.get("memory_total_bytes") else ""))
        return 0
    if args.command == "precheck":
        from .quality.prepare import read_template
        from .run.precheck import run_precheck
        items = run_precheck(args.adapter, args.model, Path(args.data), args.context, args.quality_mode,
                             config=adapter_config(args.adapter, args.config, "--config") if args.adapter else {},
                             url=args.url, served_model=args.served_model,
                             chat_template=read_template(args.chat_template), lower=not args.no_lower,
                             vram_headroom_bytes=round(headroom_gib(args) * (1 << 30)))
        if args.out:
            Path(args.out).write_text(json.dumps({"items": items}, indent=1), encoding="utf-8")
        failed = [item["name"] for item in items if item["state"] == "failed"]
        print("precheck: " + ("failed " + ", ".join(failed) if failed else "nothing failed"))
        return 4 if failed else 0
    if args.command == "status":
        from . import load_json as load
        from .reports.status import render, status_of
        from . import EVAL_DIR
        protocol = load(EVAL_DIR / "protocol.json")
        current = status_of(args.result, protocol, args.baseline, args.device, args.data, args.kernels)
        print(json.dumps(current, indent=1) if args.json else render(current))
        return 0
    from .quality.prepare import PREPARED, read_template
    from .run.runner import run as do_run
    parts = tuple(args.only or PARTS)
    data = Path(args.data)
    template = read_template(args.chat_template)
    if not (data / PREPARED).is_file():
        from .quality.packages import fetch as do_fetch, source
        from .quality.prepare import prepare as do_prepare
        try:
            # A published package brings the reference scores too; otherwise prepare here.
            if source() is None or do_fetch(args.model, data, source(), chat_template=template) is None:
                do_prepare(args.model, data, chat_template=args.chat_template)
        except ValueError as error:   # no chat template to render with, or one for a model that carries its own
            raise SystemExit(f"crit eval run: {error}")
    from . import load_json as load
    from .reports.status import render
    device = load(args.device) if args.device else None
    baseline = load(args.baseline) if args.baseline else None
    result = do_run(args.adapter, args.model, data, args.out, adapter_config(args.adapter, args.config, "--config"), parts,
                    args.warmups, args.repetitions, args.label, bandwidth_gbs=args.bandwidth, device=device,
                    baseline=baseline, quality_budget=(json_option(args.quality_budget, "--quality-budget")
                                                     if args.quality_budget else None),
                    exactness=args.exactness, context_tokens=args.context, requested_context=requested(args),
                    vram_headroom_bytes=round(headroom_gib(args) * (1 << 30)),
                    context_reserve_bytes=round(reserve_gib(args) * (1 << 30)) if reserve_gib(args) is not None else None,
                    abort_disturbed=not args.keep_disturbed, url=args.url, served_model=args.served_model,
                    profile=args.profile, chat_template=template, attribute_drift=args.attribute_drift,
                    adaptive_sampled=args.adaptive_sampled, quality_mode=args.quality_mode,
                    stage=args.stage if args.stage in (None, "full") else int(args.stage), certify=args.certify,
                    budget_seconds=args.budget_minutes * 60 if args.budget_minutes else None,
                    journaled=not args.no_journal, resume=args.resume, jitter_seed=args.jitter_seed)
    print(f"wrote {args.out}")
    if result.get("aborted"):
        print(f"aborted: {result['aborted']['detail']}")
        if result.get("journal"):
            from .run.journal import path_for
            print(f"its journal keeps what it measured: --resume {path_for(args.out)} goes on from there")
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
    """--requested-context, else settings.toml's context_tokens, else 200,000."""
    from . import requested_context
    return args.requested_context if args.requested_context is not None else requested_context()


def context_settings(settings, args):
    reserve = reserve_gib(args)
    return settings if reserve is None else {**settings, "reserve_gib": reserve}
