# Changelog

All notable changes to crit-eval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.4.0] - 2026-10-30

### Overview

crit-eval 0.4.0 starts protocol series crit-eval-v0.4: a run measures in stages within a 45-minute budget, as far as
the engine's own speed allows, and only the candidate that will ship serves its required context (`--certify`). The
default requested context is 200,000 tokens. Data prepared under 0.3.x is prepared again; the reference scores carry
over.

### Added

- Protocol crit-eval-v0.4.0: a run without a profile measures in stages within a 45-minute wall-clock budget
  (protocol `stages`, `--budget-minutes`, `--stage`). Stage 1 runs on every engine: each speed case once, quality
  on the texts up to 2,048 tokens, each priced at the rates measured so far and left out when it would pass the
  budget (`over_budget`), the request history on the shortest case, the device memory the required context
  needs against what the load left free (`context.fit`), decode projected to the required context (`context.creep`)
  and the shortest prompt's time to first token (`speed.ttft_floor_ms`). After it the run goes on to the highest
  stage whose remaining requests fit the budget at the engine's measured rates: stage 2 adds the repetitions, the
  follow-up turns, the long quality text and the curve's first point; stage 3 the sampled suite at two seeds,
  exactness on two cases, the follow-up turns again and the lower context once. The full run of a 55 tok/s prefill
  candidate projected 6.2 h; its stage 1 takes about 25 minutes. The run prints its first estimate after its first
  speed case (`stages.first_estimate`).
- A run whose engine stops answering (a request past its deadline) keeps what it measured until then, marked
  `aborted` with reason `hung`, as a disturbed round does.
- `crit eval run --certify`: the candidate that will ship runs stage 3 whatever the budget and one request at the
  required context, which the capacity gate needs. Without it no request fills the required context: capacity is
  unmeasured and the curve stops at its first point. The curve's middle points are no longer measured.

### Changed

- The default requested context is 200,000 tokens (was 262,144).

## [0.3.0] - 2026-10-30

### Overview

crit-eval 0.3.0 starts protocol series crit-eval-v0.3: the `as_quantized` quality mode judges against the exact
reference scores, each set's limits raised by the checkpoint's own activation quantization drift (the allowance),
so one `crit eval reference` run serves both quality modes. It carries protocols crit-eval-v0.2.1 and v0.2.2 as well:
prompt positions are judged by KLD alone, and the reference's spread pass probes the short texts first. Engines
that read weights from host memory get their pass floor priced at the probe's host rates, and `crit eval context`
names the placement scenario of a model on a device. Data prepared under 0.2.x is prepared again; the reference
scores carry over.

### Added

- Protocol crit-eval-v0.2.2: the reference's spread pass scores the texts shorter than 8,192 tokens first, and the
  long texts again only when that probe's spread is not tiny (below 1/20 of the level at which it would raise a
  limit, protocol `quality.spread.probe`). The second run of the 32K text measured a dense model's spread hundreds
  of times below the budget and took most of the spread pass: Nanbeige4.3-33B NVFP4's reference took 380 s instead
  of 660 s on an RTX 5090, with the same scores and the same spread on every probed set. Skipping can only keep a
  limit, never raise one; `spread.json` names the skipped texts (`spread_skipped`), and `crit eval reference`
  reports the spread as `probed`.
- `crit_eval.limits.context.placement` and `crit eval context` name the placement scenario of a model on a device
  at the required context: `fits_device`, `sparse_device_host` (experts beyond device memory fit host memory),
  `sparse_device_host_disk`, `dense_device_host` or `beyond_host`, with the bytes each tier holds, where the lookup
  tables (gathered embedding and n-gram rows) go, and one plain pass's floor and ceiling per expert source: on the
  device, misses copied over the host link, misses computed on the CPU, priced at the probe's host-to-device and host
  memory rates. Before, the probe measured those rates and nothing used them. `--ram-headroom-gib` (else
  settings.toml's `ram_headroom_gib`) sets the host memory left free.
- An engine that declares `host_bytes_per_pass` (`{"link", "cpu"}`, the part of `bytes_per_pass` read from host
  memory) gets its pass floor priced at the probe's host rates: the longest of the device's bytes at its bandwidth,
  the link's bytes at the pinned host-to-device rate and all host bytes at host memory's read rate. Before, every byte
  was priced at the device's bandwidth, so an engine streaming 1.3 GB of experts per token over a 27 GB/s link saw a
  floor near 1 ms against 90 ms measured. headroom records `host_pass`.

- `crit eval reference --quality-mode as_quantized` quantizes the inputs of fused experts (every expert's matrices
  stacked, gate and up joined along the output rows, as transformers' MoE models hold them) with each expert matrix's
  own declared scale, instead of refusing the model. NVIDIA's Qwen3.8-Flash-Next NVFP4 declares NVFP4 inputs on all
  512 experts of every layer and was refused.
- `crit eval run` says when its device file is older than this checkout's probe (`device.outdated` in the result,
  and a log line naming what it lacks: raw tensor-core rates, host rates, or another probe source). Eval prices
  nothing a device file lacks, so an older probe left the fp16_acc prefill ceiling unpriced and prefill above 100% of
  its limit without saying why (the Nanbeige4.3-33B run's device file had cuBLASLt rates only).
- `max_context(..., drafters=)` and `crit eval context --drafter DIR` keep a companion drafter's stored weights out of
  the KV cache (`basis.companion_drafter_bytes`), so an engine built for the required context can still load the
  drafter it speculates with. `companion_drafter_bytes` also gives what one drafter pass reads: its weights plus the
  output head its logits go through (the target's, where the drafter stores none). Nanbeige4.3-33B with its 1.96 GB
  DFlash2 drafter on an RTX 5090: 98,304 tokens instead of 115,712 at no reserve.
- `crit eval workbench check|bench|profile FOLDER` measures a kernel in a workbench folder crit's `crit kernels new`
  scaffolds, running the folder's harness (shipped here as `crit_eval/workbench/harness.py`) in the engine's Python:
  the kernel against its FP64 reference, repeatable and column invariant; its time and a yardstick provider's on
  inputs cycled past the L2 against the probe's floor (bench.json, a plan's yardstick evidence); the compiled kernels'
  registers, shared memory and spills, and Nsight Compute's sections with `--ncu`.

### Fixed

- `run`, `pin`, `ab` and `kernels` with `--adapter` start from the `configuration.json` packaged beside the adapter,
  the configuration `crit serve` opens the candidate with, and apply `--config` over it. Before, a package's adapter
  opened with its defaults, so `crit eval kernels` on a candidate's package profiled another engine than the one Eval
  measured.
- `prepare` encodes the parts of a render that follow other text (a long case's document and template tail, the
  long-prompt filler, the quality answers, the retrieval facts) as the whole render encodes them. Tokenizers that mark
  the start of their input (Metaspace with `prepend_scheme: "first"`, as SentencePiece-style Llama tokenizers ship)
  gave each part encoded on its own a word marker of its own, so a server rendering the frozen messages counted other
  tokens (Nanbeige4.3-33B: `agent-long` at 14,335 instead of 14,336, and Eval stopped). Byte-level tokenizers encode
  as before; data prepared for a start-marking tokenizer is prepared again, with its reference.

### Changed

- Protocol crit-eval-v0.3.0: quality mode `as_quantized` judges against the exact reference scores, each set's limits
  raised to four times the checkpoint's own quantization drift where that exceeds the mode's budget, at most to four
  times that budget (mean KLD 0.4, p99 4.0, PPL ratio 1.4; protocol `quality.allowance`), instead of against a
  second set of scores of the checkpoint run as quantized. One
  `crit eval reference` run now serves both modes: for a checkpoint that declares quantized input activations it
  scores the texts a second time with them quantized and keeps only how far that lands from the exact scores, per set
  and path (`reference/allowance.json`, carried by reference packages), so a checkpoint has one package and one
  set of scores. A checkpoint that declares none (weight-only schemes) has an empty allowance and runs no second
  pass; one that declares a form crit does not emulate (DeepSeek-V4-Flash W4A16-FP8's dynamic FP8 attention inputs in
  groups of 128) records the allowance as unsupported, and an as_quantized run of it says so. `crit eval reference --quality-mode` is accepted and ignored; the reference derives the
  allowance from an earlier crit-eval's `reference-as-quantized/` store without running again, and `crit eval fetch`
  removes that store. Results record `allowance` and `allowance_raised` per set, and status names the sets.

- A disturbed speed case's message names the engine as the likely cause when other processes kept fewer than 2 CPU
  cores busy (file-backed reads that miss the page cache, host threads, expert misses), and both causes otherwise.
  Before, it always blamed the device, and one engine's own n-gram reads cost a restart on an idle machine.
- A disturbed speed case names the device's free memory, and below 1 GiB says the engine's own memory may be the
  disturbance: under Windows' WDDM a device past its memory pages to host RAM and slows unevenly instead of failing,
  which Eval read as another process using the device (a Nanbeige4.3-33B candidate whose prefill temporaries passed
  32 GB, 45% between repetitions).
- The sampled suite's disturbance check reads an engine whose rounds draft a varying number of tokens at the median
  drafts per round: with `drafted` telemetry in three or more repetitions, each repetition's time per target pass is
  adjusted by a slope fitted over the repetitions (`timing.width_adjusted_rounds`). An adaptive budget that falls back
  to plain steps mixes 1-pass steps and wider rounds by each seed's acceptance, so its time per round followed the
  seed and Eval stopped it as a disturbed device (a Nanbeige4.3-33B candidate with DFlash2, 33% between seeds).
- An engine whose draft width varies is read round by round: with `drafted` (the drafts made in each round, from
  `generate()` or crit serve's `crit_telemetry`) beside `accepted`, a depth's acceptance counts only the rounds that
  drafted that deep, and a draft step's time divides by the drafts made. Before, every round counted as the widest
  draft, so a short round that kept all its drafts looked rejected beyond them and the width projections ran low.
- A lower-context case whose first token came later names where the time went: the engine's own first-token time and
  the time outside the engine at both contexts (`context.lower.cases[*].engine_ttft_ms`, `outside_engine_ms` and their
  `_at_required` values, from the engine's reported prefill time), so a failed `lower_context` gate says whether host
  or server time moved or the prefill did.
- Protocol crit-eval-v0.2.1 (same series: data and scores carry over): a set's prompt positions (`SET/prompt`) report
  their perplexity ratio and are judged by KLD mean and p99 only (protocol `quality.prompt_perplexity`). They predict
  the user's turn, where a chat model calls most tokens near-impossible, so the ratio follows rounding: the Qwen3.8-27B
  NVFP4 as_quantized reference scored again with its FP32 prompt in 4,096-token chunks lands at 0.68 on long-4096
  (TF32 reruns: 0.77 to 1.40 across the sets), and engines with less KLD than those reruns failed on it. Status takes a
  set's own limits for its share of the budget and its worst perplexity ratio from the sets that judge it.
- Status judges a stored result's quality again by today's rules: a result measured before crit-eval-v0.2.1 has its
  prompt sets judged without the perplexity ratio (each changed verdict says `rejudged`, and the gate names them), so
  a status rendered later, as replays do, gives the verdict a measurement today would. The prompt activations' evidence
  (which formats quality passes or rules out, the reachable limit) is priced again from that verdict.

## [0.2.0] - 2026-10-27

### Overview

crit-eval 0.2.0 starts protocol series crit-eval-v0.2: the `as_quantized` quality mode judges by a drift budget twice
`preserve`'s, measured from what 4-bit activation rounding alone moves, and every quality text is judged against the
reference's own spread. Reference packages are published on Hugging Face and fetched by default, and carry the spread
they were judged with. The package is grouped into `run`, `quality`, `limits`, `reports` and `architectures`, with
architecture-specific code in `architectures` alone. How a checkpoint stores its matrices is recognised by crit-core,
so crit-eval 0.2.0 needs crit-core 0.12.

### Added

- Adapters load as a module of a package of their own folder: code beside an adapter is imported relatively
  (`from .engine import Engine`), so `crit eval ab` loads two packages whose modules share a name; a second adapter
  whose absolute import names a module the first adapter's folder provided is refused instead of running the first
  one's code.
- docs/ADAPTER.md: a Python engine with the engine library's interface (`info`, `generate`, `score`, `close`) passes
  the function opening it to crit_core's stock adapter (`eval_engine(context, open_engine, defaults)`) and writes no
  adapter of its own.

### Changed

- JSON options take text or a file: `--config`, `--config-b`, `--quality-budget` and `--work`.
- The `crit_eval` package is grouped by what each module does: `crit_eval.run` (runner, adapters and servers,
  outcomes, deadlines, estimates, `ab`, `pin`, host), `crit_eval.quality` (prepare, reference, fixed texts,
  packages), `crit_eval.limits` (probe, ceiling, context, headroom, kernels, schemes, tensors), `crit_eval.reports`
  (metrics, status, and `compare`, formerly `report`) and `crit_eval.architectures` (the GGUF layouts per architecture
  and the indexer forms, the only architecture-specific code). Code importing `crit_eval.<module>` imports it from
  its subpackage; the `crit eval` commands are unchanged. The two largest modules are split by part: `run.runner`
  keeps `run` with the adapters, timing, speed, quality, context and history parts in `run.adapters`, `run.timing`,
  `run.speed`, `run.quality_checks`, `run.long_context` and `run.history`, and `quality.reference` keeps
  `StreamingReference` with its attention, checkpoint readers and prompt state in `quality.attention`,
  `quality.checkpoints` and `quality.prompt_state`.
- Protocol crit-eval-v0.2.0 (a new series: data prepared under 0.1.x is prepared again; reference scores carry over).
  The `as_quantized` quality mode judges by a drift budget twice `preserve`'s (`quality.mode_thresholds`: PPL ratio
  ≤ 1.10, mean KLD ≤ 0.10, p99 KLD ≤ 1.0): on Qwen3.8-27B NVFP4, 4-bit activation math swings prompt rows by a
  mean KLD of 0.045-0.08 (p99 0.5-1.4) under rounding alone, the preserve budget's size. `preserve` is unchanged.
- The reference's own spread (`quality.spread`) covers every quality text, not only those of at least 8,192
  tokens (`from_tokens` 0), so short sets are judged against it too; a spread stored under the old rule is not used
  until `crit eval reference` scores it again.
- Reference packages carry each store's `spread.json` (`crit eval pack`), so a fetched package judges by the same
  spread as data scored on the machine.
- Reference packages are published on Hugging Face
  ([crit-labs/crit-eval-references](https://huggingface.co/datasets/crit-labs/crit-eval-references)): when neither
  `--packages`, `eval_packages` nor a checkout's `reference-packages/` names a source, `crit eval fetch`, `crit eval
  run` and `crit run eval` fetch from there, and score the reference on the machine when it has no package for the
  checkout or cannot be reached.
- How a checkpoint stores each weight matrix is recognised by `crit_core.model.schemes` (from the crit-core release
  that ships it), the copy engines read too; `crit_eval.limits.schemes` keeps the decoders. No result changes: the
  recognition moved unchanged.
- crit-eval needs crit-core 0.12 (`crit-core>=0.12.0,<0.13`), the release that ships `crit_core.model.schemes`.

## [0.1.2] - 2026-10-26

### Added

- Quality modes (protocol crit-eval-v0.1.2; no score of crit-eval-v0.1.0 moves): `preserve`, the default, judges
  quality against the reference with exact activations as before; `as_quantized` against the checkpoint run as
  quantized, the reference with the activations quantized as its stored scheme declares (NVFP4 in groups of 16 with
  the stored global scale, FP8 per token or per tensor; `crit_eval.schemes.activations` and `quantize_activations`).
  `crit eval reference --quality-mode as_quantized` scores them into `reference-as-quantized/`, reference packages
  carry them, and `crit eval run --quality-mode as_quantized` judges against them; results record
  `requirements.quality_mode`. A checkpoint whose declared matrices are fused experts is refused in that mode.
- Status prices prefill and the first tokens at the quality mode's arithmetic: in `preserve` at the fastest 16-bit
  prompt activations (or a faster one quality shows passing), in `as_quantized` at the stored format. Goals and
  `kpis` carry these limits (`stored_ceiling`, `stored_floor_ms` keep the stored format's), so a quality-preserving
  engine is measured against what it can reach, and status names the mode and the trade once ("as_quantized would
  allow prefill 18,220 tok/s"). Results without a mode are read as `preserve`.

### Changed

- `crit eval kernels` classes weight-only GEMMs (`w4a16`, `marlin`) and crit's `stored_tile` projections as matmul
  by name, with the weight format the name gives (`fp4`, `e2m1`: nvfp4; `fp8`, `e4m3`: fp8). Each traced prefill
  lists its largest kernels classed other by name (`unclassed`), and the trace and `status --kernels` print them,
  so a rule can class them. Before, crit's own BF16-activation prompt GEMMs counted as other kernels, and
  matmul looked faster than its floor.
- `crit eval status --kernels` shows each matmul row's achieved TFLOPS and its floor at the engine's declared
  `prompt_activations` (also when matmul is one row of unknown format), and gives a matmul slower than that rate a
  second prefill gap, at the rate of the arithmetic the engine already uses. A matmul faster than its floor is
  flagged as kernels classed under other terms, and a trace without every prefill-scored case says which it lacks
  (`untraced`), since the gaps per term need them all.
- `crit eval status` names the prompt activation formats faster than the reachable limit that no quality evidence
  judges yet ("not judged by quality"), beside those it shows passing or rules out.

## [0.1.1] - 2026-10-26

### Fixed

- crit-eval imports on Python 3.11 again: a status line's f-string reused its own quotes inside a replacement
  field, which only Python 3.12 parses. The protocol stays crit-eval-v0.1.0.

## [0.1.0] - 2026-10-26

### Added

- `crit-eval probe` measures the raw tensor-core rates beside cuBLASLt's GEMMs (`mma_tflops`: BF16 and FP16 into
  FP32, FP16 into FP16, FP8, and on sm_120a NVFP4 and FP4 times FP8), building the probe for the device's own
  architecture. On an RTX 5090 cuBLASLt's BF16 GEMM reaches 185 to 215 TFLOPS where the BF16 tensor cores run 251,
  and FP16 accumulation runs 499.
- Prefill and TTFT ceilings by prompt activation format (`headroom.activations`): the
  floors again with every quantized matrix group's prompt GEMMs at BF16, FP16 accumulation, FP8 and the stored
  format, and at the engine's own `prompt_activations` when its adapter declares them. The quality gate on the prompt
  paths (this result's and a same-model baseline's) shows them passing or rules them out, and `status` names the
  fastest shown passing as the reachable limit next to the stored-format one, with a prefill gap of its own.
- A result records the model arithmetic it was priced on (`ceiling`). `crit eval status --device` prices prefill and
  time to first token with a newer probe, `--data` with the arithmetic of the data an older result was measured on.
- crit serve's `engine_facts` may carry `prompt_activations`; crit-eval reads it.
- Free device memory at the required context (`context.device_memory`): after the
  engine loads and after the long-context curve, the least of them beside the capacity gate in `status`, and its
  change against a baseline.
- Draft and verify apart: `status --kernels` splits a speculative engine's rounds into
  verify pass and draft steps from the trace's kernels classed `draft`, when the engine reports no `verify_ms` and
  `draft_ms`, and gives their gaps apart. An adapter's `wider_draft` opens one more engine at a wider draft width for
  its acceptance, so the width projections go past the built width.
- Sampled decode noise (`sampled_speed.adaptive`): every sampled case reports its relative
  standard error, its repetitions and the repetitions a 3% target would take, and `status` prints the largest.
  `crit eval run --adaptive-sampled` repeats a case at further seeds until its error reaches the target or 9
  repetitions (the default).
- Drift attribution (`quality.attribution`): `crit eval run --attribute-drift` scores
  quality again on an engine per adapter-declared `exact_variants` (one matrix group or arithmetic choice exact) and
  reports each variant's budget used and share of the full engine's per set; `status` prints the shares on the worst
  set.
- Each quality set and path reports `budget_spread` (`quality.position_spread`): its share of
  the drift budget over 200 block-bootstrap resamples of its positions (blocks of 64), as the 5th, 50th and 95th
  percentiles, and `status` prints the worst set's beside its share. Reported, not judged.
- `crit eval run --adapter A --only quality` scores the quality texts in process when the adapter's engine has
  `score()` (`quality.development`): judged as a served engine's, recorded as
  `quality.evidence` `development`, reported by status as development evidence, and never a gate result.
- The time to first token by part: speed samples keep the engine's own `prefill_ms` and `first_token_ms` (an
  adapter's `generate`, crit serve's streamed telemetry), `headroom.ttft` each case's medians with
  `outside_engine_ms`, the rest of the measured time (server, tokenizer, HTTP, scheduling), and `status` prints them
  per case, with a kernel trace's GPU and host split of a fresh first token under `--kernels`.
- `crit eval status --kernels kernels.json` joins a kernel trace to the prefill floor: per traced case, the GPU time of
  each floor term (matmul per stored weight format, attention, recurrent, other kernels, draft work, and the host's
  share of the first token) against its floor, and a prefill gap per term. `kernel_work` rules take a `term` and a
  `format`; kernels no rule classes are classed by name, and the report says how much. `crit eval kernels --work`
  puts rules ahead of the adapter's own.
- `crit-eval probe` measures the whole machine: device-to-host transfer rates beside host-to-device, host memory
  read, write and copy bandwidth (every hardware thread, and read on one) and CPU FP32 throughput (every hardware
  thread and one, AVX-512 or AVX2). A result's `device` carries them with the GPU's rates and the PCIe curves.
- A result records `crit_eval_version`, the `model` it measured (name, Hugging Face repository and revision when the
  path shows them, format, architectures, quantization) and its `reference`: the reference that computed the
  quality scores and the package they were fetched from.

### Changed

- The sampled speed suite repeats a case whose three seeds leave its relative standard error above 3% at further
  seeds, up to 9 repetitions (`sampled_speed.adaptive`), which moves the sampled decode score against crit-eval-v8.
  `crit eval run --no-adaptive-sampled` keeps the seeds alone (an override the result records), and a sampled decode
  compares only with one measured the same way.
- The prefill, TTFT and kernel-profile floors take each format's best probed rate, the larger of its cuBLASLt GEMM rate
  and its raw tensor-core rate, and `status` names the rate and its source beside the prefill ceiling
  (`headroom.rates`); a ceiling can rise.
- A result's KPIs (`status.kpis`) are named for what they measure: `decode_sampled` (the headline decode, at the
  model's own sampling) and `decode_greedy` instead of `decode` and `greedy_decode`, and `perplexity` and
  `kl_divergence` instead of one `quality`, each with its worst set against its limit. crit reads both namings.
- crit-eval moved out of [crit](https://github.com/crit-labs/crit) into this repository. Its protocol is now
  versioned as crit-eval is: crit-eval-v0.1.0 follows crit-eval-v8 with the changes above, and data prepared for v8
  is prepared again (reference scores carry over). The suite, protocol,
  probes and examples live inside the `crit_eval` package (installed with it), the `crit-eval` command runs what
  `crit eval` runs, and crit-eval depends on crit-core (0.10) alone. Earlier changes are in crit's CHANGELOG.

### Fixed

- `crit eval kernels` traces each case's prefill on a fresh prompt: a warmup on a prompt of the same length, then a
  request of one output token on the case's prompt with its first token changed to one no earlier request began
  with (no prompt cache serves any of it), then the whole request for its decode. The warmup on the same prompt let
  a prompt cache serve the traced request, and splitting one request at its first delivered token counted a
  speculative engine's first verify and draft kernels as prefill. Each case's prefill also reports its first-token
  time and the GPU's busy time in it.
- `crit eval kernels` takes its phase marks on CUPTI's clock (the tracer exports `crit_ktrace_timestamp`): the host
  clock drifted against it far enough under WSL to count a warmup's decode kernels as the next prefill's (short
  cases showed decode GEMVs instead of prompt GEMMs).

[Unreleased]: https://github.com/crit-labs/crit-eval/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/crit-labs/crit-eval/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crit-labs/crit-eval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/crit-labs/crit-eval/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/crit-labs/crit-eval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/crit-labs/crit-eval/compare/46c7757...v0.1.1
[0.1.0]: https://github.com/crit-labs/crit-eval/releases/tag/v0.1.0
