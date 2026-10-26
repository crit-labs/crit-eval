# Changelog

All notable changes to crit-eval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- Protocol crit-eval-v0.2.0, a new series: the sampled speed suite repeats a case whose three seeds leave its relative
  standard error above 3% at further seeds, up to 9 repetitions (`sampled_speed.adaptive`), so the sampled decode
  score moves. `crit eval run --no-adaptive-sampled` keeps the seeds alone (an override the result records), and a
  sampled decode compares only with one measured the same way. Data prepared under 0.1.x is prepared again; reference
  scores carry over.

### Added

- `crit-eval probe` measures the raw tensor-core rates beside cuBLASLt's GEMMs (`mma_tflops`: BF16 and FP16 into
  FP32, FP16 into FP16, FP8, and on sm_120a NVFP4 and FP4 times FP8), building the probe for the device's own
  architecture. On an RTX 5090 cuBLASLt's BF16 GEMM reaches 185 to 215 TFLOPS where the BF16 tensor cores run 251,
  and FP16 accumulation runs 499.
- Prefill and TTFT ceilings by prompt activation format (`headroom.activations`, protocol crit-eval-v0.1.2): the
  floors again with every quantized matrix group's prompt GEMMs at BF16, FP16 accumulation, FP8 and the stored
  format, and at the engine's own `prompt_activations` when its adapter declares them. The quality gate on the prompt
  paths (this result's and a same-model baseline's) shows them passing or rules them out, and `status` names the
  fastest shown passing as the reachable limit next to the stored-format one, with a prefill gap of its own.
- A result records the model arithmetic it was priced on (`ceiling`). `crit eval status --device` prices prefill and
  time to first token with a newer probe, `--data` with the arithmetic of the data an older result was measured on.
- crit serve's `engine_facts` may carry `prompt_activations`; crit-eval reads it.
- Free device memory at the required context (protocol crit-eval-v0.1.8, `context.device_memory`): after the
  engine loads and after the long-context curve, the least of them beside the capacity gate in `status`, and its
  change against a baseline.
- Draft and verify apart (protocol crit-eval-v0.1.7): `status --kernels` splits a speculative engine's rounds into
  verify pass and draft steps from the trace's kernels classed `draft`, when the engine reports no `verify_ms` and
  `draft_ms`, and gives their gaps apart. An adapter's `wider_draft` opens one more engine at a wider draft width for
  its acceptance, so the width projections go past the built width.
- Sampled decode noise (protocol crit-eval-v0.1.6, `sampled_speed.adaptive`): every sampled case reports its relative
  standard error, its repetitions and the repetitions a 3% target would take, and `status` prints the largest.
  `crit eval run --adaptive-sampled` repeats a case at further seeds until its error reaches the target or 9
  repetitions (the default from crit-eval-v0.2.0).
- Drift attribution (protocol crit-eval-v0.1.5, `quality.attribution`): `crit eval run --attribute-drift` scores
  quality again on an engine per adapter-declared `exact_variants` (one matrix group or arithmetic choice exact) and
  reports each variant's budget used and share of the full engine's per set; `status` prints the shares on the worst
  set.
- Each quality set and path reports `budget_spread` (protocol crit-eval-v0.1.4, `quality.position_spread`): its share of
  the drift budget over 200 block-bootstrap resamples of its positions (blocks of 64), as the 5th, 50th and 95th
  percentiles, and `status` prints the worst set's beside its share. Reported, not judged.
- `crit eval run --adapter A --only quality` scores the quality texts in process when the adapter's engine has
  `score()` (protocol crit-eval-v0.1.3, `quality.development`): judged as a served engine's, recorded as
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

### Changed

- Protocol crit-eval-v0.1.1: the prefill, TTFT and kernel-profile floors take each format's best probed rate, the
  larger of its cuBLASLt GEMM rate and its raw tensor-core rate, and `status` names the rate and its source beside
  the prefill ceiling (`headroom.rates`). Scores and gates are unchanged; a ceiling can rise.

## [0.1.0] - 2026-10-25

### Added

- `crit-eval probe` measures the whole machine: device-to-host transfer rates beside host-to-device, host memory
  read, write and copy bandwidth (every hardware thread, and read on one) and CPU FP32 throughput (every hardware
  thread and one, AVX-512 or AVX2). A result's `device` carries them with the GPU's rates and the PCIe curves.
- A result records `crit_eval_version`, the `model` it measured (name, Hugging Face repository and revision when the
  path shows them, format, architectures, quantization) and its `reference`: the reference that computed the
  quality scores and the package they were fetched from.

### Changed

- A result's KPIs (`status.kpis`) are named for what they measure: `decode_sampled` (the headline decode, at the
  model's own sampling) and `decode_greedy` instead of `decode` and `greedy_decode`, and `perplexity` and
  `kl_divergence` instead of one `quality`, each with its worst set against its limit. crit reads both namings.
- crit-eval moved out of [crit](https://github.com/crit-labs/crit) into this repository. Its protocol is now
  versioned as crit-eval is: crit-eval-v8 is renamed crit-eval-v0.1.0, otherwise as it was. The suite, protocol,
  probes and examples live inside the `crit_eval` package (installed with it), the `crit-eval` command runs what
  `crit eval` runs, and crit-eval depends on crit-core alone. Earlier changes are in crit's CHANGELOG.
