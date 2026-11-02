# Changelog

All notable changes to crit-eval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.5.2] - 2026-11-02

### Fixed

- A resumed round replays the round it resumes again: a request that raised (a served engine's load count read from a
  server without it) is a journal line, raised again on replay; a served engine the journal answers keeps the
  request deadline its requests would have given it, so the stages' first estimate reads the same rates; and the
  closed track's last load count is read only by a round that measured to its end. Before, a resumed round left the
  journal at its first request and measured everything live.

## [0.5.1] - 2026-11-02

### Fixed

- crit-eval installs beside any crit-core from 0.12.0 on (`crit-core>=0.12.0`, no upper bound). crit pins the crit-eval
  it ships with, so a crit-core release no longer needs a crit-eval release to widen the bound.

## [0.5.0] - 2026-11-02

### Overview

crit-eval 0.5.0 starts protocol series crit-eval-v0.5. The protocol changes since 0.4.1: a run measures fewer
repetitions (one greedy repetition, one sampled seed per case, held-out prompts at 64 tokens once) at steady state, and
reports stalls, a fixed-denominator decode fraction and tool-call validity; only a certified run opens the lower-context
engine, and an engine that declares no prompt cache is sent no follow-up turns. The held-out prompts and the
long-context curve's points below the requirement are measured at lengths each run draws. Quality mode `as_quantized` is
the mode for speed: its budget is eight times preserve's, runs no longer need the allowance, and it passes replies that
fork at a near-tie and served and scored paths that differ within a bound. Reported only: the concurrency part, memory
tiers in the floors, the closed track's checks and the pricing of several GPUs as one device. Results do not compare
with 0.4.x ones, and data prepared under 0.4.x is prepared again; the reference scores carry over.

### Added

- Forks in quality mode `as_quantized` (protocol `forks`, [measurements](docs/MEASUREMENTS.md#forks-in-quality-mode-as_quantized)):
  an output difference the `repeatable`, `request_history` or `prompt_cache` gate records passes when the two
  replies part at a near-tie. crit-eval scores the shared prefix once through the engine's scoring path, and the other
  reply's token must be among the top 16 within 2 nats of the first. An engine whose arithmetic depends on what it
  served before (experts on the CPU or the device, a cache's state) can then run 4-bit formats as it likes, while a
  fork one token clearly wins, a stale cache entry or a race, still fails. `preserve` is unchanged.
- In `as_quantized` the quality honesty check bounds the served and scored paths instead of asking for the same values
  (protocol `quality.honesty.modes`): each log-probability within 2 nats, and each reply's mean difference within 0.1
  nats. Paths that round apart where the engine's arithmetic depends on what it served before no longer fail quality.
- The kernel workbench's check accepts a kernel declared `"repeatable": false` in kernel.json (atomics, or a split the
  workload chooses, for `as_quantized` engines): its repeats are judged against the FP64 reference within the
  tolerance (`repeat_relative_error`) instead of bit for bit.
- Output that varies (protocol `speed.output_varies`): a speed case whose measured repetitions decoded different
  greedy replies is measured over them, with the disturbed check comparing repetitions of the same reply only, and
  records `output_varies` (the distinct replies and their decode spread); the repeatable gate fails on it as before.
  Before, such an engine stopped the round as a disturbed device and got no result.
- Fresh decode (protocol `speed.fresh_decode`, reported, not judged): each suite records `fresh_decode_score`, the
  geometric mean of every case's first request before its warmups, and `status` shows it beside the decode score
  when they differ by more than 10%. Steady state measures a prompt the engine has just served, which an engine with
  an adaptive expert cache, or one that drafts from its earlier replies, decodes faster than a new one.
- A decode above its ceiling says why in `status`: the ceiling priced the expert cache at even routing, or counted one
  token per pass, because the engine reported no `expert_reads`/`expert_hits` or `accepted`. Any OpenAI-compatible
  server can stream them in a `crit_telemetry` object, as crit serve does.
- Several GPUs (protocol `context.devices`, [measurements](docs/MEASUREMENTS.md#several-gpus)):
  `crit eval probe --devices all|N,M` measures every GPU and the copy rate between each pair (the CUDA probe's
  `--device`, `--count` and `--peer` modes). A run with that device file counts the members' memory together for the
  max context, prices its floors as one device by the faster split of the weights (tensor, with two all-reduces per
  layer over the slowest link, or pipeline, with its hand-overs) and the members' summed rates, and samples device
  memory on every member; the result records the members and the split (`device.device_set`). Checked on one GPU (the
  probe's device and peer modes, device 0 to itself) and with host tests; nothing ran on two GPUs. Before, Eval
  probed and priced GPU 0 alone.
- Closed and open tracks (protocol `tracks`, [measurements](docs/MEASUREMENTS.md#closed-and-open-tracks)):
  a result's `track.closed` checks what a run shows of the closed track, which fixes the choices that move scores for
  reasons other than engine work: the quality mode (preserve), one load for the run (a served engine's `/health`
  `engine_loads` unchanged from its first request to its last; served engines answer `loads()`), and a drafter that
  reads no more per step than the checkpoint's own draft layers (`draft_step_bytes`). It qualifies when every check
  passes; the submitter's card declares the rest. `status` names the checks. Before, nothing told a result that kept
  the checkpoint's own choices from one that tuned them.
- A result names a checkpoint in crit pull's engine store (`models/<org>--<repository>@<revision>`) by its Hugging
  Face repository and revision, as it does a hub cache's, so a result from `crit bench` says which checkpoint it ran.
- Memory tiers in the floors (protocol `headroom.tiers`,
  [measurements](docs/MEASUREMENTS.md#memory-tiers)): the weights the model files' placement leaves off the device
  are priced at host memory's tier for an engine that declares no `host_bytes_per_pass`, so its floor no longer rests
  on its own report: resident weights that do not fit next to the required context's KV cache on every pass
  (`crit eval context` now names them, `resident_off_device`), and routed experts by the share a token activates
  (S-MBU) at the device cache's miss rate, from the engine's `expert_hits` and `expert_reads` per case (adapter
  `generate()` results, crit serve telemetry) or even routing without them, on the faster of the link and host
  memory. Prompts move each off-device expert they touch once, on the faster of the link and the CPU (its arithmetic
  at the probed CPU rate): the prefill floor takes the longer of that and its arithmetic, the first-token floor the
  longest of those and the device's bytes, at every speed case and long-context point, whose rounds take their own
  hit rates too. `status` names the pass's host bytes and how they were priced. Before, only an engine that declared
  its host bytes got a host-priced decode floor, and no prompt or curve first-token floor counted the host at all.
- Prompt-length jitter (protocol `jitter`, [measurements](docs/MEASUREMENTS.md#prompt-length-jitter)):
  every run draws a seed, and from it each held-out prompt and each long-context curve point below the requirement
  a length of 80-100% of its nominal one, from the seed and its own name. A held-out prompt keeps the chat template
  and cuts its user text in tokens (prepare freezes the frame and each prompt's text tokens; a server is sent the
  cut text); a curve point fills its drawn share of the context (`filled_context`), with its floors priced there.
  The requirement's point stays exact for the capacity gate, and the speed suite, quality, retrieval-only points,
  follow-up turns and the concurrency part keep their lengths. The result records `jitter` (seed, range, every
  drawn length), `crit eval run --jitter-seed` draws a recorded run's lengths again, a resumed round keeps its
  journal's seed, and `status` reports the held-out prompts' share of their lengths. Before, every prompt had one
  fixed length, so an engine could tune CUDA graphs, tiles or prefill chunks to the suite's exact shapes and pass
  the generality gate with them.
- A served engine that runs several requests at once is measured at it (protocol `concurrency`,
  [measurements](docs/MEASUREMENTS.md#concurrency)): after every other part, closed-loop levels of 1, 2, 4, ... 64
  clients, up to the server's `max_concurrent_requests` (crit serve's model card) and the KV that N x (prompt +
  output) needs, each client sending its next greedy 128-token request when its previous one ended, over the short
  speed and the held-out prompts. Per level: completion tokens over the wall interval by `validate_cohort` (until now
  unused), the requests in flight at the peak, time to first token p50/p95 and per-request decode p50, the share of
  speed-case replies equal to the speed part's serial ones with each first difference, and the aggregate against a
  batched ceiling (`crit_eval.limits.batched`: one decode step of N rows reads the weights once, the experts its rows
  touch, E(1 - (1 - k/E)^N), and each row's KV). Staged runs price each level from the one before it; levels journal
  and resume; `status` and `compare` report them. An adapter's engine and a server that reports no capacity above 1
  stay unmeasured. Before, nothing in Eval measured an engine serving more than one request.
- Speed cases are measured at steady state (protocol `speed.steady_state`,
  [measurements](docs/MEASUREMENTS.md)):
  before a case's first measured repetition, a request whose decode speed differs from the request's before it by
  more than 10% stands as one more warmup, until two consecutive requests agree or the case has had four warmups;
  the extra repetition and the disturbed check apply from there on. The sampled suite does the same on the time per
  target pass and asks the first measured seed again. Each case records `steady_state` (its warmups, how many the
  drift added, whether it settled and its drift). Before, an engine that adapts to what it serves, a device expert
  cache filling with a case's experts, spread past `disturbed_spread` between repetitions and stopped the round, so
  an engine passed by freezing its adaptation. An engine whose warmup agrees with its first repetition measures as
  before; one still varying after four warmups is judged as before.
- An engine may declare that it keeps no prompt cache: `prompt_cache: false` in an adapter's `info()` or crit serve's
  `engine_facts`. Its run sends neither pass of the follow-up turns, which would be fresh prefills the speed cases
  already measure: `followup` is recorded as not run (`no_prompt_cache`, with the reason), the estimate and the stage
  plan leave both passes out, and the prompt_cache gate passes, as it does today for an engine without a cache. A first
  pass with no cached tokens declares nothing, since a cache of whole prompts hits only in the second pass.
- AMD GPUs (ROCm): `crit eval probe --backend hip` builds `probe/device_probe_hip.cpp` with `hipcc` and measures the
  same memory, bandwidth, transfer and host figures, with hipBLASLt GEMM rates for BF16 and, where the architecture
  has FP8 matrix instructions, FP8 (E4M3FNUZ on gfx94x, OCP E4M3 on gfx95x and gfx12xx); it measures no raw
  matrix-instruction rates yet. `crit eval kernels --backend hip` traces kernels with a rocprofiler-sdk tool library
  (`probe/kernel_trace_rocm.cpp`) in the CUPTI tracer's trace format. Both are written against ROCm 6.4 and not yet
  run on an AMD device. The probe picks hip by itself when it finds `hipcc` and no `nvcc`.
- Device files name their backend, architecture and runtime version (`backend`, `arch`, `runtime_version`; a CUDA
  probe derives `arch` from its compute capability, as `sm_120a`) and the rate keys the device has no matrix units
  for (`unsupported_formats`); results copy them when the device file has them.
- Eval rounds keep a journal and resume where they stopped ([Journal and resume](docs/README.md#journal-and-resume)):
  `crit eval run --out R.json` writes `R.journal.jsonl`, one line per engine opened, part started and request served,
  synced as each completes, and a round that stops (interrupted, crashed, a disturbed device, a hung engine, a
  failure) writes what it measured to `R.json` marked `aborted`. `--resume R.journal.jsonl` answers the journal's
  requests again with the clock readings the earlier round took, loads an engine only for the requests left and
  measures live from the first one the journal does not hold; it refuses a journal of another round's options. The
  result records the journal (`journal`), and `crit eval journal FILE` says how far a round got. Before, a round that
  stopped kept nothing but a disturbed or hung abort's partial result, and the next began from the start: a
  Qwen3.8-Flash-Next candidate's Eval interrupted after an hour, minutes before its end, kept nothing.
- Stalls (protocol `speed.stalls`, reported, not judged): every request records when each streamed chunk arrives,
  and each speed case reports the longest and p99 gap between chunks after the first token, in ms and in the case's
  median rounds (`speed.cases.*.stalls`, `status.kpis.stalls`). Speculative engines send several tokens per chunk,
  so the gaps are chunk gaps. An engine that delivers fewer than three chunks leaves them unmeasured.
- The quality gate's honesty failure names its usual cause when served and scored log-probabilities differ: a token's
  arithmetic depending on how many tokens its decode step holds (plain steps, other verify widths and the score
  call's replay running other GEMM shapes), with the fix (one step width, or column-count independent kernels) and
  the in-process check (a served reply against its replay). Status text only; results and the protocol are unchanged.
- An engine whose model weights exceed the device's memory and that declares no `host_bytes_per_pass` is named in
  `headroom.missing` and warned (`crit_eval.limits.headroom.undeclared_host_bytes`): its pass floor then prices
  every byte at the device's bandwidth, a floor it cannot reach, so its fractions of the ceiling read as out of
  reach rather than as room. Declared, the floor is priced by source and `fixed_decode` keeps the all-on-device
  fraction beside it. Reported only.
- Placement sizes the device expert cache with no fixed prompt workspace (`expert_cache_bytes`): where experts stay
  off the device, the cache also gets the reserve, since prompt (prefill) buffers are lent from cache slots while a
  prompt runs and returned for decode. Before, the reserve sat idle during decode and the cache, its hit rate and the
  floors below were sized without it. The plain-pass floors add misses split between the CPU and the host link
  (`plain_pass_floor_ms.split`), bounded by host memory's read rate: a link copy reads host memory as the CPU's
  arithmetic does, so the two paths share it rather than add up. The rendered placement says even routing is the
  pessimistic case and the cache's hit rate under real routing the main lever. Reported only; scenarios are
  unchanged.
- Placement says whether host memory holds every routed expert (`host_holds_every_expert`), not only those beyond
  the device: scenario 2's starting design, a device cache copying its slots from host memory, keeps all of them
  there. Where only the off-device share fits, the rendered placement says so and points to a static split.
  Reported only; the scenario is unchanged.
- A fixed-denominator decode fraction (protocol `headroom.fixed_decode`, reported, not judged): each decode case's
  tok/s over one token per pass at the probed bandwidth and the model files' bytes per pass plus the case's KV read,
  whatever `bytes_per_pass` the engine reports, and the cases' geometric mean (`headroom.fixed_decode`,
  `status.kpis.decode_*.fixed_fraction`). The decode headroom beside it keeps the engine's own bytes. Above 1 with
  speculation.
- Tool-call validity (protocol `speed.tool_calls`, reported, not judged): on `tool-call` and `agent-long`, the calls
  in the case's first output up to the model's first end token, in the chat template's format read from the
  prompt (`<tool_call>` with JSON, or with a `<function=...>` block of `<parameter=...>` blocks as Qwen3.8 renders),
  and the calls a server parsed itself, counted valid (they parse and name a tool of `suite/agent/tools.json` with
  declared arguments of their types and every required one), invalid, or truncated by the output limit
  (`speed.cases.*.tool_calls`, `status.kpis.tool_calls`). A served reply keeps the calls its server parsed
  (`samples[].tool_calls`).

### Changed

- `as_quantized` is the mode for speed: its quality budget is eight times preserve's (PPL ratio 1.4, mean KLD 0.4,
  p99 KLD 4.0, protocol `quality.mode_thresholds`), the cap its per-set allowance used to be raised to at most. It is
  a floor against broken engines, not a ration on precision, so the stored format's activations, lower-precision KV,
  heads and dense weights fit without per-group evidence. Runs no longer read the allowance and no longer need one:
  an `as_quantized` run of data without `allowance.json` is judged. `crit eval reference` measures the checkpoint's
  own drift only with `--allowance` (reported, not judged), which saves a second scoring pass of every quality text
  for quantized checkpoints.
- Steady state's warmups (protocol `speed.steady_state` `max_warmup_tokens`, `max_warmup_requests`): a case's warmups
  are done after four that decoded 1,024 tokens between them, and after 16 in any case, instead of after four. An
  engine that adapts per decode window adapts by the tokens it decodes, so a case with a short reply could still be
  speeding up after its fourth warmup and stop the round.
- The GGUF config rules live here, beside the reference's layouts for the same architectures: crit-core holds no
  model family's rules any more and reads them through its registry (`crit_core.registry`, kind `gguf_config`).
  `crit_eval.architectures.gguf_config` gives `llama`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe`, and
  `gguf_deepseek4` and `gguf_qwen4exp` give DeepSeek V4's and Qwen3.8-Flash-Next's next to their layouts;
  importing `crit_eval` registers them, imported only when crit-core first derives a lone GGUF's config, and the
  package names them in an entry point of group `crit_core.gguf_config`. Derived configs are unchanged. With a
  crit-core before the registry, which holds these rules itself, nothing is registered.
- DeepSeek V4's chat template lives here too, byte for byte the one crit-core shipped, so prepared data and model
  identities keep its sha256 (`crit_eval/architectures/templates/deepseek-v4.jinja`, DeepSeek's MIT notice in its
  header, with the encoder's golden tests in `tests/test_eval_deepseek4_template.py`):
  `crit_eval.architectures.CHAT_TEMPLATES` gives it to crit-core's registry (`chat_templates`, by `config.json`
  `model_type`) when `crit_eval` is imported, and the package names it in an entry point of group
  `crit_core.chat_templates`. A DeepSeek V4 checkpoint without a template renders with it as before (origin `crit`).
- The protocol: the sampled suite measures one seed per case (seed 1, was 1 and 2) after its warmup,
  staged stage 3 included (`stages.sampled_repetitions` 1). Re-scored from one seed alone, 23 Qwen3.8-27B results kept
  their sampled decode score within 0.8%, single cases moving by up to 23%. A case measured at one seed has no
  standard error (`decode_se` null), and `status` says the error is unmeasured. The warmup stays and still serves the
  sampling check's log-probabilities: over 32 recorded results its time per target pass came 1.4% above the measured
  seed's at the median, but 4 to 8% above on every case of some engines, so judging the measured seed's reply instead
  would time those engines with the log-probabilities' cost. `--repetitions N` takes seeds 1 to N, and
  `--adaptive-sampled` takes seed 2 for a case measured once before judging its error. The hub profile keeps two
  seeds (1 and 2).
- The lower-context gate runs on a certified run (`--certify`) only: stage 3 no longer opens the second engine at the
  lower context, and the gate is unmeasured, its detail pointing to `--certify`, as capacity is. A certified run opens
  it after the engine at the requirement closes, as stage 3 did; `--stage full` and profiles keep their behaviour.
- A stored format the device has no matrix units for is priced at the BF16 rate in the prefill floor, as engines
  there widen its weights for the GEMM (`crit_eval.limits.ceiling.best_rates`, source "BF16 rate (no matrix units for
  this format)"): FP8 before compute capability 8.9 and NVFP4 off 10.x and 12.x on CUDA, and what a HIP probe
  lists. Before, an NVFP4 checkpoint's prefill floor on an RTX 4090, or any FP8 floor on an RTX 3090, read as
  missing. A rate merely absent from an older probe still reads as missing.
- The protocol: a greedy speed case runs one warmup and one measured repetition (was two). When the two
  disagree (decode or time to first token spread over 5%, or different output) the case gets a second repetition,
  and a third when those two disagree in turn (`speed.extra_repetition`); the repeatable gate compares the warmup's
  output too. A staged run's stage 2 repeats its short cases once, its stage 1 request standing as their warmup.
  Re-scored from their first repetition alone, 23 Qwen3.8-27B results kept their scores within 0.1% (4% at worst).
  The run-to-run spreads come from the cases measured more than once (`headroom.noise.repeated_cases` of `cases`),
  and `status` says when a spread or the greedy standard error is unmeasured because every case was measured once.
  Data prepared under 0.4.x is prepared again; the reference scores carry over.
- The sampled speed suite runs two seeds per case (1 and 2, was 1 to 3), as the hub profile does, and repeats a
  case at further seeds only with `crit eval run --adaptive-sampled` (the default was adaptive;
  `--no-adaptive-sampled` is now the default and still accepted). Re-scored from one seed alone, 23 Qwen3.8-27B
  results kept their sampled decode score within 0.8%, single cases moving by up to 23%; `decode_se` still reports
  the seeds' spread. A repetition past the protocol's seeds (`--repetitions 3`) takes the next seed instead of
  repeating the last one.
- The held-out prompts decode 64 tokens once (were 256 tokens, twice): over 23 Qwen3.8-27B results the held-out score
  from the first 64 tokens' acceptance came within 1.4% of the 256-token score.

### Fixed

- Quality's KL no longer collapses when an engine is far off. Each reference token the engine's top-k leaves out
  was lumped with the rest of both distributions, so two lists that share no token read a KL of 0: a broken
  Qwen3.8-Flash-Next engine that ranked the reference's first token first at 0.9% of answer positions read a mean
  KLD of 0.10, and prompt positions, judged by KLD alone, had no other check to catch it. A token the engine's full
  list leaves out has at most the probability of its last entry, and the KL is now bounded with that cap; it stays a
  lower bound of the full-vocabulary KL. On Qwen3.8-27B NVFP4 run as quantized against the exact reference, the
  answers read the same and the long prompts 17-61% higher (long-4096 mean 0.056 to 0.090, p99 0.90 to 1.65). A
  spread or allowance scored before this keeps the old bound until `crit eval reference` scores it again.
- `reference` scores ModelOpt MIXED_PRECISION checkpoints (nvidia/Qwen3.8-Flash-Next-FP4-Local). Its MXFP8 matrices
  (one E8M0 scale per 32 inputs of each row) were expanded as 128 x 128 blocks, a 53 GB tensor per projection that
  failed under WSL with "Failed to create GPU mapping"; the block now comes from the scales' shape (crit_core's
  `fp8_block`). Routed experts whose NVFP4 scales are CSF-compressed decode losslessly to their E4M3 bytes
  (`csf_scales`). A table stored in shards that each keep their own row scales, beside one global scale, gathers each
  part's scales with its rows (the PLE n-gram table).
- `reference` sets expandable allocator segments on its own process's allocator instead of exporting
  `PYTORCH_ALLOC_CONF`, which every process it started inherited: `crit run eval` constructs the reference and then
  starts `crit serve`, whose engine ran under expandable segments it never asked for: a different device memory
  layout from the same engine served or run on its own, which hid a Qwen3.8-Flash-Next engine's graph-pool bug from
  its in-process tests (1.1 tokens per round served, 2.6 in process).

## [0.4.1] - 2026-10-30

### Fixed

- crit-eval installs beside crit-core 0.13 (`crit-core>=0.12.0,<0.14`). crit 0.13.0 ships crit-core 0.13.0, and 0.4.0's
  `<0.13` bound left the two with no common install; crit-eval uses nothing of crit-core that changed between them.

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

[Unreleased]: https://github.com/crit-labs/crit-eval/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/crit-labs/crit-eval/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/crit-labs/crit-eval/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/crit-labs/crit-eval/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/crit-labs/crit-eval/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/crit-labs/crit-eval/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crit-labs/crit-eval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/crit-labs/crit-eval/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/crit-labs/crit-eval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/crit-labs/crit-eval/compare/46c7757...v0.1.1
[0.1.0]: https://github.com/crit-labs/crit-eval/releases/tag/v0.1.0
