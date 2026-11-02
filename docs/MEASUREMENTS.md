# crit-eval measurements

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

## Speed

[suite/speed.json](../crit_eval/suite/speed.json) holds six greedy cases, each generating 256 tokens:

| Case | Prompt | Why |
| --- | --- | --- |
| `chat` | short prose question | everyday chat |
| `code` | write a Python module | code drafts most easily |
| `math` | multi-step word problems | reasoning |
| `tool-call` | coding agent with tool definitions and one file read, asked for edit calls | tool calls with code payloads |
| `long-summary` | 3,584-token document, summarize | long-context prefill and decode |
| `agent-long` | 14,336-token agent session: tools, task, a grep, eight file reads, edit request | agent turns over a long history |

Agent cases render [tools.json](../crit_eval/suite/agent/tools.json) and frozen tool results
([suite/agent](../crit_eval/suite/agent)) with the model's chat template. `agent-long` needs
14,592 tokens of context; an engine with less skips it and its decode score is incomplete.

Stop tokens are disabled; output past the model's end token is flagged `stop_crossed`.
Defaults: 1 warmup and 1 measured repetition, compared with each other. When they
disagree (their decode speed or time to first token spread by more than 5%, or their outputs differ) the case gets
a second repetition, and a third when those two disagree in turn ([protocol.json](../crit_eval/protocol.json)
`speed.extra_repetition`); the case's KPIs are the medians over its measured repetitions, and the repeatable gate
compares every output, the warmup's included. One repetition is enough on a quiet device: re-scored from their first
repetition alone, 23 Qwen3.8-27B results kept their scores within 0.1% (4% at worst). Until v0.4 every case ran
two repetitions. A KPI's run-to-run spread comes from the cases measured more than once (`headroom.noise`
`repeated_cases` of `cases`); with every case measured once it and the greedy standard error are unmeasured, and
`status` says so. `--repetitions N` asks for N repetitions, a third only when they disagree.

**Steady state** (`speed.steady_state`): a case is measured at the speed it settles to. Before its first measured
repetition, a request whose decode speed differs from the request's before it by more than 10% stands as one more
warmup, until two consecutive requests agree or the case's warmups are done; the extra repetition and the disturbed
check apply from the first measured repetition on. An engine that adapts to what it serves, a device cache filling
with the experts a case routes to for example, is measured once it has adapted instead of stopping the round as a
disturbed device. A case's warmups are done after four that decoded 1,024 tokens between them, and after 16 in any
case: an engine that adapts per decode window adapts by the tokens it
decodes, so a case with a short reply gets more requests to settle. One still varying then is judged as before.
Each case records `steady_state`: its `warmup_runs`, how many of them the drift `added`, whether its first measured
repetition agreed with the request before it (`steady`) and the `drift`, its decode speed over the case's first
request's, less one. The sampled suite does the same on the time per target pass, asking the first measured seed
again. An engine whose warmup agrees with its first repetition measures as before.

**Fresh decode** (`speed.fresh_decode`, reported, not judged): steady state measures a prompt
the engine has just served, and an engine that adapts its cache to a prompt or drafts from its earlier replies
decodes that faster than a new prompt. Each suite therefore records `fresh_decode_score`, the geometric mean of each
case's first request before its warmups, and `status` shows it beside the decode score when the two differ by more
than 10%.

**Output that varies** (`speed.output_varies`): an engine whose greedy reply changes between
repetitions (an expert cache that moves experts between CPU and GPU arithmetic which round differently) decodes different
texts, whose speeds differ by their content. Such a case is measured like any other: its outputs disagree, so the
extra repetitions run, and its speed is the median over them. The disturbed check compares repetitions of the same
reply only, and a case whose replies all differ is not judged disturbed. The case records `output_varies`, its
distinct replies and their decode spread, and the repeatable gate fails on it as before.

Time to first token gives prefill, the remaining tokens decode. The **decode score** is the geometric mean of the six
case medians; each case also reports tokens per round. The **prefill score** is the geometric mean of prompt tokens ÷ TTFT over `long-summary`
and `agent-long` (`prefill_cases`). The **TTFT score** is the geometric mean of TTFT in ms over
every case, where short prompts expose fixed costs.

**Stalls** (reported, not judged): a smooth decode average can hide a stream that pauses.
Every request records the host time each streamed chunk arrives (a served engine's stream events, an adapter's
`on_tokens` calls), and each speed case reports `stalls`: the longest and the p99 gap between chunks after the first
token, over its measured repetitions (the larger where it has several), in ms and over the case's median `round_ms`
(`max_gap_rounds`, `p99_gap_rounds`), so a gap of many rounds stands out from a slow but even decode. A speculative
engine sends several tokens per chunk, so these are chunk gaps, not token gaps. An engine that delivers its reply in
fewer than three chunks leaves the case's stalls unmeasured. `status` prints them per case.

**Tool calls** (reported, not judged): a fast agent turn is worth nothing if its calls do not
parse. On `tool-call` and `agent-long` crit-eval reads the calls in the case's first measured output up to the
model's first end token (stop tokens are disabled, so the output runs on past it), in the format the model's chat
template renders, which it reads from the case's own prompt: `<tool_call>` holding `{"name", "arguments"}` JSON
(Qwen3, Hermes) or a `<function=NAME>` block with one `<parameter=KEY>` block per argument (Qwen3.5 on, Qwen3-Coder;
Qwen3.8 renders this one), plus the calls a server parsed itself (streamed `tool_calls` deltas). A call is **valid**
when it parses and names a tool of [tools.json](../crit_eval/suite/agent/tools.json) with arguments it declares, of
their declared types, and every required one; **truncated** when the output-token limit cut it before it closed,
which is not counted as invalid; **invalid** otherwise, with the first reason in `status`. Each case records
`tool_calls` (`format`, `valid`, `invalid`, `truncated`, `calls`); a template in another format leaves it unmeasured.

## Sampled speed

Real requests sample, and an engine that speculates only on greedy requests decodes them one token
per pass. From crit-eval-v7 the six speed cases run a second time at the model's own sampling, the
`sampling` that `crit eval prepare` freezes from its `generation_config.json`: temperature, top_p,
top_k and min_p when it sets `do_sample` (temperature 1.0 when it names none), temperature 1.0 with
no filters when it does not sample (`sampling_source` `generation_config` or `protocol_default`).
Only the user may measure at other settings: `crit eval prepare --sampling
temperature=0.6,top_p=0.95,top_k=20` (temperature required; top_p, top_k and min_p optional)
freezes them instead, as `sampling_source` `override`; results record the source next to the
settings, `status` and `report` label an override, and decodes at different settings or sources
are not compared. Engine builders never choose the sampling: a speed won at a lower temperature is
not the speed the model's requests get. Repetition, presence and frequency penalties stay neutral,
so every engine samples the same distribution. Each repetition has its own fixed seed (warmup 0,
repetition 1; seeds 1 and 2 in v0.5, until v0.4 a third, seed 3), measured with no extra one: different seeds decode different texts,
whose tokens per round differ, so their speeds disagree by content, not noise. For the same reason a
disturbed device is judged on the time per target pass (`round_ms`) of the sampled repetitions, not
on their decode speed. Each decode score carries its relative standard error (`decode_se`: per
case, the spread of the repetitions' logarithms over the square root of their count, combined over
the cases), which for the sampled suite includes how much the drawn texts move it. Each sampled case reports its
own error, its repetitions and how many would reach the protocol's `sampled_speed.adaptive.target_se` (3%) at its
spread. One seed is enough for the score: re-scored from one seed alone, 23 Qwen3.8-27B
results kept their sampled decode score within 0.8%, while single cases moved by up to 23%. A case measured at one
seed has no standard error; its `decode_se` and the score's are null, and `status` says the error is unmeasured. The
warmup stays: it asks for the sampling check's log-probabilities (below), so the measured seed is a request without
them. Over 32 recorded results the warmup's time per target pass came 1.4% above its case's measured median at the
median, but 4 to 8% above on every case of some engines, more than one seed moves the score; judging the measured
seed's own reply would time those engines with the log-probabilities' cost. With `--adaptive-sampled` (the default
in crit-eval 0.4, off now) a case with one seed, or whose seeds leave its error above that target,
continues at the next seeds (2, 3, ...) until its error reaches the target or it has 9 repetitions, and its KPIs are
the medians over all of them (`sampled_speed.adaptive` records the rule; null without it). A fixed count
(`--repetitions`, which takes seeds past 1 in order) is never extended; an adaptive decode compares only with decodes
measured the same way.
`status` names a
baseline change within two standard errors of the difference as within the noise. The **sampled decode score**, the geometric mean of the case medians, is the decode KPI
from v7 on (`status.kpis.decode`, with `suite` "sampled"): its headroom and ceiling, the decode goal
and the stop record's decode follow it, and baselines compare it at the same sampling. The greedy
suite's decode score stays next to it (`status.kpis.greedy_decode`), and prefill, time to first
token, held-out prompts and exactness stay on the greedy suite. A result with no measured sampled
decode (v6, or an engine that cannot sample) keeps the decode KPI on the greedy suite.

Every greedy request names the penalties too, at their neutral values (presence and frequency 0,
repetition 1), where the engine applies them: a served default such as a model card's presence penalty
would otherwise change the output and send the request through another sampler.

Every sampled request names every sampling field, so no engine default changes the distribution
(an engine applies its own defaults, such as a model card's presence penalty, to the fields a
request leaves out): the prepared settings, the filters they leave unset off (`top_p` 1, `top_k` 0,
`min_p` 0), `repetition_penalty` 1, `presence_penalty` and `frequency_penalty` 0, and the seed. An
adapter's engine samples through `generate_sampled` ([Adapter](ADAPTER.md)); an engine behind a
server gets the fields in the `/chat/completions` body. An engine that lists the fields it applies
(crit serve's `supported_sampling_parameters`, an adapter's `sampling_fields`) gets only those, and
one that cannot apply a field the settings set, or cannot sample, has no sampled decode.

## Sampling check

A speculative engine must keep the distribution it samples from. Under greedy decoding the exactness
check compares tokens; under sampling there is no single right token, so crit-eval checks the
distribution. From crit-eval-v7.2 each speed case's sampled-suite warmup (seed 0), whose timing is
unused, returns each emitted token's log-probability and its top 20 alternatives, and crit-eval judges
that reply (`sampling_check.replies` "warmup"); a run without warmups, or an engine that cannot
serve the warmup with log-probabilities, gets one more sampled reply per case after the suite
(seed 7, "own request"). For every position crit-eval rebuilds the
processed distribution from those raw log-probabilities (temperature, then top_k, top_p and min_p),
with the unreported rest of the vocabulary as one last alternative: it holds the mass the reported
tokens leave, and each of its tokens is at most as likely as the last reported one. A position is
judged when the reported alternatives determine that distribution: top_k keeps only reported tokens
(Qwen's top_k 20), or the top_p or min_p edge falls among them whatever the rest holds, or the
rest's processed mass is known within 0.1% (exactly at temperature 1 without filters). The others
are reported as `unresolved`, typically flat positions under top_p with top_k above 20 (Gemma's
64) or none. Every judged token gets a randomized probability integral transform:
u = (mass of the more likely alternatives, the unreported rest last) + V × p(token), with V
uniform from a seeded generator. When every token was drawn from its processed distribution, u is
uniform on [0, 1], so over n tokens z = (mean u − ½) / √(1 / 12n) stays near 0. A sampler that
favours likely tokens, for example one that accepts drafts by greedy match, pulls z far below 0 (a
sampler taking the likeliest token half the time gives about −19 over 1,500 tokens). The
`sampling` gate fails when |z| exceeds 4 or a token falls outside its processed distribution at a
judged position, and is unmeasured with fewer than 256 judged tokens; the Kolmogorov-Smirnov
distance is reported. The log-probabilities must be the model's own, before temperature and
filters, the values the quality honesty check verifies for greedy replies; a server names each
token and alternative by its bytes, as for the honesty check.

Correct speculative sampling keeps the distribution exactly: accept a draft d with probability
min(1, p(d) / q(d)), where p is the target's processed distribution and q the drafter's, and on the
first rejection sample from the normalized residual max(p − q, 0); a greedy draft is q one-hot.

## Pinned continuation

Slightly different arithmetic yields different text and draft acceptance. `crit eval pin` records one engine's plain greedy output of every speed case (it must
repeat exactly) as the data directory's `continuations.json`, identified by the sha256 of its
token ids; a data directory keeps one. When it exists and the adapter declares `replay`, `run`
measures every speed KPI by replaying that text through the engine's speculative loop, so every
engine decodes the same tokens. Two free generations per case still feed the repeatability gate
and exactness. `speed.decode_text` records which text was decoded.

## Limits

**Device ceilings** come from `crit eval probe` on the target machine. It builds [probe/device_probe.cu](../crit_eval/probe/device_probe.cu) with
`nvcc` for the device's own architecture (`sm_120a` on an RTX 5090), runs it three times and keeps
medians: copy bandwidth (2 GiB device-to-device copies, read plus written), dense cuBLASLt GEMM rates
with FP32 accumulation at prefill-like shapes for BF16, FP8 E4M3 and NVFP4 (16-element E4M3 block
scales) as `tflops`, raw tensor-core rates as `mma_tflops` (`mma.sync` with operands in registers on
every SM: `bf16` and `fp16` into FP32, `fp16_acc` FP16 into FP16, `fp8` E4M3 into FP32 from sm_89, and
on sm_120a `nvfp4`, E2M1 with E4M3 block scales, and `fp4_fp8`, E2M1 times E4M3 through
`kind::f8f6f4`), plus total and least free device memory and host memory. cuBLASLt's GEMMs are not
every device's best kernels: on an RTX 5090 its BF16 GEMM reached 185 to 215 TFLOPS where the BF16
tensor cores run 251, and FP16 accumulation runs twice the FP32-accumulate rate (499). Every floor
therefore takes each format's best probed rate, the larger of the two (`headroom.rates` records the
rate and whether `cuBLASLt GEMM` or `raw MMA` achieved it, and `status` prints them beside the prefill
ceiling); a device file from before the raw rates has the GEMM rates alone. `host_to_device_gbs` records host-to-device rates per transfer size
(64 KiB to 256 MiB, keyed by bytes) for pinned copies, pageable copies and a kernel reading
mapped pinned memory, for engines that keep weights in host memory (offloaded layers,
host-resident experts) and size their transfers from it, and `device_to_host_gbs` the same curve back
into pinned and pageable memory. The host is measured too, for engines that keep weights or experts
in host memory or compute on the CPU: `host_memory_gbs` (read, write and copy over 1 GiB buffers on
every hardware thread, read on one, best of five passes) and `cpu_gflops` (FP32 fused multiply-adds
on every hardware thread and on one, in the widest vectors the build targets, `cpu_isa`: AVX-512 or
AVX2; the probe builds with `-march=native`, with `/arch:AVX2` on Windows), with `host_threads`. No
crit-eval score reads these yet; a result carries them in `device`. Pass it with `run --device`;
`--bandwidth GBS` overrides the bandwidth.

**Prefill ceiling.** `prepare` counts arithmetic per token from safetensors or GGUF headers and config
([crit_eval/limits/ceiling.py](../crit_eval/limits/ceiling.py)): 2 × rows × columns for every
projection in its stored format (BF16, FP8 or NVFP4; a GGUF's block types at the FP8 rate, the
rate int8 tensor cores share, and its MXFP4/NVFP4 at the NVFP4 rate), excluding gathered embeddings, the vision
tower, draft layers and the output head; 4 × heads × head dim per query-key pair in each
full-attention layer; and the chunked linear-attention recurrence. The floor is matmul FLOPs at
each format's best probed rate, attention at the faster of the BF16 and FP8 rates (FP8 Q·Kᵀ and P·V fit
the quality budget, so a BF16 attention floor can be beaten), and the recurrence at the BF16 rate.

**TTFT floor:** the larger of the prompt's compute floor and one weight read (`bytes_per_pass`,
else the language-model and output-head bytes) plus the prompt's KV writes at device bandwidth.

**Where the time to first token goes.** A short prompt's time to first token is mostly not its
prefill's arithmetic. Each speed sample keeps the engine's own `prefill_ms` and `first_token_ms` when
the engine reports them (an adapter's `generate`, crit serve's telemetry), and each case of
`headroom.ttft` the medians as `engine_prefill_ms` and `engine_first_token_ms` with
`outside_engine_ms`, the rest of the measured time to first token: a server's HTTP, templating,
tokenizer and queue, or an adapter's host work. With `--kernels`, `status` adds the trace's fresh
first token per case, split into the GPU's busy time and the host's rest ("ttft by part").

**Free device memory.** A run that builds the engine for the required context records the device memory
nvidia-smi sees free (the whole device, other processes included) after the engine loads and after the
long-context curve filled the context up to the requirement; `context.device_memory` keeps both and the
least of them, which `status` prints beside the capacity gate, with its change in GB against a baseline at
the same requirement (protocol `context.device_memory`; reported, not judged).

**Ceilings by prompt activation format.** The stored-format ceiling prices an NVFP4 matrix at the
NVFP4 rate, W4A4, and an FP8 one at the FP8 rate, W8A8: arithmetic the quality budget may not
allow (on Qwen3.8-27B, W4A4 prompt GEMMs used 252% of it). So `headroom.activations` prices the
prefill and TTFT floors again with every quantized matrix group's prompt activations at each of
`bf16` (BF16 tensor cores, FP32 accumulation), `fp16_acc` (FP16 accumulating in FP16), `fp8` (FP8;
FP4 weights times FP8 through `kind::f8f6f4`, or widened to FP8, whichever rate is faster) and `stored`
(the stored-format ceiling), each at its best probed rate, and at the engine's own `prompt_activations`
when its adapter declares them ([Adapter](ADAPTER.md)). The quality gate is the evidence: a result
whose engine declares its prompt activations and whose every prompt path (`SET/prompt`, the prefill's
own arithmetic) stays within the budget shows them passing, and one failing prompt path rules them out;
a baseline of the same model (`--baseline`) counts as well. `status` prints the four ceilings and TTFT
floors next to the stored one, which of them quality shows passing or rules out, the faster ones it has not
judged yet, and the **reachable limit**: the fastest shown passing, the engine's own declaration included, with a prefill gap of its own
("prefill at its reachable ceiling"). These are reported, not judged. Without a declaration the
reachable limit is unknown and `missing` asks for `prompt_activations`.

`crit eval status RESULT --device DEVICE.json` prices prefill and time to first token with a newer
probe's rates, and `--data DIR` with the model arithmetic of the data the result was measured on, for
results before crit-eval-v0.1.0, which do not record it (`ceiling` in the result).

**Decode headroom.** Every case reports `round_ms`, decode time per target pass. Optional
adapter fields add the ceiling:

- `draft_width`: **saturation**, tokens per round ÷ (draft width + 1).
- `bytes_per_pass` (weights read, not gathered embeddings, plus recurrent state) and the
  bandwidth: the **pass floor**, compared with the measured one-token step. Each case adds the
  KV read at its context: BF16 K and V of every full-attention layer, or `kv_bytes_per_token`.
  An engine that reports no `bytes_per_pass`, such as one behind another server, gets the stored
  model's: the weights read on every pass but draft layers, the routed experts' share
  (`num_experts_per_tok` of the experts) and the fixed recurrent and sliding-window state
  (`headroom.bytes_per_pass_source` says `engine` or `model files`). A drafting engine that reports no
  `draft_bytes_per_step` gets the checkpoint's own draft (MTP) layers, their routed experts' share and
  the output head their logits go through (`headroom.draft_bytes_per_step_source`); a drafter that is
  a model of its own has none to derive.
- `host_bytes_per_pass` (`{"link": bytes copied over the host link, "cpu": bytes the CPU computes from host
  memory}`, both part of `bytes_per_pass`): the pass floor prices those bytes at the probe's link and host memory
  rates (both together at host memory's read rate) instead of the device's bandwidth, so an engine that keeps
  weights in host memory is measured against a floor it can reach; `fixed_decode` keeps the all-on-device fraction
  beside it. An engine that declares none is priced by the model's [memory tiers](#memory-tiers) instead; only
  when Eval cannot place the model and its weights exceed the device's memory does the result list it under
  `headroom.missing` and warn that its pass floor assumes every weight on the device.
- `expert_reads` and `expert_hits` from `generate()` (or crit serve's telemetry): the decode's routed expert reads
  and those its device expert cache served, the hit rate the [memory tiers](#memory-tiers) price each case at.
- `accepted` per round from `generate()`: tokens per round at every width up to the measured one.
  With the draft cost (`draft_ms`, else round time beyond one pass ÷ width) crit-eval projects the
  **best draft width at current costs**; with `draft_bytes_per_step` also the best width at floor
  costs and the **speed-of-light decode score**, the geometric mean of tokens per round ÷ (pass
  floor + width × draft floor). Measure at a generous width.
- A better drafter raises tokens per round, so crit-eval also projects one whose later drafts are
  accepted as often as its first (each case's depth-1 rate, up to width 8) at floor costs. The
  **decode ceiling** is the larger projection.
- An engine that does not speculate decodes one token per model pass, so with `bytes_per_pass` and
  the bandwidth its **decode ceiling** is every pass at its floor: the geometric mean of 1 ÷ pass
  floor over the cases (`headroom.projection.basis` is `one token per pass`). An engine that drafts
  but reports no width keeps no ceiling.
- From v7 these describe the sampled suite. Rejection sampling lets a drafter speculate under
  sampling, so the depth-1 rate of the projection is the better of the sampled and greedy suites'
  for each case; a sampled case that drafts nothing projects from its measured tokens per round.
  An engine that speculates only on greedy requests thus keeps the ceiling its drafter allows,
  not the one-token-per-pass floor.
- The **fixed decode fraction** (`headroom.fixed_decode`, reported, not judged): every
  ceiling above starts from the engine's own `bytes_per_pass`, which an engine can under-report and so raise its
  floor. Beside them, each decode case's tok/s is divided by one token per pass at a denominator no engine
  moves: the probed bandwidth over the model files' bytes per pass (as `bytes_per_pass_source` `model files`
  computes them) plus the KV the case reads at its context, at the protocol's FP8 KV precision. The cases'
  geometric mean is the fraction `status` prints next to the decode score. It exceeds 1 when speculation decodes
  more than one token per pass, and compares engines of one model on one device whatever they report.

When `generate()` reports `verify_ms` and `draft_ms`, draft steps are compared with the drafter's
bandwidth floor and the verify pass with the ordinary pass; otherwise they form one term. A kernel
trace splits them for an engine that reports neither: `crit eval status RESULT --kernels kernels.json`
takes each case's decode GPU time in kernels classed `draft` over its draft steps and the rest over its
rounds (protocol `headroom.trace_parts`), and gives the two gaps apart.

An adapter that declares `wider_draft` (`{"width": W, "config": overrides}`) gets one more engine after
the run's own, opened at that width, which decodes each measured speed case once greedily for its drafts
accepted per round (`speed.wider_draft`); the width projections then go on past the built width, each
further draft adding what it added at the probed width (protocol `headroom.wider_draft`).

**Gaps** state what each term would add at its floor, all else as measured: model pass, draft
steps, verify pass at the ordinary pass cost, best draft width and drafter acceptance (on the
decode score), from v7 sampled requests as fast as greedy ones (each sampled case at its greedy
case's decode, where that is faster), prefill at its stored-format ceiling and at its reachable ceiling (the fastest
prompt activation format quality shows passing), and TTFT at its floor. With a kernel trace (`status --kernels`) each
prefill term gets a gap of its own (each weight format's matmul, attention, the recurrence at their floors; other
kernels, draft work and the host's share at zero), and a speculative round's verify pass and draft steps get theirs
apart where the engine reports neither. Each gain is a fraction of its own score. Results carry this under
`headroom`, the summary under `status`.

**Noise.** Every goal KPI carries its run-to-run spread (`headroom.noise`: the spread of the
repetitions' scores, max − min over the median): decode from its own suite, prefill over the prefill
cases, TTFT over every case. The long-context curve runs once per point, so its decode and first
token take the speed suite's spreads (the greedy decode's and the prefill's). A greedy case
is measured once unless its warmup disagrees, so the greedy spreads come from the cases measured more than once,
and `status` names how many of them (`repeated_cases` of `cases`) or that the spread is unmeasured. `status` prints each
goal's share of its limit and its noise, the noise beside every gap, and marks a baseline change no
larger than the larger of the two results' spreads as within the noise (decode, which has a standard
error, by two standard errors of the difference). crit-eval states the goals; ordering them is the run's.

### Several GPUs

An engine on several GPUs is measured with a device file of all of them: `crit eval probe
--devices all` (or `--devices 0,1`) measures each GPU's figures, as one GPU's file holds them, and the copy rate of
every ordered pair (`links`, peer access noted), keeping the first GPU's figures at the file's top
([devices.py](../crit_eval/limits/devices.py), protocol `context.devices`). The run prices the set as one device:

- the **max context** counts the members' memory together, each keeping the headroom and the reserve;
- a **pass floor** takes the faster split of the pass's bytes: **tensor**, each member reading its share sized to
  its bandwidth (the bytes over the bandwidths' sum) plus two all-reduces of a token's hidden activations per
  layer, of which a ring moves 2 (N - 1) / N over the slowest link; or **pipeline**, members holding consecutive
  layers that a token runs through in turn (N times the bytes over the bandwidths' sum) plus N - 1 hand-overs. The
  bandwidth that makes that the pass's time is the bandwidth every floor uses, and each format's GEMM and
  matrix-instruction rate is summed over the members that measured it;
- device memory is sampled on every member and summed.

The result records the members and the split under `device.device_set`. Collectives are priced at the copy rate
the probe measured and without latency, so the floor is a bound an engine on fast links approaches, not a schedule.

### Memory tiers

Every byte a pass or a prompt reads from host memory is priced at the tier it is read from, as
MoE-CAP's tiered cost model and sparse memory bandwidth utilization (S-MBU) price an offloading engine
([tiers.py](../crit_eval/limits/tiers.py), protocol `headroom.tiers`; reported, not judged). The tiers come from the
model files' placement on the device at the required context (the scenario `crit eval context` prints,
[CONTEXT.md](CONTEXT.md)), so a floor never rests on the engine's own report:

- The **resident weights the device cannot hold** next to the required context's KV cache (`dense_device_host`) are
  read from host memory on every pass.
- **Routed experts** count by the share a token activates (experts_per_token of the experts), those off the device
  at the device cache's miss rate: one less the engine's hit rate (`expert_hits` over `expert_reads`, per case where
  it reports them) or, without one, the share of the experts the device holds (even routing, the pessimistic case).
- A **decode pass** takes its host bytes on the faster of the host link and host memory's read rate; an engine that
  declares `host_bytes_per_pass` keeps its own path. Each case's floor uses its own hit rate (`expert_hit_rate`), and
  the long-context curve's points theirs.
- A **prompt** of n tokens moves the resident bytes off the device and each off-device expert it touches at least
  once (E(1 - (1 - k/E)^n) of a layer's E experts) at the run's miss rate, on the faster of the link and the CPU
  (host memory's read rate, and the tokens' arithmetic on those bytes at the probed CPU rate, which sends long prompts
  over the link). The prefill floor is the longer of its arithmetic and that transfer (`bound: host`), the first
  token's the longest of its arithmetic, the device-resident bytes at the device's bandwidth and that transfer, at
  every speed case and curve point.

Experts the placement leaves on disk are priced as host memory: the probe measures no disk, so their floor is a
lower bound. `headroom.tiers` records the placement's split and the run's hit rate, `headroom.host_pass` the pass's
host bytes and how they were priced, and `status` names both beside the decode's pass floor.

## Quality

Quality is what the engine loses on top of the checkpoint the user chose, nothing else, measured
on fixed text through the paths the engine serves with.
[suite/quality.json](../crit_eval/suite/quality.json) holds six conversations (prose, code, math,
German, JSON, story), each with a written answer. Each quality text is a conversation rendered by
the model's chat template with its generation prompt, followed by its answer cut to 256 tokens
(`quality.continuation_tokens`); `prepare` freezes both as token ids.

`crit eval reference` scores every position of every text once, teacher forced on the checkpoint,
and stores each position's log-probability and the reference's top 16 under the data.

**Quality modes.** A run judges quality in one of two modes, recorded in `requirements.quality_mode`.
`preserve`, the default, judges against the checkpoint's arithmetic with exact (FP32) activations:
what the engine loses on top of the weights the user chose. `as_quantized`, the mode for speed,
judges against the same scores with a budget eight times preserve's (protocol `quality.mode_thresholds`:
PPL ratio ≤ 1.4, mean KLD ≤ 0.4, p99 KLD ≤ 4.0). It is a floor that keeps out a broken engine (skipped
layers, dropped experts, a collapsed path), not a ration on precision. The checkpoint's own activations
(W4A4, W8A8, as serving stacks run them), a lower-precision KV cache, head or dense weights fit it
without separate evidence. In `as_quantized`, output differences that part at near-ties pass
([forks](#forks-in-quality-mode-as_quantized)), and the honesty check bounds the served and scored
paths. Everything else is judged as in `preserve`. 4-bit activation math turns rounding-level
differences into large KL swings: on Qwen3.8-27B NVFP4, prompt rows move by a mean KLD of 0.045-0.08
(p99 0.5-1.4) under rounding alone. The checkpoint run as quantized lands up to mean KLD 0.23 and p99 3.7
from the exact scores itself. Its share of the preserve budget on the worst set: Qwen3.8-27B NVFP4 183%
(long-32768 prompt p99), Nanbeige4.3-33B NVFP4 319% (long-32768 prompt p99), Qwen3.8-Flash-Next NVFP4 555%
(long-1024 prompt mean). The KV cache stays exact in the reference in both modes, so an engine's KV
quantization is drift in either one.

The budget is the cap that as_quantized's per-set allowance had from crit-eval-v0.3.0 to v0.4.x: four
times the checkpoint's own drift, never past these limits. `crit eval reference --allowance` measures that
drift on request when the checkpoint declares quantized input activations on any matrix or the output head (NVFP4
in groups of 16 with the stored global scale, FP8 per token or per tensor;
`crit_eval.limits.schemes.activations`). It scores the texts a second time with those activations
quantized as declared and keeps how far that lands from the exact scores, per set and path
(`allowance.json` beside the scores; reference packages carry it). That shows what the format itself
costs. Runs no longer read it. Fused experts (every expert's matrices stacked, gate and up joined
along the output rows) quantize each expert matrix's input with that matrix's own declared scale. A
model whose declared matrices the modeling code multiplies any other way than through `nn.Linear` or
such fused experts gets no measured drift (`unsupported`).

Until crit-eval-v0.3.0 `as_quantized` kept a second full set of scores (`reference-as-quantized/`, with its own
spread), judged against with twice preserve's budget: a second reference run and package per checkpoint.
`crit eval reference` derives the allowance from such a store when the data holds one, without running again, and
`crit eval fetch` removes it.

The mode also sets the limits prefill and the first tokens are measured against, since an engine that keeps the
activations exact cannot reach the stored format's rate (W4A4 runs several times faster than 16-bit activations).
In `preserve` the prefill ceiling and the time-to-first-token floors (headline and long-context) are priced at the
fastest 16-bit prompt activations (BF16 or FP16 accumulation, `headroom.mode_limits`), or at a faster arithmetic
quality shows passing (the reachable limit); in `as_quantized` at the stored format. Goals and `kpis` carry the
mode's limits (`stored_ceiling` and `stored_floor_ms` keep the stored format's), and status names the mode and, in
`preserve`, what `as_quantized` would allow, once. The engine
scores the same ids through crit serve's score call (`chat-score-v1`): every prompt position and the
first answer token from one prefill of the prompt, as it prefills any request, and every later
answer token from its decode, each step forced to the answer (a speculative engine accepts drafts
only while they equal it and takes every correction and bonus token from it). crit-eval compares each
position: the perplexity ratio (the engine's perplexity of the scored tokens over the reference's),
KL(reference ‖ engine) mean / p99 / max over the tokens in both top-16 lists, each reference token the
engine's list leaves out (its engine probability at most that of the engine's 16th token), and one bucket for
the rest (a lower bound of the full-vocabulary KL that two lists sharing no token cannot bring to 0), and top-1 agreement (the share of positions where
both rank the same token first). Each set is judged on both paths the engine computes it with, each
by the drift budget on its own: its prompt positions from the prefill (`SET/prompt`) and the positions
it serves (`SET/served`), the answer, its first token from the prefill's last position and the rest
from decode. No request returns the prompt positions' logits, but they are the prefill's arithmetic
that every reply's first token comes from, a single position among a set's served ones: an engine
whose prefill drifts far more than its decode (lower-precision prefill activations, for one) is
judged on it. The prompt positions are judged by KLD alone, their perplexity ratio reported (since
crit-eval-v0.2.1): they predict the user's own turn, which a chat model gives little probability
(Qwen3.8-27B puts nearly all of it on `<|im_end|>`, for a prompt perplexity of about 3,000 to 5,300
on the long texts), so the ratio is decided by the log-probabilities of tokens the model calls
near-impossible. Scoring the reference again in FP32 with its prompt cut into 4,096-token instead
of 8,192-token chunks moved single positions by up to 37 nats and the long-4096 prompt ratio to
0.68 (TF32: 0.77 to 1.40 across the sets), while KLD, weighted by the reference's own
probabilities, judges the same arithmetic. Both are also reported by position: absolute position in log2 buckets (1-511,
512-1023, 1024-2047, ...) for the prompt and offset in the answer (1-31, 32-63, 64-127, 128-255) for
the served positions, so a threshold where the engine switches kernels shows as a step (reported, not
judged).

**Long prompts.** A real request prefills a long prompt and then decodes, and decode reads what the
prefill left: prefetched experts and their scales, KV written by a different kernel, sparse-attention
indices. One conversation per length (`quality.long` in [protocol.json](../crit_eval/protocol.json):
`german-letter` at 1,024, `python-merge-intervals` at 4,096 and `explain-tides` at 32,768 tokens) also
goes with suite text ([long.json](../crit_eval/suite/long.json)) before its first user message, so it
renders to about that many tokens, past typical prefetch, prefill-kernel and sparse-attention
thresholds. The conversations are one set and each length another (`conversations`, `long-1024`,
`long-4096`, `long-32768`); a set whose text does not fit `max_context` is skipped. crit-eval-v7
scored three texts per length; they share their document, so the others scored the same prefill again
(99K of 117K positions were the 32K document three times).

**Fail fast.** Sets are scored in that order and judged as each completes. When a set's prompt or served
path drifts a mean KLD over ten times the budget's (`quality.fail_fast`), the remaining sets are not
scored, and the run skips the long-context curve, exactness and the repeated follow-up and history
requests (`stopped_early` in the result): nothing after it can make such a run pass, and an author with
a broken engine learns so in a third of the time.

**Honesty.** The scores must come from the paths the engine serves with. crit-eval also takes the
engine's greedy replies to the conversations over its chat API, with `logprobs` and 16 `top_logprobs`
(the server renders each prompt to the token count the model's chat template renders and names each
reply token by its bytes), and scores the same reply tokens through the score call, without the
prompt's rows, which the fixed texts already score: every token's log-probability and every
alternative in both lists must agree within `quality.honesty.tolerance` (1e-3 nats), and the lists
hold the same tokens but those within it of the 16th. The long serving path is checked on the
long-context curve's first point (32,768 tokens): its request asks for the log-probabilities too, and
its reply is scored once as the continuation of its prompt (the reply `curve@32768`), one long
prefill where a reply of its own cost two; a server that cannot stream log-probabilities keeps its
curve point and the check records why there is no long reply. `quality.honesty` records the largest
difference; a failure fails the quality gate. In `as_quantized` the two paths may round apart, as
the [forks](#forks-in-quality-mode-as_quantized) allow (`quality.honesty.modes`), so the check bounds
the difference instead of asking for the same values. Each log-probability must be within 2 nats,
and each reply's mean difference of its tokens' log-probabilities within 0.1 nats. A token in only
one list must be within 2 nats of that list's edge. The served path then stays that close to the
path quality judged.

`quality.timing` shows where the phase's time goes: for the fixed texts' score requests, the honesty
replies and the honesty rescores, the requests' seconds, the engine's own prefill and decode
seconds (`crit_telemetry`), the rest of each request's time (transfer and the server's own work),
the response bytes and the client's parse seconds, with each text's own row in `timing.texts`;
`timing.metrics_seconds` is the time the comparison with the reference took. crit-eval asks crit serve
for the scores in its packed format (int32 tokens and float32 log-probabilities, base64), a
fraction of the JSON rows' bytes, and reads JSON rows from a server that offers no packed format.
crit serve's native adapter prefills a score's prompt twice, once for the prompt rows and once
before the forced decode, and both count as the engine's prefill; the rest, about a second for
the whole set, is transfer.

**The reference's own spread.** Some models amplify rounding: a mixture of experts whose routing has
near-ties, or a hard top-k selection, flips a discrete choice under a 1e-6 difference, and a recurrent
state or attention carries the change through the rest of the text. Two equally valid FP32 reference
runs of Qwen3.8-Flash-Next's 32K text differ by a mean of 0.06 nats per position at 8K to 16K and
0.29 at 24K to 32K, so no engine can match such a reference within the budget: it cannot match itself.
`crit eval reference` therefore scores every quality text, of every length, once more with its FP32
matmuls taking TF32 inputs (10-bit mantissas, about 1e-3 relative: the size of perturbation a BF16 or FP8
engine's arithmetic introduces; FP32 rounding alone, another chunking at 1e-6, measured a spread a
hundred times smaller on Flash-Next and so would have under-estimated what any engine faces) and stores
how far that run lands from the stored scores, per set and path, by the same metrics
(`spread.json` beside the scores, protocol `quality.spread`; reference packages carry it). The spread is
measured on every text (until crit-eval-v0.2.0 only texts of at least 8,192 tokens had one; a spread stored under
that rule is not used until `crit eval reference` scores it again). A run judges such a set against the larger of
the budget and twice the reference's own spread (mean and p99 KLD; the PPL ratio's distance from 1), and
the result records the spread and the limits it raised (`reference_spread`, `limits`, `raised`);
`status` names the sets so judged. A dense model's two runs stay far inside the budget, so nothing
changes for it.

The second run costs as much as the first, and most of it goes to the 32K text. Since crit-eval-v0.2.2 the texts
shorter than 8,192 tokens run again first (protocol `quality.spread.probe`): when every set and path they hold stays
below a twentieth of the level at which its spread would raise a limit (the budget over the factor of 2; the PPL
ratio only on the served positions), the longer texts are not scored again and their sets are judged against the
budget alone. Skipping can keep a limit where it is but never raise one, so it cannot let an engine pass that a full
pass would fail. The margin of 20 covers how a dense model's spread grows from 4K to 32K tokens (12 times at most,
on Nanbeige4.3-33B, still 180 times below the level); a model that amplifies rounding shows it in the short texts
already and gets the full pass. `spread.json` then names the texts it skipped (`spread_skipped`).

**Drift attribution.** `crit eval run --adapter A --attribute-drift` (with `--only quality` during development)
scores the quality texts again after the run's own engine closes, on an engine opened with each of the adapter's
`exact_variants`: configuration overrides that each switch one matrix group or arithmetic choice back to exact (the
NVFP4 prompt GEMMs, FP16 accumulation, FP8 attention). `quality.attribution` holds, per variant and set, the budget
used with it exact and its share, the full engine's budget used minus the variant's; `status` prints each variant's
share on the worst set. Shares do not add up (the terms interact), and an engine behind a server cannot be opened
with other arithmetic, so it has no attribution. Reported, not judged (protocol `quality.attribution`).

**Position spread.** A set's share of the budget rests on its positions, and the KLD p99 on a few
knife-edge ones (on a Qwen3.8-27B engine the 4K text's KL moved by ±30% under any small numeric
change), so each set and path also reports `budget_spread` (protocol `quality.position_spread`): its
share over 200 block-bootstrap resamples of its positions, each the set's length in blocks of 64
consecutive positions drawn with replacement from a seeded generator, as the 5th, 50th and 95th
percentiles, and the KLD p99's 5th and 95th. `status` prints the worst set's 5th to 95th percentile
beside its share. It is reported, not judged: the gate judges the measured set.

The **drift budget** defaults to PPL ratio ≤ 1.05, mean KLD ≤ 0.05, p99 KLD ≤ 0.5 (`preserve`; eight times
that in `as_quantized`).
`--quality-budget '{"ppl_ratio_max": 1.02}'` sets any subset;
`{"ppl_ratio_max": 1.0, "kld_mean_max": 0, "kld_p99_max": 0}` asks for no drift. Each set
reports `budget_used`, its largest share of any limit. The budget is there to be spent on speed:
lower drift than needed is not a gain.

The **reference** is the checkpoint the data was prepared from. It runs the model's official
`transformers` code with FP32 activations, no TF32, and stored weights dequantized exactly
(unquantized, FP8 E4M3 with channel or tensor scales, NVFP4), streaming layers onto the GPU one at
a time, so a model larger than device memory still runs; it needs the GPU to itself, so stop the
engine first. The long prompts use the same arithmetic in blocks of 256 query rows, since a mask and score matrix over 32K tokens would not fit. Texts
whose leading 8,192-token chunks are the same (the long sets' texts share their document) run those chunks once:
the others resume from a copy of the first's cache after them, so their scores are the same bytes as when each
runs on its own (24,576 tokens of each 32K text; the 27B reference went from 9 to 6.4 minutes on an RTX 5090). A
fused experts parameter that would take more than half the device's free memory in FP32 is never decoded whole:
the experts' stored bytes stay on the device and each expert is decoded when the modeling code's expert loop
reads it, the same values (Qwen3.8-Flash-Next streamed this way: the same score bytes, 12 instead of 26 GB of
device memory, 23 instead of 16 minutes), so a large mixture-of-experts layer runs on a small device. Mamba-2
mixers take the dt limit config.json declares (`time_step_limit`, usually [0, inf]) as the
model's own kernels do; some transformers mixers clamp dt at `time_step_min` instead,
which makes long-memory heads forget (retrieval fails beyond about 64K tokens) and reads as
drift against an engine that follows the config.

**Reference packages.** The scores depend on the checkpoint, the suite and the reference version,
not on the machine (a run reads them under those three alone, and FP32 rounding across devices is
orders of magnitude below the drift budget). So a checkpoint is scored once and its prepared data
and scores (about 14 MB) are shared: `crit eval pack` adds a scored data folder to a package folder
(one subfolder per checkpoint and suite, listed with every file's sha256 in `packages.json`), and
`crit eval fetch` writes the package for a checkpoint into a data folder after checking every
file. The checkpoint is found by the identity `prepare` records (config, tokenizer, template, weight
file names and sizes), without hashing its weights. The package source is `eval_packages` in
`settings.toml` (a folder or an https base URL, such as a Hugging Face dataset's `resolve/main`),
else the checkout's own `reference-packages/` folder, kept out of git, else the published packages,
the Hugging Face dataset [crit-labs/crit-eval-references](https://huggingface.co/datasets/crit-labs/crit-eval-references);
`crit eval run` and `crit run eval` fetch a missing reference from it before they would score one, and score it
on the machine when the source has none for the checkpoint or cannot be reached. A package is per file: two
quantizations of a model, or the same format from two publishers, are different checkpoints.

## Exactness

Exactness covers the greedy speed suite; under sampling the [sampling check](#sampling-check) judges the
distribution instead. For a speculative engine, crit-eval compares each measured speed case's greedy output with the
same
engine's non-speculative output and reports whether they are identical, the first differing
token and the agreeing prefix. An engine declaring `plain` decodes the non-speculative output
itself before it closes; otherwise crit-eval opens a second engine with `speculative` false
(`exactness.engine` is `shared` or `separate`). A served engine is speculative when crit serve's
`engine_facts` say so, or when its telemetry reports more than one token per target pass, and decodes
plainly when its package declares `chat-plain-v1`; without it exactness is unmeasured, since a second
engine from the same server would be the same speculative one. By default (`--exactness reported`) this is
information, since verify and decode may sum in different orders and the drift budget judges the
output; `--exactness required` makes any difference fail. The drift budget compares distributions,
so it does not judge a lossy acceptance rule (accepting drafts that are not the target's argmax);
top-1 agreement shows one, and such an engine records the rule in its configuration and needs
criteria that allow it.

A difference is information, not evidence of rounding. A drop in plain-greedy equality against an
earlier candidate (status notes it when given `--baseline`) or any difference between identical
requests is a bug until a root cause is demonstrated ([AGENTS.md](https://github.com/crit-labs/crit/blob/main/AGENTS.md)): stale or
uninitialized state shows up exactly this way, and quality scores over a few prompts can miss it.

## Request history

Right after the engine built for the required context opens, before any other request, crit-eval
generates the speed cases with the shortest and the longest prompt (`agent-long`, 14,336 tokens),
and again after everything else that engine served: speed, held-out prompts, quality, the
long-context curve up to the requirement and retrieval. The tokens must be identical. A
deterministic engine's greedy output does not depend on earlier requests, so this gate holds in
every exactness mode and constrains no speed; `repeatable` cannot see state that changed before
the first measured repetition. `history.served_between` lists what ran in between.

## Follow-up turns and the prompt cache

A conversation's follow-up turns resend its history. Right after the request history's first
requests, crit-eval sends each turn of [followup.json](../crit_eval/suite/followup.json) in order: the first
its user message after a document making the prompt about `followup.prompt_tokens` (8,192)
tokens, each later one the conversation so far with the suite's fixed assistant replies and the
next user message, `output_tokens` (64) each. Each turn may be served from a prompt cache, so an
engine that keeps the earlier turns prefills only the rest. Every turn reports its time to first
token, the prompt tokens the engine reports as cached
(`usage.prompt_tokens_details.cached_tokens`) and the prefill rate of the tokens it computed;
`followup.ttft_ms` and `followup.prefill_tps` are the geometric means over the turns after the
first. After every other part the turns run again, and every turn must answer as it did first
(the **prompt_cache** gate): a cached path must give the tokens the fresh one gave. An engine
without a prompt cache reports its fresh prefill and passes. An engine that
declares it keeps no prompt cache (`prompt_cache: false` in an adapter's `info()` or crit serve's
`engine_facts`) is sent neither pass: every turn would be a fresh prefill the speed cases already
measure. `followup` records the part as not run (`no_prompt_cache`, with the reason), it costs the
run nothing, and the gate passes. crit-eval does not infer the declaration from a first pass with
no cached tokens: a cache of whole prompts hits only when the turns run again, which is what the
gate checks. The part is skipped when its last turn does not fit the engine's context.

## Forks in quality mode as_quantized

In `as_quantized`, an engine may compute the same token with arithmetic that depends on what it
served before. An expert may run on the CPU in one request and on the device in the next, a cache
may be in another state, or a prompt may come from a prompt cache. Each tier rounds the 4-bit
format its own way. Rounding can only flip a near-tie, so the `repeatable`, `request_history`
and `prompt_cache` gates allow a difference that starts at one. Before the engine closes,
crit-eval judges every difference those three gates record (each distinct reply of a case's
repetitions against the first, the request history's two outputs, each repeated follow-up turn
against its first answer). It scores the shared prefix plus the first reply's token once through
the engine's scoring path (chat-score-v1, `forks.top_k` 16 alternatives). The fork is a
**near-tie** when the other reply's token is among the alternatives within `forks.margin`
(2 nats) of the first one's. A fork where one token clearly wins fails as before: so do a token
outside the alternatives, a reply that ends where the other goes on, and an engine that cannot
score. That is how a stale cache entry or a race that reads another request's rows shows up.
Each case, history row and turn records its `fork` (or a case its `forks`): the position, both
tokens, their log-probabilities and the gap. `preserve` judges every difference as before.

## Profiles

A profile (`crit eval run --profile NAME`, protocol `profiles`) fixes what a run measures, so results
from different machines and engines compare; the result and its status record it, and `compare`
and `--baseline` do not compare results of different profiles. **hub**, for crit-hub submissions:
every engine is built for 200,000 tokens of context, whatever `--context` or `settings.toml` say;
the long-context curve measures its two ends (32,768 tokens and the requirement) and reports the
powers of two between them as `context.interpolated`, time to first token as a·n + c·n² through
the two ends and decode time per token linear in the context (on a Qwen3.8-27B engine the 64K and
128K points came within 1% and 6% of measured); retrieval is judged at 4,096 (a request of its
own), 32,768 and the requirement (the curve's own replies), not at 131,072; the engine loads once,
so the lower-context gate is unmeasured; the held-out prompts are not decoded, since their
generality gate needs a baseline a submission has none of; and the sampled suite runs two seeds,
1 and 2, where a full run has one (`decode_se` reports their spread). Every other part and gate runs as in a full
run.

## Run timing

`timing.parts` records the wall seconds of every part in the order it ran (`lower_context`, `load`,
`history`, `followup`, `speed`, `sampled_speed`, `sampling_check`, `heldout`, `quality`, `context`,
`retrieval`, `exactness`, `followup_again`, `history_again`), and `timing.total_seconds` the run from
its start to the result; the log's last lines print both. A part that did not run has no entry.

After the first measured speed case the log prints an **estimate** of every part and the total
(`estimate`, [crit_eval/run/estimate.py](../crit_eval/run/estimate.py)): every request's prompt and
output length is fixed by the prepared data and the protocol, so the requests the engine served
so far fit its prefill (t0 + a·n + c·n²) and decode (p + q·context per token) costs, and the plan
is priced with them. It leaves out third repetitions, prompt caches and replies that stop early;
on a Qwen3.8-27B engine it projected 720 s for a 606 s run, most of the excess in the long-context
curve, whose quadratic term the first requests (up to 14K tokens) fit only loosely.

## Held-out prompts and generality

[suite/heldout.json](../crit_eval/suite/heldout.json) holds ten other prompts: Chinese, Japanese,
Korean, Russian, Arabic, Hindi, German and Spanish prose, Rust code and SQL. No engine choice may
be tuned or selected on them ([AGENTS.md](https://github.com/crit-labs/crit/blob/main/AGENTS.md)). After the speed cases, each,
cut to the length the run draws for it ([prompt-length jitter](#prompt-length-jitter)), decodes
its own greedy output (64 tokens, once; until v0.4 256 tokens, twice: over 23 Qwen3.8-27B results the held-out
score from the first 64 tokens' acceptance came within 1.4% of the 256-token one); crit-eval reports per-prompt
decode tok/s and tokens per round and the **held-out decode score**, their geometric mean (`--only heldout` runs
just this).

The **generality gate** compares with a baseline (`--baseline`, normally the run's first
candidate): the held-out score may not fall more than 5%, and no prompt may decode more than 10%
slower. Each run measures at the lengths it drew, so the gate compares tok/s at two draws of the
same prompts, which a small change in prompt length moves little. For the most sensitive check also compare
with the parent (`crit eval status RESULT --baseline PARENT_RESULT`). Without a baseline, or with
data prepared before the held-out set, the gate is unmeasured.

## Prompt-length jitter

The held-out prompts and the long-context curve's points below the requirement are
measured at lengths each run draws ([jitter.py](../crit_eval/run/jitter.py), protocol `jitter`), so an
engine tuned to the suite's exact shapes (CUDA graphs captured at one token count, a tile or split choice that
fits one prompt length, a prefill chunk that divides it) shows it as a slower held-out score and curve, as
InferenceX draws each prompt at 80-100% of its nominal length. A run draws a seed; each held-out prompt and
each curve point gets a fraction of its nominal length, uniform in 80-100%, from the seed and its own name
alone, so neither the order of the parts nor which ones a stage measures changes a draw.

- A **held-out prompt** keeps the chat template around its user text and cuts the text, in tokens, so the
  whole prompt is that fraction of its prepared length (rounded up). Prepare freezes the template's frame and
  each prompt's text tokens for it (`heldout_frame`, `content_ids`); an engine behind a server is sent the cut
  text. Each prompt records `prompt_tokens` (drawn) and `nominal_tokens`.
- A **curve point** below the requirement fills that fraction of the point (prompt plus output, rounded up)
  with the same archive and request, recorded as `filled_context`; the floors are priced at its real prompt.
  The point at the requirement stays exact: the capacity gate fills it.

The speed cases keep their lengths, so their scores keep comparing run to run, and so do quality, the
retrieval check's own points (their facts sit at fractions of the filler already), the follow-up turns and
the concurrency part. The result records `jitter`: the seed, the fraction range and every drawn length.
`crit eval run --jitter-seed SEED` draws a recorded run's lengths again, and a resumed round keeps its
journal's seed. `status` reports the held-out prompts' share of their nominal lengths.

## Closed and open tracks

A result says how far it qualifies for the **closed track**, which fixes the choices that move
scores for reasons other than engine work (as MLPerf's closed division and SPEC's base do): quality mode
`preserve`; one configuration for every workload, loaded once for the run; and the checkpoint's own draft layers as
the only drafter where it has them, no drafter trained or picked elsewhere ([tracks.py](../crit_eval/run/tracks.py),
protocol `tracks`; reported, not judged). The **open track** is everything the protocol allows. `track.closed` holds
the checks Eval can make from the run, each `pass`, `fail` or `unmeasured`:

| Check | Passes when |
| --- | --- |
| `quality_mode` | The run was judged in `preserve` |
| `one_load` | Eval loaded the engine once for the run, and a server that reports its load count (crit serve's `/health` `engine_loads`) loaded no engine between the run's first request and its last |
| `own_drafter` | An engine that drafts reads per draft step at most 5% (`draft_tolerance`) more than the checkpoint's own draft layers and the head their logits go through, the model files' `draft_step_bytes`; a re-encoded head reads less. A checkpoint without draft layers has no drafter of its own to draft with |

`qualifies` is true when every check passes, false when one fails and null while one is unmeasured (a drafting
engine that declares no `draft_bytes_per_step`). The rest of the closed track, that the configuration does not change
with the case or the prompt length, is the submitter's declaration on its crit-hub card. `status` names the checks.

## Concurrency

A run measures how a served engine that runs several requests at once does it
([concurrency.py](../crit_eval/run/concurrency.py), protocol `concurrency`). It is reported, not judged, and it
leaves every single-stream figure and gate as they were: it runs on the engine built for the required context after
every other part, the request history's closing request included. It needs a server (`crit eval run --url`, or
`crit run ... eval` through crit serve) whose model card reports `max_concurrent_requests` above 1, as crit serve's
does for a package declaring `chat-concurrent-v1`
([Serving](https://github.com/crit-labs/crit/blob/main/docs/SERVING.md#concurrent-requests)); an adapter's engine
stays serial, and a server that reports none or 1 leaves the part unmeasured with the reason.

A **level** of N clients is one closed loop: the clients start together and each sends its next request when its
previous reply has ended, 4 requests per client. Client c's r-th request is prompt (c + r) mod the pool's size, so a
client never repeats a prompt and clients at the same step send different ones while there are enough. The pool is
the speed cases with prompts of at most 2,048 tokens, then the held-out prompts, each as its frozen chat messages;
every request is greedy and 128 tokens long, and the server prefills each one (a reply with cached prompt tokens
fails, as in the speed part). The levels are N = 1, 2, 4, 8, 16, 32 and 64, up to the server's capacity and while N x
(the longest pool prompt + 128) fits the required context, the KV the requests hold at once.

| Figure | Per level |
| --- | --- |
| `aggregate_tps` | Completion tokens of every request over the level's wall interval (first request sent to last reply read), by [validate_cohort](#served-benchmark-outcomes): a failed, missing or duplicated request, or fewer requests in flight at once than clients, disqualifies it (`qualified` false, the rate null, `errors`) |
| `peak_concurrency` | The requests in flight at once at the peak, as the clients see them |
| `ttft_ms_p50`, `ttft_ms_p95`, `decode_tps_p50` | Per request: time to its first reply text, and its tokens after the first over the time from first to last |
| `consistency` | Each reply to a speed-case prompt against that case's serial greedy reply from the speed part, as far as both go: the share that match and each differing reply's first different token. A batch-invariant engine matches every one ([batched steps](https://github.com/crit-labs/crit/blob/main/core/src/batch/README.md#batch-invariance)) |
| `ceiling`, `fraction_of_ceiling` | The aggregate against the batched ceiling below |

The **batched ceiling** of a level ([batched.py](../crit_eval/limits/batched.py)) is N tokens per step over the floor
of one decode step of N rows: the larger of its bytes (the weights read on every pass once, the routed experts the
rows touch, and each row's KV at the protocol's KV precision over the pool's mean context plus its fixed state, over
the probed bandwidth) and its arithmetic (each row's active matmul FLOPs, the output head and k/E of the experts, at
each stored format's best rate, plus its attention). A layer of E experts routing k per token has E(1 - (1 - k/E)^N)
of them touched by N rows, so a mixture of experts reads more of its weights the more rows a step holds. The bound
leaves speculation out, as the fixed decode denominator does.

A staged run measures a level while it fits the budget, each priced from the best aggregate measured so far (the
first has no price); one that would pass it is named in `concurrency.skipped` and the sweep stops there. A certified
run and a profile measure every level. Each level is one line of the round's [journal](README.md#journal-and-resume),
so a resumed round answers the levels it measured. `status` lists the levels with their figures and `compare` shows
each level's aggregate side by side.

## Full-vocabulary comparison

When a change trades arithmetic for speed (lower-precision activations, approximate kernels,
another accumulation order), measure the exact shift on full logit rows over a fixed corpus:

- **Host, FP64** ([metrics.py](../crit_eval/reports/metrics.py)): `log_softmax(logits, valid)`
  over the first `valid` outputs, `nll`, `kl_divergence` (KL(reference ‖ candidate) in nats,
  clamped at zero), a streaming `Summary` (perplexity, KL mean, nearest-rank quantiles, maximum)
  and `compare(reference_rows, candidate_rows, targets)`.
- **Device reference** ([logprob_compare.cu](../crit_eval/probe/logprob_compare.cu),
  [logprob_compare.h](../crit_eval/probe/logprob_compare.h)): `enqueue_row_compare(RowCompare, stream)`
  computes each row's log-softmax over a strided FP32 logits matrix and writes the target NLL,
  stores a reference, or writes the KL from a stored one. Fixed-order reductions make it bitwise
  repeatable; an out-of-range target yields NaN. Compile it beside the forward path or fuse the
  same arithmetic; [logprob.py](../crit_eval/quality/logprob.py) builds it with nvcc.

Per corpus window, run the reference path once with `store`, then every output-producing path
with `reference` on the same ids, accumulating on the host. Report the reference perplexity next
to the candidate's: the ratio measures the change, the KL tail rare moved positions. For chat
models score assistant-role tokens only.

## Served benchmark outcomes

[outcomes.py](../crit_eval/run/outcomes.py) counts only complete, valid responses.
`validate_completion` checks an OpenAI-style SSE stream: status, in-band errors, finish policy,
`[DONE]` framing and reported usage (request `stream_options.include_usage`). `validate_cohort`
disqualifies an aggregate with a missing, failed or duplicated request or too little
concurrency; its rate is completion tokens over the fixed wall interval. `telemetry_count` keeps
missing telemetry `None`.
