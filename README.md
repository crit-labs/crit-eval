<div align="center">

<p align="center">
  <img src="docs/assets/wordmark-light.png#gh-light-mode-only" width="200" alt="crit" />
  <img src="docs/assets/wordmark.png#gh-dark-mode-only" width="200" alt="crit" />
  <br />
</p>

### Speed and quality measurement tool to compare inference engines honestly.

[![Version 0.3.0](docs/assets/badge-version.svg)](CHANGELOG.md)
[![License Apache-2.0](docs/assets/badge-license.svg)](LICENSE)
[![Python 3.11+](docs/assets/badge-python.svg)](pyproject.toml)

[Summary](#summary) • [Quick start](#quick-start) • [What `crit-eval` solves](#what-crit-eval-solves) •
[What `crit-eval` measures](#what-eval-measures) • [Commands](#commands) • 
[Limitations](#limitations) • [Documentation](#documentation)

</div>

---

<a id="summary"></a>

## Summary

**`crit-eval` runs one fixed suite against every inference engine, so results compare.**
> [!IMPORTANT]
> The same prompts, output lengths, stop handling and quality references, through the same
> OpenAI-compatible API a client uses. Every score is reported against its physical limit on your
> GPU, and quality is measured as the drift the engine itself adds to the checkpoint you chose.
> Results from different runs, agents and machines therefore compare, and engines submitted to
> [crit-hub](https://github.com/crit-labs/crit-hub) are measured with its hub profile.

---

<a id="quick-start"></a>

## Quick start

```sh
python -m pip install "crit-core @ git+https://github.com/crit-labs/crit#subdirectory=core/python"
python -m pip install "crit-eval[reference] @ git+https://github.com/crit-labs/crit-eval"
```

Serve your engine behind an OpenAI-compatible API, then measure it:

```sh
crit-eval probe --out device.json
crit-eval fetch --model MODEL_DIR --data EVAL_DATA_DIR --packages PACKAGES
crit-eval run --url http://127.0.0.1:8000/v1 --model MODEL_DIR --data EVAL_DATA_DIR --device device.json \
  --profile hub --out result.json
crit-eval status result.json
```

`fetch` gets the checkpoint's [reference package](docs/MEASUREMENTS.md#quality) from `PACKAGES` (a folder or an
https base URL) when it has one; without it,
`crit-eval reference` scores the reference once on your GPU (the `reference` extra installs torch and
transformers for that). Inside crit the same commands are `crit eval ...`. An engine without a server
plugs in through a Python [adapter](docs/ADAPTER.md) instead of `--url`.

---

<a id="what-crit-eval-solves"></a>

## What `crit-eval` solves

**There is no shared way to measure a bespoke inference engine.**
> Different bespoke inference engines report vastly different types of numbers. An honest comparison requires
> downloading and running the engines yourself. `crit-eval` finally makes apples-to-apples comparison possible.

**Physics-based evaluation lets you see what performance is left on the table.**
> `crit-eval` measures your GPU's memory bandwidth and matrix-multiply rates and turns them into a limit for each
> score. Its evaluation report can tell your coding agent how much performance can still be squeezed out of your
> inference engine.

**Speed that can't hide a quality loss.**
> Many inference engines push decode and prefill speed with no regard for quality. `crit-eval` treats quality
> as a first-class KPI. It measures not only inference speed but also quality regressions.

**A hardware report rounds out the evaluation.**
> The same model on the same inference engine but on two different machines can also lead to different inference
> speeds, even if the GPU is the same. We make this transparent by also reporting the runner's hardware configuration.

---

<a id="what-eval-measures"></a>

## What `crit-eval` measures

| KPI or record | In the result | What it records |
| --- | --- | --- |
| **decode_greedy** | `status.kpis.decode_greedy` | Greedy decode tokens per second over six workloads (chat, code, math, tool calls, a long document, a 14K-token agent session), with run-to-run noise |
| **decode_sampled** | `status.kpis.decode_sampled` | Decode tokens per second at the model's own sampling, the headline decode, and tokens per round for speculative engines; a case whose three seeds disagree is repeated at further seeds (up to 9) until its standard error is within 3%, and each case's error is reported |
| **prefill** | `status.kpis.prefill` | Prompt tokens per second on the two long prompts |
| **ttft** | `status.kpis.ttft` | Milliseconds to the first token over every workload |
| **long-context decode and ttft** | `status.kpis.context` | The same scores along a curve of prompts up to the required context (262,144 tokens by default) |
| **perplexity** | `status.kpis.perplexity` | The engine's perplexity over the reference's: the worst quality set (prompt and served tokens, each long length) against its limit; per set in `quality.sets` |
| **kl_divergence** | `status.kpis.kl_divergence` | Mean and p99 KL divergence of the engine's token distributions from the reference's: the worst set against its limits; per set in `quality.sets` |
| limits and gaps | `headroom` | Each KPI's physical limit on this machine, from the probe's best rate per format (cuBLASLt GEMM or raw tensor cores), and what each term (model pass, draft steps, verify pass, draft width, drafter acceptance, prefill, first token) would give at its floor |
| prefill by activation format | `headroom.activations` | The prefill and TTFT ceilings with the quantized matrices' prompt activations at BF16, FP16 accumulation, FP8 and the stored format, which of them the quality evidence rules out, and the reachable limit: the fastest shown to pass |
| prefill by term | `crit-eval status --kernels` | With a kernel trace, each prefill term (matmul per weight format, attention, recurrence, other kernels, host) against its floor, a gap per term, and a speculative round split into verify pass and draft steps |
| time to first token by part | `headroom.ttft` | Each case's time to first token split into the engine's own prefill and first token and the rest (server, tokenizer, HTTP) |
| quality noise and attribution | `quality.sets.*.budget_spread`, `quality.attribution` | Each set's share of the drift budget with its spread over its positions; with `--attribute-drift`, each matrix group's share of the budget |
| free device memory | `context.device_memory` | Device memory left free with the engine built for the required context, beside the capacity gate |
| gates | `status.gates` | Repeatable output, request history, prompt cache, retrieval, exactness, sampling and quality, each passed or not, with every KPI against its limit (`status.kpis`) |
| retrieval | `context.retrieval` | Facts at the start, middle and end of long prompts, found or not |
| exactness | `exactness` | Whether speculative output equals plain greedy, case by case |
| sampling | `sampling_check` | Whether sampled tokens follow the model's distribution |
| request history | `history` | Whether the same greedy requests give the same tokens after a long request history |
| prompt cache | `followup` | Follow-up turns answered from a prompt cache: the same answer, and their prefill and ttft |
| held-out prompts | `heldout` | Speed on ten prompts in other languages and styles, never used for tuning |
| hardware | `host`, `device` | OS, CPU, RAM and GPUs (name, VRAM, driver, PCIe link), and the measured GPU bandwidth and matmul rates, PCIe transfer rates, host memory bandwidth and CPU throughput |
| engine | `engine` | What the engine reported: name, configuration, max context, drafting, bytes read per pass |
| model | `model`, `model_sha256` | Name, repository and revision, format, architecture and quantization |
| reference | `reference` | The reference that computed the quality scores (version, transformers, torch, GPU) and the package they came from |
| run | `crit_eval_version`, `protocol`, `profile`, `requirements`, `timing` | Which `crit-eval`, protocol and profile measured it, the drift budget it was held to, and how long each part took |

---

<a id="commands"></a>

## Commands

| Command | Does |
| --- | --- |
| `probe` | Measure the device's bandwidth, cuBLASLt GEMM rates per weight format and raw tensor-core rates per input and accumulator format (needs nvcc) |
| `context` | The max context for a model on a device: the model's positions and the device's memory |
| `prepare` | Freeze the suite as token ids for one model |
| `reference` | Score the prepared quality texts once on the checkpoint |
| `fetch` / `pack` | Get a checkpoint's reference package, or add one to a package folder |
| `pin` | Record an engine's plain greedy output of every speed case as the data's decode text |
| `run` | Evaluate one engine, through `--url` or an `--adapter`; with an adapter, `--only quality` checks quality in process (development evidence) and `--attribute-drift` attributes the drift to matrix groups |
| `status` / `compare` | Gates, goals, KPIs against their limits and gaps (`--kernels` joins a kernel trace to the prefill floor, `--device` and `--data` price a result again); results side by side |
| `ab` | Alternate two engines' speed runs in one session and report paired ratios |
| `kernels` | Trace every kernel of the speed cases (CUPTI, Linux), prefill on a fresh prompt, and rank them by GPU time per phase and prefill-floor term |

---

<a id="limitations"></a>

## Limitations

| Area | Current status |
| --- | --- |
| Quality reference | Quality is measured against the checkpoint as stored, for example an NVFP4 quantization, not against the full-precision model. `crit-eval` reports the drift the engine adds, never the quantization loss of the checkpoint itself, so quality results of two different quantizations of a model do not compare. |
| What quality covers | Drift on fixed texts: six conversations (prose, code, math, tool calls and more) and one long text each at 1K, 4K and 32K tokens, as perplexity ratio and KL divergence. It is not a task benchmark: no accuracy on MMLU-style questions, coding tests or agent tasks. |
| Reference coverage | The reference runs the model through transformers' modeling code in FP32, so it needs an architecture transformers implements and a CUDA GPU with the machine to itself for 5 to 25 minutes. Checkpoints in exl2 or exl3 are refused. Reference packages for three checkpoints are published on Hugging Face ([crit-labs/crit-eval-references](https://huggingface.co/datasets/crit-labs/crit-eval-references)) and fetched by default. |
| Rounding-sensitive models | Where a model amplifies rounding (MoE routing near-ties, sparse-attention ties, recurrent state), two valid FP32 runs of the reference disagree; quality is then judged against twice the reference's own spread instead of the budget, which is looser for those models. |
| Speed | One request at a time: `crit-eval` measures single-stream decode, prefill and time to first token, not throughput under concurrent requests or batching. A noisy sampled decode is repeated at up to nine seeds per case, so an engine whose speed swings with the drawn text takes longer to measure (`--no-adaptive-sampled` keeps three, an override the result records). |
| Hardware | NVIDIA GPUs only: the probe builds with nvcc and measures one GPU. Multi-GPU engines are not measured as such. PCIe, host memory and CPU rates are recorded; an engine that declares the bytes it reads from host memory per pass (`host_bytes_per_pass`) gets them priced at those rates, every other engine gets GPU-only limits. On Windows the CPU probe uses AVX2, which under-reports an AVX-512 CPU. |
| Long context | The hub profile measures the long-context curve at its two ends and interpolates the points between; only the full profile measures every point. |
| Comparability | Results compare only under the same protocol series and profile, on prepared data of the same checkpoint. |

---

<a id="documentation"></a>

## Documentation

| To… | Read |
| --- | --- |
| Run `crit-eval` and read its results | [Guide](docs/README.md) |
| See every case, score, limit and gate | [Measurements](docs/MEASUREMENTS.md) |
| Set the required context | [Context](docs/CONTEXT.md) |
| Plug in an engine without a server | [Adapter](docs/ADAPTER.md) |
| Build an engine to measure | [crit](https://github.com/crit-labs/crit) · [Building engines](https://github.com/crit-labs/crit/blob/main/docs/BUILDING_ENGINES.md) |
| See release history | [Changelog](CHANGELOG.md) |

