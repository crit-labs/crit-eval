# The eval-v1 adapter

[Overview](README.md) · [Measurements](MEASUREMENTS.md) · [Context](CONTEXT.md) · [Adapter](ADAPTER.md)

An engine joins with a Python file exporting `create_eval_engine(context)`;
[examples/mock_adapter.py](../crit_eval/examples/mock_adapter.py) is a complete example.

The adapter is loaded as a module of a package of its own folder: import the code beside it
relatively (`from .engine import Engine`), so `crit eval ab` can load two packages whose modules
share a name. The folder is also on `sys.path`, but Python keeps one module per top-level name, so
a second adapter whose absolute import names a module the first one's folder provided is refused.

`context` holds `model` (the `--model` directory), `config` (the `--config` JSON, given as text or a file),
`speculative` (false when the harness wants plain one-token decoding for the exactness check
from a second engine) and, with a [required context](CONTEXT.md), `context_tokens`. Size the KV
cache and every length-dependent table for it at load and report `max_context` of at least it. The returned
object provides:

| Method | Returns |
| --- | --- |
| `info()` | `name`, `configuration` (settings actually applied), `max_context`, `speculative`; optional `draft_width`, `bytes_per_pass`, `host_bytes_per_pass` (`{"link", "cpu"}` bytes a pass reads from host memory; without it an engine whose weights exceed the device's memory is priced by the model's [memory tiers](MEASUREMENTS.md#memory-tiers)), `draft_bytes_per_step`, `draft_ms`, `kv_bytes_per_token` (the device bytes one more token of context takes in the engine's caches at the precision it stores them, every cache counted, a draft layer's too) for headroom (the decode, long-context and first-token floors) and the memory fit (`context.fit`), both of which leave the KV term out and say it is missing without it; `prompt_cache` false when the engine keeps no prompt cache, so the follow-up turns are not sent (protocol `followup`; omit it when it has one); `prompt_activations`, the
activation format of each quantized matrix group's prompt GEMMs (`bf16`, `fp16_acc`, `fp8` or `stored`, one for
every quantized format, one per stored format such as `{"nvfp4": "fp16_acc", "fp8": "fp8"}`, or shares of a format's
FLOPs such as `{"nvfp4": {"fp16_acc": 0.93, "bf16": 0.07}}`), which with the quality gate names the reachable
prefill ceiling ([Measurements](MEASUREMENTS.md#limits)); `exact_variants`, configuration overrides that each switch
one matrix group or arithmetic choice back to exact (`{"nvfp4 prompt GEMMs exact": {"prompt_fp4": false}}`, or
`{name: {"config": overrides, "description": text}}`), for `crit eval run --attribute-drift`; `wider_draft`, `{"width": W, "config": overrides}` opening the engine with a
draft width W wider than its own, for the drafts accepted per round beyond it (the run opens it once more after its
own engine, [Measurements](MEASUREMENTS.md#limits)); `replay: true` when `replay()` exists; `plain: true` when `generate_plain()` exists; `native_libraries` (paths of loaded native libraries; crit-eval records sha256 and size, and stores each path relative to the checkout or `~`, or by file name inside another user's directory); `kernel_work` (for `crit eval kernels`: work estimates, and the prefill-floor term and weight format of each kernel, [Kernel profile](README.md#kernel-profile)) |
| `generate(prompt_ids, max_new_tokens, on_tokens)` | Greedy, stop tokens disabled, exactly `max_new_tokens` tokens. Call `on_tokens(list)` as soon as tokens reach the host. Return `{"tokens": [...], "rounds": int or None}` (target passes after prefill); optional `accepted` (drafts accepted per round), `verify_ms` and `draft_ms` (device time in verify passes and draft steps), `expert_reads` and `expert_hits` (the decode's routed expert reads and those its device expert cache served, integers: the hit rate the [memory tiers](MEASUREMENTS.md#memory-tiers) price), `prefill_ms` and `first_token_ms` (the engine's own time in the prefill and to its first token: the time to first token beyond them is the host's, [Measurements](MEASUREMENTS.md#limits)). An engine behind a chat API, which sees only text, returns `served` (`prompt_tokens` and `output_tokens` as its server counted them): its reply may end early, and `tokens` only identify the text |
| `replay(prompt_ids, continuation, on_tokens)` | Optional. The speculative loop on a fixed continuation: drafts accepted while they equal its next tokens, each correction or bonus token taken from it, exactly its tokens emitted. Same return fields |
| `generate_plain(prompt_ids, max_new_tokens, on_tokens)` | Optional, speculative engines. Plain greedy on the loaded engine: one token per target pass, no drafting. Same contract as `generate`. crit-eval then runs the exactness check on this engine instead of opening a second one with `speculative` false |
| `generate_sampled(prompt_ids, max_new_tokens, on_tokens, sampling, logprobs=None)` | Optional, from crit-eval-v7. Sample at `sampling`, which names every field (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`; `top_k` 0 is no filter), stop tokens disabled; same return fields as `generate`. An engine that applies only some fields lists them as the attribute `sampling_fields`: crit-eval then sends only those, and needs every field whose value is not neutral With `logprobs` n, also return `logprobs`: per token `{"token", "logprob", "top": [[token, logprob], ...n]}`, the model's own log-probabilities before temperature and filters. Without it the engine has no sampled decode ([Sampled speed](MEASUREMENTS.md#sampled-speed)) |
| `telemetry()` | Optional. The engine's own counters since it opened, `{"expert_cache": {"reads", "hits", ...}, ...}` (as Core's engine and crit serve's `/health` `engine_telemetry` report them). crit-eval reads it before and after each speed suite (`result.telemetry`); an engine whose experts the device cannot all hold and whose `generate()` results carry no `expert_reads` and `expert_hits` has its expert misses priced at that hit rate ([memory tiers](MEASUREMENTS.md#memory-tiers)). Return None, or leave it out, when there is nothing to report |
| `score(prompt_ids, continuation_ids, top_k, prompt_rows=True)` | Optional. The fixed quality texts scored as the engine computes them, for `crit eval run --only quality` without a server: `{"prompt": [...], "continuation": [...]}`, a row `{"logprob": the token's, "top": [[id, logprob], ...top_k]}` for every prompt token after the first from one prefill (none without `prompt_rows`) and for every continuation token from the decode forced to it. Development evidence ([Quality](MEASUREMENTS.md#quality)): reported, never a gate result |
| `close()` | Release the engine |

The adapter calls the engine's code; the harness computes every metric and records the
configuration, host, model and suite identities, every sample and the generated tokens. The quality
gate is judged through crit serve, on the engine's scores of fixed text from the paths it serves with
([Engines behind a server](README.md#engines-behind-a-server)). An adapter whose engine has `score()`
gets the same figures as development evidence (`quality.evidence` `development`): the sets are scored
and judged against the budget, status says so, and the gate stays unmeasured, since nothing shows the
scores come from the serving paths and there are no served replies to check them against.

One `crit eval run` opens engines one after another in the same process, never two at once: at the
lower context, at the required context and, unless the engine declares `plain`, a plain one for
exactness. Loading dominates a round for engines that stream weights from host memory, so an
adapter may keep host-side weights (for example pinned expert records) in the process and hand them
to the next engine it opens for the same model. Under `crit run ... eval` the served candidate's engines
are built in one resident host process for the same reason: the run's context is built again where
the lower context ran, and an adapter that keeps host-side weights in the process finds them there.
Build everything on the device fresh for each engine:
KV cache, expert or weight caches, workspaces and graphs. Device state carried over from an earlier
engine would bias the lower-context comparison.

Within one engine, nothing a request leaves behind may change a later result. Caches
may stay warm, but their contents must be complete and correct whenever they are read: crit-eval's
request-history gate compares outputs on the freshly opened engine with outputs after every
other part, long-context requests included.

A crit engine needs no adapter of its own: it is a plugin package (`plugins/` and the `configuration.json` whose
`plugins` selection names them, [crit's plugins](https://github.com/crit-labs/crit/blob/main/docs/PLUGINS.md)), and
`crit eval run --package DIR` (also `precheck`, `pin`, `ab --package/--package-b` and `kernels`) measures it through
Core's eval-v1 adapter, `crit_core/eval_adapter.py`, which opens the package through Core's loader with `--config`
over its `configuration.json`.

Core's adapter generates greedily with stops disabled, offers `replay`, `generate_sampled` with log-probabilities,
`score` (for `--only quality` in process) and, when the engine drafts, `generate_plain`, returns each generation's
`prefill_ms` and `first_token_ms`, and reports the engine's facts (`plugins`, `draft_width`, `bytes_per_pass`,
`prompt_activations`, `exact_variants`, `wider_draft` and the rest).

An engine behind an OpenAI-compatible server needs no adapter:
[`crit eval run --url`](README.md#engines-behind-a-server) measures it through the server, quality
included.

crit serve reaches a crit engine through Core's loader as well
([serving](https://github.com/crit-labs/crit/blob/main/docs/SERVING.md)): a package holds no adapter for either. An
adapter file of your own is for an engine outside crit.
