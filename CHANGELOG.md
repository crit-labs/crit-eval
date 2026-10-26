# Changelog

All notable changes to crit-eval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
