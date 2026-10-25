# crit Eval

One speed, quality and exactness suite that every inference engine runs the same way, so results from different
engines, agents and machines compare. Part of [crit](https://github.com/crit-labs/crit); the guide is in
[docs](docs/README.md): [Measurements](docs/MEASUREMENTS.md) (cases, scores, limits, quality),
[Context](docs/CONTEXT.md) and the [Adapter](docs/ADAPTER.md) an engine provides.

## Install

```sh
python -m pip install "crit-core @ git+https://github.com/crit-labs/crit#subdirectory=core/python"
python -m pip install "crit-eval[reference] @ git+https://github.com/crit-labs/crit-eval"
crit-eval --help
```

`crit-eval` is what `crit eval` runs inside crit, which installs it as a dependency. Without the `reference` extra
(torch, transformers) Eval measures engines against reference scores it fetches (`crit-eval fetch`) but cannot compute
them itself. To work on Eval, clone crit and crit-eval side by side and install both editable:

```sh
python -m pip install -e crit/core/python -e crit-eval
```

Machine settings (`vram_headroom_gib`, `context_reserve_gib`, `context_tokens`, `eval_packages`) are read from
`settings.toml` in this checkout, else in the crit checkout whose crit-core is installed
([settings.example.toml](https://github.com/crit-labs/crit/blob/main/settings.example.toml)).

## Layout

`crit_eval/` holds the harness and, installed with it, the fixed speed and quality suite (`suite/`), the protocol
(`protocol.json`), the device and kernel probes and the CUDA log-probability comparison reference (`probe/`) and
adapters (`examples/`): a mock to start from and one for engines behind an HTTP server. `tests/` runs with
`python -m unittest discover -s tests`.
