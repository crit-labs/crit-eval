# Changelog

All notable changes to crit Eval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
with versions following [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-25

### Changed

- crit Eval moved out of [crit](https://github.com/crit-labs/crit) into this repository, protocol crit-eval-v8 unchanged:
  the suite, protocol, probes and examples now live inside the `crit_eval` package (installed with it), the
  `crit-eval` command runs what `crit eval` runs, and Eval depends on crit-core alone. Earlier changes are in crit's
  CHANGELOG.
