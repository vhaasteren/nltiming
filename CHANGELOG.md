# Changelog

All notable changes to `nltiming` are recorded here. The project is in alpha;
the public API may change between tags.

## [0.1.0] - 2026-09-16

First tagged release.

### Fixed
- ArviZ 1.x `from_dict` no longer accepts `posterior=`
  ([#4](https://github.com/vhaasteren/nltiming/issues/4)).
  `numpyro.posterior` and `RunResults.to_arviz` use both the 0.x kwargs
  API and the 1.x nested mapping.

### Changed
- Pin `psrdata` to tag `v0.1.0` (was moving `main`).
- Discovery extra installs `vhaasteren/discovery@feat/class-tracking`
  (Transport / metamath). `@temp/nltiming` was a cho_solve-only tip and
  could not run the NUTS notebooks.
- `numpyro` extra now includes `arviz>=0.12`.
- Require `pint-pulsar>=1.1.7`.

### Changed
- Default timing engine is vela-jax (`TimingSpec()`, `normalize_engines` fill,
  `TimingEvaluator`). JUG remains available as `engines="jug"`; JUG graph-mode
  knobs (`tempo2_native`, `tempo2_jug_options`) are refused unless JUG is
  selected, and are no longer forwarded into vela-jax / PINT / libstempo
  engine builds.
- psrdata SPEC v1 alignment. `TimingEngine.gauge_provenance()` /
  `gauge_applied` are replaced by a `residual_centering` mapping (data-set
  key to `psrdata.ResidualCentering`), one entry per data set for a single
  or a combined pulsar alike; the run manifest's `gauge` block becomes
  `residual_centering` and the schema is `nlt-run-meta-v5`. `TimingPulsar`
  no longer requires `state_id()`: the context cache fingerprints the
  record's content. Engine `native_units` are PINT units and are consumed
  as declared (evaluator and manifest no longer re-derive them from the
  PINT model). `engine_support` re-exports psrdata's record engine
  (`LinearTimingEngine`, `LinearContribution`, `linear_engine`) and owns
  the bare-matrix `LinearModel` / `LinearModelEngine` used by adapters and
  tests.

### Added
- Runnable quickstart scripts under `examples/scripts/` (Discovery/vela-jax/NUTS
  and Enterprise/libstempo/PTMCMC) on the shipped J1721-2457 data.
- `nltiming.configure_logging`: the package now defaults loguru to
  `WARNING` and above instead of loguru's `DEBUG` default.
- `CITATION.cff`, `CONTRIBUTING.md`, and a `docs/` tree.

### Changed
- README rewritten as a user-facing entry point (install, quickstart, choose
  what to sample). The reference material moved to `docs/`.
