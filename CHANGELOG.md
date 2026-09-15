# Changelog

All notable changes to `nltiming` are recorded here. The project is in alpha;
the public API may change between tags.

## Unreleased

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
