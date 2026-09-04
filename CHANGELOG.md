# Changelog

All notable changes to `nltiming` are recorded here. The project is in alpha;
the public API may change between tags.

## Unreleased

### Added
- Runnable quickstart scripts under `examples/scripts/` (Discovery/JUG/NUTS
  and Enterprise/libstempo/PTMCMC) on the shipped J1721-2457 data.
- `nltiming.configure_logging`: the package now defaults loguru to
  `WARNING` and above instead of loguru's `DEBUG` default.
- `CITATION.cff`, `CONTRIBUTING.md`, and a `docs/` tree.

### Changed
- README rewritten as a user-facing entry point (install, quickstart, choose
  what to sample). The reference material moved to `docs/`.
