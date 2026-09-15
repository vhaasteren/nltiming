# Contributing

## Development setup

```bash
# matches CI (no system tempo2 / libstempo build)
pip install -e ".[dev,discovery,enterprise,numpyro]"
pip install --no-deps \
  "enterprise_extensions @ git+https://github.com/nanograv/enterprise_extensions.git@dev"
pip install healpy emcee "ptmcmcsampler>=2.1.0" "scikit-learn>=0.24" \
  ephem matplotlib pyarrow six

make fast     # tests, excluding slow
make check    # black, ruff, tests
```

Linux CI also installs `libsuitesparse-dev` so `scikit-sparse` (pulled by
enterprise) can build against CHOLMOD. Tests that need vela-jax, JUG, libstempo,
Discovery, or Enterprise skip cleanly when those packages are not installed.
libstempo-backed tests are not run on bare runners; use a tempo2-enabled
environment (devcontainer / conda) and sandbox mode for those.


## Layout

- `nonlinear_timing_model.py`, `TimingSpec` (configuration) and
  `TimingSignal` (`spec.for_pulsar(pulsar)`, all pulsar-bound queries)
- `inference.py`, `TimingInference` / `InferencePreset` / `Marginalize`, plan
  resolution and fingerprints
- `protocols.py`, `PulsarData` / `TimingPulsar` and timing engine interfaces
- `evaluator.py`, mapping-based evaluation, metadata, scans, Jacobians, and
  immutable local weighted fits
- `engine_config.py`, engine-selection vocabulary (`normalize_engines`)
- `engine_support.py`, `LinearModel` / `LinearModelEngine`, validators;
  re-exports psrdata's record engine (`LinearTimingEngine`)
  (backend adapters live in MetaPulsar's `metapulsar.engines`)
- `likelihoods/`, Discovery and Enterprise likelihood interfaces
- `sampling/`, `numpyro.joint_model` / `model` / `nuts`, PTMCMC helpers
  (model glue and recipes, not sampler ownership)
- `space.py`, `bijectors.py`, `whitening.py`, `priors.py`, `units.py` , 
  parameter-space math (charts, static affine layer, priors)
- `linearity.py`, `coordinates.py`, `linearization.py`, `expansion.py`,
  `geometry.py`, identical-linearity policy, expansion / linearization
  records, optional geometry certifier
- `metric.py`, `WhiteningConfig`, `LocalPosteriorMetric`, reference-noise metric
  builders, and the static/dynamic transport records
- `run_io.py`, the `nlt-run-meta-v5` run-metadata format, `RunResults`,
  and the static/dynamic checkpoint writers

