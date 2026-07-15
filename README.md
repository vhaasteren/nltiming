# nltiming

Nonlinear pulsar-timing likelihood components for
[Discovery](https://github.com/nanograv/discovery) and
[Enterprise](https://github.com/nanograv/enterprise).

Instead of holding every timing parameter fixed at its par-file value (or
linearizing around it), `nltiming` *numerically samples* a chosen subset of
timing parameters inside the likelihood while the remaining linear nuisance
parameters are analytically marginalized. The same model configuration drives
both likelihood frontends and both sampler stacks (NumPyro NUTS via a
JAX-capable timing engine, or PTMCMCSampler), and every run is decodable from
small on-disk artifacts with no live model objects. Note: much of this was
inspired by [Vela.jl](https://github.com/abhisrkckl/Vela.jl), and earlier work
in TempoNest.

## Quick start

```python
from nltiming import NonLinearTimingModel
from nltiming import priors, sampling

ntm = NonLinearTimingModel(
    engines="jug",                       # JAX timing engine (NUTS-capable)
    sample=["PB", "TASC", "A1"],         # base names; the rest is marginalized
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
)
binding = ntm.bind(pulsar)               # pulsar: any PulsarInterface host

# Discovery + NumPyro NUTS
import discovery as ds
likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, noisedict),
    *binding.discovery_signals(),
])
mcmc = sampling.numpyro.nuts(
    sampling.numpyro.model(likelihood, binding, fixed=noisedict), binding
)

# Enterprise + PTMCMC
pta = signal_base.PTA([(noise_model + ntm.enterprise_signal())(pulsar)])
pts = sampling.ptmcmc.sampler(pta, binding, outdir)

# Artifacts: decode chains anywhere, no live model needed
binding.write(outdir, frontend="discovery", sampler="numpyro-nuts")
from nltiming import NLTChainBundle
post = NLTChainBundle.load(outdir).posterior(burn=0.25)
```

The host object must satisfy the `TimingHost` protocol (also exported under
the original `PulsarInterface` name): frozen TOA
arrays, `pint_model()`, `timing_backend()`); single-pulsar hosts and
multi-PTA composite hosts (e.g. [MetaPulsar](https://github.com/vhaasteren/metapulsar))
both work — PTA-suffixed parameter names are matched by base name.

## Interactive evaluator

The same backend contract supports engine-independent timing inspection without
constructing a likelihood:

```python
from nltiming import TimingEvaluator
from nltiming.space import ParameterSpace

timing = TimingEvaluator.from_pulsar(
    pulsar,
    engines={"pint": "jug", "tempo2": "jug"},
    design_matrix_method="autodiff",
)

timing.parameters["F0"]
evaluation = timing.evaluate({"F0": 1e-10}, frame="delta")
scan = timing.scan("TASC", [-0.5, 0.0, 0.5], scale="PB")
jacobian = timing.jacobian(method="autodiff")
fit = timing.fit(["F0", "F1"])

# Transformed-space (z) fit: prior-bijector-scaled Jacobian + weighted LSQ,
# returning a TimingZFitResult with z_best, covariance, rank/singular values.
space = ParameterSpace.build(theta_ref_mapping=timing.reference_exact)
jacobian_z = timing.jacobian_z(space)
zfit = timing.fit_z(space, ["F0", "F1"])
```

All operations return immutable result objects. The evaluator does not mutate
TOAs, parameter fit flags, timing sessions, or input files. `white_chi2` and
the built-in fit use diagonal TOA errors only; correlated-noise inference
remains the responsibility of the Discovery or Enterprise frontend.

## Scope

`nltiming` owns the nonlinear-timing math, the timing backends (PINT,
libstempo, JUG, Vela), and the Discovery and Enterprise likelihood frontends.
Hosts (single-pulsar or multi-PTA composites such as MetaPulsar) supply the
data via the `TimingHost` protocol; the JUG package owns the JAX timing-engine
primitives.

Deliberately **out of scope**: Fourier/DM/chromatic/ECORR bases, `Phi`
inference, power-law or free-spectrum projection, and correlated-noise
likelihoods. Those belong to Discovery and Enterprise. `nltiming` supplies the
timing block and prior transform they build on, and downstream quick-look GP
tooling composes `nltiming` with those frontends rather than re-homing noise
math here.

Maintainer notes (ownership table, engineering caveats, upstream tracks) live
in [`DESIGN_NOTES.md`](DESIGN_NOTES.md). The interactive transformed-space (`z`)
timing fit (`fit_z`, `jacobian_z`, `TimingZFitResult`) is described in
[`TRANSFORMED_SPACE_FIT.md`](TRANSFORMED_SPACE_FIT.md).

## Installation

```bash
pip install nltiming

# with optional integrations
pip install "nltiming[discovery,numpyro,enterprise,ptmcmc,libstempo]"
```

The default `engines="jug"` path needs the
[JUG](https://github.com/MattTMiles/jug) JAX timing engine (`jug-timing`,
Python ≥ 3.12), currently installed from source.

## Layout

- `nonlinear_timing_model.py` — `NonLinearTimingModel` (configuration) and
  `TimingBinding` (`ntm.bind(pulsar)`, all pulsar-bound queries)
- `protocols.py` — `PulsarData` / `TimingHost` and timing backend contracts
- `evaluator.py` — mapping-based evaluation, metadata, scans, Jacobians, and
  immutable local weighted fits
- `backends/` — PINT, libstempo, and JUG engine adapters plus the multi-PTA
  composite backend
- `frontends/` — Discovery and Enterprise likelihood adapters
- `sampling/` — NumPyro and PTMCMCSampler glue
- `space.py`, `bijectors.py`, `whitening.py`, `priors.py`, `partition.py`,
  `units.py` — parameter-space math (transforms, priors, partition policy)
- `artifacts.py` — the `nlt-sidecar-v2` artifact contract and `NLTChainBundle`

## Development

```bash
pip install -e ".[dev]"
make fast     # tests, excluding slow
make check    # black, ruff, tests
```

Tests that need JUG, libstempo, Discovery, or Enterprise skip cleanly when
those packages are not installed.

## License

MIT — see [LICENSE](LICENSE).
