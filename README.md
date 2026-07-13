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
small on-disk artifacts with no live model objects.

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

The host object must satisfy the `PulsarInterface` protocol (frozen TOA
arrays, `pint_model()`, `timing_backend()`); single-pulsar hosts and
multi-PTA composite hosts (e.g. [MetaPulsar](https://github.com/vhaasteren/metapulsar))
both work — PTA-suffixed parameter names are matched by base name.

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
- `protocols.py` — `TimingBackend` / `JaxTimingBackend` / `PulsarInterface`
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
