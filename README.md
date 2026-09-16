# nltiming

**Sample the pulsar timing model, not just the noise.**

[![CI](https://github.com/vhaasteren/nltiming/actions/workflows/ci.yml/badge.svg)](https://github.com/vhaasteren/nltiming/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

`nltiming` puts the timing model into a pulsar-timing-array likelihood as a
nonlinear block. For every fit parameter in the par file you decide whether it
is **sampled** or **analytically marginalized**. One model definition drives
both [Discovery](https://github.com/nanograv/discovery) (NumPyro, NUTS) and
[Enterprise](https://github.com/nanograv/enterprise) (PTMCMCSampler), and the
sampler works in well-conditioned coordinates, never in raw par-file units.

- **Correct posteriors for nonlinear timing parameters.** Shapiro delay
  (`M2`/`SINI`, `H3`/`STIGMA`), Kopeikin terms (`KIN`/`KOM`), parallax and
  near-circular binaries are not linear in the residuals. Freezing or
  linearizing them biases the noise and GW results.
- **One model, two ecosystems.** Build the timing block once with `TimingSpec`,
  hand it to a Discovery likelihood or an Enterprise `PTA`, and decode either
  chain to physical parameters with the same tools.
- **Sampler-friendly by construction.** Prior-normalizing coordinate charts, a
  Kepler-to-Laplace binary chart, posterior whitening, and a geometry certifier
  that says before sampling whether NUTS will struggle.

The ideas follow [Vela.jl](https://github.com/abhisrkckl/Vela.jl) and
TempoNest.

## Install

Alpha, not on PyPI yet. Install from git together with the pulsar host,
[MetaPulsar](https://github.com/vhaasteren/metapulsar), and a timing engine:

```bash
# nltiming with the Discovery / NumPyro sampling stack
pip install "nltiming[discovery,numpyro] @ git+https://github.com/vhaasteren/nltiming@v0.1.0"

# pulsar host + default JAX timing engine (vela-jax)
pip install "metapulsar[vela_jax] @ git+https://github.com/vhaasteren/metapulsar"
```

For Enterprise and PTMCMC add the `enterprise,ptmcmc` extras. The full
dependency table, the tempo2 path, and the pinned development branches are in
[`docs/install.md`](docs/install.md).

## Quickstart

One pulsar, Discovery, the vela-jax engine, NUTS. This is
[`examples/scripts/quickstart_discovery.py`](examples/scripts/quickstart_discovery.py);
it runs in under a minute on the example data in the repository.

```python
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")  # timing residuals need float64

from pathlib import Path
import discovery as ds
import jax
from metapulsar import create_metapulsar
from numpyro.infer import init_to_value
from nltiming import TimingSpec
import nltiming.sampling as nlts

# 1. A pulsar: MetaPulsar reads the par/tim pair.
DATA = Path("examples/data/J1721-2457")
pulsar = create_metapulsar(
    {"combined": [{"par": DATA / "J1721-2457.par",
                   "tim": DATA / "J1721-2457.tim",
                   "timing_package": "tempo2"}]},
    combination_strategy="per_pta", use_pulse_numbers="reuse",
)

# 2. The timing model. The default plan samples the nonlinear axes and
#    marginalizes the rest analytically.
spec = TimingSpec()  # engines="vela_jax": PINT or tempo2 reads, Vela's chain evaluates
timing = spec.for_pulsar(pulsar)
print("sampled:", timing.sampled)

# 3. A Discovery likelihood with the timing signals added.
ds.config(kernels="metamath")
efac = f"{pulsar.name}_efac"
likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, add_equad=False),
    *timing.discovery_signals(),
])

# 4. A NumPyro model and a NUTS run.
model = nlts.numpyro.decentered_model(likelihood, timing, priors={efac: (0.1, 10.0)})
init = {**nlts.numpyro.decentered_init_values(timing, model.transport), efac: 1.0}
mcmc = nlts.numpyro.nuts(model, timing, num_warmup=200, num_samples=500,
                         init_strategy=init_to_value(values=init))
mcmc.run(jax.random.PRNGKey(0))

# 5. Physical posterior draws as ArviZ InferenceData.
post = nlts.numpyro.posterior(mcmc, timing)   # corner.corner(post) just works
```

The same `TimingSpec` drives Enterprise. Swap the engine, add the signal, and
sample the `PTA` as usual:

```python
import numpy as np
from enterprise.signals import parameter, signal_base, white_signals
from PTMCMCSampler.PTMCMCSampler import PTSampler
from nltiming import load_run

spec_ent = spec.with_engines({"tempo2": "libstempo"})
timing_ent = spec_ent.for_pulsar(pulsar)
white = white_signals.MeasurementNoise(efac=parameter.Uniform(0.1, 10.0))
pta = signal_base.PTA([(white + spec_ent.enterprise_signal())(pulsar)])

# Sidecar so load_run() can decode the chain to physical parameters later.
timing_ent.write("chains/J1721", likelihood="enterprise", sampler="ptmcmc",
                 chain_layout=nlts.ptmcmc.chain_layout(timing_ent, pta.param_names))

x0 = np.hstack([np.atleast_1d(p.sample()) for p in pta.params])
sampler = PTSampler(len(x0), pta.get_lnlikelihood, pta.get_lnprior,
                    np.diag(np.full(len(x0), 0.01)), outDir="chains/J1721")
sampler.sample(x0, Niter=100_000)

run = load_run("chains/J1721")
posterior = run.posterior(burn=0.25)          # dict of physical draws
```

## Choose what to sample

The inference plan names what is marginalized; every other timing axis is
sampled. Priors are per parameter and default to wide boxes scaled by the
par-file uncertainty.

```python
from nltiming import TimingSpec, TimingInference, priors

# Default: vela-jax kernel; sample the nonlinear block, marginalize the linear axes.
TimingSpec()

# Sample every timing parameter (joint full-basis NUTS).
TimingSpec(inference="all")

# Name the marginalized axes explicitly; unmentioned axes are sampled.
TimingSpec(
    inference=TimingInference.groups(delta_flat=["DM", "DM1"], z_prior=["F0", "F1"]),
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
    binary_chart="auto",      # ECC/OM/T0 -> EPS1/EPS2/TASC for near-circular orbits
)

# Optional: JUG as the JAX engine instead of vela-jax (Python >= 3.12).
TimingSpec(engines="jug")
```

`timing.plan`, `timing.sampled`, `timing.marginalized`, and
`timing.chart_summary()` show what a `TimingSpec` resolved to on a given
pulsar. See [`docs/concepts.md`](docs/concepts.md).

## Notebooks

Introductions for PTA users who have not sampled a timing model before, in
[`examples/notebooks/`](examples/notebooks/):

| # | Notebook | Focus |
|---|----------|-------|
| 1 | `01_discovery_enterprise_backends.ipynb` | Discovery + Enterprise on one pulsar, vela-jax / libstempo / Vela / JUG engines, chains and corner plots |
| 2 | `02_charts_and_binary.ipynb` | Per-axis coordinate charts and the Kepler-to-Laplace binary chart |
| 3 | `03_decentering_and_full_basis.ipynb` | Default decentered sampling vs `inference="all"` |
| 4 | `04_geometry.ipynb` | Certify the sampling geometry before running NUTS |
| 5 | `05_vela_discovery_sim.ipynb` | Three-way overlay with native Vela.jl on a simulated binary |

## Documentation

- [`docs/install.md`](docs/install.md): every extra, the tempo2 path, pinned branches.
- [`docs/concepts.md`](docs/concepts.md): inference plans, coordinate charts, binary charts, geometry.
- [`docs/discovery.md`](docs/discovery.md): Discovery workflows, checkpointing, derivative-free engines.
- [`docs/enterprise.md`](docs/enterprise.md): Enterprise and PTMCMC workflows, multi-pulsar.
- [`docs/whitening.md`](docs/whitening.md): the posterior metric and `WhiteningConfig`.
- [`docs/run-products.md`](docs/run-products.md): run metadata and decoding chains anywhere.
- [`docs/evaluator.md`](docs/evaluator.md): interactive evaluation and transformed-space fits.
- [`docs/architecture.md`](docs/architecture.md): what `nltiming` owns and what it deliberately does not.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development setup and package layout.

## Status

Alpha; the API may change between tags.

- A `TimingPulsar` host is required. Today that is MetaPulsar (even for one
  dataset) or a vela-jax `TimingPulsar`. Native Discovery and Enterprise hosts
  are planned.
- The `discovery` extra installs a fork branch with a JAX fix; `enterprise`
  installs the NANOGrav `dev` branch. Both are temporary.
- The default engine is vela-jax (Python 3.11). JUG is optional and needs
  Python 3.12. PINT, libstempo and Vela.jl work on 3.11.
- Out of scope by design: noise bases, spectra and correlated-noise
  likelihoods. Those stay with Discovery and Enterprise.

## Related projects

[MetaPulsar](https://github.com/vhaasteren/metapulsar) (multi-PTA pulsar
host), vela-jax (default JAX delay: Vela.jl's chain over a PINT or tempo2
freeze; not public yet), [JUG](https://github.com/MattTMiles/jug)
(optional JAX timing engine),
[Discovery](https://github.com/nanograv/discovery),
[Enterprise](https://github.com/nanograv/enterprise),
[Vela.jl](https://github.com/abhisrkckl/Vela.jl),
[PINT](https://github.com/nanograv/PINT),
[tempo2](https://bitbucket.org/psrsoft/tempo2).

## Citation

If `nltiming` contributes to a publication, please cite it via
[`CITATION.cff`](CITATION.cff). A methods paper is in preparation.

## License

MIT, see [LICENSE](LICENSE).
