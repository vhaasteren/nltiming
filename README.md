# nltiming

**Sample the pulsar timing model, not just the noise.**

[![CI](https://github.com/vhaasteren/nltiming/actions/workflows/ci.yml/badge.svg)](https://github.com/vhaasteren/nltiming/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

`nltiming` adds the timing model to your pulsar-timing-array likelihood as a
first-class, nonlinear block. For every fit parameter in the par file you
decide whether it is **numerically sampled** or **analytically marginalized**.
The same model definition drives both
[Discovery](https://github.com/nanograv/discovery) (NumPyro / NUTS) and
[Enterprise](https://github.com/nanograv/enterprise) (PTMCMCSampler), and the
sampler never sees raw par-file units: each sampled axis is mapped to a
well-conditioned coordinate automatically.

Why you would want this:

- **Correct posteriors for nonlinear timing parameters.** Shapiro delay
  (`M2`/`SINI`, `H3`/`STIGMA`), Kopeikin terms (`KIN`/`KOM`), parallax, and
  near-circular binaries are not linear in the residuals. Freezing them at the
  par-file value, or linearizing them, biases the noise and GW results.
- **One model, two ecosystems.** Build the timing block once with `TimingSpec`,
  then hand it to a Discovery likelihood or an Enterprise `PTA`. Chains from
  either sampler decode to physical parameters with the same tools.
- **Sampler-friendly by construction.** Prior-normalizing coordinate charts,
  Kepler-to-Laplace binary reparameterization, posterior whitening, and a
  geometry certifier that tells you *before* sampling whether NUTS will struggle.

The ideas follow [Vela.jl](https://github.com/abhisrkckl/Vela.jl) and
TempoNest; `nltiming` brings them to the Discovery and Enterprise stacks.

## Install

`nltiming` is alpha software and is not on PyPI yet. Install from git together
with the pulsar host it needs today,
[MetaPulsar](https://github.com/vhaasteren/metapulsar), and a timing engine:

```bash
# nltiming with the Discovery / NumPyro sampling stack
pip install "nltiming[discovery,numpyro] @ git+https://github.com/vhaasteren/nltiming"

# pulsar host + JAX timing engine (JUG); needs Python >= 3.12
pip install "metapulsar[jug] @ git+https://github.com/vhaasteren/metapulsar"
```

For Enterprise + PTMCMC add the `enterprise,ptmcmc` extras. Details, the
tempo2/libstempo path, and the pinned development branches are in
[`docs/install.md`](docs/install.md).

## Quickstart

Sample the timing model of one pulsar with Discovery, JUG, and NUTS. This is
[`examples/scripts/quickstart_discovery.py`](examples/scripts/quickstart_discovery.py)
and runs in under a minute on the example data shipped with the repository.

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
spec = TimingSpec(engines="jug")
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

The same `TimingSpec` drives Enterprise. Swap the engine, add the signal to
your model, and sample the `PTA` as you always do:

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

The inference plan names what is *marginalized*; every other timing axis is
sampled. Priors are per parameter and default to wide boxes scaled by the
par-file uncertainty.

```python
from nltiming import TimingSpec, TimingInference, priors

# Default: sample the nonlinear block, marginalize the linear axes.
TimingSpec(engines="jug")

# Sample every timing parameter (joint full-basis NUTS).
TimingSpec(engines="jug", inference="all")

# Name the marginalized axes explicitly; unmentioned axes are sampled.
TimingSpec(
    engines="jug",
    inference=TimingInference.groups(delta_flat=["DM", "DM1"], z_prior=["F0", "F1"]),
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
    binary_chart="auto",      # ECC/OM/T0 -> EPS1/EPS2/TASC for near-circular orbits
)
```

`timing.plan`, `timing.sampled`, `timing.marginalized`, and
`timing.chart_summary()` show what a `TimingSpec` resolved to on a given
pulsar. See [`docs/concepts.md`](docs/concepts.md).

## Notebooks

Ground-up introductions for PTA users who have not sampled a timing model
before, in [`examples/notebooks/`](examples/notebooks/):

| # | Notebook | Focus |
|---|----------|-------|
| 1 | `01_discovery_enterprise_backends.ipynb` | Discovery + Enterprise on one pulsar, JUG / libstempo / Vela engines, chains and corner plots |
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

Alpha. The API is settling but may still change between tags. Known
constraints:

- A `TimingPulsar` host is required and today that means MetaPulsar, even for
  a single dataset. Discovery and Enterprise native hosts are planned.
- The `discovery` extra installs a fork branch with a JAX fix; `enterprise`
  installs the NANOGrav `dev` branch. Both are temporary until upstream releases.
- The default JUG engine needs Python 3.12. PINT, libstempo, and Vela engines
  work on 3.11.
- Out of scope by design: noise bases, spectra, and correlated-noise
  likelihoods. Those stay with Discovery and Enterprise.

## Related projects

[MetaPulsar](https://github.com/vhaasteren/metapulsar) (multi-PTA pulsar
host), [JUG](https://github.com/MattTMiles/jug) (JAX timing engine),
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
