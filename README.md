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

## Ownership: `nltiming` owns model semantics, not sampler execution

`nltiming` supplies native objects at the two likelihood frontends:

- **Discovery:** `sampling.numpyro.model(...)` returns an ordinary
  zero-argument NumPyro model. Sample it with `sampling.numpyro.nuts` (a
  convenience recipe), raw `numpyro.infer.NUTS`/`MCMC`, or Discovery's
  `makesampler_nuts` — all three run the identical model.
- **Enterprise:** `ntm.enterprise_signal()` returns ordinary Enterprise
  `Parameter` objects (a joint vector parameter under full whitening).
  Sample the resulting `PTA` exactly like any other Enterprise analysis —
  `enterprise_extensions.sampler.setup_sampler` needs no `nltiming` import.

The `x -> z -> delta_theta` transform, the physical prior, and the Jacobian
are model semantics and live only in `nltiming.ParameterSpace`; they are
never reimplemented in a sampler wrapper. Choosing `transform="none"`,
`"standardized"`, or `"whitening"` changes the Enterprise parameter layout
(`pta.param_names`) and the NumPyro coordinate — never the top-level sampling
script.

The host object must satisfy the `TimingHost` protocol (also exported under
the original `PulsarInterface` name): frozen TOA arrays, `pint_model()`,
`timing_backend()`; single-pulsar hosts and multi-PTA composite hosts (e.g.
[MetaPulsar](https://github.com/vhaasteren/metapulsar)) both work —
PTA-suffixed parameter names are matched by base name.

## Discovery workflows

All three workflows below sample the exact same NumPyro model. Enable
float64 **before** constructing the Discovery likelihood — JAX arrays
already created as float32 stay float32:

```python
import jax
import discovery as ds

from nltiming import NonLinearTimingModel, priors, sampling

sampling.numpyro.ensure_x64()

ntm = NonLinearTimingModel(
    engines="jug",
    transform="whitening",
    sample=["PB", "TASC", "A1"],         # base names; the rest is marginalized
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
)
binding = ntm.bind(pulsar)               # pulsar: any TimingHost

likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, noisedict),
    *binding.discovery_signals(),
])

numpyro_model = sampling.numpyro.model(
    likelihood,
    binding,
    priors=noise_prior_overrides,   # optional: overrides for free non-timing params
    fixed=noisedict,                # pins numeric entries found in the likelihood
)
```

### 1. `sampling.numpyro.nuts` — shortest safe path

```python
mcmc = sampling.numpyro.nuts(
    numpyro_model,
    binding,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    dense_mass=True,
    target_accept=0.85,
    chain_method="parallel",
)

mcmc.run(jax.random.PRNGKey(42))
mcmc.print_summary()

samples = mcmc.get_samples()
posterior = mcmc.to_df()   # present because numpyro_model has .to_df
```

`mcmc` is an ordinary `numpyro.infer.MCMC` instance; the wrapper only
supplies opinionated defaults (init-at-reference, `target_accept=0.8`).

### 2. Raw `numpyro.infer.NUTS`/`MCMC` — full control

```python
from numpyro.infer import MCMC, NUTS, init_to_value

init = init_to_value(values=sampling.numpyro.timing_init_values(binding))

kernel = NUTS(
    numpyro_model,
    dense_mass=True,
    target_accept_prob=0.85,
    max_tree_depth=10,
    init_strategy=init,
)
mcmc = MCMC(
    kernel,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    chain_method="parallel",
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(42))
samples = mcmc.get_samples()
posterior = numpyro_model.to_df(samples)
```

### 3. Discovery `makesampler_nuts` — Discovery's own checkpoint runner

```python
import discovery.samplers.numpyro as ds_numpyro
from numpyro.infer import init_to_value

sampler = ds_numpyro.makesampler_nuts(
    numpyro_model,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    dense_mass=True,
    target_accept_prob=0.85,
    init_strategy=init_to_value(values=sampling.numpyro.timing_init_values(binding)),
)
sampler.run(jax.random.PRNGKey(42))
posterior = sampler.to_df()   # works because the model now has .to_df

ds_numpyro.run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint=250,
    rng_key=jax.random.PRNGKey(42),
    outdir=outdir,
    resume=False,
)
```

The Feather checkpoint contains decoded timing columns through `.to_df`. If
an `NLTChainBundle`-readable NPZ is also wanted, write it explicitly from the
latent site after sampling:

```python
artifact = binding.write(
    outdir,
    frontend="discovery",
    sampler="numpyro-nuts",
    latent={"kind": "npz", "path": "discovery_x.npz", "key_name": "x"},
)
sampling.numpyro.save_samples(
    outdir, sampler.get_samples(), binding, artifact=artifact, final=True
)
```

Do **not** sample the raw likelihood with Discovery's flat `makemodel`
helper (`ds_numpyro.makemodel(likelihood.logL)`): it samples every
`logL.params` entry as an independent `Uniform`, which cannot recover the
joint whitening transform. The correct seam is the complete NumPyro model
above, which Discovery accepts directly.

## Enterprise workflow

This is the canonical Enterprise example. It jointly samples every free
noise and timing parameter in `pta.param_names` — nothing here is
NLT-specific except adding the signal and (optionally) writing decoding
metadata:

```python
from pathlib import Path
import numpy as np

from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals
from enterprise_extensions import sampler as ee_sampler

from nltiming import NonLinearTimingModel, priors, sampling

outdir = Path("chains/J1909-3744")

efac = parameter.Uniform(0.1, 5.0)
equad = parameter.Uniform(-10.0, -4.0)
white = (
    white_signals.MeasurementNoise(efac=efac)
    + white_signals.TNEquadNoise(log10_tnequad=equad)
)

log10_A = parameter.Uniform(-20.0, -11.0)
gamma = parameter.Uniform(0.0, 7.0)
red = gp_signals.FourierBasisGP(
    utils.powerlaw(log10_A=log10_A, gamma=gamma), components=30, name="red_noise",
)

ntm = NonLinearTimingModel(
    engines={"tempo2": "jug", "pint": "jug"},
    transform="whitening",
    sample=["PB", "TASC", "A1"],
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
)
binding = ntm.bind(pulsar)

model = white + red + ntm.enterprise_signal()
pta = signal_base.PTA([model(pulsar)])
pta.set_default_params(noisedict)

sampler = ee_sampler.setup_sampler(pta, outdir=str(outdir), resume=False)

# Every NLT Enterprise Parameter implements sample(), including the joint
# whitening block; flatten scalar and vector Parameters in PTA order.
x0 = np.hstack([np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
assert x0.shape == (len(pta.param_names),)

layout = sampling.ptmcmc.chain_layout(binding, pta.param_names)
binding.write(outdir, frontend="enterprise", sampler="ptmcmc", chain_layout=layout)

sampler.sample(x0, Niter=1_000_000, SCAMweight=30, AMweight=15, DEweight=50)
```

### What whitening changes

Only the Enterprise parameter layout and proposal behavior change; the
user's model-building and sampling calls above do not, for any transform:

| `transform` | Enterprise parameters | `pta.param_names` |
|---|---|---|
| `"none"` | one scalar `UserParameter` per sampled fitpar, physical delta prior | `..._timing_A1`, `..._timing_PB`, `..._timing_TASC` |
| `"standardized"` | one scalar per fitpar, diagonal-standardized `x_i`; priors stay independent | same names, standardized values |
| `"whitening"` | **one joint vector** `UserParameter`, `size=len(sampled)`, correlated prior | `..._timing_x_0`, `..._timing_x_1`, ... |

Under `"whitening"`, `Parameter.prior_draw_mode == "joint"` on that vector
parameter, so `enterprise_extensions.JumpProposal.draw_from_prior` (and the
other generic prior-draw proposals) replace the whole correlated block
together rather than one component at a time — the block's log density does
not factor across components, so a partial update would be invalid. SCAM,
adaptive-metropolis, and differential-evolution proposals need no special
case: their acceptance ratio already runs on `pta.get_lnlikelihood`/
`get_lnprior`, which are correct for any transform.

### Direct `PTMCMCSampler`, without `enterprise_extensions`

```python
from PTMCMCSampler.PTMCMCSampler import PTSampler

ndim = len(pta.param_names)
cov = np.diag(np.full(ndim, 0.1**2))
sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=str(outdir))

x0 = np.hstack([np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
sampler.sample(x0, Niter=1_000_000)
```

This uses symmetric/adaptive PTMCMC proposals and is valid for full
whitening; `prior_draw_mode` only matters to proposal code that explicitly
calls `Parameter.sample()`.

### Multiple pulsars

Each pulsar gets its own bound NLT signal instance and, under full
whitening, its own joint vector parameter with a uniquely prefixed name:

```python
ntm = NonLinearTimingModel(...)
models = [(noise_model + ntm.enterprise_signal())(psr) for psr in pulsars]
pta = signal_base.PTA(models)
sampler = ee_sampler.setup_sampler(pta, outdir=str(outdir))
```

`sampling.ptmcmc.timing_only_sampler` is an **experimental, timing-only**
recipe: it fixes every non-timing parameter and samples only the timing
coordinates. It is not the standard Enterprise workflow above and is not
part of this quick start — see its docstring if you specifically want a
timing-only PTMCMC run with everything else pinned.

## Artifacts: decode chains anywhere, no live model needed

```python
from nltiming import NLTChainBundle
post = NLTChainBundle.load(outdir).posterior(burn=0.25)
```

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

# typical stack without tempo2 / libstempo (CI and most development)
pip install "nltiming[discovery,numpyro,enterprise,ptmcmc,jug]"

# enterprise_extensions without building libstempo (nanograv/dev hard-depends
# on libstempo≥2.4.0, which needs a system tempo2 at build time):
pip install --no-deps \
  "enterprise_extensions @ git+https://github.com/nanograv/enterprise_extensions.git@dev"
pip install healpy emcee "ptmcmcsampler>=2.1.0" "scikit-learn>=0.24" \
  ephem matplotlib pyarrow six
```

Only install the `libstempo` extra when a real Tempo2 stack is available:

```bash
# TEMPO2_PREFIX must point at the tempo2 install prefix (bin/tempo2 lives under
# $TEMPO2_PREFIX/bin). See libstempo's install docs / install_tempo2.sh.
export TEMPO2_PREFIX=/path/to/tempo2/prefix
pip install "nltiming[libstempo]"
```

When you *do* use libstempo for real `tempopulsar` evaluation, prefer
**sandbox / process isolation** (`libstempo.sandbox`, or MetaPulsar’s
`sandbox_tempo2`) so a tempo2 segfault cannot take down the host process.
`nltiming`’s `LibstempoEngine` accepts whatever session object the host
provides — it does not construct libstempo itself.

The `discovery` extra installs Discovery from
[`vhaasteren/discovery@temp/nltiming`](https://github.com/vhaasteren/discovery/tree/temp/nltiming)
(temporary JAX/`cho_solve` fix on top of NANOGrav main; the PyPI name
`discovery` is a different, unrelated package).

The `enterprise` / `enterprise_extensions` extras currently install from the
NANOGrav **`dev`** branches (git), so CI and local installs pick up
`prior_draw_mode` and related APIs before they hit PyPI. As noted above, the
`enterprise_extensions` extra currently pulls `libstempo` via that package’s
`install_requires`; use the `--no-deps` path on machines without tempo2.

The default `engines="jug"` path needs the JAX timing engine **`jug-timing`**
(import package `jug`). Until it is on PyPI, the `jug` extra installs from
[`vhaasteren/jug@tempo2-dev`](https://github.com/vhaasteren/jug/tree/tempo2-dev)
and requires **Python ≥ 3.12**.

## Layout

- `nonlinear_timing_model.py` — `NonLinearTimingModel` (configuration) and
  `TimingBinding` (`ntm.bind(pulsar)`, all pulsar-bound queries)
- `protocols.py` — `PulsarData` / `TimingHost` and timing backend contracts
- `evaluator.py` — mapping-based evaluation, metadata, scans, Jacobians, and
  immutable local weighted fits
- `backends/` — PINT, libstempo, and JUG engine adapters plus the multi-PTA
  composite backend
- `frontends/` — Discovery and Enterprise likelihood adapters
- `sampling/` — NumPyro model adapter and PTMCMC helpers (probabilistic model
  adapters and optional sampler recipes, not sampler ownership)
- `space.py`, `bijectors.py`, `whitening.py`, `priors.py`, `partition.py`,
  `units.py` — parameter-space math (transforms, priors, partition policy)
- `artifacts.py` — the `nlt-sidecar-v2` artifact contract and `NLTChainBundle`

## Development

```bash
# matches CI (no system tempo2 / libstempo build)
pip install -e ".[dev,jug,discovery,enterprise,numpyro]"
pip install --no-deps \
  "enterprise_extensions @ git+https://github.com/nanograv/enterprise_extensions.git@dev"
pip install healpy emcee "ptmcmcsampler>=2.1.0" "scikit-learn>=0.24" \
  ephem matplotlib pyarrow six

make fast     # tests, excluding slow
make check    # black, ruff, tests
```

Linux CI also installs `libsuitesparse-dev` so `scikit-sparse` (pulled by
enterprise) can build against CHOLMOD. Tests that need JUG, libstempo,
Discovery, or Enterprise skip cleanly when those packages are not installed.
libstempo-backed tests are not run on bare runners; use a tempo2-enabled
environment (devcontainer / conda) and sandbox mode for those.

## License

MIT — see [LICENSE](LICENSE).
