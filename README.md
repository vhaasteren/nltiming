# nltiming

Nonlinear pulsar-timing likelihood components for
[Discovery](https://github.com/nanograv/discovery) and
[Enterprise](https://github.com/nanograv/enterprise).

Instead of holding every timing parameter fixed at its par-file value (or
linearizing around it), `nltiming` *numerically samples* a chosen subset of
timing parameters inside the likelihood while the remaining linear nuisance
parameters are analytically marginalized. The same model configuration drives
both likelihood interfaces and both sampler stacks (NumPyro NUTS via a
JAX-capable timing engine, or PTMCMCSampler). Much of this was inspired by
[Vela.jl](https://github.com/abhisrkckl/Vela.jl), and earlier work in TempoNest.

Like Vela.jl, the timing model parameters are not sampled directly; due to high
covariances and parameter scale differences, the sampler sees a transformed set
of parameters in a latent space. Each sampled axis carries an explicit
*coordinate chart* mapping the physical parameter to a prior-normal coordinate,
and one static affine layer maps that to the sampler coordinate — `whitening=None`
(the default) is the identity static layer used by the dynamic joint transport,
and `WhiteningConfig()` is a static posterior-whitening layer. See
[Inference plans, coordinate charts, and geometry](#inference-plans-coordinate-charts-and-geometry).

## Ownership: `nltiming` owns model semantics, not sampler execution

`nltiming` supplies native objects at the two likelihood interfaces:

- **Discovery:** `sampling.numpyro.model(...)` returns an ordinary
  zero-argument NumPyro model (with `.to_df` for decoded timing columns).
  Sample it with `sampling.numpyro.nuts`, Discovery's
  `makesampler_nuts` + `run_nuts_with_checkpoints`, or — as a power-user
  option — raw `numpyro.infer.NUTS`/`MCMC`. All three run the identical model.
- **Enterprise:** `ntm.enterprise_signal()` returns ordinary Enterprise
  `Parameter` objects (a joint vector parameter under full whitening).
  Sample the resulting `PTA` exactly like any other Enterprise analysis —
  `enterprise_extensions.sampler.setup_sampler` needs no `nltiming` import.

The `xi -> z -> delta_theta` chart-and-layer, the physical prior, and the
Jacobian are model semantics and live only in `nltiming.ParameterSpace`; they are
never reimplemented in a sampler wrapper. The static-layer choice (`whitening=None`
vs `WhiteningConfig()`) changes the Enterprise parameter layout (`pta.param_names`)
and the NumPyro coordinate — never the top-level sampling script.

The pulsar object must satisfy the `TimingPulsar` protocol (also exported under
the original `TimingPulsar` name): frozen TOA arrays, `pint_model()`,
`timing_engine()`; single-pulsar and multi-PTA composite pulsars both work —
PTA-suffixed parameter names are matched by base name.

**Today, MetaPulsar is required.** The only production `TimingPulsar`
implementation is
[MetaPulsar](https://github.com/vhaasteren/metapulsar) — even for a single PTA
dataset. Examples and docs therefore build pulsars with `create_metapulsar`.
Once Discovery and/or Enterprise ship a native `TimingPulsar`, that dependency
can be dropped; the `nltiming` API does not change.

## Examples

Introductory notebooks (ground-up, for PTA users who have not sampled a timing
model before) live in [`examples/notebooks/`](examples/notebooks/):

1. `01_nonlinear_timing_charts.ipynb` — inference plan, charts, short NUTS, Enterprise
2. `02_geometry_certification_and_pivot.ipynb` — geometry certifier, `identically_linear`, pivot RN
3. `03_j1640_decentering_validation.ipynb` — full-basis on real IPTA DR2 J1640
4. `04_j1640_marginalization_validation.ipynb` — delta-flat vs z-prior on J1640

See [`examples/notebooks/README.md`](examples/notebooks/README.md) for setup
(MetaPulsar + JUG environment) and suggested order.

## Inference plans, coordinate charts, and geometry

What to sample is a **typed inference plan**, not a `sample=` list. You name what
is *marginalized*; every other timing axis is sampled:

```python
from nltiming import NonLinearTimingModel, TimingInference

# Everyday presets — string or InferencePreset also work on the model:
NonLinearTimingModel(engines="jug")                 # == inference="default"
NonLinearTimingModel(engines="jug", inference="all")

TimingInference.sample_all()                                # sample every axis (joint NUTS)
TimingInference.default()                                   # preset: marginalize the linear nuisances
TimingInference.groups(delta_flat=["DM1"], z_prior=["DM"])  # name the marginalized axes + how
```

Each fitpar gets exactly one disposition — `sample`, `marginalize_delta_flat`
(improper flat-in-δ GP) or `marginalize_z_prior` (proper unit-normal GP: a
different measure and fingerprint). Marginalization is **orthogonal** to linearity.

### Three coordinate layers

```
delta_theta   <-- chart -->   z   <-- static layer -->   xi
 (physical)     per-axis     (prior-normal,   one           (sampler)
                              N(0,1))          affine layer
```

The **physical prior lives on δ**; the per-axis chart maps δ↔z so the prior on
`z` is standard normal; one static layer maps z↔ξ. For the dynamic joint
transport use `whitening=None` (identity static layer, coordinate `z`) — the
transport is then the single affine layer.

### Charts: which physical prior gives which map

**PIT** means *probability integral transform*: map a physical draw through its
prior CDF, then through the standard-normal quantile function, so
`z ~ Normal(0, 1)` under the physical prior. For a Gaussian delta prior that map
is globally affine (`affine_normal`); for bounded or otherwise non-Gaussian
priors it is the nonlinear PIT chart, named `prior_pit` in the API (exact as a
prior transform, only local for whitening).

| Physical prior on δ | Chart | Identically-linear default? | Globally affine in z? |
|---|---|---:|---:|
| Normal | `affine_normal` | yes | yes |
| Uniform | `prior_pit` | no (explicit prior honored, warned) | no |
| Log-uniform | `prior_pit` | no | no |
| Truncated normal | `prior_pit` | no | no |

A parameter is **identically linear** when its engine waveform is exactly affine
in δ → a Gaussian delta prior → a globally-affine `affine_normal` chart. nltiming
ships a conservative, engine-independent fallback registry (`{DM, DM1, DM2,
OFFSET, PHOFF}` + `DMX`/`JUMP`/`FD` prefixes); the engine may add more; the user
may override with `identically_linear=` — **authoritative**: the explicit list
*replaces* the auto-derived set, so union with `ctx.identically_linear` to add.

### Disposition ≠ linearity — the geometry lesson

`F0`/`F1` are physically linear in *phase*, but the conservative registry does
not certify them, so **by default they are sampled on wide uniform `prior_pit`
charts**. Off the mode — exactly where the geometry certifier probes — that chart
reaches into the prior tails where spin sensitivity explodes and the joint
target's curvature blows up. This is the `F0` axis (width ≈ 2.5) that broke the
earlier decentering run.

The fix is a modeling decision, not a threshold nudge. Declaring the linear axes
identically linear flips them to `affine_normal` and collapses the off-mode
geometry — on an isolated pulsar by ~10⁶×, from a failing report (a *negative*
Hessian eigenvalue, residual RMS in the hundreds) to a clean `Hessian ≈ I` pass:

```python
# Certify BEFORE sampling — never called by nuts; passed=False is a design signal.
report = certify_joint_geometry(jm, ctx, hyper_points=box_hyper_probe_points(center, bounds))
# default:  H_eig ≈ [-4e3, 3e6]   rms ≈ 6e2    (F0/F1 on uniform prior_pit)
# declared: H_eig ≈ [1, 1]        rms ≈ 4e-3   (identically_linear unions in F0, F1, …)
```

`|z|` large is a boundary diagnostic **only** for `prior_pit` charts; an
`affine_normal` chart has no finite boundary. A certifier failure that *survives*
this fix (e.g. a white-noise-only reference that cannot precondition
timing↔red-noise cross-curvature) names the next thing to build — never a reason
to loosen `GeometryThresholds` or raise the tree depth. Worked example:
[`examples/notebooks/02_geometry_certification_and_pivot.ipynb`](examples/notebooks/02_geometry_certification_and_pivot.ipynb) §2b.

## Discovery workflows

All three workflows below sample the **exact same** NumPyro model. Enable
float64 **before** constructing the Discovery likelihood — JAX arrays
already created as float32 stay float32.

### Shared setup

```python
from pathlib import Path

import jax
import pandas as pd
import discovery as ds
import discovery.samplers.numpyro as ds_numpyro
from numpyro.infer import init_to_value

from nltiming import NonLinearTimingModel, priors, sampling

sampling.numpyro.ensure_x64()

ntm = NonLinearTimingModel(
    engines="jug",
    transform="whitening",
    sample=["PB", "TASC", "A1"],         # base names; the rest is marginalized
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
)
ctx = ntm.for_pulsar(pulsar)               # pulsar: any TimingPulsar

likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, noisedict),
    *ctx.discovery_signals(),
])

numpyro_model = sampling.numpyro.model(
    likelihood,
    context,
    priors=noise_prior_overrides,   # optional: overrides for free non-timing params
    fixed=noisedict,                # pins numeric entries found in the likelihood
)
```

`numpyro_model` is an ordinary zero-argument NumPyro model. It also exposes
`.to_df(samples)`, which decodes timing coordinates to physical columns.
That is the only NLT-specific sampling integration point: Discovery (and raw NumPyro)
never reimplement the whitening transform.

Do **not** sample the raw likelihood with Discovery's flat `makemodel`
helper (`ds_numpyro.makemodel(likelihood.logL)`): it samples every
`logL.params` entry as an independent `Uniform`, which cannot recover the
joint whitening transform.

### 1. `sampling.numpyro.nuts` — shortest path (no checkpointing)

Opinionated convenience: builds a NumPyro `MCMC` with init-at-reference and
sensible NUTS defaults. No Discovery I/O.

```python
mcmc = sampling.numpyro.nuts(
    numpyro_model,
    context,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    dense_mass=True,
    target_accept=0.85,
    chain_method="parallel",
)

mcmc.run(jax.random.PRNGKey(42))
mcmc.print_summary()

posterior = mcmc.to_df()   # wired from numpyro_model.to_df
```

### 2. Discovery checkpoint runner — recommended Discovery path

Use Discovery's own sampler factory and Feather checkpointing. No manual
`sampler.to_df = ...` boilerplate: `makesampler_nuts` attaches
`sampler.to_df` from `numpyro_model.to_df`, and
`run_nuts_with_checkpoints` recovers that attachment if needed.

```python
outdir = Path("chains/J1909-3744")

sampler = ds_numpyro.makesampler_nuts(
    numpyro_model,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    dense_mass=True,
    target_accept_prob=0.85,
    init_strategy=init_to_value(
        values=sampling.numpyro.timing_init_values(ctx)
    ),
)

# This runs the chain (do not also call sampler.run beforehand).
posterior = ds_numpyro.run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint=250,
    rng_key=jax.random.PRNGKey(42),
    outdir=outdir,
    resume=False,
)

# Equivalent on-disk read of the full chain (not sampler.get_samples(),
# which is only the last checkpoint chunk):
posterior = pd.read_feather(outdir / "numpyro-samples.feather")
```

The Feather file already contains decoded timing columns
(`{prefix}_{fitpar}_theta_display`, etc.). For nonlinear timing, that is
usually enough — no NLT run metadata required on the Discovery path.

`sampling.numpyro.timing_init_values(ctx)` is the one NLT helper used
at sampler construction: it initializes the joint timing site at the
par-file reference (zeros in sampling coordinates).

### 3. Raw `numpyro.infer.NUTS`/`MCMC` — power-user option

For nonlinear timing, prefer paths 1 or 2. Use raw NumPyro when you need
full control over the kernel/MCMC and are **not** using Discovery's
checkpoint runner.

After `mcmc.run(...)`, NumPyro only gives you latent parameters via
`mcmc.get_samples()` (the joint timing coordinate, plus any free noise
sites). Paths 1 and 2 attach a convenience `mcmc.to_df()` /
`sampler.to_df()` that turns those arrays into a DataFrame with physical
timing columns. A bare `MCMC` does **not** get that method. Call the
model's decoder instead — same function, same columns:

```python
posterior = numpyro_model.to_df(mcmc.get_samples())
```

Full example:

```python
from numpyro.infer import MCMC, NUTS, init_to_value

init = init_to_value(values=sampling.numpyro.timing_init_values(ctx))

mcmc = MCMC(
    NUTS(
        numpyro_model,
        dense_mass=True,
        target_accept_prob=0.85,
        max_tree_depth=10,
        init_strategy=init,
    ),
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    chain_method="parallel",
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(42))
posterior = numpyro_model.to_df(mcmc.get_samples())
```

If you hand a raw `MCMC` to `run_nuts_with_checkpoints`, Discovery will
attach `sampler.to_df` from `numpyro_model.to_df` automatically when the
kernel exposes `.model`. Prefer `makesampler_nuts` anyway — it is the
supported Discovery construction path.

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
ctx = ntm.for_pulsar(pulsar)

model = white + red + ntm.enterprise_signal()
pta = signal_base.PTA([model(pulsar)])
pta.set_default_params(noisedict)

sampler = ee_sampler.setup_sampler(pta, outdir=str(outdir), resume=False)

# Every NLT Enterprise Parameter implements sample(), including the joint
# whitening block; flatten scalar and vector Parameters in PTA order.
x0 = np.hstack([np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
assert x0.shape == (len(pta.param_names),)

layout = sampling.ptmcmc.chain_layout(ctx, pta.param_names)
ctx.write(outdir, likelihood="enterprise", sampler="ptmcmc", chain_layout=layout)

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

## Whitening: the posterior metric, config, and lifecycle

### The timing coordinate

Sampled timing parameters flow through two maps:

```text
delta = engine-native offset from the exact par-file reference
z     = C @ x + c            affine sampler coordinate  (x is what NUTS sees)
delta = prior_bijector(z)    probability-integral-transform (PIT) map;
                             under it the timing prior is z ~ N(0, I)
```

`ParameterSpace` owns these maps and their Jacobians. `transform=` selects how
the affine layer `(C, c)` is built:

| `transform` | affine layer `C` | sampler coordinate |
|---|---|---|
| `"none"` | identity (no whitening; sample `delta` directly) | `delta` |
| `"standardized"` | diagonal — each axis scaled to its posterior marginal σ | `x` |
| `"whitening"` (default) | lower-triangular Cholesky factor of the full posterior covariance | `x` |

### The posterior metric `F_z + I`

Because the PIT makes the prior exactly `z ~ N(0, I)`, the local posterior
precision in `z` is

```text
H = F_z + I          F_z = J_e^T F_delta J_e  (likelihood Fisher in z)
```

and whitening chooses `C C^T = H^{-1}`, so `C^T (F_z + I) C = I`. The `+ I` is
the **exact prior curvature**, not a numerical floor or ridge. nltiming whitens
the *target posterior*, never `C` itself: a likelihood-only metric `F_z` (or any
`F_z + αI` with `α ≠ 1`) mis-scales the transformed posterior direction by
direction and is deliberately **not** available — there is no likelihood-only
mode and no `numerical_floor` knob anywhere in the API.

### `WhiteningConfig`

Whitening is configured with a small frozen dataclass (there is no
stringly-typed dict):

```python
from nltiming import NonLinearTimingModel, WhiteningConfig

ntm = NonLinearTimingModel(
    transform="whitening",
    whitening=WhiteningConfig(
        reference_noise="toa_errors",   # which precision builds F_delta
        expansion_point="reference",    # where F_delta / the PIT Jacobian are evaluated
        origin="auto",                  # where x = 0 maps (the affine center c)
    ),
)
```

- **`reference_noise`** — the precision model used to build the likelihood
  Fisher `F_delta` when the model conditions itself (see the lifecycle below):
  - `"toa_errors"` *(default)* — diagonal `toaerrs**2`. Dependency-free and only
    an **approximate** preconditioner; its provenance is flagged `approximate`.
    Never describe a `toa_errors` metric as whitening a red-noise/DM/ECORR
    target.
  - `"frozen_white"` — EFAC/EQUAD white noise at declared values.
  - `"assembled_likelihood"` — the full frozen precision (marginalized
    red-noise/DM GP, ECORR, analytically-marginalized timing columns). This one
    is **not** auto-buildable from config; a likelihood interface supplies a
    `LocalPosteriorMetric` explicitly (see the two-stage lifecycle).
- **`expansion_point`** — `"reference"` (the only value): `F_delta`, the design
  matrix, and the PIT Jacobian are evaluated at the deterministic par-file
  reference `z_e = z(delta=0)`. (Evaluating the Jacobian at a WLS solution
  instead is what drives ill-conditioned PIT coordinates to their clipping
  boundaries and magnifies `C`; that historical mode is retained only for
  reproducing a pinned production commit.)
- **`origin`** — where the sampler's `x = 0` maps, i.e. the affine center `c`.
  Centering is a pure translation: it changes initialization and warmup, not the
  covariance being whitened. Options:
  - `"auto"` *(default)* — use a safeguarded local-posterior center **if** the
    metric carries a likelihood score, otherwise fall back to `"reference"`. The
    built-in `toa_errors`/`frozen_white` metrics carry no score, so `"auto"`
    resolves to `"reference"` unless an assembled metric supplies one.
  - `"reference"` — `c = z_e` (the par-file reference). Deterministic and fully
    reproducible; the recommended debug/repro setting.
  - `"local_posterior"` — one damped, trust-region Newton step toward the local
    MAP, `q = -(F_z + I)^{-1}(g_L + z_e)` where `g_L = ∇_z(-\log L)`, passed
    through a smooth interior guard (`z_max · tanh`) so it can never leave PIT
    support. Requires a metric with a `score_delta`; the run records whether the
    guard engaged. These are defaults, not mathematical invariants.

### Two-stage lifecycle: `for_pulsar` → `with_transport`

A `TimingContext` is immutable and conditioning is **finalize-once**:

```python
# Common path — conditions with the WhiteningConfig's default reference noise:
ctx = ntm.for_pulsar(pulsar)                 # conditioned; ctx.transport is set

# Assembled-metric path — supply the likelihood's own precision:
base   = ntm.for_pulsar(pulsar, condition=False)   # unconditioned; transport is None
metric = likelihood_interface.local_metric(base, reference_params)  # LocalPosteriorMetric
ctx    = base.with_transport(metric)               # conditioned, finalize-once
```

An **unconditioned** base answers every pulsar-bound query a likelihood
interface needs (`partition`, `priors`, `discovery_signals()`, the design
matrix, `local_metric` inputs) with an identity affine layer. Only sampler and
run-manifest construction require a **conditioned** context. Re-conditioning an
already-conditioned context raises; build a fresh base to re-condition.
`ctx.metric` and `ctx.transport` carry the metric provenance and the transport
record; both are folded into `ctx.fingerprint()`.

`LocalPosteriorMetric` is the typed, fingerprinted hand-off. The built-ins
`toa_errors_metric(...)` and `frozen_white_metric(...)` cover classes 1–2; an
assembled likelihood builds class 3 and marks it non-`approximate`.

## Run products: decode chains anywhere, no live model needed

A persisted run is a **scientific record**: decode it with the exact space it
was sampled with, and build a live model only for fresh calculations. A valid
read needs only the on-disk products — `nlt_run_meta.json` (schema
`nlt-run-meta-v3`) plus the serialized `ParameterSpace` and the raw chain — never
a live PTA, Discovery model, or PINT reload.

```python
import nltiming

run   = nltiming.load_run(outdir)      # RunResults, verified by default
phys  = run.load_display()             # prefer stored decoded physical values
post  = run.posterior(burn=0.25)       # decode latent draws through run.space
lat   = run.load_latent()              # raw latent chain (diagnostic)
truth = run.truths()                   # par-file reference values for overlays
```

`load_run` (sugar for `RunResults.load(outdir, verify=True)`) recomputes every
manifest **section digest** — `parameter_space`, `context`, `metric_source`,
`transport`, `chains` — and a verification failure names the section that
diverged. It refuses an unsupported schema with migration guidance, and
`RunManifest.write` refuses to overwrite an incompatible run without
`force=True`.

**Reconcile before feeding a saved point through a rebuilt likelihood** (e.g. a
GLS diagnostic):

```python
run.assert_consistent_with(ctx)   # raises, naming the diverging section, on any mismatch
```

Decode with `run.space`; never rebuild a decoder from pulsar/config for a saved
chain, and never `ctx.write(...)` an existing run before loading it.

### Static vs. dynamic transport

The manifest's `transport` section records one of two classes:

- **`static_affine`** (`latent_decodable = true`) — a fixed timing-only
  `(C, c)`. The latent chain is independently decodable through `run.space`;
  this is what the static timing whitening above produces.
- **`dynamic_transport`** (`latent_decodable = false`) — a joint full-basis
  transport `q = mu(eta) + L(eta)^{-T} xi` whose map depends on sampled
  hyperparameters, so `xi` alone has no physical meaning. `load_display()` reads
  the **required** stored per-draw physical values and refuses to reinterpret
  `xi` through `run.space`. Joint runs are written with
  `save_dynamic_checkpoint`, which refuses to promote a final checkpoint that
  lacks those canonical decoded values, and the one-affine-layer invariant keeps
  the static timing layer at identity when a dynamic transport is active.

## Interactive evaluator

The same engine interface supports engine-independent timing inspection without
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
remains the responsibility of the Discovery or Enterprise likelihood interface.

## Scope

`nltiming` owns the nonlinear-timing math, the timing engines (PINT,
libstempo, JUG, Vela), and the Discovery and Enterprise likelihood interfaces.
Pulsars (single-pulsar or multi-PTA composites such as MetaPulsar) supply the
data via the `TimingPulsar` protocol; the JUG package owns the JAX timing-engine
primitives.

Deliberately **out of scope**: Fourier/DM/chromatic/ECORR bases, `Phi`
inference, power-law or free-spectrum projection, and correlated-noise
likelihoods. Those belong to Discovery and Enterprise. `nltiming` supplies the
timing block and prior transform they build on, and downstream quick-look GP
tooling composes `nltiming` with those likelihood interfaces rather than re-homing noise
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
`nltiming`’s `LibstempoEngine` accepts whatever session object the pulsar
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
  `TimingContext` (`ntm.for_pulsar(pulsar)`, all pulsar-bound queries)
- `protocols.py` — `PulsarData` / `TimingPulsar` and timing engine interfaces
- `evaluator.py` — mapping-based evaluation, metadata, scans, Jacobians, and
  immutable local weighted fits
- `engines/` — PINT, libstempo, and JUG engines plus the multi-PTA
  composite engine
- `likelihoods/` — Discovery and Enterprise likelihood interfaces
- `sampling/` — NumPyro model helpers and PTMCMC helpers (probabilistic model
  helpers and optional sampler recipes, not sampler ownership)
- `space.py`, `bijectors.py`, `whitening.py`, `priors.py`, `partition.py`,
  `units.py` — parameter-space math (transforms, priors, partition policy)
- `metric.py` — `WhiteningConfig`, `LocalPosteriorMetric`, reference-noise metric
  builders, and the static/dynamic transport records
- `run_io.py` — the `nlt-run-meta-v3` run-metadata format, `RunResults`,
  and the static/dynamic checkpoint writers

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
