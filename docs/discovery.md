# Discovery workflows

Enable float64 **before** constructing the Discovery likelihood, JAX arrays
already created as float32 stay float32.

## Shared setup, joint full-basis (primary path)

The usual modern workflow samples every timing axis with the dynamic joint
transport (`whitening=None`, the default). See notebooks `01`-`03`.

```python
from pathlib import Path

import jax
import pandas as pd
import discovery as ds
import discovery.samplers.numpyro as ds_numpyro
from numpyro.infer import init_to_value

from nltiming import TimingSpec, sampling

sampling.numpyro.ensure_x64()

spec = TimingSpec(
    inference="all",          # sample every timing axis
    # engines="vela_jax"      # default JAX kernel
    # whitening=None          # default: identity static layer (sampler coord z)
)
timing = spec.for_pulsar(pulsar)  # pulsar: TimingPulsar (MetaPulsar today)

likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, noisedict),
    ds.makegp_fourier(pulsar, ds.powerlaw, 10, name="rednoise"),
    *timing.discovery_signals(joint=True),
])

numpyro_model = sampling.numpyro.joint_model(
    likelihood,
    timing,
    fixed=noisedict,          # pin white-noise (and optionally RN) numbers
    # priors=...,             # free non-timing param bounds when not fixed
)
```

`numpyro_model` is an ordinary zero-argument NumPyro model. It exposes
`.to_df(samples)` (physical timing columns) plus `xi_site` / `hyper_sites`
metadata used by `dense_mass="auto"`. Discovery never reimplements the chart
or transport.

**Marginalized dynamic decentering path** (the small-block sampler for
nonlinear timing): keep `whitening=None` (the default identity static layer)
and a plan that marginalizes the well-determined axes
(`inference=TimingInference.default()`, or
`groups(delta_flat=[...], z_prior=[...])`); assemble the *marginalized*
`*timing.discovery_signals()` (default `joint=False`) and build with
`sampling.numpyro.decentered_model(...)`. Only the plan's sampled timing block
(dimension `k_s`) and the free hyperparameters are sampled, every marginalized
timing axis and *all* GP coefficients stay inside `likelihood.logL` and are
whitened away against the live marginalized covariance `C(η)` by a
`discovery.transport.MarginalTransport` (the η-dependent generalization of the
static posterior-metric whitening). The sampled dimension stays at the small
`k_s` (plan-dependent, e.g. just the 6 nonlinear binary axes on J1640 once the
linear axes are marginalized, instead of the full-basis 43) while the target is
the exact marginal. Certify with `certify_decentered_geometry(...)`
(same report / thresholds as the joint certifier, measured against live `C(η)`)
and init the `ξ` site with `decentered_init_values(timing, model.transport)`.
Expansion is a geometry-plan concern (`refine_timing_expansion` /
`with_expansion`), never a `decentered_model` kwarg. Worked example:
[`examples/notebooks/03_decentering_and_full_basis.ipynb`](../examples/notebooks/03_decentering_and_full_basis.ipynb).

**Static whitening path** (Enterprise-style preconditioning in Discovery): pass
`whitening=WhiteningConfig()`, assemble `*timing.discovery_signals()` (default
`joint=False`), and build with `sampling.numpyro.model(...)` instead of
`joint_model`. Mixed marginalization uses
`inference=TimingInference.groups(delta_flat=[...], z_prior=[...])`.

Do **not** sample the raw likelihood with Discovery's flat `makemodel`
helper (`ds_numpyro.makemodel(likelihood.logL)`): it samples every
`logL.params` entry as an independent `Uniform`, which cannot recover the
chart, static layer, or joint dynamic transport.

## Derivative-free Discovery with Vela or PINT

Host timing engines (Vela, PINT) have no JAX derivatives. They can still
drive Discovery's **marginal** `logL`: `discovery_signals()` emits a
value-only delay through `jax.pure_callback`, the GP/Woodbury algebra
stays JIT-compiled, and a derivative-free sampler walks the result.
The supported sampler is **PTMCMCSampler** via `discovery_target` /
`discovery_sampler`. `DiscoveryTarget` is a transformed-density pair
`(loglikelihood, logprior)` on `[q_timing | eta]`, `q` is `timing.coord`
(`z` or `x`), `eta` is the sorted free hyperparameters, and delay keys
inside `logL` are engine-native **delta**. `q = 0` is the engine
expansion (same convention as Enterprise `initial_point` and NumPyro
`timing_init_values`). There is no unit-cube prior transform; other
derivative-free samplers may call the same density pair if they walk
that coordinate themselves.

Discovery must use its default JAX numerical backend.
`derivative_method="analytic"` is required (the default). Do **not** pass
this context to `sampling.numpyro.nuts`, `model`, `joint_model`, or
`decentered_model`. Call `ensure_x64()` before constructing the
likelihood. Delay keys are engine-native delta; do not reuse Enterprise
`eval_params`.

```python
from nltiming import TimingSpec, sampling
import discovery as ds

sampling.numpyro.ensure_x64()

spec = TimingSpec(
    engines={"pint": "vela"},
    derivative_method="analytic",
    inference="default",
)
timing = spec.for_pulsar(pulsar)  # conditioned context (the default)

fixed = {f"{pulsar.name}_efac": 1.0}
likelihood = ds.PulsarLikelihood([
    pulsar.residuals,
    ds.makenoise_measurement_simple(pulsar, fixed, add_equad=False),
    *timing.discovery_signals(),
])

target = sampling.ptmcmc.discovery_target(likelihood, timing, fixed=fixed)
timing.write(
    "chains/vela-discovery",
    likelihood="discovery",
    sampler="ptmcmc",
    chain_layout=target.chain_layout(),
)
sampler = sampling.ptmcmc.discovery_sampler(target, outdir="chains/vela-discovery")
# timing block is q=0 (engine reference); pin free hypers if any
sampler.sample(target.initial_point(), Niter=200_000)
```

For free noise hyperparameters, pass `priors=` (the same
`discovery.prior.getprior_uniform` patterns as the NumPyro path) and
supply both `target.initial_point({name: value, ...})` and an explicit
`covariance=` block. The first likelihood call includes JIT compilation;
each later PTMCMC step pays one host callback for the timing residual.

## 1. `sampling.numpyro.nuts`, shortest path (no checkpointing)

Opinionated convenience: builds a NumPyro `MCMC` with init-at-reference and
sensible NUTS defaults. `dense_mass=True` still means “full dense mass” as in
NumPyro. The `nuts` default is `dense_mass="auto"`, which densifies only
`model.hyper_sites` (when there are ≥2) and leaves the intended-white `xi` on
an identity mass, usually what you want for joint full-basis runs.

```python
mcmc = sampling.numpyro.nuts(
    numpyro_model,
    timing,
    num_warmup=1_000,
    num_samples=2_000,
    num_chains=4,
    dense_mass=True,          # or omit for dense_mass="auto"
    target_accept=0.85,
    chain_method="parallel",
)

mcmc.run(jax.random.PRNGKey(42), extra_fields=sampling.numpyro.NUTS_EXTRA_FIELDS)
mcmc.print_summary()

posterior = mcmc.to_df()   # wired from numpyro_model.to_df
diag = sampling.numpyro.chain_diagnostics(mcmc)  # per-chain; never pool first
```

## 2. Discovery checkpoint runner, recommended Discovery path

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
        values=sampling.numpyro.timing_init_values(timing)
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
usually enough, no NLT run metadata required on the Discovery path.

`sampling.numpyro.timing_init_values(timing)` is the one NLT helper used
at sampler construction: it initializes the joint timing site at the
par-file reference (zeros in sampling coordinates).

## 3. Raw `numpyro.infer.NUTS`/`MCMC`, power-user option

For nonlinear timing, prefer paths 1 or 2. Use raw NumPyro when you need
full control over the kernel/MCMC and are **not** using Discovery's
checkpoint runner.

After `mcmc.run(...)`, NumPyro only gives you latent parameters via
`mcmc.get_samples()` (the joint timing coordinate, plus any free noise
sites). Paths 1 and 2 attach a convenience `mcmc.to_df()` /
`sampler.to_df()` that turns those arrays into a DataFrame with physical
timing columns. A bare `MCMC` does **not** get that method. Call the
model's decoder instead, same function, same columns:

```python
posterior = numpyro_model.to_df(mcmc.get_samples())
```

Full example:

```python
from numpyro.infer import MCMC, NUTS, init_to_value

init = init_to_value(values=sampling.numpyro.timing_init_values(timing))

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
mcmc.run(jax.random.PRNGKey(42), extra_fields=sampling.numpyro.NUTS_EXTRA_FIELDS)
posterior = numpyro_model.to_df(mcmc.get_samples())
```

If you hand a raw `MCMC` to `run_nuts_with_checkpoints`, Discovery will
attach `sampler.to_df` from `numpyro_model.to_df` automatically when the
kernel exposes `.model`. Prefer `makesampler_nuts` anyway, it is the
supported Discovery construction path.

