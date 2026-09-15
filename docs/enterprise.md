# Enterprise workflow

Canonical Enterprise path with **static posterior whitening**
(`WhiteningConfig`) and a few DM axes analytically marginalized. Nothing else
is NLT-specific except adding the signal and (optionally) writing decoding
metadata. For identity-layer / full-basis sampling, use `inference="all"` and
omit `whitening` (per-fitpar scalar parameters in chart coordinate `z`).

```python
from pathlib import Path
import numpy as np

from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals
from enterprise_extensions import sampler as ee_sampler

from nltiming import TimingSpec, TimingInference, WhiteningConfig, sampling
from nltiming import priors

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

spec = TimingSpec(
    engines="vela_jax",
    inference=TimingInference.groups(delta_flat=["DM", "DM1"]),
    whitening=WhiteningConfig(),   # joint vector Parameter in sampler coord x
    priors={"TASC": priors.delta_uniform(-0.5, 0.5, scale="PB")},
)
timing = spec.for_pulsar(pulsar)

model = white + red + spec.enterprise_signal()
pta = signal_base.PTA([model(pulsar)])
pta.set_default_params(noisedict)

sampler = ee_sampler.setup_sampler(pta, outdir=str(outdir), resume=False)

# Every NLT Enterprise Parameter implements sample(), including the joint
# whitening block; flatten scalar and vector Parameters in PTA order.
x0 = np.hstack([np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
assert x0.shape == (len(pta.param_names),)

layout = sampling.ptmcmc.chain_layout(timing, pta.param_names)
timing.write(outdir, likelihood="enterprise", sampler="ptmcmc", chain_layout=layout)

sampler.sample(x0, Niter=1_000_000, SCAMweight=30, AMweight=15, DEweight=50)
```

## What the static layer changes (Enterprise layout)

Only the Enterprise parameter layout and proposal behavior change with the
static affine layer; the model-building and sampling calls in the canonical
path do not:

| Static layer (`whitening=`) | Enterprise parameters | `pta.param_names` |
|---|---|---|
| `None` (identity) | one scalar `UserParameter` per sampled fitpar in chart coordinate `z` | `..._timing_<fitpar>` |
| `WhiteningConfig(...)` | **one joint vector** `UserParameter`, `size=len(sampled)`, correlated prior in whitened `x` | `..._timing_x_0`, `..._timing_x_1`, ... |

(There is no separate `"standardized"` constructor flag, diagonal scaling is
not a public static-layer mode.)

Under `WhiteningConfig`, `Parameter.prior_draw_mode == "joint"` on that vector
parameter, so `enterprise_extensions.JumpProposal.draw_from_prior` (and the
other generic prior-draw proposals) replace the whole correlated block
together rather than one component at a time, the block's log density does
not factor across components, so a partial update would be invalid. SCAM,
adaptive-metropolis, and differential-evolution proposals need no special
case: their acceptance ratio already runs on `pta.get_lnlikelihood` /
`get_lnprior`, which are correct for either static layer.

## Direct `PTMCMCSampler`, without `enterprise_extensions`

```python
from PTMCMCSampler.PTMCMCSampler import PTSampler

ndim = len(pta.param_names)
cov = np.diag(np.full(ndim, 0.1**2))
sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=str(outdir))

x0 = np.hstack([np.asarray(p.sample(), dtype=float).reshape(-1) for p in pta.params])
sampler.sample(x0, Niter=1_000_000)
```

This uses symmetric/adaptive PTMCMC proposals and is valid for full
static whitening; `prior_draw_mode` only matters to proposal code that
explicitly calls `Parameter.sample()`.

## Marginalized dynamic decentering (PTMCMC)

The Enterprise/PTMCMC realization of the third sampling mode, the twin of
`sampling.numpyro.decentered_model`. Marginalize the well-determined timing
axes and **all** GP coefficients into the live `C(η)`, and sample only the small
nonlinear timing block `ξ` plus the free hyperparameters `η` with PTMCMC. No
gradients are needed: the whitened `ξ` block is an identity-covariance target,
which is a *good* PTMCMC proposal. Requires `whitening=None` (identity static
layer) and fixed white noise (the flexfit WN-first MPE).

```python
from nltiming.decentering import NumpyMarginalTransport
from nltiming.likelihoods.enterprise import enterprise_marginal_products
from nltiming.sampling import ptmcmc

# timing built with whitening=None; `pta` as above (WN + red noise +
# spec.enterprise_signal()); noisedict pins every white-noise parameter;
# eta_mpe are the WN-first MPE hyperparameters.
products = enterprise_marginal_products(pta, timing, fixed_wn_params=noisedict)
hyper_names = products.params                 # sorted; delay keys + WN excluded (E8)
transport = NumpyMarginalTransport(
    products, dimension=len(timing.plan.sampled), key=timing.joint_site, params=hyper_names)

sampler = ptmcmc.decentered_sampler(
    pta, timing, transport, outdir,
    hyper_names=hyper_names,
    hyper_bounds={n: (-20.0, -11.0) if "log10_A" in n else (0.0, 7.0)
                  for n in hyper_names},
    fixed=noisedict,
)
p0 = ptmcmc.decentered_initial_point(timing, transport, hyper_names, eta_mpe)
sampler.sample(p0, Niter=200_000)             # default jump groups: [xi block, eta block]
```

**Density accounting (E2-E4), the part you own in the log-prior callable:**

- The sampled vector *is* `[ξ | η]`; `decentered_target` builds `lnlike` /
  `lnprior` so PTMCMC samples the exact reparameterized density (no
  base-measure `+½‖ξ‖²` term, PTMCMC has no sites to cancel).
- `lnprior` carries the exact timing prior `−½‖z‖²`, the transport
  log-Jacobian `ldJ(η)`, and the `η` box normalizer, all on the **prior** side,
  so parallel tempering never scales them (only `lnlike` is tempered by `1/T`).
- **`pta.get_lnprior` is never called** in this mode: the Enterprise delay
  `UserParameter`s carry physical priors that would double-count the timing prior
  already in `−½‖z‖²`. That is why `decentered_target` builds its own `lnprior`.
  (The identity-layer delay `UserParameter`s take the prior-normal `z`, not
  physical δ; `decentered_target` injects `z` and physical-δ decoding happens
  only at checkpoint time.)

Decode / checkpoint with `run_io.save_ptmcmc_decentered_checkpoint`
(`latent_decodable=false`; row-wise `decode_decentered_chain`), optionally
recording the cold-start recipe via `decentered_reconstruction_recipe` +
`attach_decentered_reconstruction`. Certify the geometry with the same
`certify_decentered_geometry(model, timing, ...)` used on the NumPyro path:
both frontends share the same marginalized covariance `C(η)`, so one
certification covers both. The cross-frontend integration tests (Discovery
NUTS vs Enterprise PTMCMC) live in `tests/test_enterprise_decentering.py`.

## Multiple pulsars

Each pulsar gets its own bound NLT signal instance and, under
`WhiteningConfig`, its own joint vector parameter with a uniquely prefixed name:

```python
spec = TimingSpec(...)
models = [(noise_model + spec.enterprise_signal())(psr) for psr in pulsars]
pta = signal_base.PTA(models)
sampler = ee_sampler.setup_sampler(pta, outdir=str(outdir))
```

`sampling.ptmcmc.timing_only_sampler` is an **experimental, timing-only**
recipe: it fixes every non-timing parameter and samples only the timing
coordinates. It is not the canonical Enterprise workflow and is not
part of this quick start, see its docstring if you specifically want a
timing-only PTMCMC run with everything else pinned.

