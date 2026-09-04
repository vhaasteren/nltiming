# Whitening: the posterior metric, config, and lifecycle

## The timing coordinate

Sampled timing parameters flow through the
[three coordinate layers](concepts.md#three-coordinate-layers):

```text
delta  --chart-->  z  --static affine (C, c)-->  xi
```

`ParameterSpace` owns the chart and the static affine layer and their Jacobians.
The static layer is selected by the constructor kwarg **`whitening=`**:

| `whitening=` | affine layer `C` | sampler coordinate |
|---|---|---|
| `None` *(default)* | identity — required for `sampling.numpyro.joint_model` | `z` (prior-normal) |
| `WhiteningConfig(...)` | lower-triangular factor of the local posterior covariance in `z` (`C C^T = (F_z + I)^{-1}`) | `x` (statically whitened) |

## The posterior metric `F_z + I`

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

## `WhiteningConfig`

The static whitening layer is configured with a small frozen dataclass:

```python
from nltiming import TimingSpec, WhiteningConfig

spec = TimingSpec(
    inference="default",                # or TimingInference.default()
    whitening=WhiteningConfig(
        reference_noise="toa_errors",   # which precision builds F_delta
        expansion_point="reference",    # where F_delta / the chart Jacobian are evaluated
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

## Two-stage lifecycle: `for_pulsar` → `with_transport`

A `TimingSignal` is immutable and conditioning is **finalize-once**:

```python
# Common path — conditions with the WhiteningConfig's default reference noise:
timing = spec.for_pulsar(pulsar)                 # conditioned; timing.transport is set

# Assembled-metric path — supply the likelihood's own precision:
base   = spec.for_pulsar(pulsar, condition=False)   # unconditioned; transport is None
metric = likelihood_interface.local_metric(base, reference_params)  # LocalPosteriorMetric
timing    = base.with_transport(metric)               # conditioned, finalize-once
```

An **unconditioned** base answers every pulsar-bound query a likelihood
interface needs (`timing.plan`, priors, `discovery_signals()`, the design
matrix, `local_metric` inputs) with an identity affine layer. Only sampler and
run-manifest construction require a **conditioned** context when
`whitening=WhiteningConfig(...)`. Re-conditioning an already-conditioned
context raises; build a fresh base to re-condition. `timing.metric` and the static
transport record carry the metric provenance; both are folded into
`timing.fingerprint()`.

`LocalPosteriorMetric` is the typed, fingerprinted hand-off. The built-ins
`toa_errors_metric(...)` and `frozen_white_metric(...)` cover classes 1–2; an
assembled likelihood builds class 3 and marks it non-`approximate`.

