# Concepts: inference plans, coordinate charts, and geometry

**Coordinate charts.** Timing parameters are awkward to sample in their native
units: `F0` and `DM2` live on wildly different scales, and they are strongly
covariant. So `nltiming` does not walk the sampler through raw par-file values.
For each fitpar it works with an offset **δ** from the par-file reference, then
maps that axis into a standardized coordinate **z** in which the physical prior
looks like a unit Gaussian. We call that per-parameter map a **coordinate
chart**: it assigns a coordinate representation to a physical parameter point
(δ → z). A subsequent **transport** (the optional static affine layer, or the
dynamic joint map) then moves between coordinate representations chosen for
inference (z ↔ ξ). In ordinary statistical language this is a prior-normalizing
reparameterization, the same idea as
[Vela.jl](https://github.com/abhisrkckl/Vela.jl)'s prior-normal coordinates.
The geometric word “chart” (Lee 2013) emphasizes that δ is the physical point
and z is merely its coordinate; each axis is treated as a one-dimensional
interval, so we do not lean on a full manifold atlas beyond that.

Two charts appear in practice:

- **`affine_normal`**, for a Gaussian prior on δ (typical when the delay is
  identically linear, or *numerically* so for PTA MSPs, e.g. spindown): a
  simple linear rescaling, well-behaved everywhere. This is the Vela-style chart.
- **`prior_pit`**, for a bounded or non-Gaussian prior (e.g. a wide uniform
  “cheat” prior): a nonlinear map through the prior CDF so `z` is still
  standard-normal under the prior. The map itself is exact throughout the
  interior of the prior support. What is only local is a *static whitening*
  layer built from this chart’s Jacobian and the likelihood curvature at one
  expansion point. **`prior_pit` is `nltiming`-only**, Vela.jl does not use
  this transformation (priors are used directly there).

The chart is fixed by the prior you chose for that axis (one prior → one chart).
Separately, an optional static affine **layer** can map `z → ξ` for the sampler
(`whitening=None` keeps that layer as the identity, which is what joint
full-basis NUTS wants). Longer walkthrough:
[`examples/notebooks/02_charts_and_binary.ipynb`](../examples/notebooks/02_charts_and_binary.ipynb).
Reference: J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed., GTM 218,
Springer (2013), DOI
[10.1007/978-1-4419-9982-5](https://doi.org/10.1007/978-1-4419-9982-5).

What to sample is a **typed inference plan** (`inference=`). You name what is
*marginalized*; every other timing axis is sampled:

```python
from nltiming import TimingSpec, TimingInference, InferencePreset

# Everyday presets (strings or InferencePreset):
TimingSpec()                                   # vela-jax; == inference="default"
TimingSpec(inference="all")                    # sample every axis
TimingSpec(inference=InferencePreset.ALL)
TimingSpec(engines="jug")                      # optional JUG kernel instead

# Mixed mode: name marginalized axes + measure; unmentioned axes are sampled:
TimingInference.groups(delta_flat=["DM1"], z_prior=["DM"])
```

Each fitpar gets exactly one disposition, `sample`, `marginalize_delta_flat`
(improper flat-in-δ GP) or `marginalize_z_prior` (proper unit-normal GP: a
different measure and fingerprint). Marginalization is **orthogonal** to
identical linearity (`identically_linear=`).

## Three coordinate layers

```
delta_theta   <-- chart -->   z   <-- static layer -->   xi
 (physical)     per-axis     (prior-normal,   one           (sampler)
                              N(0,1))          affine layer
```

The **physical prior lives on δ**; the per-axis chart maps δ↔z so the prior on
`z` is standard normal; one static layer maps z↔ξ. Keep these three objects
distinct:

```text
δ ↦ z            exact prior reparameterization (the chart)
Dδ/Dz |_{z_e}    local Jacobian at the expansion point
z = c + C ξ      local affine whitening (static layer)
```

For the dynamic joint transport use `whitening=None` (identity static layer,
coordinate `z`), the transport is then the single affine layer.

## Kepler↔Laplace physical chart (`binary_chart="auto"`)

A **physical chart** is a different object from the per-axis prior charts:
it changes *which physical coordinates* the plan names, priors, and sampler see,
while the engine delay model and its fitpar frame stay untouched. The first one
is the low-eccentricity Kepler↔Laplace chart. When `ECC`, `OM`, `T0` are free
fit parameters, at least one of them is *sampled*, and `e_ref < e_max` (0.1, a
policy heuristic, the chart is exact for every `e > 0`), sampling happens in
`EPS1 = e sin ω`, `EPS2 = e cos ω`, `TASC = T0 − PB·ω/2π` while the engine delay
stays DD/T2/DDH. **This is not ELL1**, the eccentric `x·e²` delay term is fully
retained; only the *sampling coordinates* change. It removes the low-eccentricity
polar geometry (the thin curved `ω-T0` tube historical DD-native nonlinear J1640
analyses had to sample).

Dispositions are declared on **engine names** (`TASC` aliases `T0` *for
dispositions only*; ECC and OM must share one disposition). Fully
`marginalize_delta_flat` triples do **not** activate the physical chart (no
rename); instead, when `KeplerLaplacePolicy.marginal_basis_frame="auto"` (the
default), a **`MarginalBasisFrame`** reconditions those design-matrix columns
with the same Laplace geometry used by the chart, axis names and dispositions
stay `ECC`/`OM`/`T0`, nothing is decoded, and the improper-marginal likelihood
changes only by a recorded constant (`log_abs_det_b` / `log_volume_offset` in
the `binary_marginal_basis_frame` manifest group). Cross-run evidence
comparisons need matching frame settings (or an explicit offset correction).
Pass `marginal_basis_frame="off"` to keep the raw engine basis. The chart never
reinterprets a **deliberately specified prior**: a user or PINT prior on `ECC`,
`OM`, or `T0` demotes it, a T0 density does not transfer to TASC, so restate
it as `priors={"TASC": ...}` to use the chart. Delay keys / NumPyro node names
use the sampling names, and `RunResults.posterior()` adds derived `ECC`/`OM`/`T0`
columns (decoded on the same reference-local branch the likelihood used).

**STIGMA priors (Case D).** When a converted DDH pulsar advertises
`conversion_metadata().required_sampling` containing `STIGMA`, context build
rejects `marginalize_delta_flat` on that axis, and also rejects the quieter
violation of the axis being **absent from the plan entirely**, which would leave
ς silently pinned at the emitted prior centre and read as if it were measured.

Three composable helpers live in `nltiming.priors`:

| helper | gives |
|---|---|
| `stigma_orientation_logpdf` | density `p(ς) = 4ς/(1+ς²)²` (uniform cos *i*) |
| `stigma_mass_ceiling_lower` | lower bound `ς ≥ (H3/(T☉·M_max))^(1/3)` |
| `stigma_mass_function_support` | `(lo, hi)` from the mass-function closure over an `m_p` range |

`AxisPrior` supports only bounded and normal families, there is no
custom-logpdf family, so `4ς/(1+ς²)²` **cannot be installed directly** as
`priors={"STIGMA": ...}`. Compose the bounds, hand them to
`stigma_prior_from_support(lo, hi)` for a spec the framework can carry, and
reweight by `stigma_orientation_logpdf` in post-processing if the orientation
density matters. A declared STIGMA prior whose support leaves `(0, 1]` makes
`fw10_absorbed` inactive (`stigma_support_out_of_domain`) rather than letting
the decode run outside its domain.

**`fw10_absorbed` chart (DDH + STIGMA sampling-path conditioning).** When
`A1`, `ECC`, `OM`, `T0`, `H3`, and `STIGMA` are all free and *sampled*, PB is
frozen, and no secular dots (`PBDOT`/`A1DOT`/`EDOT`/`OMDOT`) are present, the
sampler sees absorbed-gauge coordinates
`(A1_ABS, EPS1_ABS, EPS2_ABS, TASC_ABS, H3, STIGMA)` while the engine stays on
intrinsic DDH. This flattens the curved `(A1, ECC, OM, T0, STIGMA)` valley that
gradient samplers otherwise struggle with; native DDH coordinates remain
correct without the chart. Manifest group: `fw10_absorbed_chart`.

**Prior semantics** (`policy.prior = "sampling_frame"`, recorded in the
manifest): charted-axis priors live on the sampling frame. Note the induced
measure, `dEPS1·dEPS2 = e·dECC·dOM`, so a flat EPS prior is `p(e) ∝ e` in
Kepler variables and independent Gaussian EPS priors induce a Rayleigh-like
`p(e)`, never a uniform or Gaussian `p(e)`. Because charting changes the prior
model, chart-on vs chart-off **evidence** comparisons are only meaningful with
the prior held fixed. Every accepted EPS prior support is a bounded rectangle
strictly inside the eccentricity disk (`e ≤ 1 − margin`): default WLS boxes are
shrunk to fit before sampling, and unbounded or disk-crossing user EPS priors
are rejected (use bounded families). Nothing is clipped at runtime and invalid
eccentricities never reach the engine. When the epoch-shift identity is not
exact (secular/derived orbital evolution, OMDOT/PBDOT/EDOT/A1DOT), the chart
activates only if that support provably excludes the ω-branch seam ray (which
carries an `O(rate × PB)` likelihood discontinuity); supports containing the
eccentricity origin additionally require an origin-certified engine backend. PX
and the Shapiro `(M2, SINI)` ridge are separate workstreams.

**v1 default behavior (be aware).** Until a production engine backend is
origin-certified (`BinaryChartCapability.origin_certified=True` with a
`certification_ref`, set only by a PR that lands its passing full-likelihood
origin certification), **no** backend is certified. Because a typical
low-eccentricity MSP has an EPS default box (`50·σ`) that contains the
eccentricity origin, such pulsars **demote under `auto`**, they stay on
`ECC/OM/T0` engine coordinates, identical to `binary_chart="off"`, and emit a
one-line `UserWarning` per pulsar. This is the honest, conservative default:
the chart engages once the geometry is certifiably safe. To silence the
warnings on an ensemble where you deliberately do not want charting, pass
`binary_chart="off"`.

## Charts: which physical prior gives which map

**PIT** means *probability integral transform*: map a physical draw through its
prior CDF, then through the standard-normal quantile function, so
`z ~ Normal(0, 1)` under the physical prior. For a Gaussian delta prior that map
is globally affine (`affine_normal`); for bounded or otherwise non-Gaussian
priors it is the nonlinear PIT chart, named `prior_pit` in the API. The
prior-normalizing map is exact throughout the interior of the prior support;
a static whitening transform constructed from its Jacobian and the likelihood
curvature at one expansion point is only a local approximation to posterior
geometry.

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
may override with `identically_linear=`, **authoritative**: the explicit list
*replaces* the auto-derived set, so union with `timing.identically_linear` to add.

## Disposition ≠ linearity, the geometry lesson

Spindown parameters are **not** identically linear in the timing residual.
Rotational phase is a Taylor series in `F0`/`F1`, but the residual is the phase
offset divided by frequency (`r ∼ ΔΦ/F0` in the usual timing equation), so `F0`
enters nonlinearly. For PTA MSPs near a good solution, though, `δF0/F0` is tiny
and the residual is *numerically* extremely close to linear, close enough that
treating spin (and similarly sky position) as identically linear is the right
modeling choice for sampling geometry. The conservative fallback registry still
does **not** auto-certify them, so **by default they are sampled on wide uniform
`prior_pit` charts**. Off the mode, exactly where the geometry certifier probes
,  that chart reaches into the prior tails where spin sensitivity explodes and
the joint target's curvature blows up.

The fix is a modeling decision, not a threshold nudge. Declaring those
near-linear axes identically linear flips them to `affine_normal` and collapses
the off-mode geometry, on an isolated pulsar by ~10⁶×, from a failing report (a
*negative* Hessian eigenvalue, residual RMS in the hundreds) to a clean
`Hessian ≈ I` pass:

```python
# Certify BEFORE sampling; never called by nuts; passed=False is a design signal.
report = certify_joint_geometry(jm, timing, hyper_points=box_hyper_probe_points(center, bounds))
# default:  H_eig ≈ [-4e3, 3e6]   rms ≈ 6e2    (F0/F1 on uniform prior_pit)
# declared: H_eig ≈ [1, 1]        rms ≈ 4e-3   (identically_linear unions in F0, F1, …)
```

`|z|` large is a boundary diagnostic **only** for `prior_pit` charts; an
`affine_normal` chart has no finite boundary. A certifier failure that *survives*
this fix (e.g. a white-noise-only reference that cannot precondition
timing↔red-noise cross-curvature) names the next thing to build, never a reason
to loosen `GeometryThresholds` or raise the tree depth. Worked example:
[`examples/notebooks/04_geometry.ipynb`](../examples/notebooks/04_geometry.ipynb).

