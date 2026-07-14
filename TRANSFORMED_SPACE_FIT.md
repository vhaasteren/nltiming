# Feature proposal: interactive transformed-space (`z`) timing fit

**Status:** Design proposal for review; not yet an API commitment
**Scope:** additions to the interactive evaluator, bijector, and result types.
**Explicitly out of scope:** Gaussian-process bases, `Phi` inference, spectra,
correlated-noise likelihoods, GUI/revision/provenance concepts.

## Motivation

`nltiming` already owns a physically meaningful, prior-transformed timing
coordinate system (`z`), and it already exposes an interactive, engine-neutral
timing facade. Today those two capabilities do not meet: the transformed
coordinate lives only on the *sampling* path, and the interactive facade can
only fit in physical coordinates with diagonal white errors.

Downstream tools that live **outside** `nltiming` — interactive fast timing
optimization and quick-look empirical-Bayes waveform reconstruction (see the
MetaPulsar-side `feature_flexible_fit.md` design note) — need two things from
`nltiming` and only from `nltiming`:

- a ready timing basis block in `z` coordinates (`J_z`), with an honest
  capability/fidelity label; and
- a fast, bounded, nonlinear local timing fit in those same `z` coordinates.

Providing these keeps the noise-model and GP machinery out of `nltiming` (it
stays with Discovery/Enterprise and the downstream `flexfit` module) while
giving those consumers a single, correct timing seam to build on. This proposal
covers exactly that seam.

## Current state

- **`z` transform exists, but not on the interactive path.**
  `ParameterSpace` (`src/nltiming/space.py`: `delta_from_z` / `z_from_delta` /
  `z_from_x` / `coord_from_delta` / `logjacobian` / ...) and `PriorBijector`
  (`src/nltiming/bijectors.py`, incl. `jacobian_diag_delta_from_z`) implement
  the per-axis probability-integral transform. They are constructed by
  `NonLinearTimingModel` / `TimingBinding` for the NUTS and PTMCMC frontends.

- **`TimingEvaluator` is physical-only.** `src/nltiming/evaluator.py` builds no
  `ParameterSpace`/`PriorBijector`. Its `Frame` is `Literal["delta",
  "absolute"]` — there is no `z` frame. `jacobian(frame=...)` returns
  `d residual_delta / d delta_theta` regardless of `frame` (the argument only
  moves the evaluation point). `fit(...)` hardcodes `frame="delta"`, diagonal
  white errors, plain Gauss-Newton, no bounds and no prior term.

- **Results carry no fidelity label.** `TimingEvaluation` and `TimingFitResult`
  do not record which residual-evaluation tier produced them, so a consumer
  cannot tell whether a fit that moved far from its anchor was retraced deeply
  enough to trust.

## Goals and non-goals

**Goals**

1. A genuine `z`-frame timing Jacobian `J_z` from the interactive path.
2. A bounded, nonlinear local timing fit expressed in `z`.
3. A residual-evaluation fidelity tier recorded on evaluations and fits.
4. A documented, validated prior-transform contract for the coordinate map.
5. A README/`UPSTREAM_INTEGRATION.md` statement pinning the charter boundary.

**Non-goals**

- No Fourier / DM / chromatic / ECORR bases, no `Phi` estimation, no
  power-law or free-spectrum projection. Those belong to Discovery/Enterprise
  and the downstream `flexfit` consumer, never here.
- No correlated-noise or generalized-least-squares likelihood; the interactive
  fit stays diagonal-white, matching the existing `fit` contract.
- No GUI, revision, caching, or provenance concepts.

## Proposed changes

### 1. `z`-frame Jacobian `J_z`

Add a `z` option to the interactive Jacobian so consumers get the timing block
of `T` directly, assembled as the product of the two owned factors:

```
J_z = (d residual_delta / d delta_theta) * diag(d delta_theta / d z)
```

The first factor is the existing backend Jacobian (`autodiff` for a
JAX-capable backend, otherwise the reference design matrix or a
finite-difference fallback). The second is
`PriorBijector.jacobian_diag_delta_from_z`, evaluated in NumPy/SciPy outside
any JAX trace. Concretely:

- extend the interactive `Frame` (or add a dedicated `jacobian(..., coord="z")`
  entry point) so the returned matrix is `d residual_delta / d z`;
- return it alongside a small capability/fidelity descriptor:
  `{method: "autodiff" | "analytic" | "finite-difference", tier: <T0..T4>}`;
- the evaluator needs a `PriorBijector`/`ParameterSpace` to do this — see
  "Wiring" below.

This lets a downstream fast-fit build its timing basis block from one call
instead of re-deriving the transform, and keeps the guarantee that JUG (or any
autodiff backend) never has to trace through a SciPy prior.

### 2. Bounded nonlinear `z`-space local fit

Add a `z`-coordinate fit — either a `coord="z"` mode on `fit(...)` or a sibling
method — that performs Gauss-Newton / trust-region iterations in `z` with:

- optional per-parameter bounds (from the prior support / transform);
- optional damping or a trust-region radius for large nonlinear moves;
- relinearization at each step using the `z`-frame `J_z` above;
- the same diagonal-white weighting the current `fit` uses.

The absolute-`z` parameterization keeps a zero-centered coefficient meaning
after relinearization (the same property the downstream affine timing model
relies on). The result stays an immutable `TimingFitResult`, extended per
change 3.

This is the "transformed-space iterative least squares" that both
`feature_flexible_fit.md` and `feature_pylk.md` name as the default fast timing
optimizer.

### 3. Residual-evaluation fidelity tier on results

Record the residual-evaluation tier (the T0-T4 vocabulary documented in
`ref-packages/jug/TEMPO2_NATIVE_MODES.md`) on `TimingEvaluation` and
`TimingFitResult`, plus on the Jacobian descriptor from change 1. This lets a
caller enforce escalation-at-acceptance: a proposed state that moved far in an
astrometric / binary / phase-connection direction can be re-evaluated at a
higher tier before it is trusted. `TimingCapabilities` should advertise which
tiers the resolved backend can produce.

### 4. Documented prior-transform contract

Promote the prior-transform map to a documented, validated public surface:

- publish `PriorBijector.jacobian_diag_delta_from_z` (and the `delta_from_z` /
  `z_from_delta` pair) as supported API;
- define the accepted-prior protocols:

  ```python
  class ScalarPrior(Protocol):       # minimal contract
      def ppf(self, u: np.ndarray) -> np.ndarray: ...
      def logpdf(self, value: np.ndarray) -> np.ndarray: ...

  class ScalarPriorTransform(Protocol):   # advanced: supply the map directly
      def delta_from_z(self, z: np.ndarray) -> np.ndarray: ...
      def derivative_delta_from_z(self, z: np.ndarray) -> np.ndarray: ...
  ```

- validate priors up front and **reject with a clear capability error** (never
  silently substitute a default) when a prior is improper (e.g. PINT's
  unbounded uniform), PDF-only (no reliable `cdf`/`ppf`), discrete / mixed /
  point-mass, singular (density vanishes on a relevant region), or correlated
  multivariate (needs a full transport Jacobian, not the per-axis diagonal).

For every accepted proper, continuous scalar prior the transform Jacobian is
analytic via the PIT identity `d delta / d z = phi_N(z) / p(delta)`, which needs
only `ppf` and `logpdf` — no differentiation through SciPy. The families
`nltiming` already supports (normal, uniform, log-uniform, truncated normal)
satisfy this today.

### 5. Charter statement (docs only)

Add a short statement to `README.md` and `UPSTREAM_INTEGRATION.md`: `nltiming`
provides the timing block (`J_z`), the prior transform, and the interactive
`z`-space fit; it does **not** own GP bases, `Phi` inference, spectra, or
correlated-noise likelihoods. This keeps the placement decision from reopening.

## Wiring

`TimingEvaluator` currently constructs no `ParameterSpace`. Changes 1-2 need
one. Two coherent options, not mutually exclusive:

- **Drive it from `TimingBinding`.** `NonLinearTimingModel.bind(pulsar)` already
  owns priors + `ParameterSpace` bound to a pulsar. The `z`-frame Jacobian and
  `z`-space fit fit naturally as binding methods (or an evaluator obtained from
  the binding), so the priors that define `z` are exactly those already
  configured for sampling. **Recommended as the primary path** — it keeps one
  source of truth for the coordinate.
- **Optional `space=` on the evaluator.** Let `TimingEvaluator` accept an
  optional `ParameterSpace`/`PriorBijector` (or a `priors=` spec it builds one
  from) for callers that use the interactive facade directly without a binding.

Physical-`delta` behavior is unchanged when no space is supplied; the `z`
capabilities simply become available once one is.

## API sketch

```python
# via the binding (recommended)
binding = ntm.bind(pulsar)

Jz, info = binding.jacobian(coord="z", method="auto")   # info: {method, tier}
fit = binding.fit(
    ["F0", "F1", "PB", "TASC"],
    coord="z",
    bounds="prior",            # or explicit per-parameter bounds
    trust_region=...,          # optional damping
    max_iter=...,
)
fit.best_fit.tier              # residual-evaluation fidelity actually used
```

The names are illustrative; the exact surface is chosen during implementation.

## Testing

- **Prior-transform Jacobian:** finite-difference `d delta / d z` against the
  analytic diagonal for every supported family; assert rejection errors for
  improper / PDF-only / discrete / singular / multivariate priors.
- **`J_z` composition:** compare `J_z` against a finite-difference of
  `residual_delta` w.r.t. `z`; check `autodiff` vs `reference` agreement within
  each backend's fidelity claim.
- **`z`-space fit:** recover injected parameters on a simulated pulsar; verify
  bound clipping and trust-region damping; confirm physical-`delta` and
  `z`-space fits agree where the transform is affine.
- **Fidelity tier:** assert the recorded tier matches the path taken and that
  `TimingCapabilities` advertises attainable tiers.
- Reuse the existing `tests/test_timing_evaluator.py` fixtures where possible.

## Relationship to downstream work

This proposal is the `nltiming` half ("Piece A") of the split described in the
MetaPulsar-side `feature_flexible_fit.md`. The flexible-`Phi` GP fit ("Piece B")
— joint reduced-rank solve, variance-group empirical Bayes, spectrum projection
— is deliberately **not** part of `nltiming`; it consumes `J_z` and the
`z`-space fit from here and lives in a downstream headless module. Implementing
this proposal unblocks that consumer without importing any GP, spectrum, or
frontend-noise machinery into `nltiming`.
