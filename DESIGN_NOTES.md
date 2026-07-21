# nltiming design notes

Maintainer-facing notes: the ownership boundary, hard-won engineering caveats,
and the remaining upstream tracks. Not end-user documentation — see
[`README.md`](README.md) for that, and
[`TRANSFORMED_SPACE_FIT.md`](TRANSFORMED_SPACE_FIT.md) for the proposed
interactive `z`-space fit.

## Ownership boundary

`nltiming` owns the nonlinear-timing math, the timing engines, and both
likelihood interfaces. It is a satellite of the timing ecosystem, not a
reimplementation of it.

| Layer | Owner |
|-------|-------|
| NLT math (ParameterSpace, bijectors, whitening, priors, inference plan) | **nltiming** |
| Engine interface + engines (pint, libstempo, jug, vela) | **nltiming** |
| Multi-PTA composite engine (session assembly over row slices) | **nltiming** |
| Discovery + Enterprise likelihood interfaces, probabilistic model helpers and optional sampler recipes, run products | **nltiming** |
| JAX / nonlinear timing-engine primitives | **JUG** |
| Multi-PTA pulsar, session construction, data combination | pulsar (e.g. MetaPulsar) |
| GP bases, `Phi` inference, spectra, correlated-noise likelihoods | **Discovery / Enterprise** |

**Charter (do not cross):** `nltiming` provides the timing block, the prior
transform, and interactive timing evaluation/fitting. It does **not** own
Fourier/DM/chromatic/ECORR bases, `Phi` estimation, power-law or free-spectrum
projection, or correlated-noise likelihoods. Those stay with Discovery and
Enterprise; downstream quick-look GP tooling composes `nltiming` (timing) with
those likelihood interfaces (noise) rather than re-homing noise math here. See
`TRANSFORMED_SPACE_FIT.md` for the interactive-fit integration point this exposes to such
consumers.

## Engineering caveats

**Tempo2 / JUG nonlinear path** — default tempo2 nonlinear fits should use
`design_matrix_method="autodiff"`, not analytic columns for the full nonlinear
tangent. Pulsar residual agreement is at the picosecond tier on curated fixtures;
see JUG's agreement checks for production coverage.

**Prior semantics** — `prior_policy="fallback"` uses wide **uniform** cheat
boxes in delta space (`cheat_prior_scale × σ`), not Gaussians at the WLS
scale. Whitening / standardized coordinates are sampler reparameterization
only; they do not change the physical prior measure. Document identically in
every likelihood interface.

**Analytical marginalization** — the typed inference plan
(`TimingInference` / `TimingParameterPlan`: dispositions `sample` /
`marginalize_delta_flat` / `marginalize_z_prior`) and the GP basis columns must
match across likelihood interfaces for the same pulsar and config. Both
likelihood interfaces column-normalize the marginalized basis
(`whitening.normalized_basis`) — span-preserving under the improper prior,
required for float64 conditioning with the 1e40 weight. There is no
`sample=` / `analytically_marginalize=` / `transform=` constructor surface.

**Frontend consistency (hard-won)** — the Enterprise likelihood interface must consume the
`TimingContext` (engine, `plan`, space, design matrix), never re-query
`pulsar.timing_engine(...)` with partial kwargs: a engine rebuilt without
`subtract_tzr=False` evaluates a delay wrong by its own order of magnitude.
Both likelihood interfaces evaluate the delay via `residual_delta_jax` when the engine
provides it; the deprecated JUG NumPy path drifts from the JAX path
nonlinearly at ~2σ deltas, which is enough to visibly distort posteriors.
The three-way check (exact projected likelihood vs Discovery vs Enterprise on
one context) is the regression probe of record for this class of bug.

**JUG tempo2 zero-delta tolerance** — the relaxed 1e-7 s tolerance for the
known G1 reference gap is validation policy here; upstream consumers may
choose stricter or documented relaxed checks.

## Physical charts (Kepler↔Laplace) — the two-frame model

Distinct from the per-axis **prior charts** (`affine_normal` / `prior_pit`,
`ResolvedTimingAxis.prior_chart`), a **physical chart** (`kepler_laplace`) is a
multivariate reparameterization between the *sampling frame* (plan names,
priors, sampler) and the *engine frame* (canonical fitpars). The engine delay
model and its fitpar frame are untouched — sampling happens in
`EPS1/EPS2/TASC`, the engine still evaluates the exact DD/T2/DDH delay at
`ECC/OM/T0`. Every sampling→engine conversion flows through one seam,
`EngineDeltaMap` (`frames.py`), which consumes only the generic `PhysicalChart`
protocol (`physical_charts.py`). Charts are *slot-preserving*
(`EPS1`@ECC-slot, `EPS2`@OM-slot, `TASC`@T0-slot), so `fitpar_index` is valid in
both frames and every index-based path (Schur-WLS, metrics, improper GP) works
unchanged.

**Evaluation-point policy for the design matrix.** `ctx.design_matrix` is the
sampling-frame `M_s = M_e·B`; `ctx.engine_design_matrix` is the engine-frame
`M_e`. `M_e` is fixed at the engine reference and the analytic frame-change
block `B` is evaluated at the **reference** in production, so `M_s =
M_e(ref)·B(ref)` is a *consistent reference pair* — the hybrid `M_e(ref)·B(exp)`
is never formed (it is wrong at O(1) exactly where the chart's `1/e` rows move).
When the expansion moves and charted delta-flat axes exist, those `M_s` columns
are replaced by the **exact** composed-Jacobian columns (`jax.jvp` of
`residual_delta_jax(apply_charts(·))`), never the hybrid; a non-JAX engine keeps
them at the reference (a documented local approximation). `W_s`/`W_m` are always
exact because `build_linearization` differentiates the residual *through* the
sampling→engine composition.

**Prior semantics and the moved singularity.** There is **no chart-Jacobian term
in any posterior density**: charted-axis priors are declared on the sampling
frame (`KeplerLaplacePolicy.prior = "sampling_frame"`, recorded in the manifest)
via the existing per-axis machinery; the physical chart sits inside the
deterministic likelihood map. The induced measure is disclosed
(`dEPS1·dEPS2 = e·dECC·dOM`); an exact `prior = "pushforward"` mode is reserved.
Deliberate Kepler-axis priors (user or informative PINT) always win and demote
the chart. The chart *moves* the coordinate singularity into the decode (`atan2`
and `1/e`-scale intermediates near `ε = 0`) and introduces an `O(rate × PB)`
ω-branch seam discontinuity when secular terms are present; both are handled by
exact **activation** guards over the resolved EPS reachability rectangle (never
runtime/in-density guards) and certified at the composed-likelihood level.

**Engine capability (`§2.4`) — current state.** Candidacy consumes a normalized
`BinaryChartCapability` from the engine when present, else a conservative
name-search fallback (`_present_secular_terms`) that inspects the pulsar's PINT
model — including a binary-*type* check that flags GR-derived models (DDGR) whose
post-Keplerian rates are computed internally and are invisible to a name search.
Authoritative per-group `binary_chart_capability` on the engine adapters is
**deferred, not blocked** (review correction): the `PintEngine` already wraps the
PINT model (`from_contribution(model, toas, …)`), so a capability there needs
**no MetaPulsar change** — it can inspect the binary instance directly for
convention and active (explicit or derived) secular terms; the `JUG` engine needs
JUG-session introspection, still nltiming-side. The strengthened name-search +
DDGR fallback is the sanctioned interim. Separately, until an adapter both
implements the method **and** passes the §12.6 origin certification,
`origin_certified` stays `False`, so low-e binaries whose EPS box contains the
origin demote under `auto` (a conservative, honest default; the chart benefit is
unreachable on such pulsars until certification lands).

## Upstream tracks (parallel, non-blocking)

**JUG.** `JaxTimingState` / `export_jax_timing_state` now live in `jug.timing`
(landed on the `tempo2-dev` branch); `nltiming`'s jug engine imports them and
the previously vendored copy is retired. Remaining: merge `tempo2-dev` into
`MattTMiles/jug` main so the `[jug]` extra can point at a release rather than a
branch.

**Discovery.** The `make_uind` fix (empty ECORR bases, variable per-epoch TOA
counts) currently ships pulsar-side as
`metapulsar.discovery_compat.apply_discovery_compat_patches` and on a
`fix/make-uind-empty-basis` branch; the compatibility patch stays until it merges upstream.
Optionally register NLT parameter-name prior patterns in
`discovery.prior.priordict_standard`.

**enterprise_extensions.** A nonlinear-timing signal depending on `nltiming`,
positioned as the successor to `enterprise_extensions.timing.tm_delay`. Later
and optional; not a blocker.

**Vela / pyvela.** Vela.jl is now a first-class engine (`engines/vela.py`,
`VelaEngine`, `engines={"pint": "vela"}`) via pyvela's existing
`time_residuals` / `unscale_params` / `param_offsets`; no upstream change was
required. It is cross-validated against the JUG JAX path (binary params agree
to ~1e-11 s). Note it enters through the black-box juliacall boundary, so it is
not JAX-differentiable — usable for evaluation and cross-validation, not as an
autodiff engine.
