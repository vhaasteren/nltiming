# nltiming design notes

Maintainer-facing notes: the ownership boundary, hard-won engineering caveats,
and the remaining upstream tracks. Not end-user documentation — see
[`README.md`](README.md) for that, and
[`TRANSFORMED_SPACE_FIT.md`](TRANSFORMED_SPACE_FIT.md) for the proposed
interactive `z`-space fit.

## Ownership boundary

`nltiming` owns the nonlinear-timing math, engine-selection vocabulary,
backend-neutral engine support, and both likelihood interfaces. Concrete
backend adapters and composite assembly live in MetaPulsar. It is a satellite
of the timing ecosystem, not a reimplementation of it.

| Layer | Owner |
|-------|-------|
| NLT math (ParameterSpace, bijectors, whitening, priors, inference plan) | **nltiming** |
| Engine protocols, selection vocabulary (`engine_config`), validators / `LinearModel` (`engine_support`) | **nltiming** |
| Backend adapters (pint, libstempo, jug, vela) + multi-PTA composite | **MetaPulsar** (`metapulsar.engines`) |
| Discovery + Enterprise likelihood interfaces, probabilistic model helpers and optional sampler recipes, run products | **nltiming** |
| JAX / nonlinear timing-engine primitives | **JUG** |
| Multi-PTA pulsar, session construction, data combination | **MetaPulsar** |
| GP bases, `Phi` inference, spectra, correlated-noise likelihoods | **Discovery / Enterprise** |

`nltiming` never imports the `metapulsar` package. At runtime it calls a supplied
`TimingPulsar.timing_engine(...)` which returns a MetaPulsar-owned engine.

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
`derivative_method="autodiff"`, not analytic columns for the full nonlinear
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

**MarginalBasisFrame (not a chart).** When a Kepler triple is fully
`marginalize_delta_flat` (no sampled axis), the physical chart stays off
(`reason=no_sampled_axis`) and names/dispositions never change. Under
`KeplerLaplacePolicy.marginal_basis_frame="auto"`, a `MarginalBasisFrame`
reuses the same reference-local `B` block to recondition only the analytically
marginalized design-matrix columns (`M_s = M_e @ B` on those slots). The
improper-flat marginal shifts by a constant (`log_abs_det_b`, and after column
normalization the operative `log_volume_offset`); posteriors within a run are
unaffected. z-prior or mixed dispositions keep the engine basis and every
existing chart/z-prior guard. Module constant `SUPPORTS_CONVERSION_METADATA`
gates MetaPulsar Case-D `conversion_metadata()` probing in `for_pulsar`.

**Evaluation-point policy for the design matrix.** `ctx.design_matrix` is the
sampling-frame `M_s = M_e·B`; `ctx.engine_design_matrix` is the engine-frame
`M_e`. `M_e` is fixed at the engine reference and the analytic frame-change
block `B` is evaluated at the **reference** in production, so `M_s =
M_e(ref)·B(ref)` is a *consistent reference pair* — the hybrid `M_e(ref)·B(exp)`
is never formed (it is wrong at O(1) exactly where the chart's `1/e` rows move).
When the expansion moves and charted delta-flat axes exist, those `M_s` columns
are replaced by the **exact** composed-Jacobian columns (`jax.jvp` of
`residual_delta_jax(apply_charts(·))`), never the hybrid; a non-JAX engine keeps
them at the reference (a documented local approximation). `W_s`/`W_m` follow
`derivative_method` with the same meaning as `design_matrix`: analytic mode is
`M_s ∂δ/∂z` (reference `M_s`, prior Jacobian at the expansion); autodiff mode
is `jacfwd` of `residual_delta_jax` through the sampling→engine composition.
There is no finite-difference fallback; autodiff on a non-JAX engine is
rejected.

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

**`fw10_absorbed` (sampling-path sibling).** For DDH blocks that free the full
orthometric sextet including `STIGMA`, `FW10AbsorbedChart` (`fw10_absorbed.py`)
is tried *before* `kepler_laplace` at activation. It is the exact inverse of the
Freire & Wex absorbed-gauge transfer (dot-free; PB frozen at `pb_ref`, never a
`dependency_slot`). Disposition rule for v1: all six engine axes must be
`sample`; any mix → inactive (`unsupported_disposition_mix`). Secular dots →
`secular_terms_present`; sampled PB → `dependency_sampled`. Manifest:
`fw10_absorbed_chart`. Native DDH sampling without the chart remains correct;
the chart is conditioning for gradient samplers.

**Engine capability (`§2.4`) — the authoritative chain (landed).** The source of
truth for binary physics facts is the **timing backend**, and the capability is
built by a layered chain, with the name-search fallback as genuine last resort:

1. **JUG owns the facts.** `jug.fitting.binary_delay_plan.binary_chart_facts`
   (a general JUG API, no nltiming/MetaPulsar knowledge) resolves the binary
   (`T2 → DD/ELL1/DDK`; GR-derived DDGR via the original `BINARY`) and reports
   `{convention_family, epoch_shift_exact, secular_terms}`.
2. **Leaf engines translate.** `JugEngine` caches those facts at
   `from_contribution` and maps them → `BinaryChartCapability` (adding the
   nltiming-owned `origin_certified`/`supports_domain`). `PintEngine` reads the
   PINT model directly (PINT exposes the binary type + PK params on the model),
   so it needs no separate facts helper. **No MetaPulsar change** for leaf
   engines.
3. **Composite forwards per group.** `PulsarTimingEngine.binary_chart_capability`
   (`engines/composite.py`) delegates to the contribution owning a suffix,
   returning `None` on no-owner / missing-method / shared-binary disagreement —
   nltiming-side, zero MetaPulsar changes.
4. **Fallback is last-resort only.** `_present_secular_terms` (pulsar/name
   search + DDGR binary-type check) covers test doubles and engines that
   implement neither the facts query nor the capability. It is **no longer** the
   production source of truth on the JUG path.

Separately, until an adapter's backend passes the §12.6 real-DD origin
certification, `origin_certified` stays `False`, so low-e binaries whose EPS box
contains the origin demote under `auto` (a conservative, honest default; the
chart benefit is unreachable on such pulsars until certification lands). Note:
nltiming now depends on JUG's `binary_chart_facts` — that jug tip must be
upstreamed for clean checkouts.

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
