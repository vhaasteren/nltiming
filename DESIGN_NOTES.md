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
| NLT math (ParameterSpace, bijectors, whitening, priors, partition) | **nltiming** |
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

**Analytical marginalization** — the partition policy
(`sample=`/`analytically_marginalize=`, default linear nuisances +
position-only astrometry) and the improper-GP basis columns must match across
likelihood interfaces for the same pulsar and config. Both likelihood interfaces column-normalize the
marginalized basis (`whitening.normalized_basis`) — span-preserving under the
improper prior, required for float64 conditioning with the 1e40 weight.

**Frontend consistency (hard-won)** — the Enterprise likelihood interface must consume the
`TimingContext` (engine, partition, space, design matrix), never re-query
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
