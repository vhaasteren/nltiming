"""NumPyro model helpers for nonlinear timing contexts.

Owns everything NumPyro-specific: the joint timing sample site and its
density (:func:`sample_timing`, via the private coordinate-distribution
builder :func:`_sample_timing_coord`), physical value post-processing
(:func:`record_physical_postprocess`), the standard Discovery model closure
(:func:`model`), and an optional NUTS convenience recipe with
init-at-reference (:func:`nuts`).

All functions take a :class:`~nltiming.nonlinear_timing_model.TimingContext`
(from ``NonLinearTimingModel.for_pulsar``); none of this leaks into the model config.
Sampler construction (NUTS/MCMC) is an opinionated convenience, not the
canonical integration path — see the module README for the native NumPyro and
Discovery ``makesampler_nuts`` workflows.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def ensure_x64() -> None:
    """Enable JAX float64 and fail loudly if it cannot take effect.

    Timing deltas need float64; silently sampling in float32 produces wrong
    posteriors. Call this before building likelihoods; it is invoked by
    :func:`nuts` as a safety net.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if jnp.zeros(1).dtype != jnp.float64:
        raise RuntimeError(
            "JAX x64 mode could not be enabled (arrays still default to "
            "float32). Set JAX_ENABLE_X64=1 or call ensure_x64() before any "
            "JAX arrays are created."
        )


def _static_pullback_distribution_cls():
    """Build (once) the JAX-safe static timing pullback distribution class.

    Defined lazily so importing ``nltiming`` never requires numpyro/JAX. The
    class implements the exact pullback of ``z ~ N(0, I)`` under the affine map
    ``z = C x + z0`` — ``log p(x) = -1/2 ||C x + z0||^2 + log|det C| + const`` —
    **without** forming ``C.T @ C`` (which squares ``cond(C)``; the pinned
    production commit did exactly that, §3). ``C`` is lower triangular, so
    ``sample`` maps a standard-normal draw back with one triangular solve.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular
    from numpyro.distributions import Distribution, constraints

    class StaticTimingPullback(Distribution):
        arg_constraints: dict = {}
        support = constraints.real_vector
        reparametrized_params: list = []

        def __init__(self, C, z0, logabsdet, *, validate_args=None):
            self._C = jnp.asarray(C)
            self._z0 = jnp.asarray(z0)
            self._logabsdet = jnp.asarray(logabsdet)
            self._ndim = int(self._C.shape[-1])
            super().__init__(
                batch_shape=(),
                event_shape=(self._ndim,),
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            eps = jax.random.normal(
                key, tuple(sample_shape) + self.event_shape, dtype=self._C.dtype
            )
            rhs = eps - self._z0

            def _solve(vec):
                return solve_triangular(self._C, vec, lower=True)

            flat = rhs.reshape((-1, self._ndim))
            solved = jax.vmap(_solve)(flat)
            return solved.reshape(rhs.shape)

        def log_prob(self, value):
            z = jnp.matmul(value, jnp.swapaxes(self._C, -1, -2)) + self._z0
            quad = jnp.sum(z * z, axis=-1)
            norm = 0.5 * self._ndim * jnp.log(2.0 * jnp.pi)
            return -0.5 * quad - norm + self._logabsdet

    return StaticTimingPullback


def _sample_timing_coord(ctx, *, coord: str | None = None):
    """Sample the joint timing site with a distribution matching ``coord``.

    ``x`` uses a custom static pullback distribution (:func:`
    _static_pullback_distribution_cls`) whose ``log_prob`` equals
    ``space.logprior_coord`` exactly but never builds ``C.T @ C``; ``z`` uses a
    standard normal; both carry their own normalized ``log_prob`` and must not
    add a second prior factor. ``delta`` has no closed-form NumPyro
    distribution (an unbounded physical prior is not a location-scale family in
    general), so it samples an ``ImproperUniform`` placeholder and adds the
    physical-prior density as an explicit factor.
    """
    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist
    from numpyro.distributions import constraints

    coord = ctx.coord if coord is None else coord
    site = ctx.latent_name_for_coord(coord)
    ndim = len(ctx.sampled)

    if coord == "x":
        C = jnp.asarray(ctx.space.linear.C)
        z0 = jnp.asarray(ctx.space.linear.z0)
        logabsdet = jnp.asarray(ctx.space.linear.logabsdet)
        pullback = _static_pullback_distribution_cls()
        return numpyro.sample(site, pullback(C, z0, logabsdet))

    if coord == "z":
        return numpyro.sample(site, dist.Normal(0.0, 1.0).expand((ndim,)).to_event(1))

    if coord == "delta":
        q = numpyro.sample(site, dist.ImproperUniform(constraints.real, (), (ndim,)))
        numpyro.factor(
            f"{site}_logprior", ctx.space.logprior_coord(q, jnp, coord="delta")
        )
        return q

    raise ValueError(f"Unsupported coord: {coord}")


def sample_timing(
    ctx,
    params: Mapping[str, Any],
    *,
    coord: str | None = None,
) -> Mapping[str, Any]:
    """Sample the joint timing site and inject per-parameter delta values.

    Adds one timing sample site (``ctx.latent_name_for_coord()``) whose density
    matches ``coord`` (see :func:`_sample_timing_coord`), one JAX-safe
    ``numpyro.deterministic`` per sampled parameter carrying its engine-native
    offset (``{prefix}_{fitpar}_delta``), and returns ``params`` extended with
    those same engine-facing delta keys (``{pulsar}_{model}_{fitpar}``) for
    the likelihood.
    """
    sampled = ctx.plan.sampled
    if not sampled:
        return params

    coord = ctx.coord if coord is None else coord
    if coord not in {"delta", "z", "x"}:
        raise ValueError(f"Unsupported coord: {coord}")

    import jax.numpy as jnp
    import numpyro

    q = _sample_timing_coord(ctx, coord=coord)
    delta = ctx.space.delta_from_coord(q, jnp, coord=coord)

    out = dict(params)
    for i, name in enumerate(sampled):
        numpyro.deterministic(f"{ctx.name_stem}_{name}_delta", delta[i])
        out[f"{ctx.name_stem}_{name}"] = delta[i]
    return out


def record_physical_postprocess(
    ctx,
    params: Mapping[str, Any],
    *,
    scope: str = "timing",
    coord: str | None = None,
) -> dict[str, Any]:
    """Compute ``{prefix}_{fitpar}_theta_{native,display}`` physical values.

    Pure post-processing over concrete (un-traced) parameter values —
    posterior draws or a reconstructed parameter dict. This is not part of a
    NumPyro trace and must not call ``numpyro.deterministic``; callers that
    want a deterministic site should do so themselves inside an active trace.
    Returns a flat dict keyed by the site names above.
    """
    coord_was_explicit = coord is not None
    if coord is not None and coord not in {"delta", "z", "x"}:
        raise ValueError("coord must be one of {'delta', 'z', 'x'}")
    if coord is None:
        coord = ctx.coord

    if scope == "all":
        raise NotImplementedError("scope='all' is deferred")
    if scope != "timing":
        raise ValueError("scope must be one of {'timing', 'all'}")

    sampled = ctx.plan.sampled
    if not sampled:
        return {}

    delta = ctx.delta_from_params(
        params,
        coord=coord,
        coord_explicit=coord_was_explicit,
    )
    name_stem = ctx.name_stem
    theta_native = ctx.space.to_physical(delta[None, :], units="native", coord="delta")
    theta_display = ctx.space.to_physical(
        delta[None, :], units="display", coord="delta"
    )
    out: dict[str, Any] = {}
    for name in sampled:
        out[f"{name_stem}_{name}_theta_native"] = theta_native[name][0]
        out[f"{name_stem}_{name}_theta_display"] = theta_display[name][0]
    return out


def _flatten_chain_major(arr: np.ndarray, *, grouped: bool, n_rows: int) -> np.ndarray:
    """Merge a leading (chain, draw) pair into one chain-major row axis."""
    arr = np.asarray(arr)
    if grouped:
        return arr.reshape((n_rows,) + arr.shape[2:])
    return arr


def samples_to_frame(samples: Mapping[str, Any], ctx):
    """Flatten NumPyro samples into a DataFrame and append exact timing decodes.

    Accepts both ordinary ``(draw, ...)`` and grouped ``(chain, draw, ...)``
    sample arrays (grouping is detected from the joint timing site, which
    always has event shape ``(ndim,)``); grouped arrays flatten to rows in
    chain-major order, matching :func:`timing_draws`.

    Column naming:

    - scalar non-timing sites: unchanged (e.g. ``red_noise_gamma``);
    - vector non-timing sites: ``site[0]``, ``site[1]``, ...;
    - latent timing columns: ``{latent_name_for_coord()}[0]``, ``[1]``, ...;
    - timing offsets: ``{prefix}_{fitpar}_delta``;
    - exact native/display values: ``{prefix}_{fitpar}_theta_{native,display}``,
      always recomputed through ``ParameterSpace.to_physical`` — never trusted
      from a traced value.

    NumPyro factor sites (which never appear in ``mcmc.get_samples()`` output)
    are not included; a stray ``*_logprior`` key is skipped defensively.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "samples_to_frame requires pandas; install the 'discovery' extra "
            "(pip install nltiming[discovery])"
        ) from exc

    site = ctx.latent_name_for_coord()
    if site not in samples:
        raise KeyError(f"{site!r} (joint timing site) not found in samples")
    timing = np.asarray(samples[site], dtype=float)
    if timing.ndim == 3:
        grouped, n_rows = True, timing.shape[0] * timing.shape[1]
    elif timing.ndim == 2:
        grouped, n_rows = False, timing.shape[0]
    else:
        raise ValueError(f"unexpected shape for {site!r}: {timing.shape}")
    timing = _flatten_chain_major(timing, grouped=grouped, n_rows=n_rows)

    columns: dict[str, np.ndarray] = {}
    for name, value in samples.items():
        if name.endswith("_logprior"):
            continue
        flat = _flatten_chain_major(np.asarray(value), grouped=grouped, n_rows=n_rows)
        if flat.ndim == 1:
            columns[name] = flat
        else:
            flat = flat.reshape(n_rows, -1)
            for i in range(flat.shape[1]):
                columns[f"{name}[{i}]"] = flat[:, i]

    delta_keys = [f"{ctx.name_stem}_{name}_delta" for name in ctx.sampled]
    if all(key in samples for key in delta_keys):
        delta = np.stack([columns[key] for key in delta_keys], axis=1)
    else:
        delta = np.stack(
            [
                np.asarray(ctx.space.delta_from_coord(row, np, coord=ctx.coord))
                for row in timing
            ],
            axis=0,
        )
        for i, key in enumerate(delta_keys):
            columns[key] = delta[:, i]

    theta_native = ctx.space.to_physical(delta, units="native", coord="delta")
    theta_display = ctx.space.to_physical(delta, units="display", coord="delta")
    for name in ctx.sampled:
        columns[f"{ctx.name_stem}_{name}_theta_native"] = theta_native[name]
        columns[f"{ctx.name_stem}_{name}_theta_display"] = theta_display[name]

    return pd.DataFrame(columns)


def model(
    likelihood,
    ctx,
    *,
    priors: Mapping[str, Any] | None = None,
    fixed: Mapping[str, float] | None = None,
) -> Callable[[], None]:
    """Build the standard NumPyro model for a Discovery likelihood.

    Non-timing likelihood parameters are sampled from uniform priors resolved
    via ``discovery.prior.getprior_uniform(par, priors)``, unless pinned in
    ``fixed``. Timing parameters enter through :func:`sample_timing`.

    Args:
        likelihood: Discovery ``PulsarLikelihood`` (or anything exposing
            ``logL`` with a ``params`` attribute).
        ctx: ``TimingContext`` for the pulsar in the likelihood.
        priors: prior overrides / noise dictionary for non-timing parameters.
        fixed: parameter values held constant (not sampled).

    Raises:
        ValueError: ``likelihood.logL.params`` has duplicate names, is missing
            a ``ctx.delay_keys`` entry (ctx and likelihood were not
            assembled together), contains the joint latent site
            (``ctx.latent_name_for_coord()`` — Discovery consumes the derived
            delay keys, never the latent coordinate), or ``fixed`` pins a
            timing parameter (owned by the ctx, not the caller).
        TypeError: a ``fixed`` value is not numeric.
    """
    import numpyro
    from numpyro import distributions as dist

    logL_params = list(likelihood.logL.params)
    seen: set[str] = set()
    dupes = sorted({par for par in logL_params if par in seen or seen.add(par)})
    if dupes:
        raise ValueError(f"duplicate likelihood parameter names: {dupes}")

    site_name = ctx.latent_name_for_coord()
    if site_name in logL_params:
        raise ValueError(
            f"{site_name!r} is the joint latent timing site and must not "
            "appear in likelihood.logL.params; Discovery consumes the "
            "derived delay keys, not the latent coordinate"
        )
    missing_delay_keys = [key for key in ctx.delay_keys if key not in logL_params]
    if missing_delay_keys:
        raise ValueError(
            "ctx and likelihood were not assembled together: "
            f"missing delay keys in likelihood.logL.params: {missing_delay_keys}"
        )

    timing_keys = set(ctx.timing_param_keys())
    fixed_in = dict(fixed or {})
    bad_timing_fixed = sorted(timing_keys & fixed_in.keys())
    if bad_timing_fixed:
        raise ValueError(
            f"fixed cannot pin timing parameters (owned by the ctx): {bad_timing_fixed}"
        )
    fixed_params: dict[str, float] = {}
    for key, value in fixed_in.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"fixed[{key!r}] must be numeric, got {type(value).__name__}"
            )
        fixed_params[key] = float(value)

    priordict = dict(priors or {})
    free = [
        par for par in ctx.non_timing_params(logL_params) if par not in fixed_params
    ]
    if free:
        from discovery import prior as ds_prior

        bounds = {par: tuple(ds_prior.getprior_uniform(par, priordict)) for par in free}

    def nlt_model() -> None:
        params = dict(fixed_params)
        for par in free:
            params[par] = numpyro.sample(par, dist.Uniform(*bounds[par]))
        numpyro.factor("ll", likelihood.logL(sample_timing(ctx, params)))

    nlt_model.to_df = lambda samples: samples_to_frame(samples, ctx)
    return nlt_model


def _resolve_reference_noise(reference_noise, pulsar):
    """Resolve the ``reference_noise=`` argument of :func:`joint_model` into a
    discovery frozen-solve operator (a ``_FrozenSolve``-like with ``.solve`` and
    ``.description``).

    - ``"toa_errors"`` -> ``discovery.transport.reference_noise(pulsar)`` (the
      diagonal ``toaerrs**2`` default, §4.4 / cleanup §5.5);
    - an object already exposing ``.solve`` -> passed through (e.g. a
      ``reference_noise_frozen(kernel, params0)`` for a pinned noisedict).
    """
    if hasattr(reference_noise, "solve"):
        return reference_noise
    if reference_noise == "toa_errors":
        from discovery import transport as dst

        return dst.reference_noise(pulsar)
    raise ValueError(
        "reference_noise must be 'toa_errors' or a discovery frozen-solve "
        f"object (reference_noise_frozen(...)); got {reference_noise!r}"
    )


def build_joint_transport(
    likelihood,
    ctx,
    *,
    reference_noise="toa_errors",
    center=True,
    softclip_zmax=None,
    global_gp=None,
    psr_slot=None,
    npsr=None,
    center_extsignals=None,
):
    """Build the per-pulsar dynamic :class:`discovery.transport.Transport` for a
    joint full-basis run (§6.4, §7).

    Blocks, in order: the external timing block (``array_block`` on the local
    timing waveform Jacobian ``W_z`` with identity ``z`` precision), then one
    ``gp_block`` per sampled intrinsic GP, then — for a correlated global GP —
    the per-pulsar ``globalgp_curn_block`` (the diagonal inverse-marginal
    conditioner view of the dense cross-pulsar prior; §4.3 rule 3, §7). Centering
    is the damped local posterior Newton step at the reference residual, with an
    optional timing soft-clamp and, for a deterministic ExtSignal (e.g. CW),
    template-subtracted centering (§4.4).

    ``softclip_zmax`` is **off by default**. With matched reference noise the
    centering ``mu`` equals the conditional mode ``q_hat`` and the joint density
    factorizes exactly as ``N(xi; 0, I) x p_marginal(eta)``; a clamp makes
    ``mu != q_hat``, adding an eta-dependent ``-1/2 ||L^T(mu - q_hat)||^2`` to
    the ``xi = 0`` slice. When the timing centering is large (e.g. zeroed
    inter-PTA JUMPs), that term reaches ~1e5 and destroys the hyperparameter
    geometry (measured on IPTA J1640+2224: hyper curvature ~1e4-1e6 with a
    clamp at 4, vs the marginal's ~1e2 without). Clamp only when a PIT-bounded
    coordinate genuinely saturates, and then with a generous ``zmax``.
    """
    from discovery import transport as dst

    block_t = ctx.local_timing_block()
    blocks = [
        dst.array_block(
            block_t.basis,
            index={block_t.joint_site: slice(0, block_t.dimension)},
            conditioner_precision=block_t.prior_precision,
            name="timing",
        )
    ]
    blocks += [dst.gp_block(gp) for gp in likelihood.sampled_gps]

    if global_gp is not None:
        if psr_slot is None or npsr is None:
            raise ValueError(
                "global_gp requires psr_slot and npsr (the pulsar's index in the "
                "global GP and the total pulsar count)"
            )
        blocks.append(dst.globalgp_curn_block(global_gp, psr_slot, npsr))

    return dst.Transport(
        blocks,
        reference_noise=_resolve_reference_noise(reference_noise, ctx.pulsar),
        reference_residual=np.asarray(ctx.pulsar.residuals, dtype=float),
        center=center,
        softclip=(
            {"timing": float(softclip_zmax)}
            if (center and softclip_zmax is not None)
            else None
        ),
        center_extsignals=center_extsignals,
        psr_slot=(psr_slot if center_extsignals else None),
    )


def gw_residual_delay(global_gp, psr_slot):
    """A discovery delay that subtracts one pulsar's global-GP waveform from its
    residual **without** a prior (§7).

    The global (HD) GW coefficients are sampled by the transport and conditioned
    per-pulsar, but their prior is the *dense* cross-pulsar Gaussian added once in
    the joint model — so the GW enters each pulsar's ``clogL`` only through the
    data term. Add this to the pulsar's ``PulsarLikelihood`` alongside the
    intrinsic signals; its ``.params`` is the single global-GP coefficient key
    ``{pulsar}_{name}_coefficients(k)`` that the transport injects.
    """
    import jax.numpy as jnp

    F = jnp.asarray(np.asarray(global_gp.Fs[psr_slot], dtype=float))
    key = list(global_gp.index)[psr_slot]

    def delay(params):
        return F @ params[key]

    delay.params = [key]
    return delay


def global_gp_logprior(coeff_flat, global_gp, params, xp=None):
    """Exact dense cross-pulsar coefficient prior for a global (HD) GP (§7).

    ``-½ cᵀ Φ_gw⁻¹ c - ½ log|Φ_gw|`` with the Kronecker-structured inverse
    ``Φ_gw⁻¹ = ORF⁻¹ ⊗ diag(φ⁻¹)`` and log-determinant taken straight from
    ``global_gp.Phi_inv`` (cost ``O(n_psr² · k_gw)`` — no dense factorization).
    ``coeff_flat`` is the pulsar-major stack of per-pulsar GW coefficients, in
    the same order as ``global_gp.index``.
    """
    if xp is None:
        import jax.numpy as xp
    phi_inv, logdet = global_gp.Phi_inv(params)
    c = xp.asarray(coeff_flat)
    return -0.5 * (c @ (xp.asarray(phi_inv) @ c)) - 0.5 * xp.asarray(logdet)


def joint_model(
    likelihood,
    ctx,
    *,
    reference_noise: str = "toa_errors",
    center: bool = True,
    softclip_zmax: float | None = None,
    center_extsignals=None,
    priors: Mapping[str, Any] | None = None,
    fixed: Mapping[str, float] | None = None,
) -> Callable[[], None]:
    """Build the joint full-basis NumPyro model (§6.4).

    One standard-normal ``xi`` site is mapped through the dynamic transport to
    the joint coordinate ``q = (z, coefficients)``; ``z`` decodes to the exact
    nonlinear timing residual via the PIT bijector and the GP coefficient blocks
    feed discovery's residual-form ``clogL`` directly. The exact timing prior
    ``-½‖z‖²``, the transport log-Jacobian, and the ``N(0, I)`` base-measure
    cancellation are all added explicitly (§4.5).

    ``center_extsignals`` is an optional list of deterministic ExtSignals (e.g. a
    CW model) used for template-subtracted centering (§4.4); the same ExtSignal
    must also be subtracted from the residual in ``likelihood``.

    Requires ``ctx`` to be built with ``whitening=None`` (or otherwise carry an
    identity static affine layer): the dynamic transport is the ONE affine layer
    (§5.5). The existing static :func:`model` builder is untouched.
    """
    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist

    from ..metric import assert_static_layer_identity

    if not ctx.sampled:
        raise ValueError("joint_model requires at least one sampled timing parameter")
    assert_static_layer_identity(ctx.space)

    transport = build_joint_transport(
        likelihood,
        ctx,
        reference_noise=reference_noise,
        center=center,
        softclip_zmax=softclip_zmax,
        center_extsignals=center_extsignals,
        psr_slot=0 if center_extsignals else None,
    )

    clogL_params = list(likelihood.clogL.params)
    seen: set[str] = set()
    dupes = sorted({p for p in clogL_params if p in seen or seen.add(p)})
    if dupes:
        raise ValueError(f"duplicate likelihood parameter names: {dupes}")

    coeff_keys = [k for k in transport.index if k != ctx.joint_site]
    missing_delay = [k for k in ctx.delay_keys if k not in clogL_params]
    if missing_delay:
        raise ValueError(
            "ctx and likelihood were not assembled together: missing delay keys "
            f"in likelihood.clogL.params: {missing_delay}"
        )
    missing_coeff = [k for k in coeff_keys if k not in clogL_params]
    if missing_coeff:
        raise ValueError(
            "transport GP coefficient keys absent from likelihood.clogL.params "
            f"(likelihood and transport disagree): {missing_coeff}"
        )

    owned = set(ctx.delay_keys) | set(coeff_keys)
    fixed_params: dict[str, float] = {}
    for key, value in dict(fixed or {}).items():
        if key in owned:
            raise ValueError(
                f"fixed cannot pin timing/coefficient parameters (owned by the "
                f"joint model): {key!r}"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(f"fixed[{key!r}] must be numeric")
        fixed_params[key] = float(value)

    priordict = dict(priors or {})
    # Sorted so hyper_sites (and the sample loop) have a deterministic order:
    # marginalized D20 / Enterprise E8 and the §10 mass matrix bind to it.
    free = sorted(p for p in clogL_params if p not in owned and p not in fixed_params)
    if free:
        from discovery import prior as ds_prior

        bounds = {p: tuple(ds_prior.getprior_uniform(p, priordict)) for p in free}

    xi_site = f"{ctx.name_stem}_joint_xi"
    dim = transport.dimension
    sampled_all = ctx.sampled_all
    name_stem = ctx.name_stem
    joint_site = ctx.joint_site

    def nlt_joint_model() -> None:
        params = dict(fixed_params)
        for par in free:
            params[par] = numpyro.sample(par, dist.Uniform(*bounds[par]))

        xi = numpyro.sample(xi_site, dist.Normal(0.0, 1.0).expand([dim]).to_event(1))
        q, ldj = transport.apply(params, xi)
        parts = transport.split(q)

        z = parts[joint_site]
        delta = ctx.space.delta_from_z(z, jnp)
        for i, fitpar in enumerate(sampled_all):
            params[f"{name_stem}_{fitpar}"] = delta[i]
            numpyro.deterministic(f"{name_stem}_{fitpar}_delta", delta[i])
        for key in coeff_keys:
            params[key] = parts[key]

        logtarget = likelihood.clogL(params)  # data + exact GP priors
        logtarget = logtarget - 0.5 * jnp.sum(z * z)  # exact timing prior
        numpyro.factor("nlt_joint", logtarget + ldj + 0.5 * jnp.sum(xi * xi))

    nlt_joint_model.transport = transport
    nlt_joint_model.xi_site = xi_site
    nlt_joint_model.hyper_sites = tuple(free)
    nlt_joint_model.to_df = lambda samples: joint_samples_to_frame(samples, ctx)
    return nlt_joint_model


def _joint_pulsar_entry(
    likelihood,
    ctx,
    *,
    reference_noise,
    center,
    softclip_zmax,
    global_gp=None,
    psr_slot=None,
    npsr=None,
    center_extsignals=None,
):
    """Resolve one pulsar's joint pieces: transport, owned keys, clogL params."""
    from ..metric import assert_static_layer_identity

    if not ctx.sampled:
        raise ValueError(
            f"joint model requires >=1 sampled timing parameter for pulsar "
            f"{ctx.pulsar.name}"
        )
    assert_static_layer_identity(
        ctx.space, context=f"joint sampling ({ctx.pulsar.name})"
    )
    transport = build_joint_transport(
        likelihood,
        ctx,
        reference_noise=reference_noise,
        center=center,
        softclip_zmax=softclip_zmax,
        global_gp=global_gp,
        psr_slot=psr_slot,
        npsr=npsr,
        center_extsignals=center_extsignals,
    )
    clogL_params = list(likelihood.clogL.params)
    coeff_keys = [k for k in transport.index if k != ctx.joint_site]
    missing_delay = [k for k in ctx.delay_keys if k not in clogL_params]
    if missing_delay:
        raise ValueError(
            f"pulsar {ctx.pulsar.name}: ctx and likelihood not assembled "
            f"together; missing delay keys in clogL.params: {missing_delay}"
        )
    missing_coeff = [k for k in coeff_keys if k not in clogL_params]
    if missing_coeff:
        raise ValueError(
            f"pulsar {ctx.pulsar.name}: transport GP coefficient keys absent "
            f"from clogL.params: {missing_coeff} (a global-GP block needs "
            f"gw_residual_delay(global_gp, psr_slot) added to the likelihood)"
        )
    gw_key = None
    if global_gp is not None:
        gw_key = list(global_gp.index)[psr_slot]
        if gw_key not in coeff_keys:
            raise ValueError(
                f"pulsar {ctx.pulsar.name}: global-GP coefficient key {gw_key!r} "
                f"is not a transport block key {coeff_keys}"
            )
    return {
        "likelihood": likelihood,
        "ctx": ctx,
        "transport": transport,
        "coeff_keys": coeff_keys,
        "clogL_params": clogL_params,
        "owned": set(ctx.delay_keys) | set(coeff_keys),
        "xi_site": f"{ctx.name_stem}_joint_xi",
        "gw_key": gw_key,
    }


def joint_model_multi(
    likelihoods,
    ctxs,
    *,
    reference_noise: str = "toa_errors",
    center: bool = True,
    softclip_zmax: float | None = None,
    global_gp=None,
    center_extsignals=None,
    priors: Mapping[str, Any] | None = None,
    fixed: Mapping[str, float] | None = None,
) -> Callable[[], None]:
    """Joint full-basis model over several pulsars (§6.4, §7).

    One :class:`discovery.transport.Transport` and one ``xi`` site per pulsar —
    per-pulsar timing widths differ (ragged is expected) and the joint path never
    routes through ``ArrayTransport``. Per-pulsar ``clogL`` terms sum;
    hyperparameters shared across pulsars (e.g. a CURN amplitude that every
    pulsar's GP names identically) are declared **once** and flow into every
    pulsar's transport and likelihood. Each pulsar must be built with an identity
    static affine layer (``whitening=None``).

    ``global_gp`` (a discovery ``makeglobalgp_fourier`` object, HD or any ORF)
    adds a correlated GW block: each pulsar's transport conditions its GW
    coefficients through the per-pulsar ``globalgp_curn_block`` diagonal, the GW
    waveform is subtracted from each residual by ``gw_residual_delay`` (which the
    caller adds to each ``PulsarLikelihood``), and the **exact dense** cross-pulsar
    coefficient prior is added once via :func:`global_gp_logprior`. Its pulsar
    order must match ``likelihoods``/``ctxs``.

    ``center_extsignals`` is an optional per-pulsar sequence (one entry per
    pulsar; ``None`` to skip that pulsar) of deterministic ExtSignals used for
    template-subtracted centering (§4.4). The same ExtSignal must also be
    subtracted from that pulsar's residual in its ``PulsarLikelihood``.
    """
    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist

    likelihoods = list(likelihoods)
    ctxs = list(ctxs)
    if len(likelihoods) != len(ctxs):
        raise ValueError("likelihoods and ctxs must have equal length")
    if not likelihoods:
        raise ValueError("joint_model_multi requires at least one pulsar")
    npsr = len(likelihoods)
    if global_gp is not None and len(global_gp.index) != npsr:
        raise ValueError(
            f"global_gp spans {len(global_gp.index)} pulsars but "
            f"{npsr} likelihoods were given"
        )
    if center_extsignals is not None and len(center_extsignals) != npsr:
        raise ValueError(
            f"center_extsignals must have one entry per pulsar ({npsr}); "
            f"got {len(center_extsignals)}"
        )

    entries = [
        _joint_pulsar_entry(
            lk,
            ctx,
            reference_noise=reference_noise,
            center=center,
            softclip_zmax=softclip_zmax,
            global_gp=global_gp,
            psr_slot=(i if global_gp is not None else None),
            npsr=(npsr if global_gp is not None else None),
            center_extsignals=(
                center_extsignals[i] if center_extsignals is not None else None
            ),
        )
        for i, (lk, ctx) in enumerate(zip(likelihoods, ctxs))
    ]

    xi_sites = [e["xi_site"] for e in entries]
    if len(set(xi_sites)) != len(xi_sites):
        raise ValueError(
            f"duplicate pulsar/model name_stem across the joint set (xi sites "
            f"collide): {xi_sites}"
        )

    owned = set().union(*[e["owned"] for e in entries])
    all_clogL = set().union(*[set(e["clogL_params"]) for e in entries])
    if global_gp is not None:
        # The global-GP hyperparameters drive the transport conditioner and the
        # dense prior but never appear in any per-pulsar clogL (the GW enters
        # prior-free, as a delay), so add them explicitly to the free set.
        all_clogL |= set(getattr(global_gp.Phi_inv, "params", []))

    fixed_params: dict[str, float] = {}
    for key, value in dict(fixed or {}).items():
        if key in owned:
            raise ValueError(f"fixed cannot pin a timing/coefficient key: {key!r}")
        if not isinstance(value, (int, float)):
            raise TypeError(f"fixed[{key!r}] must be numeric")
        fixed_params[key] = float(value)

    priordict = dict(priors or {})
    free = sorted(all_clogL - owned - set(fixed_params))
    if free:
        from discovery import prior as ds_prior

        bounds = {p: tuple(ds_prior.getprior_uniform(p, priordict)) for p in free}

    def nlt_joint_model_multi() -> None:
        params = dict(fixed_params)
        for par in free:
            params[par] = numpyro.sample(par, dist.Uniform(*bounds[par]))

        total = 0.0
        gw_coeffs = []
        for e in entries:
            ctx, tr = e["ctx"], e["transport"]
            xi = numpyro.sample(
                e["xi_site"],
                dist.Normal(0.0, 1.0).expand([tr.dimension]).to_event(1),
            )
            q, ldj = tr.apply(params, xi)
            parts = tr.split(q)
            z = parts[ctx.joint_site]
            delta = ctx.space.delta_from_z(z, jnp)
            for i, fitpar in enumerate(ctx.sampled_all):
                params[f"{ctx.name_stem}_{fitpar}"] = delta[i]
                numpyro.deterministic(f"{ctx.name_stem}_{fitpar}_delta", delta[i])
            for key in e["coeff_keys"]:
                params[key] = parts[key]
            if e["gw_key"] is not None:
                gw_coeffs.append(parts[e["gw_key"]])
            total = (
                total
                + e["likelihood"].clogL(params)
                - 0.5 * jnp.sum(z * z)
                + ldj
                + 0.5 * jnp.sum(xi * xi)
            )

        # Exact dense cross-pulsar coefficient prior for the global (HD) GP,
        # added ONCE (the per-pulsar clogL carries no GW prior; the transport's
        # per-pulsar diagonal only conditions it). Pulsar-major stack (§7).
        if global_gp is not None:
            c_flat = jnp.concatenate([jnp.asarray(c) for c in gw_coeffs])
            total = total + global_gp_logprior(c_flat, global_gp, params, jnp)

        numpyro.factor("nlt_joint", total)

    nlt_joint_model_multi.transports = [e["transport"] for e in entries]
    return nlt_joint_model_multi


def joint_samples_to_frame(samples: Mapping[str, Any], ctx):
    """Flatten joint-model samples into a DataFrame with exact physical decodes.

    The canonical timing decode for a dynamic-transport run is the recorded
    ``{name_stem}_{fitpar}_delta`` deterministics (the latent ``xi`` is not
    decodable on its own, §6.6); physical values are recomputed from them
    through ``ParameterSpace.to_physical``.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("joint_samples_to_frame requires pandas") from exc

    name_stem = ctx.name_stem
    delta_keys = [f"{name_stem}_{name}_delta" for name in ctx.sampled_all]
    missing = [k for k in delta_keys if k not in samples]
    if missing:
        raise KeyError(f"joint samples missing timing delta sites: {missing}")

    def _rows(arr):
        arr = np.asarray(arr, dtype=float)
        return arr.reshape(-1) if arr.ndim <= 1 else arr.reshape(arr.shape[0], -1)

    n_rows = np.asarray(samples[delta_keys[0]], dtype=float).reshape(-1).shape[0]
    columns: dict[str, np.ndarray] = {}
    for name, value in samples.items():
        flat = np.asarray(value, dtype=float)
        flat = flat.reshape(n_rows, -1) if flat.ndim > 1 else flat.reshape(-1)
        if flat.ndim == 1:
            columns[name] = flat
        else:
            for i in range(flat.shape[1]):
                columns[f"{name}[{i}]"] = flat[:, i]

    delta = np.stack([columns[k] for k in delta_keys], axis=1)
    theta_native = ctx.space.to_physical(delta, units="native", coord="delta")
    theta_display = ctx.space.to_physical(delta, units="display", coord="delta")
    for name in ctx.sampled_all:
        columns[f"{name_stem}_{name}_theta_native"] = theta_native[name]
        columns[f"{name_stem}_{name}_theta_display"] = theta_display[name]
    return pd.DataFrame(columns)


def joint_run_manifest(ctx, transport, **kwargs):
    """Build the dynamic-transport :class:`RunManifest` for a joint run (§6.6).

    Records the transport structure/digest and ``latent_decodable=false`` in the
    manifest ``transport`` section; the caller supplies ``likelihood=``,
    ``sampler=`` and any run metadata forwarded to ``ctx.run_manifest``.
    """
    from ..metric import dynamic_transport_record

    return ctx.run_manifest(
        dynamic_transport=dynamic_transport_record(transport), **kwargs
    )


def timing_init_values(ctx) -> dict[str, Any]:
    """Init-at-reference values for the joint timing site (zero coordinates)."""
    ndim = len(ctx.sampled)
    if not ndim:
        return {}
    import jax.numpy as jnp

    return {ctx.latent_name_for_coord(): jnp.zeros((ndim,), dtype=jnp.float64)}


def nuts(
    model_fn: Callable[[], None],
    ctx,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    dense_mass: bool = True,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    init_strategy: Any = None,
    chain_method: str = "vectorized",
    progress_bar: bool = True,
    **mcmc_kwargs: Any,
) -> Any:
    """Construct, but do not run, a configured NUTS ``MCMC``.

    This is an opinionated convenience recipe, not the canonical integration
    path (see the module docstring). Timing coordinates initialize at the
    engine reference (zero in sampling coordinates) unless ``init_strategy``
    is given explicitly, in which case it wins outright. Run the returned
    object with ``mcmc.run(jax.random.PRNGKey(seed))``.

    Callers must invoke :func:`ensure_x64` before constructing the Discovery
    likelihood, not just before calling this function — JAX arrays already
    created as float32 stay float32.
    """
    ensure_x64()
    from numpyro.infer import MCMC, NUTS, init_to_value

    if init_strategy is None:
        init_strategy = init_to_value(values=timing_init_values(ctx))

    kernel = NUTS(
        model_fn,
        dense_mass=dense_mass,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
        **mcmc_kwargs,
    )
    if hasattr(model_fn, "to_df"):
        mcmc.to_df = lambda: model_fn.to_df(mcmc.get_samples())
    return mcmc


def timing_draws(samples: Mapping[str, Any], ctx) -> np.ndarray:
    """Extract flattened joint-site draws ``(n_draws, ndim)`` from MCMC samples."""
    x = np.asarray(samples[ctx.latent_name_for_coord()], dtype=float)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    return x


def save_samples(
    run_dir,
    samples: Mapping[str, Any],
    ctx,
    *,
    manifest,
    final: bool,
    n_target: int | None = None,
):
    """Decode timing draws and write a Discovery checkpoint/final npz.

    Thin public wrapper around :func:`timing_draws` and
    ``nltiming.run_io.save_discovery_checkpoint``. Performs no sampling and
    does not depend on an ``MCMC`` object. ``manifest`` is the ``RunManifest``
    returned by ``ctx.write(...)``, which must be written before the
    first sample so the run metadata always precedes data.
    """
    from ..run_io import save_discovery_checkpoint

    x = timing_draws(samples, ctx)
    return save_discovery_checkpoint(
        run_dir, x, manifest, final=final, n_target=n_target
    )
