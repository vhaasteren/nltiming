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
    sampled = ctx.partition.sampled
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

    sampled = ctx.partition.sampled
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
