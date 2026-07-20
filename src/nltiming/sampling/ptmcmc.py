"""Sampler-neutral PTMCMC helpers, plus one optional timing-only recipe.

``timing_param_names`` and ``chain_layout`` are sampler-neutral: they map the
timing block's coordinate layout (whitening joint site vs standardized scalar
columns) onto a sampler's flat parameter-name order — Enterprise's
``pta.param_names`` for a normal full-PTA analysis, or a timing-only vector
for :func:`timing_only_sampler`. ``eval_params``/:func:`timing_only_sampler`
are an optional, experimental recipe that fixes every non-timing parameter
and samples only the timing coordinates; they are not the standard Enterprise
workflow, where a normal ``enterprise_extensions.sampler.setup_sampler``
samples the complete PTA using the native Enterprise parameters from
``ntm.enterprise_signal()`` (see the package README).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CHAIN_FILENAME = "chain_1.txt"


def eval_params(
    ctx,
    vec: np.ndarray,
    *,
    fixed: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Map a timing-block sampler vector to a likelihood parameter dict.

    Under the identity static layer (``whitening=None``) the vector entries are
    scalar prior-normal ``z`` coordinates keyed by delay-key names (Enterprise
    scalar ``UserParameter`` layout); under static whitening the whole vector is
    the joint ``x`` coordinate site.
    """
    params: dict[str, Any] = {
        key: float(value)
        for key, value in dict(fixed or {}).items()
        if isinstance(value, (int, float))
    }
    x = np.asarray(vec, dtype=float).reshape(-1)
    if x.size != len(ctx.sampled):
        raise ValueError(f"expected vector of length {len(ctx.sampled)}, got {x.size}")
    if not ctx.sampled:
        return params
    if ctx.model.static_layer == "identity":
        for i, key in enumerate(ctx.delay_keys):
            params[key] = float(x[i])
    else:
        params[ctx.latent_name_for_coord()] = x
    return params


def initial_point(ctx) -> np.ndarray:
    """Reference initial point: zero in sampling coordinates."""
    return np.zeros(len(ctx.sampled), dtype=float)


def initial_cov(ctx, *, nsamples: int = 2000, seed: int = 0) -> np.ndarray:
    """Proposal covariance from the WLS covariance, in sampling coordinates.

    The default coordinate transforms rescale axes but do not rotate away
    cross-parameter correlations, so the posterior in sampling coordinates can
    be a narrow correlated ridge. Seeding PTMCMC's jump proposals with the
    whitened-least-squares covariance (mapped through the coordinate stack by
    sampling) makes the chain mix immediately instead of relying on long
    adaptation.
    """
    from ..whitening import schur_delta_wls

    ndim = len(ctx.sampled)
    if ndim == 0:
        raise ValueError("ctx has no sampled timing parameters")
    wls = schur_delta_wls(
        pulsar=ctx.pulsar,
        partition=ctx.plan,
        variance=np.asarray(ctx.pulsar.toaerrs, dtype=float) ** 2,
        design_matrix=ctx.design_matrix,
    )
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        np.zeros(ndim), np.asarray(wls.covariance, dtype=float), size=nsamples
    )
    coords = np.stack(
        [
            np.asarray(
                ctx.space.coord_from_delta(delta, np, coord=ctx.coord),
                dtype=float,
            )
            for delta in draws
        ]
    )
    cov = np.atleast_2d(np.cov(coords.T))
    # Guard against numerically singular proposals.
    cov[np.diag_indices_from(cov)] += 1e-12
    return cov


def timing_param_names(ctx) -> tuple[str, ...]:
    """Sampler-visible timing parameter names in vector order."""
    if not ctx.sampled:
        return tuple()
    if ctx.model.static_layer == "identity":
        return tuple(ctx.delay_keys)
    site = ctx.latent_name_for_coord()
    return tuple(f"{site}_{i}" for i in range(len(ctx.sampled)))


def chain_layout(
    ctx,
    param_names: Sequence[str],
    *,
    chain_file: str = CHAIN_FILENAME,
) -> dict[str, Any]:
    """Run-metadata ``chain_layout`` spec locating timing columns in a PTMCMC chain.

    ``param_names`` is the sampler's parameter-name order (for a
    timing-only run, ``timing_param_names(ctx)``; for an Enterprise PTA
    with free noise, ``pta.param_names``).
    """
    names = list(param_names)
    columns = []
    for key in timing_param_names(ctx):
        try:
            columns.append(names.index(key))
        except ValueError as exc:
            raise ValueError(
                f"timing parameter {key!r} not found in sampler param names"
            ) from exc
    return {"kind": "ptmcmc", "file": chain_file, "columns": columns}


def timing_only_sampler(
    pta,
    ctx,
    outdir: str | Path,
    *,
    fixed: Mapping[str, float] | None = None,
    cov: np.ndarray | None = None,
    verbose: bool = True,
    **ptmcmc_kwargs: Any,
):
    """Configured ``PTSampler`` over ONLY the timing block; experimental recipe.

    This fixes every non-timing parameter via ``fixed`` and samples only the
    timing coordinates — it is not the standard Enterprise workflow, where a
    normal ``enterprise_extensions.sampler.setup_sampler(pta, ...)`` samples
    the complete PTA vector (noise and timing jointly) using the native
    Enterprise parameters from ``ntm.enterprise_signal()``. Use this only for
    timing-only experiments with all other parameters pinned.

    Returns the sampler; run it with
    ``pts.sample(p0=initial_point(ctx), Niter=..., burn=...)``. The chain
    lands in ``outdir/chain_1.txt`` (PTMCMC layout: ndim columns + lnpost,
    lnlik, accept, pt-accept). The default proposal covariance is
    :func:`initial_cov` — the WLS covariance mapped into sampling coordinates —
    which captures cross-parameter correlations the coordinate transform does
    not remove.
    """
    from PTMCMCSampler.PTMCMCSampler import PTSampler

    ndim = len(ctx.sampled)
    if ndim == 0:
        raise ValueError("ctx has no sampled timing parameters")
    if cov is None:
        cov = initial_cov(ctx)

    def _params(vec: np.ndarray) -> dict[str, Any]:
        return eval_params(ctx, vec, fixed=fixed)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return PTSampler(
        ndim,
        lambda vec: pta.get_lnlikelihood(_params(vec)),
        lambda vec: pta.get_lnprior(_params(vec)),
        cov,
        outDir=str(outdir),
        verbose=verbose,
        **ptmcmc_kwargs,
    )


# ---------------------------------------------------------------------------
# Marginalized dynamic decentering (Enterprise/PTMCMC parity, feature §6)
#
# NOTE (E19 deferred): the static helpers above (eval_params/timing_param_names/
# initial_cov/timing_only_sampler) are NOT migrated here. The proposal §6.0
# assumed the geometry plan deleted ctx.sampled / ctx.model.transform, but
# ctx.sampled is retained (== ctx.plan.sampled) and eval_params already keys off
# ctx.model.static_layer; the static-whitening Enterprise assembly moreover uses
# per-axis `_x_i` parameters (not ctx.delay_keys), so the §6.0 "delta scalars
# keyed by delay_keys" claim holds only for whitening=None. Migration is a doc
# reconciliation, not a code fix; it is out of scope for the decentered mode
# below (which never touches those helpers).
# ---------------------------------------------------------------------------


def decentered_param_names(ctx, hyper_names) -> tuple[str, ...]:
    """PTMCMC vector layout ``[xi_0 .. xi_{k-1}, eta ..]`` (E20, sorted eta)."""
    k = len(ctx.plan.sampled)
    xi = tuple(f"{ctx.name_stem}_timing_xi_{i}" for i in range(k))
    return xi + tuple(hyper_names)


def decentered_target(pta, ctx, transport, *, hyper_names, hyper_bounds, fixed):
    """``(lnlike_fn, lnprior_fn)`` over ``vec = [xi | eta]`` for ``PTSampler``.

    Accounting (§3, E2-E4): ``lnlike`` is the Enterprise marginalized likelihood
    at ``delta(z(xi, eta))``; ``lnprior`` carries the eta boxes, the exact timing
    prior ``-1/2||z||^2``, and ``ldJ(eta)`` — untempered. ``pta.get_lnprior`` is
    NEVER called in this mode (the delay ``UserParameter``s carry physical priors
    that would double-count the timing prior, E4).
    """
    hyper_names = tuple(hyper_names)
    if hyper_names != tuple(sorted(hyper_names)):
        raise ValueError("hyper_names must be in sorted order (E8/D20)")
    if hyper_names != tuple(transport.params):
        raise ValueError(
            f"hyper_names {hyper_names} != transport.params "
            f"{tuple(transport.params)}; build both from "
            f"enterprise_marginal_products(...).params"
        )
    overlap = set(fixed) & (set(hyper_names) | set(ctx.delay_keys))
    if overlap:
        raise ValueError(
            f"fixed must not pin sampled hypers or delay keys: {sorted(overlap)}"
        )
    k = transport.dimension
    lo = np.array([hyper_bounds[n][0] for n in hyper_names])
    hi = np.array([hyper_bounds[n][1] for n in hyper_names])
    logwidth = float(np.sum(np.log(hi - lo)))
    fixed = dict(fixed)

    def _split(vec):
        vec = np.asarray(vec, dtype=float)
        return vec[:k], dict(zip(hyper_names, vec[k:]))

    def lnprior(vec):
        xi, eta = _split(vec)
        ev = np.asarray(list(eta.values()))
        if np.any(ev < lo) or np.any(ev > hi):
            return -np.inf
        z, ldj = transport.apply(eta, xi)
        return -0.5 * float(z @ z) + ldj - logwidth

    def lnlike(vec):
        xi, eta = _split(vec)
        ev = np.asarray(list(eta.values()))
        if np.any(ev < lo) or np.any(ev > hi):
            return -np.inf
        z, _ = transport.apply(eta, xi)  # memo hit after lnprior (E9)
        delta = np.asarray(ctx.space.delta_from_z(z, np), dtype=float)
        params = {**fixed, **eta}
        for key, value in zip(ctx.delay_keys, delta):
            params[key] = float(value)
        return float(pta.get_lnlikelihood(params))

    return lnlike, lnprior


def decentered_initial_cov(ctx, hyper_names, hyper_sigmas=None, default=0.3):
    """Proposal covariance: identity on the xi block (E22), diag(sigma^2) on eta.

    The xi block is identity BY CONSTRUCTION — that is the point of the dynamic
    decentering. ``hyper_sigmas`` comes from the marginal-Hessian sigma of the
    WN-first MPE pipeline when available.
    """
    k = len(ctx.plan.sampled)
    sig = np.array([(hyper_sigmas or {}).get(n, default) for n in hyper_names])
    return np.diag(np.concatenate([np.ones(k), sig**2]))


def decentered_initial_point(ctx, transport, hyper_names, eta_mpe) -> np.ndarray:
    """Init at ``(xi=0, eta=MPE)`` (E22, the validated J1640 recipe)."""
    missing = sorted(set(hyper_names) - set(eta_mpe))
    if missing:
        raise ValueError(f"eta_mpe missing hyperparameters: {missing}")
    return np.concatenate(
        [
            np.zeros(transport.dimension),
            np.array([float(eta_mpe[n]) for n in hyper_names]),
        ]
    )


def decentered_chain_layout(ctx, hyper_names, *, chain_file: str = CHAIN_FILENAME):
    """Run-metadata layout for a decentered PTMCMC chain (E23).

    ``chain_1.txt`` carries 4 trailing bookkeeping columns
    (``lnpost, lnlike, accept, pt-accept``); the columns below index the
    parameter block only, and the decoder strips the trailing columns by count.
    """
    k = len(ctx.plan.sampled)
    return {
        "kind": "ptmcmc-decentered",
        "file": chain_file,
        "xi_columns": list(range(k)),
        "hyper_columns": list(range(k, k + len(hyper_names))),
        "hyper_names": list(hyper_names),
    }


def decentered_sampler(
    pta,
    ctx,
    transport,
    outdir,
    *,
    hyper_names,
    hyper_bounds,
    fixed,
    cov=None,
    groups=None,
    verbose: bool = True,
    **ptmcmc_kwargs,
):
    """Configured ``PTSampler`` over ``vec = [xi | eta]`` (E14).

    Default jump ``groups`` separate the xi block from the eta block: eta moves
    refactorize ``Sigma(eta)`` (expensive), while xi-only moves at fixed eta are
    near-free (memo hit, E9) and mix the whitened block rapidly. A caller-supplied
    ``groups`` is forwarded unchanged.
    """
    from PTMCMCSampler.PTMCMCSampler import PTSampler

    lnlike, lnprior = decentered_target(
        pta,
        ctx,
        transport,
        hyper_names=hyper_names,
        hyper_bounds=hyper_bounds,
        fixed=fixed,
    )
    k, m = transport.dimension, len(hyper_names)
    if cov is None:
        cov = decentered_initial_cov(ctx, hyper_names)
    if groups is None:
        groups = [list(range(k)), list(range(k, k + m))]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return PTSampler(
        k + m,
        lnlike,
        lnprior,
        cov,
        groups=groups,
        outDir=str(outdir),
        verbose=verbose,
        **ptmcmc_kwargs,
    )
