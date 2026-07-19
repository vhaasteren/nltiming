"""Off-zero geometry diagnostics for joint timing models.

This module owns the exact-target geometry metric kernel used by the optional
geometry certifier (feature: timing-coordinate charts and geometry certification,
§8). The kernel differentiates the *actual* NumPyro model's unconstrained
potential — it never accepts a surrogate function — and is never invoked by model
construction or by :func:`nltiming.sampling.numpyro.nuts`.

Stage 1 delivers the internal metric kernel only:

- :func:`target_metrics_at` — target-only gradient, conditional Hessian,
  cross-Hessian, and conditional-identity spread at one ``(xi, eta)`` point in
  the coordinate NUTS actually sees (unconstrained-logit hyperparameters);

The public certifier (``certify_joint_geometry``, ``GeometryThresholds``,
``JointGeometryReport``, report I/O) is layered on top of this kernel in a later
stage. The kernel requires a built joint model exposing ``xi_site: str`` and
``hyper_sites: tuple[str, ...]`` (see
:func:`nltiming.sampling.numpyro.joint_model`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _require_site_metadata(model) -> tuple[str, tuple[str, ...]]:
    xi_site = getattr(model, "xi_site", None)
    hyper_sites = getattr(model, "hyper_sites", None)
    if not isinstance(xi_site, str):
        raise ValueError(
            "geometry kernel requires model.xi_site (a str); the built joint "
            "model does not expose it"
        )
    if hyper_sites is None:
        raise ValueError(
            "geometry kernel requires model.hyper_sites (a tuple of str); the "
            "built joint model does not expose it"
        )
    return xi_site, tuple(hyper_sites)


def _unconstrained_potential(model, xi, hyper):
    """Return ``(potential_fn, z0_dict, order, shapes)`` at the requested point.

    ``z0_dict`` is the unconstrained representation of the constrained point
    ``{xi_site: xi, **hyper}`` (hyper values are constrained/in-box). ``order``
    is ``[xi_site, *hyper_sites]`` and ``shapes`` maps each site to its
    unconstrained shape; both fix the block-flatten order NUTS sees (§8.3).
    """
    import jax
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    xi_site, hyper_sites = _require_site_metadata(model)
    point = {xi_site: xi, **{k: hyper[k] for k in hyper_sites}}
    info = initialize_model(
        jax.random.PRNGKey(0),
        model,
        init_strategy=init_to_value(values=point),
    )
    z0 = dict(info.param_info.z)
    potential_fn = info.potential_fn
    order = [xi_site, *hyper_sites]
    missing = [k for k in order if k not in z0]
    if missing:
        raise ValueError(
            f"model trace is missing expected sites {missing}; xi_site/"
            "hyper_sites disagree with the model"
        )
    shapes = {k: tuple(np.asarray(z0[k]).shape) for k in order}
    return potential_fn, z0, order, shapes


def _flatten(zdict, order):
    import jax.numpy as jnp

    parts = [jnp.reshape(jnp.asarray(zdict[k], dtype=float), (-1,)) for k in order]
    return jnp.concatenate(parts) if parts else jnp.zeros((0,))


def _unflatten(vec, order, shapes):
    import jax.numpy as jnp

    out = {}
    i = 0
    for k in order:
        shp = shapes[k]
        n = int(np.prod(shp)) if shp else 1
        out[k] = jnp.reshape(vec[i : i + n], shp)
        i += n
    return out


@dataclass(frozen=True)
class TargetMetrics:
    """Exact-target geometry metrics at one ``(xi, eta)`` probe point (§8.3).

    All quantities are computed on the model's unconstrained potential in the
    coordinate NUTS sees (hyperparameters in their unconstrained-logit frame).
    """

    xi_gradient_inf_norm: float
    xi_hessian_eigen_min: float
    xi_hessian_eigen_max: float
    xi_eta_cross_operator_norm: float
    conditional_identity: float


def target_metrics_at(
    model,
    *,
    xi: np.ndarray,
    hyper: Mapping[str, float],
) -> TargetMetrics:
    """Target-only geometry metrics at ``(xi, hyper)`` (§8.3 items 3–6).

    - ``xi_gradient_inf_norm``: infinity norm of ``d(-log p)/d xi`` (the hyper
      Uniform constants drop out of the xi-gradient).
    - ``xi_hessian_eigen_{min,max}``: extreme eigenvalues of ``H_xixi``.
    - ``xi_eta_cross_operator_norm``: operator 2-norm of ``H_xieta`` in the
      unconstrained-logit hyper coordinate.
    - ``conditional_identity``: ``D = log p(xi, eta) - log p(0, eta) +
      0.5||xi||^2`` via ``numpyro.infer.util.log_density`` on the actual model.

    The Hessians differentiate the unconstrained ``potential_fn`` returned by
    ``numpyro.infer.util.initialize_model``; the interval transform is never
    hand-coded (I-item, §8.3).
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    xi_site, hyper_sites = _require_site_metadata(model)
    xi = jnp.asarray(np.asarray(xi, dtype=float))
    hyper = {k: float(hyper[k]) for k in hyper_sites}

    potential_fn, z0, order, shapes = _unconstrained_potential(model, xi, hyper)
    u0 = _flatten(z0, order)
    xi_dim = int(np.prod(shapes[xi_site])) if shapes[xi_site] else 1

    def pot(u):
        return potential_fn(_unflatten(u, order, shapes))

    grad = np.asarray(jax.grad(pot)(u0), dtype=float)
    hess = np.asarray(jax.hessian(pot)(u0), dtype=float)

    grad_xi = grad[:xi_dim]
    h_xixi = hess[:xi_dim, :xi_dim]
    h_xieta = hess[:xi_dim, xi_dim:]

    eigs = np.linalg.eigvalsh(0.5 * (h_xixi + h_xixi.T))
    if h_xieta.size:
        cross_norm = float(np.linalg.norm(h_xieta, 2))
    else:
        cross_norm = 0.0

    # Conditional identity uses the constrained log density (xi is a real site,
    # eta cancels between the two evaluations, so no transform Jacobian needed).
    zeros = jnp.zeros_like(xi)
    constrained = {xi_site: xi, **hyper}
    constrained0 = {xi_site: zeros, **hyper}
    lp_xi, _ = log_density(model, (), {}, constrained)
    lp_0, _ = log_density(model, (), {}, constrained0)
    identity = float(lp_xi - lp_0 + 0.5 * jnp.sum(xi * xi))

    return TargetMetrics(
        xi_gradient_inf_norm=float(np.max(np.abs(grad_xi))) if grad_xi.size else 0.0,
        xi_hessian_eigen_min=float(eigs.min()) if eigs.size else 0.0,
        xi_hessian_eigen_max=float(eigs.max()) if eigs.size else 0.0,
        xi_eta_cross_operator_norm=cross_norm,
        conditional_identity=identity,
    )


def deterministic_xi_probes(dim: int) -> list[np.ndarray]:
    """The deterministic ``2K + 9`` sampler-space probe set (§8.2).

    Zero, the ``+/-`` unit axes, then eight fixed pseudo-random draws from a
    seeded generator. Deterministic across processes (fixed seed 8675309).
    """
    if dim < 0:
        raise ValueError("dim must be non-negative")
    points = [np.zeros(dim)]
    eye = np.eye(dim)
    for i in range(dim):
        points += [eye[i].copy(), -eye[i].copy()]
    rng = np.random.default_rng(8675309)
    points += [rng.standard_normal(dim) for _ in range(8)]
    return points


def conditional_identity_spread(
    model,
    *,
    hyper: Mapping[str, float],
    xi_points: Sequence[np.ndarray],
) -> float:
    """``max(D) - min(D)`` of the conditional-identity metric over probe points."""
    values = [
        target_metrics_at(model, xi=xi, hyper=hyper).conditional_identity
        for xi in xi_points
    ]
    return float(max(values) - min(values)) if values else 0.0
