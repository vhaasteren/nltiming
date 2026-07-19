"""Off-zero geometry diagnostics for joint timing models.

This module owns the exact-target geometry metric kernel used by the optional
geometry certifier (feature: timing-coordinate charts and geometry certification,
§8). The kernel differentiates the *actual* NumPyro model's unconstrained
potential — it never accepts a surrogate function — and is never invoked by model
construction or by :func:`nltiming.sampling.numpyro.nuts`.

The internal metric kernel:

- :func:`target_metrics_at` — target-only gradient, conditional Hessian,
  cross-Hessian, and conditional-identity spread at one ``(xi, eta)`` point in
  the coordinate NUTS actually sees (unconstrained-logit hyperparameters);

The public certifier (:func:`certify_joint_geometry`, :class:`GeometryThresholds`,
:class:`JointGeometryReport`, :func:`transport_center_report`,
:func:`box_hyper_probe_points`, and the standalone
:func:`write_geometry_report` / :func:`read_geometry_report` products) is layered
on top of this kernel. It requires a built joint model exposing ``xi_site: str``,
``hyper_sites: tuple[str, ...]``, and ``transport`` (see
:func:`nltiming.sampling.numpyro.joint_model`). None of it is ever called by
model construction or by :func:`nltiming.sampling.numpyro.nuts`.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class GeometryDiagnosticWarning(UserWarning):
    """Non-fatal concern raised by ``JointGeometryReport.warn`` (§8.5)."""


class GeometryCertificationError(RuntimeError):
    """Raised by ``JointGeometryReport.require_passed`` on a failed report (§8)."""


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


def _target_arrays_at(
    model,
    *,
    xi: np.ndarray,
    hyper: Mapping[str, float],
) -> dict:
    """Shared core for the target-only metrics (§8.3 items 3–6).

    Returns the raw ``xi`` gradient vector, the symmetrized ``H_xixi``
    eigenvalues, the ``H_xieta`` operator 2-norm, and the conditional-identity
    value, all in the coordinate NUTS sees (unconstrained-logit hyper). The
    scalar :class:`TargetMetrics` reductions and the standalone-report NPZ
    payload both build on this.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    xi_site, hyper_sites = _require_site_metadata(model)
    xi_j = jnp.asarray(np.asarray(xi, dtype=float))
    hyper = {k: float(hyper[k]) for k in hyper_sites}

    potential_fn, z0, order, shapes = _unconstrained_potential(model, xi_j, hyper)
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
    cross_norm = float(np.linalg.norm(h_xieta, 2)) if h_xieta.size else 0.0

    # Conditional identity uses the constrained log density (xi is a real site,
    # eta cancels between the two evaluations, so no transform Jacobian needed).
    zeros = jnp.zeros_like(xi_j)
    constrained = {xi_site: xi_j, **hyper}
    constrained0 = {xi_site: zeros, **hyper}
    lp_xi, _ = log_density(model, (), {}, constrained)
    lp_0, _ = log_density(model, (), {}, constrained0)
    identity = float(lp_xi - lp_0 + 0.5 * jnp.sum(xi * xi))

    return {
        "grad_xi": grad_xi,
        "hxixi_eigs": eigs,
        "cross_norm": cross_norm,
        "identity": identity,
    }


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
    arrays = _target_arrays_at(model, xi=xi, hyper=hyper)
    grad_xi = arrays["grad_xi"]
    eigs = arrays["hxixi_eigs"]
    return TargetMetrics(
        xi_gradient_inf_norm=float(np.max(np.abs(grad_xi))) if grad_xi.size else 0.0,
        xi_hessian_eigen_min=float(eigs.min()) if eigs.size else 0.0,
        xi_hessian_eigen_max=float(eigs.max()) if eigs.size else 0.0,
        xi_eta_cross_operator_norm=arrays["cross_norm"],
        conditional_identity=arrays["identity"],
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


# ---------------------------------------------------------------------------
# §7 Transport-center diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportCenterAxis:
    """Per-sampled-axis transport-center record (§7).

    ``center_z`` is the timing slice of ``transport.apply(params, 0)`` — where
    the conditional centering places the coordinate when the sampler is at the
    origin. Its meaning differs by chart: a large ``center_z`` on an
    ``affine_normal`` axis is an ordinary Gaussian mean shift (always interior),
    while on a ``prior_pit`` axis it approaches a bounded-prior edge.
    """

    name: str
    chart: str
    expansion_z: float
    center_z: float
    center_delta: float
    local_chart_ratio: float | None
    interior: bool


def transport_center_report(
    ctx, transport, params, *, pit_interior_limit: float = 5.0
) -> tuple[TransportCenterAxis, ...]:
    """Per-axis transport-center report over the sampled timing block (§7).

    ``params`` is a constrained hyperparameter point. Only ``prior_pit`` axes
    participate in the interior limit; ``affine_normal`` axes are interior for
    every finite center and carry ``local_chart_ratio=None``.
    """
    import jax.numpy as jnp

    bijector = ctx.space.prior_bijector
    names = tuple(ctx.space.names)
    charts = bijector.chart_kinds()
    z_e = np.asarray(ctx.linearization.sampled_z_expansion, dtype=float)

    q, _ = transport.apply(dict(params), jnp.zeros(transport.dimension))
    center_z = np.asarray(transport.split(q)[ctx.joint_site], dtype=float)
    center_delta = np.asarray(ctx.space.delta_from_z(center_z, np), dtype=float)

    jac_center = np.asarray(bijector.jacobian_diag_delta_from_z(center_z, np))
    jac_expansion = np.asarray(bijector.jacobian_diag_delta_from_z(z_e, np))

    axes: list[TransportCenterAxis] = []
    for i, name in enumerate(names):
        chart = charts[i]
        if chart == "affine_normal":
            ratio: float | None = None
            interior = True
        else:
            denom = float(jac_expansion[i])
            ratio = float(jac_center[i] / denom) if denom != 0.0 else float("inf")
            interior = bool(abs(float(center_z[i])) <= pit_interior_limit)
        axes.append(
            TransportCenterAxis(
                name=name,
                chart=chart,
                expansion_z=float(z_e[i]),
                center_z=float(center_z[i]),
                center_delta=float(center_delta[i]),
                local_chart_ratio=ratio,
                interior=interior,
            )
        )
    return tuple(axes)


# ---------------------------------------------------------------------------
# §8 Geometry certification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryThresholds:
    """Recommended default thresholds for :func:`certify_joint_geometry` (§8.1).

    Every field is user-overrideable and recorded verbatim in the report.
    """

    pit_interior_limit: float = 5.0
    residual_remainder_rms: float = 0.10
    residual_remainder_max_standardized_toa: float = 1.00
    xi_gradient_inf_norm: float = 0.20
    xi_hessian_eigen_min: float = 0.50
    xi_hessian_eigen_max: float = 2.00
    xi_eta_cross_operator_norm: float = 0.25
    conditional_identity_spread: float = 0.10


@dataclass(frozen=True)
class JointGeometryReport:
    """Result of an off-zero joint-geometry certification (§8.1).

    ``passed`` is advisory only: it never controls ``nuts``. Callers inspect the
    raw metrics, call :meth:`warn`, or opt into :meth:`require_passed`.
    """

    passed: bool
    failures: tuple[str, ...]
    hyper_points: tuple[dict[str, float], ...]
    xi_points_digest: str
    center_axes: tuple[TransportCenterAxis, ...]
    max_residual_remainder_rms: float
    max_residual_remainder_standardized_toa: float
    max_xi_gradient_inf_norm: float
    xi_hessian_eigen_min: float
    xi_hessian_eigen_max: float
    max_xi_eta_cross_operator_norm: float
    max_conditional_identity_spread: float
    per_point: tuple[dict[str, Any], ...]
    thresholds: GeometryThresholds
    context_fingerprint: str
    model_fingerprint: str

    def warn(self) -> None:
        """Emit one :class:`GeometryDiagnosticWarning` per reported concern."""
        for failure in self.failures:
            warnings.warn(failure, GeometryDiagnosticWarning, stacklevel=2)

    def require_passed(self) -> None:
        if not self.passed:
            raise GeometryCertificationError("; ".join(self.failures))


def box_hyper_probe_points(
    center: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], ...]:
    """Center plus per-parameter box-quantile probes (§8.4).

    For each parameter independently, points are placed at normalized box
    quantiles ``0.05, 0.25, 0.75, 0.95`` with all other parameters held at
    center. Exact bounds are never evaluated and no Cartesian product is formed.
    """
    center = {k: float(v) for k, v in center.items()}
    missing = [k for k in center if k not in bounds]
    if missing:
        raise ValueError(f"box_hyper_probe_points: no bounds for {missing}")

    points: list[dict[str, float]] = [dict(center)]
    for name in center:
        lo, hi = (float(b) for b in bounds[name])
        if not hi > lo:
            raise ValueError(
                f"box_hyper_probe_points: bounds for {name!r} are not lo<hi: "
                f"({lo}, {hi})"
            )
        for q in (0.05, 0.25, 0.75, 0.95):
            point = dict(center)
            point[name] = lo + q * (hi - lo)
            points.append(point)
    return tuple(points)


def _digest_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _model_fingerprint(model, ctx) -> str:
    xi_site, hyper_sites = _require_site_metadata(model)
    payload = json.dumps(
        {
            "xi_site": xi_site,
            "hyper_sites": list(hyper_sites),
            "dimension": int(model.transport.dimension),
            "index": sorted(model.transport.index),
            "linearization": ctx.linearization.fingerprint(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def certify_joint_geometry(
    model,
    ctx,
    *,
    hyper_points: Sequence[Mapping[str, float]],
    thresholds: GeometryThresholds | None = None,
) -> JointGeometryReport:
    """Certify the exact joint target geometry at supplied hyper points (§8.3).

    ``model`` is the built NumPyro joint model (it must expose ``.transport``,
    ``.xi_site``, ``.hyper_sites``). At each hyper point the exact target is
    probed at the deterministic ``2K+9`` sampler points; the certifier reports
    center interiority, exact-vs-local residual remainder (global RMS and
    localized per-TOA), the ``xi`` gradient/Hessian/cross-Hessian at zero, and
    the conditional-identity spread. Nothing here is ever called by ``nuts``.
    """
    import jax.numpy as jnp

    from .linearization import _waveform_of_z

    if not hyper_points:
        raise ValueError("certify_joint_geometry requires at least one hyper point")
    thr = thresholds if thresholds is not None else GeometryThresholds()
    _require_site_metadata(model)

    transport = model.transport
    dim = int(transport.dimension)
    probes = deterministic_xi_probes(dim)
    xi_points = np.stack(probes) if probes else np.zeros((0, dim))

    # Exact vs local timing waveform machinery (§8.3 item 2).
    lin = ctx.linearization
    idx = np.asarray(ctx.plan.idx_sampled, dtype=int)
    nfit = len(ctx.plan.fitpars)
    d_of_z = _waveform_of_z(ctx.engine, ctx.space, idx, nfit, jnp)
    d_e = np.asarray(lin.sampled_waveform_expansion, dtype=float)
    W_e = np.asarray(lin.sampled_basis, dtype=float)
    z_e = np.asarray(lin.sampled_z_expansion, dtype=float)
    sd = np.asarray(transport.reference_noise_standard_deviation(), dtype=float)
    n_toa = int(sd.shape[0])

    failures: list[str] = []
    per_point: list[dict[str, Any]] = []
    center_axes_first: tuple[TransportCenterAxis, ...] = ()

    max_rms = 0.0
    max_std_toa = 0.0
    max_grad = 0.0
    eig_min = float("inf")
    eig_max = float("-inf")
    max_cross = 0.0
    max_spread = 0.0

    for pi, raw_point in enumerate(hyper_points):
        point = {k: float(v) for k, v in raw_point.items()}

        center_axes = transport_center_report(
            ctx, transport, point, pit_interior_limit=thr.pit_interior_limit
        )
        if pi == 0:
            center_axes_first = center_axes
        for axis in center_axes:
            if not axis.interior:
                failures.append(
                    f"center_interior[point {pi}]: axis {axis.name} "
                    f"({axis.chart}) center_z={axis.center_z:.3g} exceeds "
                    f"pit_interior_limit={thr.pit_interior_limit}"
                )

        # Residual remainder over every probe point.
        point_rms = 0.0
        point_std = 0.0
        for xi in probes:
            q, _ = transport.apply(point, jnp.asarray(xi))
            z = np.asarray(transport.split(q)[ctx.joint_site], dtype=float)
            d_exact = np.asarray(d_of_z(jnp.asarray(z)), dtype=float)
            rem = d_exact - (d_e + W_e @ (z - z_e))
            quad = float(transport.reference_noise_quadratic(jnp.asarray(rem)))
            rms = float(np.sqrt(quad / n_toa))
            std_toa = float(np.max(np.abs(rem) / sd)) if n_toa else 0.0
            point_rms = max(point_rms, rms)
            point_std = max(point_std, std_toa)

        # Target metrics at xi=0 (§8.3 items 3–5) and identity spread (item 6).
        arrays = _target_arrays_at(model, xi=np.zeros(dim), hyper=point)
        grad_xi = arrays["grad_xi"]
        eigs = arrays["hxixi_eigs"]
        grad_inf = float(np.max(np.abs(grad_xi))) if grad_xi.size else 0.0
        e_min = float(eigs.min()) if eigs.size else 0.0
        e_max = float(eigs.max()) if eigs.size else 0.0
        cross = float(arrays["cross_norm"])
        spread = conditional_identity_spread(model, hyper=point, xi_points=probes)

        max_rms = max(max_rms, point_rms)
        max_std_toa = max(max_std_toa, point_std)
        max_grad = max(max_grad, grad_inf)
        eig_min = min(eig_min, e_min)
        eig_max = max(eig_max, e_max)
        max_cross = max(max_cross, cross)
        max_spread = max(max_spread, spread)

        per_point.append(
            {
                "hyper": point,
                "residual_remainder_rms": point_rms,
                "residual_remainder_standardized_toa": point_std,
                "xi_gradient_inf_norm": grad_inf,
                "xi_hessian_eigen_min": e_min,
                "xi_hessian_eigen_max": e_max,
                "xi_eta_cross_operator_norm": cross,
                "conditional_identity_spread": spread,
            }
        )

    if eig_min == float("inf"):
        eig_min = 0.0
    if eig_max == float("-inf"):
        eig_max = 0.0

    if max_rms > thr.residual_remainder_rms:
        failures.append(
            f"residual_remainder_rms={max_rms:.3g} > {thr.residual_remainder_rms}"
        )
    if max_std_toa > thr.residual_remainder_max_standardized_toa:
        failures.append(
            f"residual_remainder_max_standardized_toa={max_std_toa:.3g} > "
            f"{thr.residual_remainder_max_standardized_toa}"
        )
    if max_grad > thr.xi_gradient_inf_norm:
        failures.append(
            f"xi_gradient_inf_norm={max_grad:.3g} > {thr.xi_gradient_inf_norm}"
        )
    if eig_min < thr.xi_hessian_eigen_min:
        failures.append(
            f"xi_hessian_eigen_min={eig_min:.3g} < {thr.xi_hessian_eigen_min}"
        )
    if eig_max > thr.xi_hessian_eigen_max:
        failures.append(
            f"xi_hessian_eigen_max={eig_max:.3g} > {thr.xi_hessian_eigen_max}"
        )
    if max_cross > thr.xi_eta_cross_operator_norm:
        failures.append(
            f"xi_eta_cross_operator_norm={max_cross:.3g} > "
            f"{thr.xi_eta_cross_operator_norm}"
        )
    if max_spread > thr.conditional_identity_spread:
        failures.append(
            f"conditional_identity_spread={max_spread:.3g} > "
            f"{thr.conditional_identity_spread}"
        )

    return JointGeometryReport(
        passed=not failures,
        failures=tuple(failures),
        hyper_points=tuple({k: float(v) for k, v in p.items()} for p in hyper_points),
        xi_points_digest=_digest_array(xi_points),
        center_axes=center_axes_first,
        max_residual_remainder_rms=max_rms,
        max_residual_remainder_standardized_toa=max_std_toa,
        max_xi_gradient_inf_norm=max_grad,
        xi_hessian_eigen_min=eig_min,
        xi_hessian_eigen_max=eig_max,
        max_xi_eta_cross_operator_norm=max_cross,
        max_conditional_identity_spread=max_spread,
        per_point=tuple(per_point),
        thresholds=thr,
        context_fingerprint=ctx.fingerprint(),
        model_fingerprint=_model_fingerprint(model, ctx),
    )


# ---------------------------------------------------------------------------
# §8.5 Standalone report persistence (atomic JSON + NPZ)
# ---------------------------------------------------------------------------

_REPORT_SCHEMA = "nlt-geometry-report-v1"


def _axis_to_dict(axis: TransportCenterAxis) -> dict:
    return {
        "name": axis.name,
        "chart": axis.chart,
        "expansion_z": axis.expansion_z,
        "center_z": axis.center_z,
        "center_delta": axis.center_delta,
        "local_chart_ratio": axis.local_chart_ratio,
        "interior": axis.interior,
    }


def _axis_from_dict(d: Mapping[str, Any]) -> TransportCenterAxis:
    return TransportCenterAxis(
        name=str(d["name"]),
        chart=str(d["chart"]),
        expansion_z=float(d["expansion_z"]),
        center_z=float(d["center_z"]),
        center_delta=float(d["center_delta"]),
        local_chart_ratio=(
            None if d["local_chart_ratio"] is None else float(d["local_chart_ratio"])
        ),
        interior=bool(d["interior"]),
    )


def _atomic_write_bytes(path: pathlib.Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_geometry_report(
    report: JointGeometryReport,
    stem: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Atomically write ``<stem>.json`` and ``<stem>.npz`` (§8.5).

    JSON holds configuration, scalar results, failures, fingerprints, and
    digests (including the NPZ filename and its SHA-256). NPZ holds the numeric
    payload — probe points and per-point metric arrays — and contains no pickled
    Python objects. Both files are written through temporary files and renamed
    into place only after the payload is assembled. Existing files raise
    ``FileExistsError`` unless ``overwrite=True``.
    """
    stem = pathlib.Path(stem)
    json_path = stem.with_suffix(".json")
    npz_path = stem.with_suffix(".npz")
    if not overwrite:
        for p in (json_path, npz_path):
            if p.exists():
                raise FileExistsError(f"{p} exists; pass overwrite=True to replace")

    # Numeric payload for the NPZ (no pickled objects).
    def _col(key):
        return np.asarray([float(pp[key]) for pp in report.per_point], dtype=float)

    npz_arrays = {
        "residual_remainder_rms": _col("residual_remainder_rms"),
        "residual_remainder_standardized_toa": _col(
            "residual_remainder_standardized_toa"
        ),
        "xi_gradient_inf_norm": _col("xi_gradient_inf_norm"),
        "xi_hessian_eigen_min": _col("xi_hessian_eigen_min"),
        "xi_hessian_eigen_max": _col("xi_hessian_eigen_max"),
        "xi_eta_cross_operator_norm": _col("xi_eta_cross_operator_norm"),
        "conditional_identity_spread": _col("conditional_identity_spread"),
    }
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **npz_arrays)
    npz_bytes = buf.getvalue()
    npz_digest = hashlib.sha256(npz_bytes).hexdigest()

    meta = {
        "schema": _REPORT_SCHEMA,
        "passed": report.passed,
        "failures": list(report.failures),
        "hyper_points": [dict(p) for p in report.hyper_points],
        "xi_points_digest": report.xi_points_digest,
        "center_axes": [_axis_to_dict(a) for a in report.center_axes],
        "max_residual_remainder_rms": report.max_residual_remainder_rms,
        "max_residual_remainder_standardized_toa": (
            report.max_residual_remainder_standardized_toa
        ),
        "max_xi_gradient_inf_norm": report.max_xi_gradient_inf_norm,
        "xi_hessian_eigen_min": report.xi_hessian_eigen_min,
        "xi_hessian_eigen_max": report.xi_hessian_eigen_max,
        "max_xi_eta_cross_operator_norm": report.max_xi_eta_cross_operator_norm,
        "max_conditional_identity_spread": report.max_conditional_identity_spread,
        "per_point": [
            {**{k: v for k, v in pp.items() if k != "hyper"}, "hyper": dict(pp["hyper"])}
            for pp in report.per_point
        ],
        "thresholds": vars(report.thresholds),
        "context_fingerprint": report.context_fingerprint,
        "model_fingerprint": report.model_fingerprint,
        "npz_filename": npz_path.name,
        "npz_sha256": npz_digest,
    }
    json_bytes = json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")

    _atomic_write_bytes(npz_path, npz_bytes)
    _atomic_write_bytes(json_path, json_bytes)
    return json_path, npz_path


def read_geometry_report(
    stem: str | os.PathLike[str],
) -> JointGeometryReport:
    """Read ``<stem>.json`` + ``<stem>.npz`` and verify all digests (§8.5)."""
    stem = pathlib.Path(stem)
    json_path = stem.with_suffix(".json")
    npz_path = stem.with_suffix(".npz")

    with open(json_path, "rb") as fh:
        meta = json.loads(fh.read().decode("utf-8"))
    if meta.get("schema") != _REPORT_SCHEMA:
        raise ValueError(
            f"unexpected geometry-report schema {meta.get('schema')!r}; "
            f"expected {_REPORT_SCHEMA!r}"
        )

    with open(npz_path, "rb") as fh:
        npz_bytes = fh.read()
    if hashlib.sha256(npz_bytes).hexdigest() != meta["npz_sha256"]:
        raise ValueError(
            f"NPZ digest mismatch for {npz_path}: file does not match the "
            f"sha256 recorded in {json_path}"
        )
    if npz_path.name != meta["npz_filename"]:
        raise ValueError(
            f"NPZ filename {npz_path.name!r} does not match the recorded "
            f"{meta['npz_filename']!r}"
        )

    per_point = tuple(
        {**{k: v for k, v in pp.items() if k != "hyper"}, "hyper": dict(pp["hyper"])}
        for pp in meta["per_point"]
    )
    return JointGeometryReport(
        passed=bool(meta["passed"]),
        failures=tuple(meta["failures"]),
        hyper_points=tuple(dict(p) for p in meta["hyper_points"]),
        xi_points_digest=str(meta["xi_points_digest"]),
        center_axes=tuple(_axis_from_dict(a) for a in meta["center_axes"]),
        max_residual_remainder_rms=float(meta["max_residual_remainder_rms"]),
        max_residual_remainder_standardized_toa=float(
            meta["max_residual_remainder_standardized_toa"]
        ),
        max_xi_gradient_inf_norm=float(meta["max_xi_gradient_inf_norm"]),
        xi_hessian_eigen_min=float(meta["xi_hessian_eigen_min"]),
        xi_hessian_eigen_max=float(meta["xi_hessian_eigen_max"]),
        max_xi_eta_cross_operator_norm=float(meta["max_xi_eta_cross_operator_norm"]),
        max_conditional_identity_spread=float(meta["max_conditional_identity_spread"]),
        per_point=per_point,
        thresholds=GeometryThresholds(**meta["thresholds"]),
        context_fingerprint=str(meta["context_fingerprint"]),
        model_fingerprint=str(meta["model_fingerprint"]),
    )
