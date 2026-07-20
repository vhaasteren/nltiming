"""Linear transform builders for timing-space whitening/standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sl

from .bijectors import WhiteningLinear


@dataclass(frozen=True)
class DeltaWLS:
    """Schur-complement WLS approximation in sampled delta coordinates."""

    fisher: np.ndarray
    covariance: np.ndarray
    mean: np.ndarray


def _as_columns(matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    return np.asarray(matrix[:, indices], dtype=float)


def normalized_basis(matrix: np.ndarray) -> np.ndarray:
    """Unit-normalize basis columns for stable improper-GP marginalization.

    Timing design-matrix column norms span ~15 orders of magnitude; with a
    large finite prior weight (``1e40``) the unnormalized Woodbury/Cholesky
    solve loses the marginalization to float64 roundoff. Normalization leaves
    the column span (and hence the improper-prior marginalization) unchanged.
    """
    basis = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return basis / norms


def _weighted_cross(
    left: np.ndarray, weights: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return left.T @ (weights[:, None] * right)


def _cho_factor_pd(matrix: np.ndarray, *, context: str) -> tuple[np.ndarray, bool]:
    """Cholesky factorization with a clear error when the matrix is not PD."""
    try:
        return sl.cho_factor(matrix)
    except sl.LinAlgError as exc:
        raise ValueError(
            f"{context} is not numerically positive definite. This usually indicates "
            "collinear timing parameters, invalid TOA weights, or an ill-posed "
            "partition/model."
        ) from exc


def schur_delta_wls(
    *,
    pulsar,
    partition,
    variance: np.ndarray,
    design_matrix: np.ndarray | None = None,
    idx_kept=None,
    idx_marginalized=None,
) -> DeltaWLS:
    """Return kept-block Fisher, covariance, and WLS mean in delta units.

    The kept block defaults to the sampled axes and the marginalized block to the
    analytically delta-flat axes; pass ``idx_kept``/``idx_marginalized`` to Schur
    over a different split (e.g. the full proper set for cheat-prior widths).
    """
    mmat = (
        np.asarray(design_matrix, dtype=float)
        if design_matrix is not None
        else np.asarray(pulsar.Mmat, dtype=float)
    )
    residuals = np.asarray(pulsar.residuals, dtype=float)
    weights = 1.0 / np.asarray(variance, dtype=float)

    if idx_kept is None:
        idx_kept = partition.idx_sampled
    if idx_marginalized is None:
        idx_marginalized = partition.idx_analytically_marginalized
    sampled = _as_columns(mmat, tuple(idx_kept))
    analytically_marginalized_cols = _as_columns(mmat, tuple(idx_marginalized))
    ndim = sampled.shape[1]
    if ndim == 0:
        empty = np.eye(0, dtype=float)
        return DeltaWLS(
            fisher=empty,
            covariance=empty,
            mean=np.zeros(0, dtype=float),
        )

    fisher_ss = _weighted_cross(sampled, weights, sampled)
    rhs_s = sampled.T @ (weights * residuals)

    if analytically_marginalized_cols.shape[1]:
        fisher_sm = _weighted_cross(sampled, weights, analytically_marginalized_cols)
        fisher_mm = _weighted_cross(
            analytically_marginalized_cols, weights, analytically_marginalized_cols
        )
        rhs_m = analytically_marginalized_cols.T @ (weights * residuals)
        cf_mm = _cho_factor_pd(
            fisher_mm, context="Analytically marginalized timing Fisher block"
        )
        fisher_ss = fisher_ss - fisher_sm @ sl.cho_solve(cf_mm, fisher_sm.T)
        rhs_s = rhs_s - fisher_sm @ sl.cho_solve(cf_mm, rhs_m)

    cf_ss = _cho_factor_pd(fisher_ss, context="Sampled-block Schur Fisher")
    mean_delta = sl.cho_solve(cf_ss, rhs_s)
    covariance = sl.cho_solve(cf_ss, np.eye(ndim, dtype=float))
    return DeltaWLS(fisher=fisher_ss, covariance=covariance, mean=mean_delta)


def schur_marginalized_mean_given_sampled(
    *,
    pulsar,
    partition,
    variance: np.ndarray,
    design_matrix: np.ndarray,
    sampled_mean: np.ndarray,
) -> np.ndarray:
    """Conditional WLS mean for the analytically marginalized block at ``sampled_mean``."""
    mmat = np.asarray(design_matrix, dtype=float)
    residuals = np.asarray(pulsar.residuals, dtype=float)
    weights = 1.0 / np.asarray(variance, dtype=float)
    sampled_mean = np.asarray(sampled_mean, dtype=float)

    sampled = _as_columns(mmat, tuple(partition.idx_sampled))
    marginalized = _as_columns(mmat, tuple(partition.idx_analytically_marginalized))
    ndim = marginalized.shape[1]
    if ndim == 0:
        return np.zeros(0, dtype=float)

    fisher_sm = _weighted_cross(sampled, weights, marginalized)
    fisher_mm = _weighted_cross(marginalized, weights, marginalized)
    rhs_m = marginalized.T @ (weights * residuals)
    cf_mm = _cho_factor_pd(
        fisher_mm, context="Analytically marginalized timing Fisher block"
    )
    return sl.cho_solve(cf_mm, rhs_m - fisher_sm.T @ sampled_mean)


def _z_space_wls(wls: DeltaWLS, prior_bijector) -> tuple[np.ndarray, np.ndarray]:
    """Map delta-space WLS mean/covariance into local probability-integral-transform (PIT) z coordinates."""
    if prior_bijector is None:
        return wls.mean, wls.covariance
    mean_z = np.asarray(prior_bijector.z_from_delta(wls.mean, np), dtype=float)
    jac = np.asarray(prior_bijector.jacobian_diag_delta_from_z(mean_z, np), dtype=float)
    if np.any(~np.isfinite(jac)) or np.any(jac <= 0.0):
        raise ValueError(
            "Invalid prior bijector Jacobian while building timing transform"
        )
    inv_jac = np.diag(1.0 / jac)
    covariance_z = inv_jac @ wls.covariance @ inv_jac
    return mean_z, covariance_z


def _linear_from_z_covariance(covariance_z: np.ndarray, *, mode: str = "whitening") -> np.ndarray:
    # The only static whitening is the full posterior whitening (§4.4.1); the old
    # diagonal "standardized" mode was retired in favor of the chart system.
    if mode != "whitening":
        raise ValueError(f"Unsupported static whitening mode: {mode!r}")
    try:
        return sl.cholesky(covariance_z, lower=True)
    except sl.LinAlgError as exc:
        raise ValueError(
            "Timing covariance in z coordinates is not numerically positive "
            "definite while building the whitening transform."
        ) from exc


def _linear_transform_from_wls(
    wls: DeltaWLS,
    *,
    prior_bijector,
    mode: str,
) -> WhiteningLinear:
    """Historical WLS-centered likelihood-only transform (§4.3).

    Retained only for reproducing the pinned production commit; it centers on
    the unconstrained WLS point and whitens ``F_z`` alone. It is not the
    future default (see :func:`posterior_linear_transform`).
    """
    z0, covariance_z = _z_space_wls(wls, prior_bijector)
    return WhiteningLinear(
        C=_linear_from_z_covariance(covariance_z, mode=mode),
        z0=z0,
    )


# Interior guard for a proposed local-posterior center in z coordinates. The
# probability-integral-transform (PIT) delta<->z maps clip the CDF at 1e-12, so
# |z| ~ 7.03 is the support edge; 6.0 keeps a smooth margin inside it (§4.1/§4.2).
_Z_GUARD = 6.0


def _reference_z_and_jac(prior_bijector, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Expansion point ``z_e = z(delta=0)`` and ``d(delta)/d(z)`` diag there.

    The probability-integral-transform (PIT) Jacobian is evaluated at the
    deterministic reference, never at a WLS point (§4.3: evaluating it at the
    WLS solution is what drives J1640 PIT coordinates to clipping boundaries and
    magnifies ``C``). With no prior bijector the sampled coordinate is delta
    itself (identity PIT): ``z_e`` is the origin and the Jacobian is unity.
    """
    if prior_bijector is None:
        return np.zeros(ndim, dtype=float), np.ones(ndim, dtype=float)
    z_e = np.asarray(
        prior_bijector.z_from_delta(np.zeros(ndim, dtype=float), np), dtype=float
    )
    jac = np.asarray(prior_bijector.jacobian_diag_delta_from_z(z_e, np), dtype=float)
    if np.any(~np.isfinite(jac)) or np.any(jac <= 0.0):
        raise ValueError(
            "Invalid prior bijector Jacobian at the expansion point while "
            "building the posterior whitening transform"
        )
    return z_e, jac


def _fisher_z(fisher_delta: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """Map a delta-space Fisher to z coordinates: ``F_z = J_e^T F_delta J_e``."""
    scale = jac[:, None] * jac[None, :]
    return np.asarray(fisher_delta, dtype=float) * scale


def _guarded_local_center(
    z_e: np.ndarray,
    fisher_z: np.ndarray,
    posterior_precision: np.ndarray,
    score_delta: np.ndarray,
    jac: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Damped local-posterior Newton origin with a smooth interior guard.

    ``score_delta`` is ``g_delta = d(-log L)/d(delta)`` at the reference. In z,
    ``g_L = J_e^T g_delta`` and the negative-log-posterior gradient at ``z_e``
    is ``g_L + z_e`` (standard-normal PIT prior), giving the local Newton step
    ``q_MAP = -(F_z + I)^-1 (g_L + z_e)`` (§4.2). The raw center ``z_e+q_MAP``
    is passed through ``z_max * tanh(z / z_max)`` so it can never leave PIT
    support; the caller records whether the guard engaged.
    """
    g_L = np.asarray(jac, dtype=float) * np.asarray(score_delta, dtype=float)
    grad_u = g_L + z_e
    cf = _cho_factor_pd(
        posterior_precision, context="Local-posterior Newton metric (F_z + I)"
    )
    q_map = -sl.cho_solve(cf, grad_u)
    raw = z_e + q_map
    guarded = _Z_GUARD * np.tanh(raw / _Z_GUARD)
    engaged = bool(np.any(np.abs(raw) > np.abs(guarded) + 1e-12))
    return guarded, engaged


def posterior_linear_transform(
    fisher_delta: np.ndarray,
    *,
    prior_bijector,
    mode: str,
    score_delta: np.ndarray | None = None,
    origin: str = "reference",
) -> tuple[WhiteningLinear, dict]:
    """Build the posterior whitening transform ``CC^T = (F_z + I)^-1`` (§5.3).

    ``F_z = J_e^T F_delta J_e`` is the local Fisher in the standard-normal PIT
    coordinate; adding ``I`` is the exact prior curvature, not a numerical
    floor. For ``mode='whitening'`` the lower-triangular ``C`` satisfies
    ``C^T (F_z + I) C = I``; for ``mode='standardized'`` ``C`` is the diagonal
    of posterior marginal scales. The affine origin ``z0`` defaults to the
    reference expansion point; ``origin='local_posterior'`` (which needs a
    ``score_delta``) uses a guarded single damped Newton step instead.

    Returns the transform and a diagnostics dict (expansion point, origin
    policy, whether the interior guard engaged).
    """
    fisher_delta = np.asarray(fisher_delta, dtype=float)
    ndim = fisher_delta.shape[0]
    z_e, jac = _reference_z_and_jac(prior_bijector, ndim)
    fisher_z = _fisher_z(fisher_delta, jac)
    posterior_precision = fisher_z + np.eye(ndim, dtype=float)
    cf = _cho_factor_pd(
        posterior_precision, context="Posterior whitening metric (F_z + I)"
    )
    covariance_z = sl.cho_solve(cf, np.eye(ndim, dtype=float))
    C = _linear_from_z_covariance(covariance_z, mode=mode)

    if origin not in {"reference", "auto", "local_posterior"}:
        raise ValueError(f"Unsupported whitening origin: {origin}")
    resolved_origin = origin
    if origin == "auto":
        resolved_origin = "local_posterior" if score_delta is not None else "reference"
    if resolved_origin == "local_posterior" and score_delta is None:
        raise ValueError(
            "origin='local_posterior' requires a score_delta (the likelihood "
            "gradient of -log L at the reference)"
        )

    guard_engaged = False
    if resolved_origin == "local_posterior":
        z0, guard_engaged = _guarded_local_center(
            z_e,
            fisher_z,
            posterior_precision,
            np.asarray(score_delta, dtype=float),
            jac,
        )
    else:
        z0 = z_e

    diagnostics = {
        "expansion_point": "reference",
        "origin": resolved_origin,
        "guard_engaged": guard_engaged,
    }
    return WhiteningLinear(C=C, z0=np.asarray(z0, dtype=float)), diagnostics


def diagonal_white(
    ndim: int | None = None,
    *,
    pulsar=None,
    partition=None,
    prior_bijector=None,
    mode: str = "whitening",
    design_matrix: np.ndarray | None = None,
    origin: str = "reference",
) -> WhiteningLinear:
    """Default TOA-errors reference posterior preconditioner (§5.1 class 1).

    Builds the posterior whitening metric ``CC^T = (F_z + I)^-1`` from the
    diagonal ``toaerrs**2`` reference Fisher. This is only an approximate
    preconditioner for a correlated/marginalized likelihood (§5.1).
    """
    if pulsar is None or partition is None:
        if ndim is None:
            raise ValueError("ndim is required when pulsar/partition are not provided")
        return WhiteningLinear(
            C=np.eye(ndim, dtype=float),
            z0=np.zeros(ndim, dtype=float),
        )

    variance = np.asarray(pulsar.toaerrs, dtype=float) ** 2
    wls = schur_delta_wls(
        pulsar=pulsar,
        partition=partition,
        variance=variance,
        design_matrix=design_matrix,
    )
    linear, _ = posterior_linear_transform(
        wls.fisher,
        prior_bijector=prior_bijector,
        mode=mode,
        origin=origin,
    )
    return linear


def _resolve_noise_value(value, labels: np.ndarray, default: float) -> np.ndarray:
    if value is None:
        return np.full(len(labels), default, dtype=float)
    if isinstance(value, dict):
        return np.asarray(
            [float(value.get(label, default)) for label in labels], dtype=float
        )
    return np.full(len(labels), float(value), dtype=float)


def fixed_hyperparameters(
    ndim: int | None = None,
    hyperparameters: dict | None = None,
    *,
    pulsar=None,
    partition=None,
    prior_bijector=None,
    mode: str = "whitening",
    design_matrix: np.ndarray | None = None,
) -> WhiteningLinear:
    """Deterministic linear transform from fixed hyperparameter snapshot.

    This mirrors diagonal-white construction with serialized EFAC/EQUAD values.
    Red-noise covariance support is intentionally rejected until a named pure
    NumPy builder is added.
    """
    hyperparameters = hyperparameters or {}
    if pulsar is None or partition is None:
        if ndim is None:
            raise ValueError("ndim is required when pulsar/partition are not provided")
        center = hyperparameters.get("center", None)
        z0 = (
            np.zeros(ndim, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )
        if z0.shape != (ndim,):
            raise ValueError("fixed_hyperparameters center must match ndim")
        return WhiteningLinear(C=np.eye(ndim, dtype=float), z0=z0)

    red_noise = hyperparameters.get("red_noise", None)
    if red_noise is not None:
        raise NotImplementedError(
            "fixed_hyperparameters red_noise requires a named pure NumPy builder"
        )

    labels = np.asarray(pulsar.backend_flags)
    efac = _resolve_noise_value(hyperparameters.get("efac", 1.0), labels, 1.0)
    equad = _resolve_noise_value(hyperparameters.get("equad", 0.0), labels, 0.0)
    toaerrs = np.asarray(pulsar.toaerrs, dtype=float)
    variance = (efac * toaerrs) ** 2 + equad**2
    wls = schur_delta_wls(
        pulsar=pulsar,
        partition=partition,
        variance=variance,
        design_matrix=design_matrix,
    )
    linear, _ = posterior_linear_transform(
        wls.fisher,
        prior_bijector=prior_bijector,
        mode=mode,
        origin="reference",
    )
    return linear
