"""Dynamic decentering transport for samplers without autodiff (PTMCMC).

NumPy/SciPy only — no discovery, enterprise, or JAX imports. The live marginal
products (``G = W_sᵀC(η)⁻¹W_s``, ``b = W_sᵀC(η)⁻¹y_t``) are injected as a
``products_fn`` callable, so this module is frontend-agnostic. The public surface
is duck-compatible with the discovery ``MarginalTransport`` pieces that
downstream code consumes (feature §5, S11), so
:func:`nltiming.metric.dynamic_transport_record` works on it unmodified.

Nothing in the transport may depend on ``xi`` or on the current ``z`` (D-INV,
marginalized D19): ``W_s`` and ``y_t`` are the sealed geometry-plan arrays, baked
into ``products_fn`` at build time. There are no clamps, floors, or translation
modifiers (E12); ``center`` is the only knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sl


@dataclass(frozen=True)
class MarginalProducts:
    """One evaluation of the live marginal products (all NumPy)."""

    G: np.ndarray  # (k, k)  = W_s^T C(eta)^-1 W_s
    b: np.ndarray  # (k,)    = W_s^T C(eta)^-1 y_t


class NumpyMarginalTransport:
    """Live-kernel decentering for ONE external block: ``xi -> (z, ldJ)``.

    ``z = mu(eta) + L(eta)^-T xi``,
    ``A(eta) = G(eta) + diag(prior_precision)``,  ``A = L L^T``,
    ``mu = A^-1 b`` (``center=True``) or ``0``,
    ``ldJ = -sum(log diag L)``.

    ``products_fn(params) -> MarginalProducts`` supplies ``G`` and ``b`` computed
    from the sealed geometry-plan arrays (``W_s``, ``y_t``). Nothing here may
    depend on ``xi`` or ``z`` (D-INV). There are no clamps, floors, or
    translation modifiers; ``center`` is the only knob (E12).
    """

    def __init__(
        self,
        products_fn,
        *,
        dimension,
        key,
        params,
        prior_precision=1.0,
        center=True,
        description="live_kernel_numpy",
    ):
        k = int(dimension)
        if k <= 0:
            raise ValueError(f"dimension must be positive; got {dimension}")
        p = np.asarray(prior_precision, dtype=float)
        p = np.full(k, float(p)) if p.ndim == 0 else p
        if p.shape != (k,) or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(
                "prior_precision must be a finite scalar or (k,) vector >= 0 "
                "(no floors are applied)"
            )
        self.dimension = k
        self.index = {str(key): slice(0, k)}
        self.center = bool(center)
        self.params = tuple(params)
        self._products_fn = products_fn
        self._pinv = p
        self._description = str(description)

    def _factor(self, params):
        pr = self._products_fn(params)
        A = np.asarray(pr.G, dtype=float) + np.diag(self._pinv)
        L = np.linalg.cholesky(A)
        return L, np.asarray(pr.b, dtype=float)

    def apply(self, params, xi):
        L, b = self._factor(params)
        z = sl.solve_triangular(L, np.asarray(xi, dtype=float), lower=True, trans=1)
        ldJ = -float(np.sum(np.log(np.diag(L))))
        if self.center:
            z = z + sl.cho_solve((L, True), b)
        return z, ldJ

    def split(self, z):
        return {key: z[sli] for key, sli in self.index.items()}

    def validate(self, params):
        L, _ = self._factor(params)  # raises LinAlgError if not PD
        if not np.all(np.isfinite(np.diag(L))):
            raise ValueError("transport factor is not finite at params")

    def diagnostics(self, params=None):
        (key,) = self.index
        out = {
            "blocks": [
                {
                    "name": "timing",
                    "k": self.dimension,
                    "params": list(self.params),
                    "keys": [key],
                    "conditioner_kind": "exact_diagonal",
                }
            ],
            "dimension": self.dimension,
            "center": self.center,
            "reference_noise": self._description,
        }
        if params is not None:
            L, _ = self._factor(params)
            d = np.diag(L)
            out["chol_diag_min"] = float(np.min(d))
            out["chol_diag_max"] = float(np.max(d))
        return out

    def fingerprint(self):
        import hashlib
        import json

        payload = {"schema": "nltiming-marginal-transport-v1", **self.diagnostics()}
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        )


def decode_decentered_chain(chain_xi, chain_eta, eta_names, transport, space):
    """Row-wise ``(xi, eta) -> delta``. ``chain_xi``: (n, k); ``chain_eta``: (n, m).

    Dependency-free decode used by both the checkpoint writer and any cold-start
    decoder: each row's ``eta`` reconstructs the transport, ``apply`` maps
    ``xi -> z``, and the chart maps ``z -> delta`` (E26 / §5).
    """
    chain_xi = np.asarray(chain_xi, dtype=float)
    chain_eta = np.asarray(chain_eta, dtype=float)
    out = np.empty_like(chain_xi)
    for i in range(chain_xi.shape[0]):
        params = dict(zip(eta_names, chain_eta[i]))
        z, _ = transport.apply(params, chain_xi[i])
        out[i] = np.asarray(space.delta_from_z(z, np), dtype=float)
    return out
