"""Pulsar timing engine that assembles per-contribution engines in pulsar row order."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any, Mapping

import numpy as np

from nltiming.protocols import JaxTimingEngine, TimingEngine


@dataclass(frozen=True)
class PtaContribution:
    """One per-PTA contribution (row slice + engine) to a pulsar timing engine."""

    name: str
    row_indices: np.ndarray
    engine: TimingEngine
    exact_linear_fitpars: frozenset[str] = frozenset()
    fallback_reference_exact: Mapping[str, str] = field(default_factory=dict)


def _to_exact_str(value: str) -> str:
    with localcontext() as ctx:
        ctx.prec = 50
        return format(Decimal(value), "f")


class PulsarTimingEngine:
    """Canonical-row-order timing engine over per-contribution engines."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        contributions: list[PtaContribution],
        design_matrix: np.ndarray | None = None,
    ):
        self.fitpars = fitpars
        self.native_units = {name: "native" for name in fitpars}
        self._nrows = int(nrows)
        self._contributions = list(contributions)
        self._global_index = {name: i for i, name in enumerate(self.fitpars)}
        for contribution in self._contributions:
            unknown_exact_linear = [
                name
                for name in contribution.exact_linear_fitpars
                if name not in self._global_index
            ]
            if unknown_exact_linear:
                raise ValueError(
                    f"Contribution '{contribution.name}' declares exact-linear evaluation for "
                    f"unknown fitpars: {unknown_exact_linear}"
                )
        self._design_matrix = (
            None if design_matrix is None else np.asarray(design_matrix, dtype=float)
        )
        self._ref_exact = self._merge_reference_theta_exact()

    @property
    def contributions(self) -> list[PtaContribution]:
        """Per-PTA contributions in pulsar row order."""
        return list(self._contributions)

    def identically_linear_fitpars(self) -> frozenset[str]:
        """Union of per-contribution identically-linear fitpars (§4.3)."""
        out: set[str] = set()
        for contribution in self._contributions:
            out.update(contribution.exact_linear_fitpars)
        return frozenset(out)

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """Forward the §2.4 binary-chart capability to the contribution that owns
        this binary group (ownership split, §2.4.1).

        Composite forwarding is nltiming-side: candidacy calls this on the whole
        pulsar engine, and we delegate to the leaf engine (``JugEngine`` /
        ``PintEngine``) of the contribution owning ``suffix``. Returns ``None``
        (→ candidacy uses its conservative pulsar/name-search fallback) when no
        contribution owns the group, the owner's leaf engine does not implement
        the query, or two contributions sharing an unsuffixed binary DISAGREE —
        we never guess across disagreeing owners. Leaf engines that lack the
        method (e.g. a JugEngine before its translator lands) therefore keep the
        whole group on the fallback, unchanged.
        """
        caps = []
        for contribution in self._contributions:
            cap_fn = getattr(contribution.engine, "binary_chart_capability", None)
            if cap_fn is None or not self._owns_binary_group(contribution, suffix):
                continue
            cap = cap_fn(chart_family, suffix)
            if cap is not None:
                caps.append(cap)
        if not caps:
            return None
        first = caps[0]
        if any(cap != first for cap in caps[1:]):
            return None  # shared-binary contributions disagree -> fall back
        return first

    @staticmethod
    def _owns_binary_group(contribution: PtaContribution, suffix: str) -> bool:
        fitpars = tuple(getattr(contribution.engine, "fitpars", ()))
        if suffix:
            return any(name.endswith(suffix) for name in fitpars)
        # Unsuffixed group: the contribution carries an unsuffixed binary
        # (Kepler or Laplace) coordinate. Multiple contributions may share it;
        # the disagreement guard above keeps that safe.
        binary = {"ECC", "OM", "T0", "EPS1", "EPS2", "TASC"}
        return any(name in binary for name in fitpars)

    def _merge_reference_theta_exact(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for contribution in self._contributions:
            ref = contribution.engine.reference_theta_exact()
            ref = dict(ref) | dict(contribution.fallback_reference_exact)
            for name in self.fitpars:
                if name not in ref:
                    continue
                exact = _to_exact_str(str(ref[name]))
                if name in merged and merged[name] != exact:
                    raise ValueError(
                        f"Shared fitpar '{name}' disagrees across contributions: "
                        f"{merged[name]} != {exact} (contribution={contribution.name})"
                    )
                merged[name] = exact
        for name in self.fitpars:
            if name not in merged:
                raise ValueError(
                    f"No contribution provides reference_theta_exact for '{name}'"
                )
        return merged

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self._ref_exact)

    def reference_theta(self) -> np.ndarray:
        return np.asarray(
            [float(self._ref_exact[name]) for name in self.fitpars], dtype=float
        )

    def _contribution_delta_and_exact_linear(
        self, contribution: PtaContribution, delta_theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        local = np.zeros(len(contribution.engine.fitpars), dtype=float)

        for i, name in enumerate(contribution.engine.fitpars):
            if name in self._global_index:
                local[i] = delta_theta[self._global_index[name]]
            else:
                raise ValueError(
                    f"Contribution '{contribution.name}' fitpar '{name}' is not a canonical pulsar fitpar"
                )

        exact_linear = [
            name
            for name in contribution.exact_linear_fitpars
            if delta_theta[self._global_index[name]] != 0.0
        ]
        if exact_linear and self._design_matrix is not None:
            rows = np.asarray(contribution.row_indices, dtype=int)
            exact_delta = np.zeros(len(rows), dtype=float)
            for name in exact_linear:
                exact_delta += (
                    self._design_matrix[rows, self._global_index[name]]
                    * delta_theta[self._global_index[name]]
                )
        elif exact_linear:
            raise ValueError(
                f"Contribution '{contribution.name}' requires exact-linear evaluation for "
                f"{exact_linear}, but no pulsar design matrix was provided"
            )
        else:
            exact_delta = np.zeros(len(contribution.row_indices), dtype=float)
        return local, exact_delta

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float)
        if delta.shape != (len(self.fitpars),):
            raise ValueError("delta_theta shape mismatch with fitpars")
        out = np.zeros(self._nrows, dtype=float)
        for contribution in self._contributions:
            local_delta, exact_linear = self._contribution_delta_and_exact_linear(
                contribution, delta
            )
            block = (
                np.asarray(contribution.engine.residual_delta(local_delta), dtype=float)
                + exact_linear
            )
            out[np.asarray(contribution.row_indices, dtype=int)] = block
        return out

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        out = np.zeros((self._nrows, len(self.fitpars)), dtype=float)
        for contribution in self._contributions:
            block = np.asarray(
                contribution.engine.design_matrix(params=params), dtype=float
            )
            rows = np.asarray(contribution.row_indices, dtype=int)
            for local_j, name in enumerate(contribution.engine.fitpars):
                if name not in self._global_index:
                    raise ValueError(
                        f"Contribution '{contribution.name}' fitpar '{name}' is not a canonical pulsar fitpar"
                    )
                out[rows, self._global_index[name]] = block[:, local_j]
            if contribution.exact_linear_fitpars:
                if self._design_matrix is None:
                    raise ValueError(
                        f"Contribution '{contribution.name}' requires exact-linear evaluation but no "
                        "pulsar design matrix was provided"
                    )
                for name in contribution.exact_linear_fitpars:
                    out[rows, self._global_index[name]] = self._design_matrix[
                        rows, self._global_index[name]
                    ]
        return out

    def linearized_design_matrix(self, params: Any | None = None) -> np.ndarray:
        """Assemble each contribution's selected linearized residual basis."""
        out = np.zeros((self._nrows, len(self.fitpars)), dtype=float)
        for contribution in self._contributions:
            matrix_fn = getattr(
                contribution.engine,
                "linearized_design_matrix",
                contribution.engine.design_matrix,
            )
            block = np.asarray(matrix_fn(params=params), dtype=float)
            rows = np.asarray(contribution.row_indices, dtype=int)
            for local_j, name in enumerate(contribution.engine.fitpars):
                if name not in self._global_index:
                    raise ValueError(
                        f"Contribution '{contribution.name}' fitpar '{name}' is not a canonical pulsar fitpar"
                    )
                out[rows, self._global_index[name]] = block[:, local_j]
            if contribution.exact_linear_fitpars:
                if self._design_matrix is None:
                    raise ValueError(
                        f"Contribution '{contribution.name}' requires exact-linear evaluation but no "
                        "pulsar design matrix was provided"
                    )
                for name in contribution.exact_linear_fitpars:
                    out[rows, self._global_index[name]] = self._design_matrix[
                        rows, self._global_index[name]
                    ]
        return out


class PulsarJaxTimingEngine(PulsarTimingEngine):
    """Pulsar timing engine with JAX-capable path and precision-critical union."""

    def __init__(
        self,
        *,
        fitpars: tuple[str, ...],
        nrows: int,
        contributions: list[PtaContribution],
        design_matrix: np.ndarray | None = None,
    ):
        super().__init__(
            fitpars=fitpars,
            nrows=nrows,
            contributions=contributions,
            design_matrix=design_matrix,
        )
        self._precision_union = frozenset().union(
            *[
                contribution.engine.precision_critical_fitpars()
                for contribution in contributions
                if isinstance(contribution.engine, JaxTimingEngine)
            ]
        )

    def residual_delta_jax(self, delta_theta):
        import jax.numpy as jnp

        delta = jnp.asarray(delta_theta)
        out = jnp.zeros((self._nrows,), dtype=delta.dtype)
        for contribution in self._contributions:
            if not isinstance(contribution.engine, JaxTimingEngine):
                raise ValueError(
                    f"Contribution '{contribution.name}' does not provide a JAX engine path"
                )
            local = jnp.zeros((len(contribution.engine.fitpars),), dtype=delta.dtype)
            for i, name in enumerate(contribution.engine.fitpars):
                if name in self._global_index:
                    local = local.at[i].set(delta[self._global_index[name]])
                else:
                    raise ValueError(
                        f"Contribution '{contribution.name}' missing global mapping for fitpar '{name}'"
                    )
            block = jnp.asarray(
                contribution.engine.residual_delta_jax(local), dtype=delta.dtype
            )
            out = out.at[jnp.asarray(contribution.row_indices, dtype=int)].set(block)
        return out

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self._precision_union


def build_composite_engine(
    *,
    fitpars: tuple[str, ...],
    nrows: int,
    contributions: list[PtaContribution],
    design_matrix: np.ndarray | None = None,
) -> TimingEngine:
    """Return JAX-capable composite only when all contributions are JAX-capable."""
    all_jax = all(isinstance(s.engine, JaxTimingEngine) for s in contributions)
    cls = PulsarJaxTimingEngine if all_jax else PulsarTimingEngine
    return cls(
        fitpars=fitpars,
        nrows=nrows,
        contributions=contributions,
        design_matrix=design_matrix,
    )
