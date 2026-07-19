"""Engine-independent interactive timing-model evaluation.

This module turns the low-level :class:`~nltiming.protocols.TimingEngine`
vector contract into a mapping-oriented, immutable user API. It deliberately
does not mutate pulsars, timing sessions, TOAs, or par files.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from .selection import match_fitpars
from .protocols import JaxTimingEngine
from .space import ParameterSpace
from .units import lookup_pint_param, normalize_param_name, units_map

Frame = Literal["delta", "absolute"]
JacobianMethod = Literal["auto", "reference", "analytic", "autodiff"]


def _readonly_array(values: Any, *, dtype: Any = float) -> np.ndarray:
    """Return an owned, read-only array for an immutable result object."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TimingParameter:
    """Metadata for one canonical timing-engine axis."""

    name: str
    base_name: str
    index: int
    reference_exact: str
    reference: float
    native_unit: str
    display_unit: str
    uncertainty: float | None
    sessions: tuple[str, ...]
    aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))


class TimingParameters(Mapping[str, TimingParameter]):
    """Ordered, name-addressable timing parameter metadata."""

    def __init__(self, parameters: Sequence[TimingParameter]):
        self._parameters = tuple(parameters)
        self._by_name = {parameter.name: parameter for parameter in self._parameters}

    def __getitem__(self, name: str) -> TimingParameter:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def select(
        self,
        *,
        base_name: str | None = None,
        session: str | None = None,
    ) -> tuple[TimingParameter, ...]:
        """Select parameters by normalized base name and/or source session."""
        selected = self._parameters
        if base_name is not None:
            normalized = normalize_param_name(base_name)
            selected = tuple(p for p in selected if p.base_name == normalized)
        if session is not None:
            selected = tuple(p for p in selected if session in p.sessions)
        return selected


@dataclass(frozen=True)
class TimingCapabilities:
    """Inspectable capabilities of a resolved timing evaluator."""

    nonlinear: bool
    jax: bool
    autodiff_jacobian: bool
    reference_jacobian: bool
    session_engines: Mapping[str, str]
    exact_linear: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_engines", MappingProxyType(dict(self.session_engines))
        )


@dataclass(frozen=True)
class TimingEvaluation:
    """One immutable timing-model evaluation."""

    fitpars: tuple[str, ...]
    theta: np.ndarray
    delta: np.ndarray
    reference_residuals: np.ndarray
    residual_delta: np.ndarray

    def __post_init__(self) -> None:
        for field in ("theta", "delta", "reference_residuals", "residual_delta"):
            object.__setattr__(self, field, _readonly_array(getattr(self, field)))

    @property
    def residuals(self) -> np.ndarray:
        """Absolute residuals ``r(theta_ref) + residual_delta`` in seconds."""
        return np.asarray(self.reference_residuals + self.residual_delta, dtype=float)

    @property
    def delay(self) -> np.ndarray:
        """Likelihood delay convention, ``-residual_delta``, in seconds."""
        return -self.residual_delta

    @property
    def parameter_values(self) -> dict[str, float]:
        return dict(zip(self.fitpars, self.theta, strict=True))

    @property
    def parameter_deltas(self) -> dict[str, float]:
        return dict(zip(self.fitpars, self.delta, strict=True))

    def rms(self) -> float:
        """Unweighted RMS of absolute residuals in seconds."""
        return float(np.sqrt(np.mean(self.residuals**2)))

    def weighted_rms(self, toaerrs: np.ndarray) -> float:
        """Weighted RMS after removing the weighted mean, in seconds."""
        errors = np.asarray(toaerrs, dtype=float)
        if errors.shape != self.residuals.shape or np.any(errors <= 0):
            raise ValueError("toaerrs must be positive and match residual shape")
        weights = errors**-2
        centered = self.residuals - np.average(self.residuals, weights=weights)
        return float(np.sqrt(np.average(centered**2, weights=weights)))

    def white_chi2(self, toaerrs: np.ndarray) -> float:
        """White-noise chi-square using only the supplied TOA errors."""
        errors = np.asarray(toaerrs, dtype=float)
        if errors.shape != self.residuals.shape or np.any(errors <= 0):
            raise ValueError("toaerrs must be positive and match residual shape")
        return float(np.sum((self.residuals / errors) ** 2))


@dataclass(frozen=True)
class TimingScan:
    """Evaluations along one requested parameter axis."""

    parameter: str
    frame: Frame
    values: np.ndarray
    evaluations: tuple[TimingEvaluation, ...]
    toaerrs: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_array(self.values))
        object.__setattr__(self, "toaerrs", _readonly_array(self.toaerrs))

    @property
    def residuals(self) -> np.ndarray:
        return np.stack([evaluation.residuals for evaluation in self.evaluations])

    @property
    def residual_deltas(self) -> np.ndarray:
        return np.stack([evaluation.residual_delta for evaluation in self.evaluations])

    @property
    def rms(self) -> np.ndarray:
        return np.asarray([evaluation.rms() for evaluation in self.evaluations])

    @property
    def weighted_rms(self) -> np.ndarray:
        return np.asarray(
            [evaluation.weighted_rms(self.toaerrs) for evaluation in self.evaluations]
        )

    @property
    def white_chi2(self) -> np.ndarray:
        return np.asarray(
            [evaluation.white_chi2(self.toaerrs) for evaluation in self.evaluations]
        )


@dataclass(frozen=True)
class TimingFitResult:
    """Immutable result of a local weighted nonlinear least-squares fit."""

    parameters: tuple[str, ...]
    initial: TimingEvaluation
    best_fit: TimingEvaluation
    covariance: np.ndarray
    uncertainties: Mapping[str, float]
    converged: bool
    iterations: int
    chi2: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariance", _readonly_array(self.covariance))
        object.__setattr__(
            self, "uncertainties", MappingProxyType(dict(self.uncertainties))
        )


@dataclass(frozen=True)
class TimingZFitResult:
    """Immutable result of a local weighted least-squares fit in z coordinates."""

    parameters: tuple[str, ...]
    initial: TimingEvaluation
    best_fit: TimingEvaluation
    z_initial: np.ndarray
    z_best: np.ndarray
    covariance: np.ndarray
    covariance_coord: str
    covariance_delta: np.ndarray
    uncertainties: Mapping[str, float]
    uncertainties_delta: Mapping[str, float]
    rank: int
    singular_values: np.ndarray
    iterations: int
    chi2: float

    def __post_init__(self) -> None:
        for field in (
            "z_initial",
            "z_best",
            "covariance",
            "covariance_delta",
            "singular_values",
        ):
            object.__setattr__(self, field, _readonly_array(getattr(self, field)))
        object.__setattr__(self, "covariance_coord", str(self.covariance_coord))
        object.__setattr__(
            self, "uncertainties", MappingProxyType(dict(self.uncertainties))
        )
        object.__setattr__(
            self,
            "uncertainties_delta",
            MappingProxyType(dict(self.uncertainties_delta)),
        )


class TimingEvaluator:
    """Mapping-oriented interactive API over a pulsar timing engine."""

    def __init__(
        self,
        pulsar: Any,
        *,
        engines: str | Mapping[str, str] = "jug",
        prefer_jax: bool = True,
        **engine_kwargs: Any,
    ):
        self.pulsar = pulsar
        self.engines = engines
        self.prefer_jax = bool(prefer_jax)
        self.engine_kwargs = dict(engine_kwargs)
        self.engine = pulsar.timing_engine(engines, **self.engine_kwargs)
        self.fitpars = tuple(self.engine.fitpars)
        self._index = {name: i for i, name in enumerate(self.fitpars)}
        self._reference_exact = dict(self.engine.reference_theta_exact())
        self._reference = np.asarray(self.engine.reference_theta(), dtype=float)
        if self._reference.shape != (len(self.fitpars),):
            raise ValueError("engine reference_theta shape does not match fitpars")
        self.parameters = TimingParameters(self._build_parameters())
        self.capabilities = self._build_capabilities()

    @classmethod
    def from_pulsar(cls, pulsar: Any, **kwargs: Any) -> "TimingEvaluator":
        return cls(pulsar, **kwargs)

    @property
    def reference(self) -> dict[str, float]:
        return dict(zip(self.fitpars, self._reference, strict=True))

    @property
    def reference_exact(self) -> dict[str, str]:
        return dict(self._reference_exact)

    def _parameter_mapping(self) -> Mapping[str, Mapping[str, str]]:
        provider = getattr(self.pulsar, "timing_parameter_mapping", None)
        if provider is not None:
            return cast(Mapping[str, Mapping[str, str]], provider())
        return cast(
            Mapping[str, Mapping[str, str]],
            getattr(self.pulsar, "_fitparameters", {}) or {},
        )

    def _build_parameters(self) -> tuple[TimingParameter, ...]:
        model = self.pulsar.pint_model()
        native_units = units_map(self.fitpars, model, kind="native")
        display_units = units_map(self.fitpars, model, kind="display")
        mapping = self._parameter_mapping()
        parameters = []
        for i, name in enumerate(self.fitpars):
            aliases = dict(mapping.get(name, {}))
            param = lookup_pint_param(model, name)
            uncertainty = getattr(param, "uncertainty_value", None)
            parameters.append(
                TimingParameter(
                    name=name,
                    base_name=normalize_param_name(name),
                    index=i,
                    reference_exact=self._reference_exact[name],
                    reference=float(self._reference[i]),
                    native_unit=native_units[name],
                    display_unit=display_units[name],
                    uncertainty=None if uncertainty is None else float(uncertainty),
                    sessions=tuple(aliases),
                    aliases=aliases,
                )
            )
        return tuple(parameters)

    def _build_capabilities(self) -> TimingCapabilities:
        sessions = getattr(self.engine, "_sessions", ())
        session_engines = {
            str(session.name): str(
                getattr(session.engine, "backend_name", type(session.engine).__name__)
            )
            for session in sessions
        }
        if not session_engines:
            session_engines = {
                str(getattr(self.pulsar, "name", "pulsar")): str(
                    getattr(self.engine, "backend_name", type(self.engine).__name__)
                )
            }
        exact_linear = sorted(
            {
                name
                for session in sessions
                for name in getattr(session, "exact_linear_fitpars", ())
            }
        )
        return TimingCapabilities(
            nonlinear=hasattr(self.engine, "residual_delta"),
            jax=isinstance(self.engine, JaxTimingEngine),
            autodiff_jacobian=isinstance(self.engine, JaxTimingEngine),
            reference_jacobian=hasattr(self.engine, "design_matrix"),
            session_engines=session_engines,
            exact_linear=tuple(exact_linear),
        )

    def _resolve_requested(self, requested: Mapping[str, float]) -> dict[str, float]:
        resolved: dict[str, float] = {}
        for requested_name, value in requested.items():
            hits = match_fitpars(self.pulsar, str(requested_name), self.fitpars)
            if not hits:
                raise KeyError(
                    f"timing parameter {requested_name!r} matches no fitpar; "
                    f"available: {list(self.fitpars)}"
                )
            overlap = set(hits) & set(resolved)
            if overlap:
                raise ValueError(
                    f"parameter request {requested_name!r} overlaps prior requests: "
                    f"{sorted(overlap)}"
                )
            for name in hits:
                resolved[name] = float(value)
        return resolved

    def delta_vector(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        *,
        frame: Frame = "delta",
    ) -> np.ndarray:
        """Convert a partial mapping or full vector to canonical delta order."""
        if frame not in {"delta", "absolute"}:
            raise ValueError("frame must be 'delta' or 'absolute'")
        if values is None:
            return np.zeros(len(self.fitpars), dtype=float)
        if isinstance(values, Mapping):
            resolved = self._resolve_requested(values)
            delta = np.zeros(len(self.fitpars), dtype=float)
            for name, value in resolved.items():
                i = self._index[name]
                delta[i] = value if frame == "delta" else value - self._reference[i]
            return delta
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (len(self.fitpars),):
            raise ValueError(
                f"timing vector must have length {len(self.fitpars)}, got {vector.size}"
            )
        return vector.copy() if frame == "delta" else vector - self._reference

    def evaluate(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        *,
        frame: Frame = "delta",
        use_jax: bool | None = None,
    ) -> TimingEvaluation:
        """Evaluate residuals at a partial or complete timing parameter point."""
        delta = self.delta_vector(values, frame=frame)
        use_jax = self.prefer_jax if use_jax is None else bool(use_jax)
        jax_fn = getattr(self.engine, "residual_delta_jax", None)
        if use_jax and jax_fn is not None:
            from .sampling.numpyro import ensure_x64

            ensure_x64()
            residual_delta = np.asarray(jax_fn(delta), dtype=float)
        else:
            residual_delta = np.asarray(self.engine.residual_delta(delta), dtype=float)
        reference_residuals = np.asarray(self.pulsar.residuals, dtype=float)
        if residual_delta.shape != reference_residuals.shape:
            raise ValueError("engine residual shape does not match pulsar residuals")
        return TimingEvaluation(
            fitpars=self.fitpars,
            theta=self._reference + delta,
            delta=delta,
            reference_residuals=reference_residuals,
            residual_delta=residual_delta,
        )

    def jacobian(
        self,
        at: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        *,
        frame: Frame = "delta",
        method: JacobianMethod = "auto",
    ) -> np.ndarray:
        """Return ``d residual_delta / d delta_theta`` in canonical order."""
        if method == "auto":
            method = "autodiff" if self.capabilities.autodiff_jacobian else "reference"
        if method in {"reference", "analytic"}:
            if at is not None and np.any(self.delta_vector(at, frame=frame)):
                raise ValueError(
                    f"method={method!r} only provides the reference-point matrix; "
                    "use method='autodiff' for an arbitrary point"
                )
            return np.asarray(self.engine.design_matrix(), dtype=float)
        if method != "autodiff":
            raise ValueError(
                "method must be 'auto', 'reference', 'analytic', or 'autodiff'"
            )
        if not isinstance(self.engine, JaxTimingEngine):
            raise ValueError("autodiff requires a JAX-capable timing engine")
        from .sampling.numpyro import ensure_x64

        ensure_x64()
        import jax
        import jax.numpy as jnp

        delta = self.delta_vector(at, frame=frame)
        return np.asarray(
            jax.jacfwd(self.engine.residual_delta_jax)(jnp.asarray(delta)),
            dtype=float,
        )

    def scan(
        self,
        parameter: str,
        values: Sequence[float] | np.ndarray,
        *,
        frame: Frame = "delta",
        scale: str | float | None = None,
        use_jax: bool | None = None,
    ) -> TimingScan:
        """Evaluate a one-dimensional timing parameter scan."""
        hits = match_fitpars(self.pulsar, parameter, self.fitpars)
        if len(hits) != 1:
            raise ValueError(
                f"scan parameter {parameter!r} must resolve to exactly one fitpar; "
                f"matches: {list(hits)}"
            )
        resolved = hits[0]
        axis = np.asarray(values, dtype=float).reshape(-1)
        if scale is not None:
            if frame != "delta":
                raise ValueError("scale is only supported for delta-frame scans")
            if isinstance(scale, str):
                scale_hits = match_fitpars(self.pulsar, scale, self.fitpars)
                if len(scale_hits) != 1:
                    raise ValueError(
                        f"scale parameter {scale!r} must resolve to one fitpar; "
                        f"matches: {list(scale_hits)}"
                    )
                factor = self.reference[scale_hits[0]]
            else:
                factor = float(scale)
            axis = axis * factor
        evaluations = tuple(
            self.evaluate({resolved: value}, frame=frame, use_jax=use_jax)
            for value in axis
        )
        return TimingScan(
            parameter=resolved,
            frame=frame,
            values=axis,
            evaluations=evaluations,
            toaerrs=np.asarray(self.pulsar.toaerrs, dtype=float),
        )

    def _resolve_space_parameters(
        self, space: ParameterSpace, parameters: Sequence[str] | None
    ) -> tuple[list[str], np.ndarray]:
        if parameters is None:
            selected = list(space.names)
        else:
            selected = []
            for requested in parameters:
                hits = [
                    hit for hit in match_fitpars(self.pulsar, requested, space.names)
                ]
                if not hits and requested in space.names:
                    hits = [requested]
                for hit in hits:
                    if hit not in selected:
                        selected.append(hit)
        if not selected:
            raise ValueError("parameters matches no timing fitpars in ParameterSpace")
        missing = [name for name in selected if name not in self._index]
        if missing:
            raise ValueError(
                "ParameterSpace contains names absent from evaluator fitpars: "
                f"{missing}"
            )
        space_index = {name: i for i, name in enumerate(space.names)}
        space_indices = np.asarray([space_index[name] for name in selected], dtype=int)
        return selected, space_indices

    def _z_vector(
        self,
        space: ParameterSpace,
        values: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        if values is None:
            return np.zeros(space.ndim, dtype=float)
        if isinstance(values, Mapping):
            z = np.zeros(space.ndim, dtype=float)
            space_index = {name: i for i, name in enumerate(space.names)}
            for requested_name, value in values.items():
                hits = [
                    hit
                    for hit in match_fitpars(
                        self.pulsar, str(requested_name), space.names
                    )
                ]
                if not hits and requested_name in space_index:
                    hits = [str(requested_name)]
                if not hits:
                    raise KeyError(
                        f"z parameter {requested_name!r} matches no ParameterSpace "
                        f"name; available: {list(space.names)}"
                    )
                for hit in hits:
                    z[space_index[hit]] = float(value)
            return z
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.shape != (space.ndim,):
            raise ValueError(
                f"z vector must have length {space.ndim}, got {vector.size}"
            )
        return vector.copy()

    def _delta_from_space_z(self, space: ParameterSpace, z: np.ndarray) -> np.ndarray:
        delta = np.zeros(len(self.fitpars), dtype=float)
        transformed_delta = np.asarray(space.delta_from_z(z, np), dtype=float)
        for i, name in enumerate(space.names):
            if name not in self._index:
                raise ValueError(
                    "ParameterSpace contains names absent from evaluator fitpars: "
                    f"{name!r}"
                )
            delta[self._index[name]] = transformed_delta[i]
        return delta

    def jacobian_z(
        self,
        space: ParameterSpace,
        z: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        *,
        method: JacobianMethod = "auto",
    ) -> np.ndarray:
        """Return ``d residual_delta / d z`` for the supplied ParameterSpace."""
        z_vector = self._z_vector(space, z)
        delta = self._delta_from_space_z(space, z_vector)
        jacobian_at = None if method in {"reference", "analytic"} else delta
        jacobian_delta = self.jacobian(jacobian_at, frame="delta", method=method)
        timing_indices = np.asarray(
            [self._index[name] for name in space.names], dtype=int
        )
        d_delta_d_z = np.asarray(
            space.prior_bijector.jacobian_diag_delta_from_z(z_vector, np),
            dtype=float,
        )
        return jacobian_delta[:, timing_indices] * d_delta_d_z[None, :]

    def fit_z(
        self,
        space: ParameterSpace,
        parameters: Sequence[str] | None = None,
        *,
        initial: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        toaerrs: np.ndarray | None = None,
        jacobian_method: JacobianMethod = "auto",
        iterations: int = 1,
    ) -> TimingZFitResult:
        """Run a fixed-iteration local weighted least-squares fit in z space."""
        selected, space_indices = self._resolve_space_parameters(space, parameters)
        errors = np.asarray(
            self.pulsar.toaerrs if toaerrs is None else toaerrs, dtype=float
        )
        if errors.shape != np.asarray(self.pulsar.residuals).shape or np.any(
            errors <= 0
        ):
            raise ValueError("toaerrs must be positive and match pulsar residuals")
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        z = self._z_vector(space, initial)
        z_initial = z.copy()
        initial_evaluation = self.evaluate(
            self._delta_from_space_z(space, z), frame="delta"
        )
        for _ in range(iterations):
            evaluation = self.evaluate(
                self._delta_from_space_z(space, z), frame="delta"
            )
            jacobian = self.jacobian_z(space, z, method=jacobian_method)
            weighted_jacobian = jacobian[:, space_indices] / errors[:, None]
            weighted_residuals = evaluation.residuals / errors
            step, *_ = np.linalg.lstsq(
                weighted_jacobian, -weighted_residuals, rcond=None
            )
            z[space_indices] += step

        best = self.evaluate(self._delta_from_space_z(space, z), frame="delta")
        final_jacobian = self.jacobian_z(space, z, method=jacobian_method)[
            :, space_indices
        ]
        weighted_final = final_jacobian / errors[:, None]
        singular_values = np.linalg.svd(weighted_final, compute_uv=False)
        eps = np.finfo(weighted_final.dtype).eps
        rank_threshold = (
            singular_values[0] * max(weighted_final.shape) * eps
            if singular_values.size
            else 0.0
        )
        rank = int(np.sum(singular_values > rank_threshold))
        covariance_z = np.linalg.pinv(weighted_final.T @ weighted_final)
        d_delta_d_z = np.asarray(
            space.prior_bijector.jacobian_diag_delta_from_z(z, np),
            dtype=float,
        )[space_indices]
        transform = np.diag(d_delta_d_z)
        covariance_delta = transform @ covariance_z @ transform.T
        uncertainty_z = np.sqrt(np.clip(np.diag(covariance_z), 0.0, None))
        uncertainty_delta = np.sqrt(np.clip(np.diag(covariance_delta), 0.0, None))
        return TimingZFitResult(
            parameters=tuple(selected),
            initial=initial_evaluation,
            best_fit=best,
            z_initial=z_initial,
            z_best=z,
            covariance=covariance_z,
            covariance_coord="z",
            covariance_delta=covariance_delta,
            uncertainties=dict(zip(selected, uncertainty_z, strict=True)),
            uncertainties_delta=dict(zip(selected, uncertainty_delta, strict=True)),
            rank=int(rank),
            singular_values=singular_values,
            iterations=iterations,
            chi2=best.white_chi2(errors),
        )

    def fit(
        self,
        parameters: Sequence[str],
        *,
        initial: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
        frame: Frame = "delta",
        toaerrs: np.ndarray | None = None,
        jacobian_method: JacobianMethod = "auto",
        max_iter: int = 12,
        tolerance: float = 1e-12,
    ) -> TimingFitResult:
        """Run an immutable local weighted Gauss-Newton timing fit.

        This intentionally uses only diagonal TOA uncertainties. Correlated
        noise and generalized likelihood fits remain likelihood-interface responsibilities.
        """
        selected: list[str] = []
        for requested in parameters:
            for hit in match_fitpars(self.pulsar, requested, self.fitpars):
                if hit not in selected:
                    selected.append(hit)
        if not selected:
            raise ValueError("parameters matches no timing fitpars")
        indices = np.asarray([self._index[name] for name in selected], dtype=int)
        errors = np.asarray(
            self.pulsar.toaerrs if toaerrs is None else toaerrs, dtype=float
        )
        if errors.shape != np.asarray(self.pulsar.residuals).shape or np.any(
            errors <= 0
        ):
            raise ValueError("toaerrs must be positive and match pulsar residuals")
        delta = self.delta_vector(initial, frame=frame)
        initial_evaluation = self.evaluate(delta, frame="delta")
        converged = False
        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            evaluation = self.evaluate(delta, frame="delta")
            jacobian_at = (
                None if jacobian_method in {"reference", "analytic"} else delta
            )
            jacobian = self.jacobian(jacobian_at, frame="delta", method=jacobian_method)
            weighted_jacobian = jacobian[:, indices] / errors[:, None]
            weighted_residuals = evaluation.residuals / errors
            step, *_ = np.linalg.lstsq(
                weighted_jacobian, -weighted_residuals, rcond=None
            )
            delta[indices] += step
            if float(np.linalg.norm(step)) <= float(tolerance) * (
                1.0 + float(np.linalg.norm(delta[indices]))
            ):
                converged = True
                break
        best = self.evaluate(delta, frame="delta")
        final_at = None if jacobian_method in {"reference", "analytic"} else delta
        final_jacobian = self.jacobian(final_at, frame="delta", method=jacobian_method)[
            :, indices
        ]
        weighted_final = final_jacobian / errors[:, None]
        covariance = np.linalg.pinv(weighted_final.T @ weighted_final)
        uncertainty_values = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        return TimingFitResult(
            parameters=tuple(selected),
            initial=initial_evaluation,
            best_fit=best,
            covariance=covariance,
            uncertainties=dict(zip(selected, uncertainty_values, strict=True)),
            converged=converged,
            iterations=iterations,
            chi2=best.white_chi2(errors),
        )
