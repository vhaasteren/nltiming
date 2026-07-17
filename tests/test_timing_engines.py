"""Slice-3 tests for per-PTA timing engines and validators."""

import numpy as np
import pytest

from nltiming.engines.base import (
    LinearModel,
    is_exact_linear_param,
    validate_engine_shapes,
    validate_engine_zero_delta,
    zero_delta_tolerance,
)
from nltiming.engines.jug import (
    JugEngine,
    LinearizedJugEngine,
)
from nltiming.engines.pint import (
    PintEngine,
    LinearizedPintEngine,
)
from nltiming.engines.tempo2 import (
    LibstempoEngine,
    LinearizedLibstempoEngine,
)


class _FakeDeltaEngine:
    def delta_residuals(self, delta_params):
        delta = np.array([delta_params["F0"], delta_params["F1"]], dtype=float)
        return _linear_model().design @ delta


class _StrictTempo2Engine:
    def __init__(self):
        self._reference_values = {"PB": 1.0}
        self.calls: list[dict[str, float]] = []

    def delta_residuals(self, delta_params):
        unknown = set(delta_params) - set(self._reference_values)
        if unknown:
            raise KeyError(f"unexpected native params: {unknown}")
        self.calls.append(dict(delta_params))
        return np.array([2.0, 3.0, 5.0], dtype=float) * delta_params.get("PB", 0.0)


class _FakeLTPulsarParam:
    def __init__(self, val: float):
        self.val = val


class _FakeLTPulsarWithJump:
    def __init__(self):
        self._params = {"PB": _FakeLTPulsarParam(1.0)}

    def pars(self, which=None):
        if which == "set":
            return ["PB", "JUMP"]
        return ["PB", "JUMP"]

    def __getitem__(self, name):
        if name not in self._params:
            raise KeyError(name)
        return self._params[name]

    def residuals(self):
        return np.zeros(3, dtype=float)

    def designmatrix(self):
        return np.array(
            [
                [1.0, 10.0],
                [1.0, 11.0],
                [1.0, 13.0],
            ],
            dtype=float,
        )

    def formbats(self):
        return None


class _FakeJaxState:
    def residual_delta_np(self, delta):
        return _linear_model().design @ np.asarray(delta, dtype=float)

    def residual_delta_jax(self, delta):
        import jax.numpy as jnp

        return jnp.asarray(_linear_model().design) @ jnp.asarray(delta)


def _linear_model():
    fitpars = ("F0", "F1")
    design = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    theta_exact = {"F0": "1234.567890123456789", "F1": "-1.0e-15"}
    return LinearModel.from_design(
        fitpars=fitpars, design=design, theta_exact=theta_exact
    )


def _assert_linear_tangent(engine):
    delta = np.array([0.2, -0.5], dtype=float)
    np.testing.assert_allclose(
        engine.residual_delta(delta),
        engine.design_matrix() @ delta,
        atol=1e-12,
    )
    validate_engine_zero_delta(engine)
    validate_engine_shapes(engine)


def test_pint_engine_linear_contract():
    engine = LinearizedPintEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(engine)
    assert engine.fitpars == ("F0", "F1")
    assert set(engine.reference_theta_exact()) == {"F0", "F1"}


def test_tempo2_engine_linear_contract():
    engine = LinearizedLibstempoEngine.from_linear_model(_linear_model())
    _assert_linear_tangent(engine)
    assert engine.fitpars == ("F0", "F1")


def test_jug_engine_jax_surface_and_precision_metadata():
    engine = LinearizedJugEngine.from_linear_model(
        _linear_model(),
        compatibility="tempo2",
        precision_critical=frozenset({"F0"}),
    )
    _assert_linear_tangent(engine)
    assert engine.compatibility == "tempo2"
    assert engine.precision_critical_fitpars() == frozenset({"F0"})

    jnp = __import__("jax.numpy", fromlist=["*"])
    delta = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(delta)),
        engine.design_matrix() @ np.asarray(delta),
        atol=1e-12,
    )


def test_native_engines_wrap_engines_with_pulsar_metadata():
    model = _linear_model()
    engines = [
        PintEngine(engine=_FakeDeltaEngine(), linear_model=model),
        LibstempoEngine(engine=_FakeDeltaEngine(), linear_model=model),
        JugEngine(state=_FakeJaxState(), linear_model=model),
    ]
    for engine in engines:
        _assert_linear_tangent(engine)
        assert engine.reference_theta_exact()["F0"] == "1234.567890123456789"


def test_jug_engine_adds_exact_linear_to_numpy_and_jax_paths():
    model = LinearModel.from_design(
        fitpars=("PB", "Offset"),
        design=np.array(
            [
                [2.0, 1.0],
                [3.0, 1.0],
                [5.0, 1.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "Offset": "0.0"},
    )
    state = _FakeJaxState()
    state.residual_delta_np = lambda delta: model.design[:, :1] @ np.asarray(
        delta, dtype=float
    )

    def residual_delta_jax(delta):
        import jax.numpy as jnp

        return jnp.asarray(model.design[:, :1]) @ jnp.asarray(delta)

    state.residual_delta_jax = residual_delta_jax
    engine = JugEngine(state=state, linear_model=model)
    engine._jug_indices = (0,)
    engine._jug_fitpars = ("PB",)
    engine._exact_linear_indices = (1,)
    engine._exact_linear_fitpars = frozenset({"Offset"})

    delta = np.array([0.5, -0.25], dtype=float)
    expected = model.design @ delta
    np.testing.assert_allclose(engine.residual_delta(delta), expected)

    jnp = __import__("jax.numpy", fromlist=["*"])
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(jnp.asarray(delta))),
        expected,
    )


def test_jug_engine_converts_astrometry_fit_units_to_native():
    """RAJ/DECJ deltas are scaled from pulsar fit units to JUG native radians.

    ``MetaPulsar.Mmat`` carries RAJ in hourangle and DECJ in degrees, while the
    frozen ``JaxTimingState`` is native (radians). Without the conversion the
    residual response is over-scaled by ``12/pi`` (RAJ) / ``180/pi`` (DECJ);
    with it, ``residual_delta == design_matrix @ delta`` holds for every axis.
    """
    pytest.importorskip("jax")
    pytest.importorskip("jug.utils.units")
    import jax.numpy as jnp
    from jug.utils.units import native_to_fit_value

    fitpars = ("RAJ", "DECJ", "F0")
    pulsar_design = np.array(
        [
            [2.0, 1.0, 0.5],
            [3.0, -1.0, 1.0],
            [5.0, 0.5, -0.5],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    model = LinearModel.from_design(
        fitpars=fitpars,
        design=pulsar_design,
        theta_exact={"RAJ": "0.0", "DECJ": "0.0", "F0": "100.0"},
    )
    # Native state: same physical derivative, re-expressed per column in native
    # units (host_col * fit_per_native), acting linearly on the native delta.
    scale = np.array([native_to_fit_value(name, 1.0) for name in fitpars])
    native_design = pulsar_design * scale

    class _NativeState:
        design_matrix = native_design
        fit_params = fitpars
        param_mapping = ()

        def residual_delta_np(self, delta):
            return native_design @ np.asarray(delta, dtype=float)

        def residual_delta_jax(self, delta):
            return jnp.asarray(native_design) @ jnp.asarray(delta)

    engine = JugEngine(state=_NativeState(), linear_model=model)
    fit_delta = np.array([7.0e-8, 5.0e-7, 1.0e-9], dtype=float)
    expected = pulsar_design @ fit_delta

    np.testing.assert_allclose(engine.residual_delta(fit_delta), expected, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(engine.residual_delta_jax(jnp.asarray(fit_delta))),
        expected,
        rtol=1e-12,
    )
    # linearized_design_matrix is served in pulsar fit units too (native / scale).
    np.testing.assert_allclose(
        engine.linearized_design_matrix(), pulsar_design, rtol=1e-12
    )


def test_exact_linear_policy_does_not_capture_spin_frequency_params():
    assert not is_exact_linear_param("F0")
    assert not is_exact_linear_param("F1")
    assert not is_exact_linear_param("F12")
    assert is_exact_linear_param("Offset")
    assert is_exact_linear_param("DMX_0001")
    assert is_exact_linear_param("JUMP1")


def test_libstempo_engine_routes_jump_through_exact_linear_design_column():
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array(
            [
                [2.0, 10.0],
                [3.0, 11.0],
                [5.0, 13.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )
    strict = _StrictTempo2Engine()
    engine = LibstempoEngine(
        engine=strict,
        linear_model=model,
        native_fitpars=("PB",),
        exact_linear_fitpars=frozenset({"JUMP"}),
    )

    delta = np.array([0.25, -0.5], dtype=float)
    np.testing.assert_allclose(engine.residual_delta(delta), model.design @ delta)
    assert strict.calls == [{"PB": 0.25}]
    assert engine.exact_linear_fitpars() == frozenset({"JUMP"})


def test_libstempo_from_contribution_marks_unsettable_jump_exact_linear():
    model = LinearModel.from_design(
        fitpars=("PB", "JUMP"),
        design=np.array(
            [
                [2.0, 10.0],
                [3.0, 11.0],
                [5.0, 13.0],
            ],
            dtype=float,
        ),
        theta_exact={"PB": "1.0", "JUMP": "0.0"},
    )
    engine = LibstempoEngine.from_contribution(
        _FakeLTPulsarWithJump(), linear_model=model
    )

    assert engine.exact_linear_fitpars() == frozenset({"JUMP"})
    np.testing.assert_allclose(
        engine.residual_delta(np.array([0.0, 0.5], dtype=float)),
        model.design[:, 1] * 0.5,
    )


class _OffsetZeroDeltaBackend:
    backend_name = "jug"
    fitpars = ("F0",)

    def __init__(self, *, compatibility: str, offset_sec: float):
        self.compatibility = compatibility
        self._offset_sec = float(offset_sec)

    def residual_delta(self, delta_theta):
        return np.full(3, self._offset_sec, dtype=float)


def test_zero_delta_tolerance_is_strict_for_jug_tempo2():
    engine = _OffsetZeroDeltaBackend(compatibility="tempo2", offset_sec=2.7e-8)
    assert zero_delta_tolerance(engine, 1e-9) == 1e-9
    with pytest.raises(ValueError, match="residual_delta\\(0\\)"):
        validate_engine_zero_delta(engine, tol=1e-9)

    strict = _OffsetZeroDeltaBackend(compatibility="pint", offset_sec=2.7e-8)
    assert zero_delta_tolerance(strict, 1e-9) == 1e-9
    with pytest.raises(ValueError, match="residual_delta\\(0\\)"):
        validate_engine_zero_delta(strict, tol=1e-9)


def test_zero_delta_tolerance_fails_large_jug_tempo2_offset():
    engine = _OffsetZeroDeltaBackend(compatibility="tempo2", offset_sec=1e-3)
    with pytest.raises(ValueError, match="residual_delta\\(0\\)"):
        validate_engine_zero_delta(engine, tol=1e-9)
