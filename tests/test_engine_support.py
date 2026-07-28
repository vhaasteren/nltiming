"""Backend-neutral engine support tests."""

from __future__ import annotations

import numpy as np
import pytest

from _engine_stubs import JaxLinearTestEngine
from nltiming.engine_support import (
    LinearModel,
    LinearTimingEngine,
    is_exact_linear_param,
    validate_engine_zero_delta,
    zero_delta_tolerance,
)
from nltiming.nonlinear_timing_model import (
    NonLinearTimingModel,
    _normalize_gauge_provenance,
)
from nltiming.protocols import GaugeProvenance
from nltiming.run_io import build_run_manifest
from nltiming import WhiteningConfig, TimingInference


def _gf(**kwargs):
    base = dict(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )
    base.update(kwargs)
    return GaugeProvenance(**base)


def _model():
    return LinearModel.from_design(
        fitpars=("F0", "Offset"),
        design=np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float),
        theta_exact={"F0": "1.0", "Offset": "0.0"},
    )


def test_linear_timing_engine_requires_gauge_provenance():
    with pytest.raises(TypeError):
        LinearTimingEngine(_model())


def test_context_gauge_provenance_for_direct_engine(tmp_path):
    class _Pulsar:
        def __init__(self):
            self.name = "J1234+5678"
            self.fitpars = ("F0", "Offset")
            self._toas = np.linspace(0, 1, 2)
            self._residuals = np.zeros(2)
            self._toaerrs = np.ones(2) * 1e-6
            self._freqs = np.full(2, 1400.0)
            self._flags = {"pta": np.array(["x", "x"])}
            self._backend_flags = np.array(["x", "x"])
            self._backend = JaxLinearTestEngine.from_linear_model(_model())

        @property
        def toas(self):
            return self._toas

        @property
        def residuals(self):
            return self._residuals

        @property
        def toaerrs(self):
            return self._toaerrs

        @property
        def freqs(self):
            return self._freqs

        @property
        def Mmat(self):
            return self._backend.design_matrix()

        @property
        def flags(self):
            return self._flags

        @property
        def backend_flags(self):
            return self._backend_flags

        def state_id(self):
            return "g14a"

        def pint_model(self):
            return None

        def timing_engine(self, engines="jug", **kwargs):
            return self._backend

    pulsar = _Pulsar()
    ntm = NonLinearTimingModel(
        engines="jug",
        whitening=WhiteningConfig(),
        inference=TimingInference.groups(delta_flat=["Offset"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar, condition=True)
    assert len(ctx.gauge_provenance) == 1
    assert ctx.gauge_provenance[0][0] == "J1234+5678"
    assert ctx.gauge_provenance[0][1].export == "none"

    manifest = build_run_manifest(ctx, likelihood="discovery", sampler="test")
    meta = manifest.run_meta()
    assert meta["schema"] == "nlt-run-meta-v4"
    assert "J1234+5678" in meta["gauge"]["contributions"]
    assert meta["gauge"]["export"] == "none"


def test_normalize_raises_when_leaf_omits_gauge_provenance():
    class _NoProv:
        fitpars = ("F0", "Offset")

    class _Pulsar:
        name = "x"
        fitpars = ("F0", "Offset")

    with pytest.raises(ValueError, match="gauge_provenance"):
        _normalize_gauge_provenance(_Pulsar(), _NoProv())


def _gauge_free() -> GaugeProvenance:
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


def _assert_sign_contract(engine, *, modulo_constant: bool = False, rtol=1e-10, atol=1e-12):
    """Δr(δ) ≈ -M δ; on pre-gauged blocks, compare modulo an additive constant."""
    M = np.asarray(engine.design_matrix(), dtype=float)
    delta = np.linspace(0.1, 0.1 * len(engine.fitpars), len(engine.fitpars))
    predicted = -(M @ delta)
    actual = np.asarray(engine.residual_delta(delta), dtype=float)
    if modulo_constant:
        actual = actual - np.mean(actual)
        predicted = predicted - np.mean(predicted)
    np.testing.assert_allclose(actual, predicted, rtol=rtol, atol=atol)


def _sign_model(fitpars=("F0", "Offset"), n=4):
    design = np.ones((n, len(fitpars)), dtype=float)
    for j in range(len(fitpars)):
        design[:, j] = np.linspace(1.0 + j, 2.0 + j, n)
    return LinearModel.from_design(
        fitpars=fitpars,
        design=design,
        theta_exact={name: "0.0" for name in fitpars},
    )


def test_linear_timing_engine_sign_contract():
    eng = LinearTimingEngine(_sign_model(), gauge_provenance=_gauge_free())
    _assert_sign_contract(eng)


def test_exact_linear_policy_does_not_capture_spin_frequency_params():
    assert not is_exact_linear_param("F0")
    assert not is_exact_linear_param("F1")
    assert not is_exact_linear_param("F12")
    assert is_exact_linear_param("Offset")
    # Documented asymmetry vs JUMP*: suffixed Offset is not exact-linear yet.
    assert not is_exact_linear_param("Offset_epta")
    assert is_exact_linear_param("DMX_0001")
    assert is_exact_linear_param("JUMP1")
    assert is_exact_linear_param("JUMP1_epta")


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
