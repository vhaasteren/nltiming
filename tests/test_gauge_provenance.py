"""Gauge provenance population at context construction."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engines.base import LinearModel, LinearTimingEngine
from nltiming.engines.composite import PtaContribution, build_composite_engine
from nltiming.engines.jug import JugEngine, LinearizedJugEngine
from nltiming.engines.pint import LinearizedPintEngine, PintEngine
from nltiming.engines.tempo2 import LibstempoEngine, LinearizedLibstempoEngine
from nltiming.nonlinear_timing_model import (
    NonLinearTimingModel,
    _normalize_gauge_provenance,
)
from nltiming.protocols import GaugeProvenance
from nltiming.run_io import build_run_manifest
from nltiming import TimingInference, WhiteningConfig


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


def test_leaf_provenance_table():
    model = _model()
    cases = [
        (
            LinearizedPintEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LinearizedJugEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LinearizedLibstempoEngine.from_linear_model(model),
            "none",
            "unknown",
            False,
        ),
        (
            LibstempoEngine(
                engine=type(
                    "E",
                    (),
                    {"delta_residuals": lambda self, d: np.zeros(2)},
                )(),
                linear_model=model,
            ),
            "applied-unknown",
            "unknown",
            True,
        ),
    ]
    for eng, export, ref_mode, applied in cases:
        prov = eng.gauge_provenance()
        assert prov.export == export
        assert prov.reference_mode == ref_mode
        assert eng.gauge_applied is applied
        assert eng.gauge_applied == (prov.export != "none")


def test_pint_engine_wrapper_provenance():
    model = _model()

    class _Fake:
        def delta_residuals(self, delta_params):
            return np.zeros(2)

    eng = PintEngine(engine=_Fake(), linear_model=model)
    assert eng.gauge_provenance().export == "none"
    assert eng.gauge_applied is False


def test_jug_engine_translates_reference_gauge_without_exporting_jug_type():
    model = _model()

    class _RefGauge:
        mode = "mean"
        weights = np.array([0.5, 0.5])

    class _State:
        compatibility = "pint"
        reference_gauge = _RefGauge()
        param_mapping = ()

        def residual_delta_jax(self, delta):
            import jax.numpy as jnp

            return jnp.zeros((2,))

    eng = JugEngine(state=_State(), linear_model=model)
    prov = eng.gauge_provenance()
    assert prov.export == "none"
    assert prov.reference_mode == "mean"
    assert prov.reference_weighted is True
    # Provenance is backend-neutral — not a JUG ReferenceGauge.
    assert type(prov).__name__ == "GaugeProvenance"
    assert "jug" not in type(prov).__module__


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
            self._backend = LinearizedJugEngine.from_linear_model(_model())

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


def test_composite_or_gauge_applied():
    jug = LinearizedJugEngine.from_linear_model(_model(), gauge_provenance=_gf())
    lib = LibstempoEngine(
        engine=type("E", (), {"delta_residuals": lambda self, d: np.zeros(2)})(),
        linear_model=_model(),
    )
    engine = build_composite_engine(
        fitpars=("F0", "Offset"),
        nrows=4,
        contributions=[
            PtaContribution(name="epta", row_indices=np.array([0, 1]), engine=jug),
            PtaContribution(name="ppta", row_indices=np.array([2, 3]), engine=lib),
        ],
    )
    assert engine.gauge_applied is True
