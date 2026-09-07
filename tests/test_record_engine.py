"""The file-only T0 path over a psrdata record, single-leg and composite.

Everything the linearized model is travels in the record: ``-M @ delta`` over
the record's own matrix. A composite record needs no partition declared
anywhere, because the matrix carries it -- a leg's rows are the support of
its named gauge column and the parameters it owns are the columns nonzero
on those rows. These tests are synthetic on purpose: no timing package.
"""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engine_support import (
    LinearContribution,
    LinearModel,
    LinearTimingEngine,
    RecordLinearTimingEngine,
)
from nltiming.nonlinear_timing_model import (
    GaugeColumnMissingError,
    _normalize_gauge_provenance,
    assert_gauge_column_present,
)
from nltiming.protocols import GaugeProvenance

psrdata = pytest.importorskip("psrdata")

_GAUGE_FREE = {
    "export": "none",
    "reference_mode": "none",
    "reporting_mode": "mean",
    "reporting_weighted": True,
}


def _record(fitpars, design, *, gauge, timing_package, name="J0000+0000"):
    """A psrdata record with the given matrix and the shapes the schema needs."""
    n = design.shape[0]
    rng = np.random.default_rng(1)
    planetssb = np.full((n, 9, 6), np.nan)
    return psrdata.PulsarData(
        name=name,
        fitpars=tuple(fitpars),
        setpars=("PSR",),
        toas=np.linspace(5e9, 5e9 + 6e8, n),
        stoas=np.linspace(5e9, 5e9 + 6e8, n),
        toaerrs=np.full(n, 1e-6),
        residuals=rng.normal(size=n) * 1e-6,
        freqs=np.full(n, 1400.0),
        Mmat=np.asarray(design, dtype=float),
        flags={},
        backend_flags=np.array([""] * n),
        telescope=np.array(["ao"] * n),
        pos=np.array([0.6, 0.8, 0.0]),
        pos_t=np.tile([0.6, 0.8, 0.0], (n, 1)),
        sunssb=np.zeros((n, 6)),
        planetssb=planetssb,
        theta=1.2,
        phi=0.3,
        pdist=(1.0, 0.2),
        dm=12.5,
        dmx=None,
        state_id="synthetic",
        software="test",
        timing_package=timing_package,
        gauge=gauge,
        reference_theta_exact={name: "1.0" for name in fitpars},
        native_units={name: "native" for name in fitpars},
    )


def _composite_record():
    """Two legs, one shared F0, one local parameter each, per-leg offsets."""
    n_epta, n_ppta = 6, 4
    n = n_epta + n_ppta
    epta = np.arange(n_epta)
    ppta = np.arange(n_epta, n)
    fitpars = ("F0", "F1_epta", "Offset_epta", "Offset_ppta", "JUMP1_ppta")
    M = np.zeros((n, len(fitpars)))
    M[:, 0] = np.linspace(1.0, 2.0, n)
    M[epta, 1] = np.linspace(0.1, 0.5, n_epta)
    M[epta, 2] = 1.0
    M[ppta, 3] = 1.0
    M[ppta, 4] = 1.0
    gauge = {
        "epta": dict(_GAUGE_FREE),
        "ppta": {**_GAUGE_FREE, "reporting_weighted": False},
    }
    return _record(fitpars, M, gauge=gauge, timing_package="composite"), epta, ppta


def test_a_composite_record_is_its_own_linear_engine():
    record, epta, ppta = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)

    delta = np.array([1e-3, 2e-3, -1e-3, 4e-3, 5e-3])
    np.testing.assert_allclose(
        engine.residual_delta(delta), -(np.asarray(record.Mmat) @ delta)
    )
    np.testing.assert_array_equal(engine.design_matrix(), record.Mmat)

    by_name = {c.name: c for c in engine.contributions}
    assert set(by_name) == {"epta", "ppta"}
    assert all(isinstance(c, LinearContribution) for c in engine.contributions)
    np.testing.assert_array_equal(by_name["epta"].row_indices, epta)
    np.testing.assert_array_equal(by_name["ppta"].row_indices, ppta)
    # Ownership is read off the matrix: nonzero on the leg's rows.
    assert by_name["epta"].engine.fitpars == ("F0", "F1_epta", "Offset_epta")
    assert by_name["ppta"].engine.fitpars == ("F0", "Offset_ppta", "JUMP1_ppta")
    assert by_name["epta"].engine.gauge_column == "Offset_epta"
    assert by_name["ppta"].engine.gauge_column == "Offset_ppta"


def test_a_composite_record_carries_one_gauge_provenance_per_leg():
    record, _, _ = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)
    provenance = dict(_normalize_gauge_provenance(record, engine))
    assert set(provenance) == {"epta", "ppta"}
    assert provenance["epta"].reporting_weighted is True
    assert provenance["ppta"].reporting_weighted is False
    with pytest.raises(AttributeError, match="per contribution"):
        engine.gauge_provenance()
    assert engine.gauge_applied is False


def test_a_composite_record_passes_the_gauge_check_without_a_partition():
    record, _, _ = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)
    assert_gauge_column_present(record, engine, np.asarray(record.Mmat))

    # And so does a plain linear engine over the same matrix that declares
    # no contributions at all: the check reads the partition off the columns.
    plain = LinearTimingEngine(
        LinearModel.from_design(fitpars=record.fitpars, design=record.Mmat),
        gauge_provenance=GaugeProvenance(**_GAUGE_FREE),
    )
    assert_gauge_column_present(record, plain, np.asarray(record.Mmat))


def test_a_leg_without_its_own_gauge_column_is_refused():
    record, _, _ = _composite_record()
    fitpars = tuple(
        name.replace("Offset_ppta", "Offset_other") for name in record.fitpars
    )
    broken = _record(
        fitpars, record.Mmat, gauge=dict(record.gauge), timing_package="composite"
    )
    with pytest.raises(ValueError, match="leg 'ppta' must carry exactly one"):
        LinearTimingEngine.from_pulsar_data(broken)


def test_legs_whose_gauge_columns_overlap_are_refused():
    record, _, _ = _composite_record()
    M = np.array(record.Mmat)
    M[:, 3] = 1.0  # Offset_ppta now claims every row
    broken = _record(
        record.fitpars, M, gauge=dict(record.gauge), timing_package="composite"
    )
    with pytest.raises(ValueError, match="partition the rows"):
        LinearTimingEngine.from_pulsar_data(broken)


def test_a_single_leg_record_declares_its_gauge_column_as_the_direction():
    """A vela-jax record's PHOFF column is 1/F(t), constant to F1*T/F0.

    That is a few 1e-8 on a high-F1 millisecond pulsar over twenty years,
    above the 1e-8 the gauge check lives at. The record engine declares the
    column, so the check tests the record's own direction; an engine that
    declares nothing is held to the constant and fails, which is the gap the
    declaration closes.
    """
    n = 24
    drift = 1e-7
    phoff = 1.0 / (300.0 * (1.0 + drift * np.linspace(0.0, 1.0, n)))
    M = np.column_stack([np.linspace(1.0, 2.0, n), phoff])
    record = _record(("F0", "PHOFF"), M, gauge=dict(_GAUGE_FREE), timing_package="pint")
    engine = LinearTimingEngine.from_pulsar_data(record)
    assert isinstance(engine, RecordLinearTimingEngine)
    assert engine.contributions is None
    np.testing.assert_array_equal(engine.gauge_direction(), phoff)
    assert_gauge_column_present(record, engine, M)

    silent = LinearTimingEngine(
        LinearModel.from_design(fitpars=record.fitpars, design=M),
        gauge_provenance=GaugeProvenance(**_GAUGE_FREE),
    )
    with pytest.raises(GaugeColumnMissingError, match="constant direction"):
        assert_gauge_column_present(record, silent, M)


def test_a_record_engine_still_fails_on_a_zero_gauge_column():
    """Declaring the column is not an escape hatch: a zero column is refused."""
    n = 8
    M = np.column_stack([np.linspace(1.0, 2.0, n), np.zeros(n)])
    record = _record(
        ("F0", "Offset"), M, gauge=dict(_GAUGE_FREE), timing_package="pint"
    )
    engine = LinearTimingEngine.from_pulsar_data(record)
    with pytest.raises(ValueError, match="all zero"):
        assert_gauge_column_present(record, engine, M)


def test_a_record_without_a_named_gauge_column_is_refused():
    M = np.column_stack([np.linspace(1.0, 2.0, 5), np.ones(5)])
    record = _record(("F0", "DM"), M, gauge=dict(_GAUGE_FREE), timing_package="pint")
    with pytest.raises(ValueError, match="exactly one named gauge column"):
        LinearTimingEngine.from_pulsar_data(record)


def test_the_composite_round_trips_through_a_feather(tmp_path):
    record, epta, ppta = _composite_record()
    path = tmp_path / "composite.feather"
    record.to_feather(path)
    engine = LinearTimingEngine.from_feather(path)
    by_name = {c.name: c for c in engine.contributions}
    np.testing.assert_array_equal(by_name["epta"].row_indices, epta)
    np.testing.assert_array_equal(by_name["ppta"].row_indices, ppta)
    delta = np.ones(len(record.fitpars)) * 1e-3
    np.testing.assert_allclose(
        engine.residual_delta(delta), -(np.asarray(record.Mmat) @ delta)
    )
