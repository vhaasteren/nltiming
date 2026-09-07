"""psrdata's record engine against this package's engine interface.

This is the consumer side of psrdata SPEC R-2.3 / R-5.4.1 / section 13 item 11:
the record's own linear engine, ``Δr = -Mmat @ δ`` over one ``PulsarData``,
must pass the same runtime and behavioural checks every other engine passes,
for one data set and for several alike, and a composite record needs no
partition declared anywhere because the matrix carries it. These tests are
synthetic on purpose: no timing package.
"""

from __future__ import annotations

import numpy as np
import pytest

from nltiming import TimingInference, TimingSpec, WhiteningConfig
from nltiming.engine_support import (
    LinearContribution,
    LinearTimingEngine,
    validate_engine_against_pulsar,
)
from nltiming.nonlinear_timing_model import (
    GaugeColumnMissingError,
    _normalize_residual_centering,
    assert_gauge_column_present,
)
from nltiming.protocols import JacobianTimingEngine, TimingEngine
from nltiming.run_io import build_run_manifest

psrdata = pytest.importorskip("psrdata")
from psrdata import LinearEngineError, ParameterFact, ResidualCentering  # noqa: E402

PINT_CENTERING = ResidualCentering(
    stored_residuals="none", standard_output="mean_removed", standard_weighted=True
)
TEMPO2_CENTERING = ResidualCentering(
    stored_residuals="none", standard_output="mean_removed", standard_weighted=False
)

UNITS = {
    "F0": "Hz",
    "F1": "Hz / s",
    "DM": "pc / cm3",
    "JUMP1": "s",
    "PHOFF": "dimensionless",
}


def _fact(name: str) -> ParameterFact:
    base = name.split("_")[0]
    if base in ("Offset", "PHOFF"):
        return ParameterFact("0", "s" if base == "Offset" else "dimensionless")
    return ParameterFact("1.0", UNITS.get(base, "s"))


def _record(
    fitpars,
    design,
    *,
    timing_package,
    partim_compatibility,
    residual_centering,
    name="J0000+0000",
):
    """A psrdata record with the given matrix and the shapes the schema needs."""
    n = design.shape[0]
    rng = np.random.default_rng(1)
    planetssb = np.full((n, 9, 6), np.nan)
    fitpars = tuple(fitpars)
    return psrdata.PulsarData(
        name=name,
        fitpars=fitpars,
        setpars=fitpars + ("PSR",),
        parameters={**{p: _fact(p) for p in fitpars}, "PSR": ParameterFact(name, None)},
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
        timing_package=timing_package,
        partim_compatibility=partim_compatibility,
        residual_centering=residual_centering,
        producer="test",
    )


def _composite_record():
    """Two data sets, one shared F0, one local parameter each, per-set offsets."""
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
    M[ppta[: len(ppta) // 2], 4] = 1.0  # a JUMP over half the PPTA rows
    record = _record(
        fitpars,
        M,
        timing_package={"epta": "pint", "ppta": "vela_jax"},
        partim_compatibility={"epta": "pint", "ppta": "tempo2"},
        residual_centering={"epta": PINT_CENTERING, "ppta": TEMPO2_CENTERING},
    )
    return record, epta, ppta


def _single_record(fitpars, M):
    return _record(
        fitpars,
        M,
        timing_package="pint",
        partim_compatibility="pint",
        residual_centering=PINT_CENTERING,
    )


class _RecordPulsar:
    """The smallest ``TimingPulsar`` over a psrdata record.

    Every array is the record's; the engine is the record's own. This is what
    a file-only consumer looks like.
    """

    def __init__(self, record):
        self.record = record
        for field in (
            "name", "fitpars", "toas", "residuals", "toaerrs", "freqs", "Mmat",
            "flags", "backend_flags",
        ):  # fmt: skip
            setattr(self, field, getattr(record, field))

    def pint_model(self):
        return None

    def timing_engine(self, engines="jug", **kwargs):
        return self.record.linear_engine()

    def can_use_engines(self, engines="jug") -> bool:
        return True


# --- the interface, structurally ---------------------------------------------


@pytest.mark.parametrize("which", ["single", "composite"])
def test_the_record_engine_is_a_timing_engine(which):
    if which == "single":
        M = np.column_stack([np.linspace(1.0, 2.0, 5), np.ones(5)])
        record = _single_record(("F0", "Offset"), M)
    else:
        record, _, _ = _composite_record()
    engine = record.linear_engine()
    assert isinstance(engine, TimingEngine)
    assert isinstance(engine, JacobianTimingEngine)
    validate_engine_against_pulsar(engine, record)
    assert engine.engine_name == "linear"
    assert engine.nonlinear_params is None
    assert engine.binary_chart_capability("kepler_laplace", "") is None
    assert set(engine.identically_linear_fitpars()) == set(record.fitpars)
    assert engine.native_units == {
        name: record.parameters[name].units for name in record.fitpars
    }


def test_a_composite_record_is_its_own_linear_engine():
    record, epta, ppta = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)

    delta = np.array([1e-3, 2e-3, -1e-3, 4e-3, 5e-3])
    np.testing.assert_allclose(
        engine.residual_delta(delta), -(np.asarray(record.Mmat) @ delta)
    )
    np.testing.assert_array_equal(engine.design_matrix(), record.Mmat)
    np.testing.assert_array_equal(engine.residual_jacobian(), -record.Mmat)

    blocks = engine.contributions()
    assert tuple(blocks) == ("epta", "ppta")
    assert all(isinstance(c, LinearContribution) for c in blocks.values())
    np.testing.assert_array_equal(blocks["epta"].rows, epta)
    np.testing.assert_array_equal(blocks["ppta"].rows, ppta)
    # Active columns are read off the matrix: nonzero on the data set's rows.
    assert blocks["epta"].fitpars == ("F0", "F1_epta", "Offset_epta")
    assert blocks["ppta"].fitpars == ("F0", "Offset_ppta", "JUMP1_ppta")
    assert blocks["epta"].phase_offset == "Offset_epta"
    assert blocks["ppta"].phase_offset == "Offset_ppta"
    # Each block is a timing engine over its own rows and columns.
    for block in blocks.values():
        assert isinstance(block, TimingEngine)
        sub = delta[list(block.column_indices)]
        np.testing.assert_allclose(
            block.residual_delta(sub), engine.residual_delta(delta)[block.rows]
        )


def test_residual_centering_is_one_entry_per_data_set():
    record, _, _ = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)
    centering = dict(_normalize_residual_centering(record, engine))
    assert centering == {"epta": PINT_CENTERING, "ppta": TEMPO2_CENTERING}
    assert engine.timing_package == {"epta": "pint", "ppta": "vela_jax"}
    assert engine.partim_compatibility == {"epta": "pint", "ppta": "tempo2"}

    M = np.column_stack([np.linspace(1.0, 2.0, 5), np.ones(5)])
    single = _single_record(("F0", "Offset"), M).linear_engine()
    assert dict(_normalize_residual_centering(None, single)) == {
        "single": PINT_CENTERING
    }


# --- the gauge check reads the partition off the matrix --------------------------


def test_a_composite_record_passes_the_gauge_check_without_a_partition():
    record, _, _ = _composite_record()
    engine = LinearTimingEngine.from_pulsar_data(record)
    assert_gauge_column_present(record, engine, np.asarray(record.Mmat))


def test_a_data_set_without_its_own_phase_offset_column_is_refused():
    record, _, _ = _composite_record()
    fitpars = tuple(
        name.replace("Offset_ppta", "Offset_other") for name in record.fitpars
    )
    broken = _record(
        fitpars,
        record.Mmat,
        timing_package=record.timing_package,
        partim_compatibility=record.partim_compatibility,
        residual_centering=record.residual_centering,
    )
    with pytest.raises(LinearEngineError, match="'ppta'"):
        LinearTimingEngine.from_pulsar_data(broken)


def test_data_sets_whose_phase_offset_columns_overlap_are_refused():
    record, _, _ = _composite_record()
    M = np.array(record.Mmat)
    M[:, 3] = 1.0  # Offset_ppta now claims every row
    broken = _record(
        record.fitpars,
        M,
        timing_package=record.timing_package,
        partim_compatibility=record.partim_compatibility,
        residual_centering=record.residual_centering,
    )
    with pytest.raises(LinearEngineError, match="partition"):
        LinearTimingEngine.from_pulsar_data(broken)


def test_the_phase_offset_column_is_the_gauge_direction():
    """A vela-jax record's PHOFF column is 1/F(t), constant to F1*T/F0.

    That is a few 1e-8 on a high-F1 millisecond pulsar over twenty years,
    above the 1e-8 the gauge check lives at. The record stores no separate
    direction (psrdata R-3.6.4): the named column is the direction, and the
    check tests the record's own column. A synthetic engine over the same
    matrix that declares nothing is held to the constant and fails.
    """
    from _engine_stubs import JaxLinearTestEngine
    from nltiming.engine_support import LinearModel

    n = 24
    drift = 1e-7
    phoff = 1.0 / (300.0 * (1.0 + drift * np.linspace(0.0, 1.0, n)))
    M = np.column_stack([np.linspace(1.0, 2.0, n), phoff])
    record = _single_record(("F0", "PHOFF"), M)
    engine = LinearTimingEngine.from_pulsar_data(record)
    assert_gauge_column_present(record, engine, M)

    silent = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(fitpars=record.fitpars, design=M)
    )
    with pytest.raises(GaugeColumnMissingError, match="constant direction"):
        assert_gauge_column_present(record, silent, M)


def test_a_zero_phase_offset_column_is_refused_by_the_record_engine():
    n = 8
    M = np.column_stack([np.linspace(1.0, 2.0, n), np.zeros(n)])
    record = _single_record(("F0", "Offset"), M)
    with pytest.raises(LinearEngineError, match="partition"):
        record.linear_engine()


def test_a_record_without_a_named_phase_offset_column_is_refused():
    M = np.column_stack([np.linspace(1.0, 2.0, 5), np.ones(5)])
    record = _single_record(("F0", "DM"), M)
    with pytest.raises(LinearEngineError, match="phase-offset"):
        record.linear_engine()


# --- end to end: a file-only pulsar through TimingSpec -------------------------


def test_the_composite_round_trips_through_a_feather(tmp_path):
    record, epta, ppta = _composite_record()
    path = tmp_path / "composite.feather"
    record.to_feather(path)
    engine = LinearTimingEngine.from_feather(path)
    blocks = engine.contributions()
    np.testing.assert_array_equal(blocks["epta"].rows, epta)
    np.testing.assert_array_equal(blocks["ppta"].rows, ppta)
    delta = np.ones(len(record.fitpars)) * 1e-3
    np.testing.assert_allclose(
        engine.residual_delta(delta), -(np.asarray(record.Mmat) @ delta)
    )


@pytest.mark.parametrize("which", ["single", "composite"])
def test_timing_spec_builds_a_context_over_the_record_engine(which):
    """The whole consumer path, one data set or several alike (R-5.3.4)."""
    if which == "single":
        M = np.column_stack([np.linspace(1.0, 2.0, 6), np.ones(6)])
        record = _single_record(("F0", "Offset"), M)
        delta_flat = ["Offset"]
        keys = ("single",)
    else:
        record, _, _ = _composite_record()
        delta_flat = ["Offset_epta", "Offset_ppta"]
        keys = ("epta", "ppta")
    pulsar = _RecordPulsar(record)
    ntm = TimingSpec(
        engines="jug",
        whitening=WhiteningConfig(),
        inference=TimingInference.groups(delta_flat=delta_flat),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar, condition=True)
    assert tuple(name for name, _ in ctx.residual_centering) == keys
    assert ntm.for_pulsar(pulsar, condition=True) is ctx

    manifest = build_run_manifest(ctx, likelihood="discovery", sampler="test")
    meta = manifest.run_meta()
    assert meta["schema"] == "nlt-run-meta-v5"
    assert tuple(meta["residual_centering"]["contributions"]) == keys
    assert meta["residual_centering"]["stored_residuals"] == "none"
    assert all(
        meta["native_units"][name] == record.parameters[name].units
        for name in meta["sampled"]
        if name in record.parameters
    )
