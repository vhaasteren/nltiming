"""Per-contribution gauge-column assertion."""

from __future__ import annotations

import numpy as np
import pytest

from _engine_stubs import JaxLinearTestEngine, TestContribution, CompositeView
from nltiming.engine_support import LinearModel
from nltiming.nonlinear_timing_model import (
    GaugeColumnMissingError,
    assert_gauge_column_present,
)
from nltiming.protocols import GaugeProvenance


def _gf():
    return GaugeProvenance(
        export="none",
        reference_mode="none",
        reporting_mode="mean",
        reporting_weighted=True,
    )


class _Pulsar:
    def __init__(self, fitpars, Mmat, name="J0000+0000"):
        self.name = name
        self.fitpars = list(fitpars)
        self.Mmat = np.asarray(Mmat, dtype=float)


def test_passes_on_per_pta_offset_layout():
    n = 6
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)  # F0
    M[:3, 1] = 1.0  # Offset_epta
    M[3:, 2] = 1.0  # Offset_ppta
    fitpars = ("F0", "Offset_epta", "Offset_ppta")
    a = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_epta"),
            design=M[:3, :2],
            theta_exact={"F0": "1.0", "Offset_epta": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    b = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=("F0", "Offset_ppta"),
            design=np.column_stack([M[3:, 0], M[3:, 2]]),
            theta_exact={"F0": "1.0", "Offset_ppta": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    engine = CompositeView([
            TestContribution(name="epta", row_indices=np.arange(3), engine=a),
            TestContribution(name="ppta", row_indices=np.arange(3, 6), engine=b),
        ])
    pulsar = _Pulsar(fitpars, M)
    assert_gauge_column_present(pulsar, engine, M)


def test_fails_when_named_column_dropped():
    M = np.column_stack([np.ones(4), np.linspace(0, 1, 4)])
    fitpars = ("F0", "DM")  # no Offset
    eng = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars, design=M, theta_exact={"F0": "1.0", "DM": "0.0"}
        ),
        gauge_provenance=_gf(),
    )
    with pytest.raises(GaugeColumnMissingError, match="no named gauge column"):
        assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)


def test_fails_when_named_column_zeroed():
    M = np.column_stack([np.linspace(1, 2, 4), np.zeros(4)])
    fitpars = ("F0", "Offset")
    eng = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars, design=M, theta_exact={"F0": "1.0", "Offset": "0.0"}
        ),
        gauge_provenance=_gf(),
    )
    with pytest.raises(GaugeColumnMissingError, match="local numeric"):
        assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)


def test_fails_when_only_unnamed_near_constant_spans():
    # Near-constant F0 spans 1, but Offset is absent.
    M = np.column_stack([np.ones(4) + 1e-12 * np.arange(4), np.linspace(0, 1, 4)])
    fitpars = ("F0", "DM")
    eng = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars, design=M, theta_exact={"F0": "1.0", "DM": "0.0"}
        ),
        gauge_provenance=_gf(),
    )
    with pytest.raises(GaugeColumnMissingError, match="no named gauge column"):
        assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)


def test_fails_when_named_zeroed_while_others_span():
    # Offset zeroed; JUMP is constant and would span under a full-basis check.
    M = np.column_stack([np.linspace(1, 2, 4), np.zeros(4), np.ones(4)])
    fitpars = ("F0", "Offset", "JUMP1")
    eng = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars,
            design=M,
            theta_exact={"F0": "1.0", "Offset": "0.0", "JUMP1": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    with pytest.raises(GaugeColumnMissingError, match="named gauge column"):
        assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)


def test_wrong_pta_offset_name_does_not_satisfy():
    n = 4
    M = np.zeros((n, 2), dtype=float)
    M[:, 0] = 1.0
    M[:, 1] = 1.0  # Offset_other
    fitpars = ("F0", "Offset_other")
    a = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars,
            design=M,
            theta_exact={"F0": "1.0", "Offset_other": "0.0"},
        ),
        gauge_provenance=_gf(),
    )
    engine = CompositeView([
            TestContribution(name="epta", row_indices=np.arange(n), engine=a),
        ])
    with pytest.raises(GaugeColumnMissingError, match="no named gauge column"):
        assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)
