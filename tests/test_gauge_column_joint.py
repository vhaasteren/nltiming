"""Joint gauge-column assertion across contributions."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engines.base import LinearModel
from nltiming.engines.composite import PtaContribution, build_composite_engine
from nltiming.engines.jug import LinearizedJugEngine
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
    def __init__(self, fitpars, Mmat):
        self.name = "J0000+0000"
        self.fitpars = list(fitpars)
        self.Mmat = np.asarray(Mmat, dtype=float)


def _two_leaf_composite(fitpars, M, leaf_fitpars_a, leaf_fitpars_b, design_a, design_b):
    a = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=leaf_fitpars_a,
            design=design_a,
            theta_exact={n: "0.0" for n in leaf_fitpars_a},
        ),
        gauge_provenance=_gf(),
    )
    b = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=leaf_fitpars_b,
            design=design_b,
            theta_exact={n: "0.0" for n in leaf_fitpars_b},
        ),
        gauge_provenance=_gf(),
    )
    n = M.shape[0]
    mid = n // 2
    return build_composite_engine(
        fitpars=fitpars,
        nrows=n,
        contributions=[
            PtaContribution(name="epta", row_indices=np.arange(mid), engine=a),
            PtaContribution(name="ppta", row_indices=np.arange(mid, n), engine=b),
        ],
    )


def test_shared_unsuffixed_offset_fails_joint():
    n = 6
    mid = 3
    M = np.zeros((n, 2), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:, 1] = 1.0  # global Offset — locally constant on each block
    fitpars = ("F0", "Offset")
    engine = _two_leaf_composite(
        fitpars,
        M,
        ("F0", "Offset"),
        ("F0", "Offset"),
        M[:mid],
        M[mid:],
    )
    with pytest.raises(GaugeColumnMissingError, match="joint|deficit"):
        assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)


def test_identical_suffixed_columns_fail_rank():
    n = 6
    mid = 3
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:, 1] = 1.0  # Offset_epta present on ALL rows
    M[:, 2] = 1.0  # Offset_ppta identical over all N rows
    fitpars = ("F0", "Offset_epta", "Offset_ppta")
    engine = _two_leaf_composite(
        fitpars,
        M,
        ("F0", "Offset_epta"),
        ("F0", "Offset_ppta"),
        M[:mid, :2],
        np.column_stack([M[mid:, 0], M[mid:, 2]]),
    )
    with pytest.raises(GaugeColumnMissingError, match="joint"):
        assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)


def test_block_local_columns_pass():
    n = 6
    mid = 3
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:mid, 1] = 1.0
    M[mid:, 2] = 1.0
    fitpars = ("F0", "Offset_epta", "Offset_ppta")
    engine = _two_leaf_composite(
        fitpars,
        M,
        ("F0", "Offset_epta"),
        ("F0", "Offset_ppta"),
        M[:mid, :2],
        np.column_stack([M[mid:, 0], M[mid:, 2]]),
    )
    assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)


def test_k1_reduces_to_per_contribution():
    M = np.column_stack([np.linspace(1, 2, 4), np.ones(4)])
    fitpars = ("F0", "Offset")
    eng = LinearizedJugEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars, design=M, theta_exact={"F0": "1.0", "Offset": "0.0"}
        ),
        gauge_provenance=_gf(),
    )
    assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)
