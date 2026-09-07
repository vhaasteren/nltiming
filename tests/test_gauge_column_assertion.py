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
    engine = CompositeView(
        [
            TestContribution(name="epta", row_indices=np.arange(3), engine=a),
            TestContribution(name="ppta", row_indices=np.arange(3, 6), engine=b),
        ]
    )
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
    engine = CompositeView(
        [
            TestContribution(name="epta", row_indices=np.arange(n), engine=a),
        ]
    )
    with pytest.raises(GaugeColumnMissingError, match="no named gauge column"):
        assert_gauge_column_present(_Pulsar(fitpars, M), engine, M)


def test_without_contributions_each_named_column_is_checked_on_its_support():
    """The partition is in the matrix: ``Offset_<pta>`` is nonzero on its rows.

    An engine that exposes no contributions -- a composite record read back
    from a file, say -- is checked one named gauge column at a time on the
    rows it supports, which asks the per-contribution question from nothing
    but the matrix and the names.
    """
    n = 6
    M = np.zeros((n, 3), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)  # F0
    M[:3, 1] = 1.0  # Offset_epta
    M[3:, 2] = 1.0  # Offset_ppta
    fitpars = ("F0", "Offset_epta", "Offset_ppta")
    eng = JaxLinearTestEngine.from_linear_model(
        LinearModel.from_design(
            fitpars=fitpars,
            design=M,
            theta_exact={name: "0.0" for name in fitpars},
        ),
        gauge_provenance=_gf(),
    )
    assert_gauge_column_present(_Pulsar(fitpars, M), eng, M)

    # A per-PTA column that is not constant on its own support still fails.
    bad = M.copy()
    bad[:3, 1] = np.linspace(1.0, 1.1, 3)
    with pytest.raises(GaugeColumnMissingError, match="constant direction"):
        assert_gauge_column_present(_Pulsar(fitpars, bad), eng, bad)


# --- the declared gauge direction (optional capability) --------------------


class _DeclaringEngine:
    """A leaf whose phase gauge is not the constant direction.

    An engine whose design matrix is ``-J`` of a residual divided by the spin
    Taylor series ``F(t)`` (vela-jax) has a ``PHOFF`` column of ``1/F(t_i)``,
    constant to ``F1*T/F0``: a few 1e-8 on a high-F1 MSP, 7% on this fixture.
    Vela.jl's topocentric divisor moves it by a part in 1e4. Either way this
    check lives at 1e-8, so the engine has to say.
    """

    def __init__(self, direction):
        self._direction = np.asarray(direction, dtype=float)
        self.gauge_applied = False

    def gauge_direction(self):
        return self._direction


def test_a_declared_gauge_direction_is_what_gets_tested():
    n = 8
    direction = 1.0 / np.linspace(100.0, 107.0, n)  # 1/F_i, 7% of drift
    M = np.zeros((n, 2), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:, 1] = direction
    pulsar = _Pulsar(("F0", "PHOFF"), M)
    engine = CompositeView(
        [
            TestContribution(
                name="J0000+0000",
                row_indices=np.arange(n),
                engine=_DeclaringEngine(direction),
            )
        ]
    )
    assert_gauge_column_present(pulsar, engine, M)

    # The same matrix fails for an engine that declares nothing, because for
    # that engine the constant vector *is* the claim being made. Defaulting to
    # the constant is what keeps every existing engine's guard as strict as it
    # was.
    class _Silent:
        gauge_applied = False

    plain = CompositeView(
        [
            TestContribution(
                name="J0000+0000", row_indices=np.arange(n), engine=_Silent()
            )
        ]
    )
    with pytest.raises(GaugeColumnMissingError, match="constant direction"):
        assert_gauge_column_present(pulsar, plain, M)


def test_a_constant_column_still_fails_a_declared_direction():
    """The capability is not an escape hatch: a wrong column is still wrong."""
    n = 8
    direction = 1.0 / np.linspace(100.0, 107.0, n)
    M = np.zeros((n, 2), dtype=float)
    M[:, 0] = np.linspace(1, 2, n)
    M[:, 1] = 1.0  # a constant Offset column, which is *not* this gauge
    pulsar = _Pulsar(("F0", "PHOFF"), M)
    engine = CompositeView(
        [
            TestContribution(
                name="J0000+0000",
                row_indices=np.arange(n),
                engine=_DeclaringEngine(direction),
            )
        ]
    )
    with pytest.raises(GaugeColumnMissingError, match="constant direction"):
        assert_gauge_column_present(pulsar, engine, M)


def test_a_malformed_declared_direction_raises():
    n = 8
    M = np.zeros((n, 2), dtype=float)
    M[:, 1] = 1.0
    pulsar = _Pulsar(("F0", "PHOFF"), M)
    for bad in (np.zeros(n), np.ones(3), np.full(n, np.nan)):
        engine = CompositeView(
            [
                TestContribution(
                    name="J0000+0000",
                    row_indices=np.arange(n),
                    engine=_DeclaringEngine(bad),
                )
            ]
        )
        with pytest.raises(ValueError):
            assert_gauge_column_present(pulsar, engine, M)
