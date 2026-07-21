"""Authoritative §2.4 engine capability on the PINT adapter (review: this lands
without any MetaPulsar change — PintEngine already wraps the PINT model)."""

import pytest

from nltiming.engines.pint import PintEngine
from nltiming.protocols import BinaryChartCapability


class _LM:
    fitpars = ("ECC", "OM", "T0", "PB")
    native_units: dict[str, str] = {}


class _P:
    def __init__(self, value):
        self.value = value


class _FakeModel:
    def __init__(self, binary, **secular):
        self.BINARY = _P(binary)
        for k, v in secular.items():
            setattr(self, k, _P(v))


def _cap(model):
    eng = PintEngine(engine=None, linear_model=_LM(), pint_model=model)
    return eng.binary_chart_capability("kepler_laplace", "")


def test_pint_capability_returns_none_when_inapplicable():
    m = _FakeModel("DD")
    eng = PintEngine(engine=None, linear_model=_LM(), pint_model=m)
    # wrong family, or no model held -> None (candidacy falls back).
    assert eng.binary_chart_capability("shapiro", "") is None
    assert (
        PintEngine(
            engine=None, linear_model=_LM(), pint_model=None
        ).binary_chart_capability("kepler_laplace", "")
        is None
    )


def test_pint_capability_dd_family_no_secular():
    cap = _cap(_FakeModel("DD"))
    assert isinstance(cap, BinaryChartCapability)
    assert cap.kepler_convention == "dd"
    assert cap.epoch_shift_exact is True
    assert cap.secular_terms == ()
    assert cap.origin_certified is False and cap.supports_domain is True


def test_pint_capability_explicit_secular_not_exact():
    cap = _cap(_FakeModel("DD", OMDOT=1.2e-3, PBDOT=0.0))
    assert cap.epoch_shift_exact is False
    assert "OMDOT" in cap.secular_terms and "PBDOT" not in cap.secular_terms


def test_pint_capability_ddgr_derived_secular():
    # DDGR derives OMDOT/PBDOT internally (no explicit fitpar) -> not exact.
    cap = _cap(_FakeModel("DDGR"))
    assert cap.kepler_convention == "dd"
    assert cap.epoch_shift_exact is False
    assert {"OMDOT", "PBDOT"} <= set(cap.secular_terms)


def test_pint_capability_non_kepler_convention():
    # An ELL1(-family) binary is not the ECC/OM/T0 convention (in practice it is
    # filtered earlier as already_laplace, but the capability is honest).
    cap = _cap(_FakeModel("ELL1"))
    assert cap.kepler_convention == "other"


def test_pint_capability_real_bt_model():
    pytest.importorskip("pint")
    import pint.config as pint_config
    from pint.models import get_model

    model = get_model(pint_config.examplefile("J0613-sim.par"))
    cap = _cap(model)
    assert cap.kepler_convention == "dd"  # BINARY == BT
    assert cap.epoch_shift_exact is True  # no active secular rates
    assert cap.secular_terms == ()
