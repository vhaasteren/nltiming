"""Slice-2 tests for timing fit-parameter partition policy."""

from dataclasses import dataclass

import pytest

from nltiming.partition import (
    default_analytically_marginalized_fitpars,
    resolve_partition,
)


@dataclass
class _FakeComponent:
    category: str
    params: tuple[str, ...]


class _FakeModel:
    def __init__(self):
        self.components = {
            "astro": _FakeComponent("astrometry", ("RAJ", "DECJ", "PMRA", "PX")),
            "spin": _FakeComponent("spindown", ("F0", "F1", "EDOT")),
            "disp": _FakeComponent("dispersion_constant", ("DM", "DM1")),
            "dmx": _FakeComponent("dispersion_dmx", ("DMX_0001",)),
            "binary": _FakeComponent("pulsar_system", ("A1", "PB")),
        }


class _FakePulsar:
    def __init__(self):
        self.fitpars = (
            "RAJ",
            "DECJ",
            "PMRA",
            "PX",
            "F0",
            "F1",
            "DM",
            "DM1",
            "DMX_0001",
            "A1",
            "PB",
            "JUMP1",
        )
        self._model = _FakeModel()

    def pint_model(self):
        return self._model


def test_default_partition_uses_pint_components():
    pulsar = _FakePulsar()
    analytically_marginalized = default_analytically_marginalized_fitpars(pulsar)
    assert analytically_marginalized == (
        "RAJ",
        "DECJ",
        "F0",
        "F1",
        "DM",
        "DM1",
        "DMX_0001",
        "JUMP1",
    )


def test_resolve_partition_default_and_indices():
    pulsar = _FakePulsar()
    part = resolve_partition(pulsar, analytically_marginalize="default")
    assert part.analytically_marginalized == (
        "RAJ",
        "DECJ",
        "F0",
        "F1",
        "DM",
        "DM1",
        "DMX_0001",
        "JUMP1",
    )
    assert part.sampled == ("PMRA", "PX", "A1", "PB")
    assert part.idx_analytically_marginalized == (0, 1, 4, 5, 6, 7, 8, 11)
    assert part.idx_sampled == (2, 3, 9, 10)


def test_resolve_partition_explicit_list():
    pulsar = _FakePulsar()
    part = resolve_partition(pulsar, analytically_marginalize=["F0", "PB"])
    assert part.analytically_marginalized == ("F0", "PB")
    assert part.sampled == (
        "RAJ",
        "DECJ",
        "PMRA",
        "PX",
        "F1",
        "DM",
        "DM1",
        "DMX_0001",
        "A1",
        "JUMP1",
    )


def test_resolve_partition_none_analytically_marginalized():
    pulsar = _FakePulsar()
    part = resolve_partition(pulsar, analytically_marginalize=None)
    assert part.analytically_marginalized == ()
    assert part.sampled == pulsar.fitpars


def test_resolve_partition_unknown_and_duplicate_errors():
    pulsar = _FakePulsar()
    with pytest.raises(ValueError, match="Unknown fit parameters"):
        resolve_partition(pulsar, analytically_marginalize=["DOES_NOT_EXIST"])
    with pytest.raises(ValueError, match="Duplicate entries"):
        resolve_partition(pulsar, analytically_marginalize=["F0", "F0"])
    with pytest.raises(ValueError, match="analytically_marginalize must be"):
        resolve_partition(pulsar, analytically_marginalize="F0")


class _FakeCompositeModel:
    """Model whose components carry canonical (unsuffixed) PINT param names."""

    def __init__(self):
        self.components = {
            "astro": _FakeComponent("astrometry", ("RAJ", "DECJ", "PMRA", "PMDEC")),
            "spin": _FakeComponent("spindown", ("F0", "F1")),
            "dmx": _FakeComponent("dispersion_dmx", ("DMX_0001", "DMX_0002")),
            "jumps": _FakeComponent("phase_jump", ("JUMP1", "JUMP2", "JUMP3")),
            "binary": _FakeComponent("pulsar_system", ("PB", "A1", "ECC", "T0")),
        }


class _FakeCompositePulsar:
    """Composite pulsar with PTA-suffixed fitpars and a base-name mapping."""

    def __init__(self, *, with_mapping: bool = True):
        bases = (
            "RAJ",
            "DECJ",
            "PMRA",
            "PMDEC",
            "F0",
            "F1",
            "DMX_0001",
            "DMX_0002",
            "JUMP1",
            "JUMP2",
            "JUMP3",
            "PB",
            "A1",
            "ECC",
            "T0",
            "Offset",
        )
        self.fitpars = tuple(f"{b}_ng5" for b in bases)
        self._fitparameters = (
            {f"{b}_ng5": {"ng5": b} for b in bases} if with_mapping else {}
        )
        self._model = _FakeCompositeModel()

    def pint_model(self):
        return self._model


def test_default_analytically_marginalizes_linear_block_on_suffixed_composite_pulsar():
    """Regression: PTA-suffixed names must still map to canonical categories.

    The binary params plus the explicit Offset stay sampled; every discovered
    linear nuisance family (astrometry/spindown/DMX/JUMP) is analytically marginalized.
    """
    pulsar = _FakeCompositePulsar()
    part = resolve_partition(pulsar, analytically_marginalize="default")
    assert part.sampled == (
        "PMRA_ng5",
        "PMDEC_ng5",
        "PB_ng5",
        "A1_ng5",
        "ECC_ng5",
        "T0_ng5",
    )
    assert "JUMP1_ng5" in part.analytically_marginalized
    assert "DMX_0001_ng5" in part.analytically_marginalized
    assert "RAJ_ng5" in part.analytically_marginalized
    assert "PMRA_ng5" not in part.analytically_marginalized
    assert "PMDEC_ng5" not in part.analytically_marginalized
    assert "Offset_ng5" in part.analytically_marginalized
    assert len(part.analytically_marginalized) == 10


def test_default_astrometry_samples_kinematics_and_marginalizes_position():
    pulsar = _FakePulsar()
    part = resolve_partition(pulsar, analytically_marginalize="default")

    assert "RAJ" in part.analytically_marginalized
    assert "DECJ" in part.analytically_marginalized
    assert "PMRA" in part.sampled
    assert "PX" in part.sampled


def test_default_guard_raises_when_suffix_mapping_unavailable():
    """If name matching silently fails, refuse to sample every timing param."""
    pulsar = _FakeCompositePulsar(with_mapping=False)
    with pytest.raises(ValueError, match="empty analytically marginalized set"):
        resolve_partition(pulsar, analytically_marginalize="default")
