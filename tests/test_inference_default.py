"""Default delta-flat inference preset (§4.2).

Ported from the former partition-policy tests: these exercise
``inference.default_delta_flat_fitpars`` and ``TimingInference.default()``, the
highest-risk resolution path (PINT category discovery, suffixed composites,
astrometry position vs kinematics, and the broken-suffix guard).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nltiming.coordinates import TimingCoordinatePolicy
from nltiming.inference import (
    TimingInference,
    default_delta_flat_fitpars,
    resolve_inference_plan,
)
from nltiming.linearity import resolve_linearity


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
            "RAJ", "DECJ", "PMRA", "PX", "F0", "F1", "DM", "DM1",
            "DMX_0001", "A1", "PB", "JUMP1",
        )
        self._model = _FakeModel()

    def pint_model(self):
        return self._model


def test_default_preset_resolves_only_documented_default_families():
    delta_flat = default_delta_flat_fitpars(_FakePulsar())
    assert delta_flat == (
        "RAJ", "DECJ", "F0", "F1", "DM", "DM1", "DMX_0001", "JUMP1",
    )


def test_default_preset_plan_dispositions_and_indices():
    pulsar = _FakePulsar()
    plan = resolve_inference_plan(
        pulsar,
        inference=TimingInference.default(),
        linearity=resolve_linearity(pulsar, None),
        coordinate_policy=TimingCoordinatePolicy(),
    )
    assert plan.marginalized_delta == (
        "RAJ", "DECJ", "F0", "F1", "DM", "DM1", "DMX_0001", "JUMP1",
    )
    assert plan.sampled == ("PMRA", "PX", "A1", "PB")
    assert plan.indices("marginalize_delta_flat") == (0, 1, 4, 5, 6, 7, 8, 11)
    assert plan.indices("sample") == (2, 3, 9, 10)
    assert plan.marginalized_z == ()


def test_default_astrometry_samples_kinematics_and_marginalizes_position():
    delta_flat = set(default_delta_flat_fitpars(_FakePulsar()))
    assert {"RAJ", "DECJ"} <= delta_flat
    assert not ({"PMRA", "PX"} & delta_flat)  # kinematics/parallax stay sampled


class _FakeCompositeModel:
    def __init__(self):
        self.components = {
            "astro": _FakeComponent("astrometry", ("RAJ", "DECJ", "PMRA", "PMDEC")),
            "spin": _FakeComponent("spindown", ("F0", "F1")),
            "dmx": _FakeComponent("dispersion_dmx", ("DMX_0001", "DMX_0002")),
            "jumps": _FakeComponent("phase_jump", ("JUMP1", "JUMP2", "JUMP3")),
            "binary": _FakeComponent("pulsar_system", ("PB", "A1", "ECC", "T0")),
        }


class _FakeCompositePulsar:
    def __init__(self, *, with_mapping: bool = True):
        bases = (
            "RAJ", "DECJ", "PMRA", "PMDEC", "F0", "F1", "DMX_0001", "DMX_0002",
            "JUMP1", "JUMP2", "JUMP3", "PB", "A1", "ECC", "T0", "Offset",
        )
        self.fitpars = tuple(f"{b}_ng5" for b in bases)
        self._fitparameters = (
            {f"{b}_ng5": {"ng5": b} for b in bases} if with_mapping else {}
        )
        self._model = _FakeCompositeModel()

    def pint_model(self):
        return self._model


def test_default_marginalizes_linear_block_on_suffixed_composite_pulsar():
    plan = resolve_inference_plan(
        _FakeCompositePulsar(),
        inference=TimingInference.default(),
        linearity=resolve_linearity(_FakeCompositePulsar(), None),
        coordinate_policy=TimingCoordinatePolicy(),
    )
    assert plan.sampled == (
        "PMRA_ng5", "PMDEC_ng5", "PB_ng5", "A1_ng5", "ECC_ng5", "T0_ng5",
    )
    marg = set(plan.marginalized_delta)
    assert {"JUMP1_ng5", "DMX_0001_ng5", "RAJ_ng5", "Offset_ng5"} <= marg
    assert not ({"PMRA_ng5", "PMDEC_ng5"} & marg)
    assert len(plan.marginalized_delta) == 10


def test_default_guard_raises_when_suffix_mapping_unavailable():
    with pytest.raises(ValueError, match="empty delta-flat set"):
        default_delta_flat_fitpars(_FakeCompositePulsar(with_mapping=False))


def test_default_preset_requires_pint_model():
    class _NoModel:
        fitpars = ("F0", "DM")

        def pint_model(self):
            return None

    with pytest.raises(ValueError, match="pint_model"):
        default_delta_flat_fitpars(_NoModel())
