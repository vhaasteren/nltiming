"""One per-parameter timing inference plan (§4.2, §4.5)."""

from __future__ import annotations

import pytest

from nltiming.coordinates import TimingCoordinatePolicy
from nltiming.inference import (
    InferencePreset,
    Marginalize,
    TimingInference,
    coerce_timing_inference,
    resolve_inference_plan,
)
from nltiming.linearity import resolve_linearity


class _Pulsar:
    def __init__(self, fitpars, fitparameters=None):
        self.name = "FAKE"
        self.fitpars = tuple(fitpars)
        if fitparameters is not None:
            self._fitparameters = dict(fitparameters)

    def pint_model(self):
        return None


def _host():
    # F1 sampled-only; JUMP1_a/JUMP1_b are PTA-suffixed JUMPs (base JUMP1); DM.
    return _Pulsar(
        ("F1", "DM", "JUMP1_a", "JUMP1_b"),
        fitparameters={
            "F1": {"shared": "F1"},
            "DM": {"shared": "DM"},
            "JUMP1_a": {"pta_a": "JUMP1"},
            "JUMP1_b": {"pta_b": "JUMP1"},
        },
    )


def _plan(host, inference, *, identically_linear=None):
    linearity = resolve_linearity(host, None, identically_linear=identically_linear)
    return resolve_inference_plan(
        host,
        inference=inference,
        linearity=linearity,
        coordinate_policy=TimingCoordinatePolicy(),
    )


def test_unmentioned_axis_is_sampled():
    plan = _plan(_host(), TimingInference.sample_all())
    assert plan.sampled == ("F1", "DM", "JUMP1_a", "JUMP1_b")
    assert plan.marginalized_delta == ()
    assert plan.marginalized_z == ()


def test_mixed_delta_flat_and_z_prior_groups_resolve_in_fitpar_order():
    plan = _plan(
        _host(),
        TimingInference.groups(delta_flat=["JUMP1"], z_prior=["DM"]),
    )
    assert plan.marginalized_delta == ("JUMP1_a", "JUMP1_b")
    assert plan.marginalized_z == ("DM",)
    assert plan.sampled == ("F1",)
    # proper = sampled + z-marginalized, in fitpar order.
    assert plan.proper == ("F1", "DM")
    assert plan.indices("marginalize_delta_flat") == (2, 3)


def test_group_overlap_after_suffix_expansion_raises():
    inference = TimingInference(
        marginalize={
            "JUMP1": Marginalize.delta_flat(),
            "JUMP1_a": Marginalize.z_prior(),
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        _plan(_host(), inference)


def test_missing_or_ambiguous_selector_raises():
    with pytest.raises(ValueError, match="matches no fit parameter"):
        _plan(_host(), TimingInference.groups(delta_flat=["NOPE"]))


def test_plan_is_exhaustive_disjoint_and_fingerprinted():
    plan = _plan(_host(), TimingInference.groups(delta_flat=["JUMP1"], z_prior=["DM"]))
    dispositions = [a.disposition for a in plan.axes]
    assert len(plan.axes) == len(plan.fitpars) == 4
    covered = set(plan.sampled) | set(plan.marginalized_delta) | set(plan.marginalized_z)
    assert covered == set(plan.fitpars)
    assert len(covered) == len(plan.fitpars)  # disjoint
    assert all(d in {"sample", "marginalize_delta_flat", "marginalize_z_prior"}
               for d in dispositions)

    other = _plan(_host(), TimingInference.sample_all())
    assert plan.fingerprint() != other.fingerprint()
    assert plan.fingerprint() == plan.fingerprint()


def test_linearity_declaration_never_changes_inference_disposition():
    inference = TimingInference.groups(delta_flat=["JUMP1"], z_prior=["DM"])
    a = _plan(_host(), inference)
    b = _plan(_host(), inference, identically_linear=[])  # assert none linear
    assert [x.disposition for x in a.axes] == [x.disposition for x in b.axes]


def test_identically_linear_applies_to_sampled_delta_flat_and_z_marginal_axes():
    # JUMP1_a -> delta_flat; DM -> z_prior; JUMP1_b left sampled but fallback-linear.
    plan = _plan(
        _host(),
        TimingInference.groups(delta_flat=["JUMP1_a"], z_prior=["DM"]),
    )
    assert plan.axis("JUMP1_b").disposition == "sample"
    assert "fallback" in plan.axis("JUMP1_b").linearity_sources  # sampled + linear
    assert plan.axis("JUMP1_a").disposition == "marginalize_delta_flat"
    assert "fallback" in plan.axis("JUMP1_a").linearity_sources  # delta_flat + linear
    assert plan.axis("DM").disposition == "marginalize_z_prior"
    assert "fallback" in plan.axis("DM").linearity_sources  # z-marginal + linear
    # F1 is neither identically linear nor mentioned.
    assert plan.axis("F1").disposition == "sample"
    assert plan.axis("F1").linearity_sources == ()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "default",
        "DEFAULT",
        InferencePreset.DEFAULT,
        TimingInference.default(),
    ],
)
def test_coerce_timing_inference_default_presets(value):
    assert coerce_timing_inference(value) == TimingInference.default()


@pytest.mark.parametrize(
    "value",
    [
        "all",
        "sample_all",
        "ALL",
        InferencePreset.ALL,
        TimingInference.sample_all(),
    ],
)
def test_coerce_timing_inference_sample_all_presets(value):
    assert coerce_timing_inference(value) == TimingInference.sample_all()


def test_coerce_timing_inference_rejects_unknown_and_wrong_type():
    with pytest.raises(ValueError, match="unknown inference preset"):
        coerce_timing_inference("PB")
    with pytest.raises(TypeError, match="TimingInference"):
        coerce_timing_inference(123)


def test_timing_inference_repr_round_trips_common_forms():
    assert repr(TimingInference.default()) == "TimingInference.default()"
    assert repr(TimingInference.sample_all()) == "TimingInference.sample_all()"
    assert (
        repr(TimingInference.groups(delta_flat=["DM1", "DM2"], z_prior=["DM"]))
        == "TimingInference.groups(delta_flat=['DM1', 'DM2'], z_prior=['DM'])"
    )


def test_timing_parameter_plan_repr_summarizes_dispositions():
    plan = _plan(
        _host(),
        TimingInference.groups(delta_flat=["JUMP1"], z_prior=["DM"]),
    )
    text = repr(plan)
    assert text.startswith("TimingParameterPlan(")
    assert "sampled=('F1',)" in text
    assert "marginalized_delta=('JUMP1_a', 'JUMP1_b')" in text
    assert "marginalized_z=('DM',)" in text
