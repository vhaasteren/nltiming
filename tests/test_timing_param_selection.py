"""Tests for inference-based plan selection and constructor priors."""

import numpy as np
import pytest

from nltiming import priors as prior_specs
from nltiming import TimingInference
from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.selection import (
    fitpar_suffixes,
    match_fitpars,
    select_fitpars,
)


class _SuffixHost:
    """Composite-style pulsar with PTA-suffixed fitpars and _fitparameters mapping."""

    def __init__(self):
        self.name = "J2222+2222"
        self.fitpars = ("F1", "PB_a", "TASC_a", "PB_b", "TASC_b")
        self._fitparameters = {
            "PB_a": {"pta_a": "PB"},
            "TASC_a": {"pta_a": "TASC"},
            "PB_b": {"pta_b": "PB"},
            "TASC_b": {"pta_b": "TASC"},
        }
        n = 7
        self._toas = np.linspace(0.0, 1.0, n)
        self._residuals = np.zeros(n)
        self._toaerrs = np.full(n, 1.0e-6)
        self._freqs = np.full(n, 1400.0)
        self._flags = {"pta": np.array(["demo"] * n, dtype="U8")}
        self._backend_flags = np.array(["demo"] * n, dtype="U8")
        rng = np.random.default_rng(42)
        design = np.column_stack(
            [np.linspace(-0.5, 0.5, n)]
            + [rng.normal(size=n) for _ in range(len(self.fitpars) - 1)]
        )
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=design,
            theta_exact={
                "F1": "1.0",
                "PB_a": "10.0",
                "TASC_a": "55000.0",
                "PB_b": "20.0",
                "TASC_b": "56000.0",
            },
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

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
        return "suffix-token"

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def pulsar():
    return _SuffixHost()


def test_match_fitpars_base_name_matches_all_suffixed(pulsar):
    fitpars = pulsar.fitpars
    assert match_fitpars(pulsar, "PB", fitpars) == ("PB_a", "PB_b")
    assert match_fitpars(pulsar, "PB_a", fitpars) == ("PB_a",)
    assert match_fitpars(pulsar, "F1", fitpars) == ("F1",)
    assert match_fitpars(pulsar, "DMX_0001", fitpars) == ()


def test_fitpar_suffixes(pulsar):
    assert fitpar_suffixes(pulsar, "PB_a") == {"_a"}
    assert fitpar_suffixes(pulsar, "F1") == {""}


def test_select_fitpars_preserves_order_and_raises_on_miss(pulsar):
    assert select_fitpars(pulsar, ["TASC", "PB"]) == (
        "PB_a",
        "TASC_a",
        "PB_b",
        "TASC_b",
    )
    with pytest.raises(ValueError, match="matches no fit parameter"):
        select_fitpars(pulsar, ["ECC"])


def test_model_inference_groups_selects_plan(pulsar):
    ntm = NonLinearTimingModel(engines="jug", inference=TimingInference.groups(delta_flat=["F1"]), name="timing")
    ctx = ntm.for_pulsar(pulsar)
    assert ctx.sampled == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    assert ctx.marginalized == ("F1",)


def test_model_inference_string_and_enum_presets(pulsar):
    from nltiming import InferencePreset

    default_ctx = NonLinearTimingModel(engines="jug", name="timing").for_pulsar(pulsar)
    assert (
        NonLinearTimingModel(engines="jug", inference="default", name="timing")
        .for_pulsar(pulsar)
        .plan.fingerprint()
        == default_ctx.plan.fingerprint()
    )
    all_ctx = NonLinearTimingModel(
        engines="jug", inference="all", name="timing"
    ).for_pulsar(pulsar)
    assert all_ctx.sampled == tuple(pulsar.fitpars)
    assert all_ctx.plan.marginalized_delta == ()
    assert (
        NonLinearTimingModel(
            engines="jug", inference=InferencePreset.ALL, name="timing"
        )
        .for_pulsar(pulsar)
        .plan.fingerprint()
        == all_ctx.plan.fingerprint()
    )


def test_model_inference_type_rejected():
    with pytest.raises(ValueError, match="unknown inference preset"):
        NonLinearTimingModel(engines="jug", inference="PB")
    with pytest.raises(TypeError, match="TimingInference"):
        NonLinearTimingModel(engines="jug", inference=123)


def test_constructor_priors_expand_to_suffixed_targets(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["F1"]),
        priors={"TASC": prior_specs.delta_uniform(-0.5, 0.5, scale="PB")},
        name="timing",
    )
    block = ntm.for_pulsar(pulsar).priors
    by_name = dict(zip(block.names, block.priors))
    assert block.sources["TASC_a"] == "override"
    assert block.sources["TASC_b"] == "override"
    # scale resolves suffix-consistently: PB_a ref = 10.0, PB_b ref = 20.0
    np.testing.assert_allclose(
        (by_name["TASC_a"].lower, by_name["TASC_a"].upper), (-5.0, 5.0)
    )
    np.testing.assert_allclose(
        (by_name["TASC_b"].lower, by_name["TASC_b"].upper), (-10.0, 10.0)
    )


def test_constructor_priors_reject_non_spec_values():
    with pytest.raises(TypeError, match="PriorOverrideSpec"):
        NonLinearTimingModel(engines="jug", priors={"PB": ("uniform", -1, 1)})


def test_prior_spec_helpers_validate_scale_frame():
    with pytest.raises(ValueError, match="frame='delta'"):
        prior_specs.uniform(-1.0, 1.0, scale="PB")
    spec = prior_specs.delta_normal(0.0, 1.0, scale="PB")
    assert spec.frame == "delta"
    assert spec.scale == "PB"


def test_with_engines_carries_inference_and_priors(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["F1"]),
        priors={"TASC": prior_specs.delta_uniform(-0.5, 0.5, scale="PB")},
        name="timing",
    )
    other = ntm.with_engines("jug")
    ctx = other.for_pulsar(pulsar)
    assert ctx.sampled == ("PB_a", "TASC_a", "PB_b", "TASC_b")
    assert ctx.priors.sources["TASC_a"] == "override"


# ---------------------------------------------------------------------------
# tempo2_native default resolution (§18)


def test_omitted_tempo2_native_resolves_to_fixed_state_stripped():
    ntm = NonLinearTimingModel(engines="jug", name="timing")
    # Raw field stays None (the "user set a mode" signal for _uses_jug);
    # the resolved mode is the production default and is what layers see.
    assert ntm.tempo2_native is None
    assert ntm.resolved_tempo2_native == "fixed_state_stripped"
    assert ntm._timing_engine_kwargs()["tempo2_native"] == "fixed_state_stripped"


def test_explicit_tempo2_native_is_an_explicit_choice():
    ntm = NonLinearTimingModel(
        engines="jug", tempo2_native="fixed_state", name="timing")
    assert ntm.resolved_tempo2_native == "fixed_state"
    assert ntm._timing_engine_kwargs()["tempo2_native"] == "fixed_state"


def test_resolved_tempo2_native_is_fingerprinted():
    default = NonLinearTimingModel(engines="jug", name="timing")
    explicit = NonLinearTimingModel(
        engines="jug", tempo2_native="fixed_state", name="timing")
    # The resolved mode enters the config fingerprint, so a non-default mode
    # produces a distinct fingerprint from the resolved default.
    assert default._tempo2_native_fingerprint() == "fixed_state_stripped"
    assert default._config_fingerprint() != explicit._config_fingerprint()
