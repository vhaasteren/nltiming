"""Tests for inference-based plan selection and constructor priors."""

import numpy as np
import pytest

from nltiming import priors as prior_specs
from nltiming import TimingInference
from _engine_stubs import JaxLinearTestEngine
from nltiming.engine_support import LinearModel
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.selection import (
    ParameterMappingError,
    fitpar_suffix,
    match_fitpars,
    select_fitpars,
)


class _SuffixHost:
    """Composite-style pulsar with PTA-suffixed fitpars and a total public mapping."""

    def __init__(self):
        self.name = "J2222+2222"
        self.fitpars = ("Offset", "F1", "PB_a", "TASC_a", "PB_b", "TASC_b")
        self._mapping = {
            "Offset": {"shared": "Offset"},
            "F1": {"shared": "F1"},
            "PB_a": {"a": "PB"},
            "TASC_a": {"a": "TASC"},
            "PB_b": {"b": "PB"},
            "TASC_b": {"b": "TASC"},
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
            [np.ones(n), np.linspace(-0.5, 0.5, n)]
            + [rng.normal(size=n) for _ in range(len(self.fitpars) - 2)]
        )
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=design,
            theta_exact={
                "Offset": "0.0",
                "F1": "1.0",
                "PB_a": "10.0",
                "TASC_a": "55000.0",
                "PB_b": "20.0",
                "TASC_b": "56000.0",
            },
        )
        self._backend = JaxLinearTestEngine.from_linear_model(model)

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

    def timing_parameter_mapping(self):
        return {name: dict(owners) for name, owners in self._mapping.items()}


@pytest.fixture
def pulsar():
    return _SuffixHost()


def test_match_fitpars_base_name_matches_all_suffixed(pulsar):
    fitpars = pulsar.fitpars
    assert match_fitpars(pulsar, "PB", fitpars) == ("PB_a", "PB_b")
    assert match_fitpars(pulsar, "PB_a", fitpars) == ("PB_a",)
    assert match_fitpars(pulsar, "F1", fitpars) == ("F1",)
    assert match_fitpars(pulsar, "DMX_0001", fitpars) == ()


def test_fitpar_suffix(pulsar):
    assert fitpar_suffix(pulsar, "PB_a") == "_a"
    assert fitpar_suffix(pulsar, "F1") == ""


def _mapping_host(mapping):
    class _Host:
        def timing_parameter_mapping(self):
            return mapping

    return _Host()


def test_fitpar_suffix_alias_native_base_is_unsuffixed():
    # Native Tempo2 spelling `E` recorded under canonical `ECC` (BUG 001):
    # the alias base must not be mistaken for a stripable prefix.
    psr = _mapping_host({"ECC": {"ng9": "E"}})
    assert fitpar_suffix(psr, "ECC") == ""
    # Same defect class, ELL1H orthometric spelling.
    psr = _mapping_host({"STIGMA": {"mpta": "STIG"}})
    assert fitpar_suffix(psr, "STIGMA") == ""


def test_fitpar_suffix_suffixed_name_with_alias_native():
    # PTA-specific composite whose par file spells the base as an alias.
    psr = _mapping_host({"ECC_ng9": {"ng9": "E"}})
    assert fitpar_suffix(psr, "ECC_ng9") == "_ng9"


def test_fitpar_suffix_merged_mixed_spellings():
    psr = _mapping_host({"ECC": {"ng9": "E", "epta": "ECC"}})
    assert fitpar_suffix(psr, "ECC") == ""


def test_fitpar_suffix_indexed_and_offset_natives():
    psr = _mapping_host({"JUMP1_epta": {"epta": "JUMP1"}})
    assert fitpar_suffix(psr, "JUMP1_epta") == "_epta"
    # `Offset` is unknown to PINT alias resolution on both sides.
    psr = _mapping_host({"Offset_ng9": {"ng9": "Offset"}})
    assert fitpar_suffix(psr, "Offset_ng9") == "_ng9"


def test_fitpar_suffix_identity_wins_over_coincidental_tail():
    # A PTA literally named "0001" must not turn the indexed parameter
    # DMX_0001 into a suffixed name: alias identity is checked first.
    psr = _mapping_host({"DMX_0001": {"0001": "DMX_0001"}})
    assert fitpar_suffix(psr, "DMX_0001") == ""


def test_fitpar_suffix_inconsistent_mapping_raises():
    psr = _mapping_host({"ECC": {"ng9": "PB"}})
    with pytest.raises(ParameterMappingError):
        fitpar_suffix(psr, "ECC")


def test_fitpar_suffix_no_mapping_capability_is_unsuffixed():
    class _PlainHost:
        pass

    assert fitpar_suffix(_PlainHost(), "F1") == ""


def test_fitpar_suffix_supplied_mapping_is_total():
    with pytest.raises(ParameterMappingError, match="absent"):
        fitpar_suffix(_mapping_host({}), "F1")


def test_fitpar_suffix_rejects_mixed_or_multiple_local_identity():
    psr = _mapping_host({"ECC_ng9": {"ng9": "ECC", "other": "ECC_ng9"}})
    with pytest.raises(ParameterMappingError, match="mixes"):
        fitpar_suffix(psr, "ECC_ng9")

    psr = _mapping_host({"ECC_ng9": {"ng9": "ECC", "x": "ECC"}})
    with pytest.raises(ParameterMappingError, match="multiple"):
        fitpar_suffix(psr, "ECC_ng9")


def test_public_parameter_mapping_wins_over_private_field():
    psr = _mapping_host({"ECC": {"ng9": "E"}})
    psr._fitparameters = {"ECC": {"wrong": "PB"}}
    assert fitpar_suffix(psr, "ECC") == ""


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"ECC": None},
        {"ECC": {}},
        {"ECC": {"": "E"}},
        {"ECC": {"ng9": ""}},
    ],
)
def test_malformed_public_mapping_raises(bad):
    psr = _mapping_host(bad)
    with pytest.raises(ParameterMappingError):
        fitpar_suffix(psr, "ECC")


def _fitpar_mapping_host(fitpars, mapping):
    class _Host:
        def __init__(self):
            self.fitpars = tuple(fitpars)

        def timing_parameter_mapping(self):
            return mapping

    return _Host()


def test_match_fitpars_rejects_partial_mapping():
    # Supplied {} must not silently miss PB_a and change inference disposition.
    psr = _fitpar_mapping_host(("PB_a",), {})
    with pytest.raises(ParameterMappingError, match="absent"):
        match_fitpars(psr, "PB", psr.fitpars)


def test_match_fitpars_rejects_malformed_owner_entry():
    psr = _fitpar_mapping_host(("PB_a",), {"PB_a": None})
    with pytest.raises(ParameterMappingError):
        match_fitpars(psr, "PB", psr.fitpars)


def test_select_fitpars_rejects_partial_and_malformed_mapping():
    with pytest.raises(ParameterMappingError, match="absent"):
        select_fitpars(
            _fitpar_mapping_host(("PB_a", "F1"), {"F1": {"s": "F1"}}), ["PB"]
        )
    with pytest.raises(ParameterMappingError):
        select_fitpars(_fitpar_mapping_host(("PB_a",), {"PB_a": None}), ["PB"])


def test_select_fitpars_fetches_mapping_once(pulsar):
    calls = {"n": 0}
    base = pulsar.timing_parameter_mapping

    def _counting():
        calls["n"] += 1
        return base()

    pulsar.timing_parameter_mapping = _counting
    select_fitpars(pulsar, ["PB", "TASC"])
    assert calls["n"] == 1


def test_select_fitpars_preserves_order_and_raises_on_miss(pulsar):
    assert select_fitpars(pulsar, ["TASC", "PB"]) == (
        "PB_a",
        "TASC_a",
        "PB_b",
        "TASC_b",
    )
    with pytest.raises(ValueError, match="matches no fit parameter"):
        select_fitpars(pulsar, ["ECC"])


def test_timing_parameter_mapping_provider_is_package_export():
    from nltiming import TimingParameterMappingProvider
    from nltiming.protocols import TimingParameterMappingProvider as Proto

    assert TimingParameterMappingProvider is Proto


def test_model_inference_groups_selects_plan(pulsar):
    ntm = NonLinearTimingModel(
        engines="jug",
        inference=TimingInference.groups(delta_flat=["F1"]),
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar)
    assert ctx.sampled == ("Offset", "PB_a", "TASC_a", "PB_b", "TASC_b")
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
    assert ctx.sampled == ("Offset", "PB_a", "TASC_a", "PB_b", "TASC_b")
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
        engines="jug", tempo2_native="fixed_state", name="timing"
    )
    assert ntm.resolved_tempo2_native == "fixed_state"
    assert ntm._timing_engine_kwargs()["tempo2_native"] == "fixed_state"


def test_resolved_tempo2_native_is_fingerprinted():
    default = NonLinearTimingModel(engines="jug", name="timing")
    explicit = NonLinearTimingModel(
        engines="jug", tempo2_native="fixed_state", name="timing"
    )
    # The resolved mode enters the config fingerprint, so a non-default mode
    # produces a distinct fingerprint from the resolved default.
    assert default._tempo2_native_fingerprint() == "fixed_state_stripped"
    assert default._config_fingerprint() != explicit._config_fingerprint()


# ---------------------------------------------------------------------------
# nonlinear_params residual-linearization forwarding (§10.1)
#


def test_omitted_nonlinear_params_stays_none():
    ntm = NonLinearTimingModel(engines="jug", name="timing")
    assert ntm.nonlinear_params is None
    assert ntm._timing_engine_kwargs()["nonlinear_params"] is None
    assert ntm._nonlinear_params_fingerprint() is None


def test_explicit_nonlinear_params_forwarded_and_fingerprinted():
    native = NonLinearTimingModel(engines="jug", name="timing")
    hybrid = NonLinearTimingModel(
        engines="jug", nonlinear_params="binary", name="timing"
    )
    assert hybrid.nonlinear_params == "binary"
    assert hybrid._timing_engine_kwargs()["nonlinear_params"] == "binary"
    assert hybrid._nonlinear_params_fingerprint() == "binary"
    assert hybrid._config_fingerprint() != native._config_fingerprint()
    carried = hybrid.with_engines("jug")
    assert carried.nonlinear_params == "binary"


def test_nonlinear_params_rejects_unknown_mode():
    import pytest

    with pytest.raises(ValueError, match="nonlinear_params"):
        NonLinearTimingModel(engines="jug", nonlinear_params="all", name="timing")


def test_nonlinear_params_implies_jug_use():
    # Hybrid mode requires jug even if engines omit it in the sense that
    # _uses_jug is true whenever nonlinear_params is set.
    ntm = NonLinearTimingModel(
        engines={"pint": "pint", "tempo2": "libstempo"},
        nonlinear_params="binary+",
        name="timing",
    )
    assert ntm._uses_jug() is True


def test_run_meta_records_nonlinear_params():
    from nltiming.run_io import _run_meta_nonlinear_params

    omitted = NonLinearTimingModel(engines="jug", name="timing")
    hybrid = NonLinearTimingModel(
        engines="jug", nonlinear_params="binary", name="timing"
    )
    assert _run_meta_nonlinear_params(omitted) is None
    assert _run_meta_nonlinear_params(hybrid) == "binary"
