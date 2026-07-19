"""Two-stage TimingContext lifecycle and metric provenance (§5.1, §5.2, §10).

Covers the unconditioned -> conditioned split, finalize-once immutability,
metric order/dimension validation, and reference-noise provenance/digests.
"""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.metric import (
    LocalPosteriorMetric,
    WhiteningConfig,
    frozen_white_metric,
    toa_errors_metric,
)
from nltiming.nonlinear_timing_model import NonLinearTimingModel


class _Pulsar:
    def __init__(self):
        self.name = "J2222+2222"
        self.fitpars = ("F0", "F1")
        self._toaerrs = np.full(6, 1.0e-6)
        self._backend_flags = np.array(["demo"] * 6, dtype="U8")
        design = np.column_stack([np.ones(6), np.linspace(-0.5, 0.5, 6)])
        model = LinearModel.from_design(
            fitpars=self.fitpars,
            design=design,
            theta_exact={"F0": "100.0", "F1": "1.0"},
        )
        self._backend = LinearizedJugEngine.from_linear_model(model)

    @property
    def toas(self):
        return np.linspace(0.0, 1.0, 6)

    @property
    def residuals(self):
        return np.linspace(-1e-7, 1e-7, 6)

    @property
    def toaerrs(self):
        return self._toaerrs

    @property
    def freqs(self):
        return np.full(6, 1400.0)

    @property
    def Mmat(self):
        return self._backend.design_matrix()

    @property
    def flags(self):
        return {"pta": np.array(["demo"] * 6, dtype="U8")}

    @property
    def backend_flags(self):
        return self._backend_flags

    def state_id(self):
        return "lifecycle-token"

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def pulsar():
    return _Pulsar()


def _model(**kwargs):
    return NonLinearTimingModel(
        engines="jug", whitening=WhiteningConfig(), name="timing", **kwargs
    )


def test_for_pulsar_unconditioned_base_has_no_transport(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    assert base.conditioned is False
    assert base.transport is None
    assert base.metric is None
    # Unconditioned linear layer is identity until with_transport runs.
    np.testing.assert_allclose(base.space.linear.C, np.eye(len(base.sampled)))


def test_with_transport_conditions_and_records_provenance(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    metric = base.default_metric()
    ctx = base.with_transport(metric)

    assert ctx.conditioned is True
    assert ctx.transport.kind == "static_affine"
    assert ctx.transport.latent_decodable is True
    assert ctx.metric is metric
    assert ctx.transport.metric_source["reference_noise"] == "toa_errors"
    assert ctx.transport.metric_source["approximate"] is True
    assert ctx.transport.metric_source["digest"] == metric.fingerprint()
    # A non-identity whitening transform was actually built.
    assert not np.allclose(ctx.space.linear.C, np.eye(len(ctx.sampled)))


def test_conditioning_is_finalize_once(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    ctx = base.with_transport(base.default_metric())
    with pytest.raises(ValueError, match="already conditioned"):
        ctx.with_transport(ctx.default_metric())
    # The default for_pulsar path is already conditioned, too.
    conditioned = _model().for_pulsar(pulsar)
    with pytest.raises(ValueError, match="already conditioned"):
        conditioned.with_transport(conditioned.default_metric())


def test_with_transport_rejects_wrong_metric_order_and_dimension(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    good = base.default_metric()

    wrong_dim = LocalPosteriorMetric(
        fisher_delta=np.eye(1),
        sampled=("F0",),
        expansion_delta=np.zeros(1),
        reference_noise="toa_errors",
        source="toa_errors",
        source_description="wrong dim",
    )
    with pytest.raises(ValueError, match="do not match"):
        base.with_transport(wrong_dim)

    reordered = LocalPosteriorMetric(
        fisher_delta=good.fisher_delta,
        sampled=tuple(reversed(base.sampled)),
        expansion_delta=np.zeros(len(base.sampled)),
        reference_noise="toa_errors",
        source="toa_errors",
        source_description="reordered",
    )
    if len(base.sampled) > 1:
        with pytest.raises(ValueError, match="do not match"):
            base.with_transport(reordered)


def test_reference_noise_sources_are_distinguishable(pulsar):
    """§10.7: raw-TOA and frozen-white sources differ in values, provenance,
    and digests."""
    base = _model().for_pulsar(pulsar, condition=False)
    toa = toa_errors_metric(
        pulsar=base.pulsar, partition=base.plan, design_matrix=base.design_matrix
    )
    frozen = frozen_white_metric(
        pulsar=base.pulsar,
        partition=base.plan,
        efac={"demo": 2.0},
        equad={"demo": 1.0e-7},
        design_matrix=base.design_matrix,
    )

    assert toa.reference_noise == "toa_errors"
    assert frozen.reference_noise == "frozen_white"
    assert toa.fingerprint() != frozen.fingerprint()
    assert not np.allclose(toa.fisher_delta, frozen.fisher_delta)
    assert toa.provenance()["approximate"] is True
    assert frozen.provenance()["noise_snapshot"]["efac"] == {"demo": 2.0}

    # Conditioning with different metrics changes the context fingerprint.
    fp_toa = base.with_transport(toa).fingerprint()
    fp_frozen = base.with_transport(frozen).fingerprint()
    assert fp_toa != fp_frozen


def test_assembled_metric_is_not_flagged_approximate(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    assembled = LocalPosteriorMetric(
        fisher_delta=base.default_metric().fisher_delta,
        sampled=base.sampled,
        expansion_delta=np.zeros(len(base.sampled)),
        reference_noise="assembled_likelihood",
        source="discovery-frozen-full-likelihood",
        source_description="assembled full-noise precision",
    )
    ctx = base.with_transport(assembled)
    assert assembled.approximate is False
    assert ctx.transport.metric_source["approximate"] is False


def test_run_manifest_requires_conditioned_context(pulsar):
    base = _model().for_pulsar(pulsar, condition=False)
    with pytest.raises(ValueError, match="requires a conditioned"):
        base.run_manifest(likelihood="discovery", sampler="numpyro-nuts")


def test_whitening_config_rejects_stringly_dict(pulsar):
    with pytest.raises(TypeError, match="WhiteningConfig"):
        NonLinearTimingModel(engines="jug", whitening={"name": "diagonal_white"})
    with pytest.raises(ValueError, match="reference_noise"):
        WhiteningConfig(reference_noise="not_a_class")


def test_whitening_none_conditions_with_identity_transport_and_no_metric(pulsar):
    """whitening=None is an identity map: it conditions with an identity
    transport and no reference-noise metric, so its provenance never claims a
    (never-applied) toa_errors whitening (§5.5, provenance honesty)."""
    none_ctx = NonLinearTimingModel(
        engines="jug", name="t"
    ).for_pulsar(pulsar)
    assert none_ctx.conditioned is True
    assert none_ctx.metric is None
    assert none_ctx.transport.kind == "static_affine"
    assert none_ctx.transport.metric_source["reference_noise"] == "identity"
    np.testing.assert_allclose(none_ctx.space.linear.C, np.eye(len(none_ctx.sampled)))
    # An explicit unconditioned base can still be conditioned with None only.
    base = NonLinearTimingModel(engines="jug", name="t").for_pulsar(
        pulsar, condition=False
    )
    assert base.with_transport().metric is None
