"""Track J: dynamic transport record, one-affine-layer guard, dynamic writer.

The nltiming dynamic writer / record / guard depend only on a transport that
exposes ``diagnostics()`` and ``fingerprint()``. These tests exercise them
against a lightweight structural stub matching the discovery ``Transport``
contract, plus the JAX map behaviour (§7.3, §5.5, §10).
"""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.bijectors import WhiteningLinear
from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.metric import (
    DynamicTransportRecord,
    OneAffineLayerError,
    assert_static_layer_identity,
    dynamic_transport_record,
)
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.run_io import RunIOError, RunResults, save_dynamic_checkpoint
from nltiming.space import ParameterSpace


class _FakeTransport:
    """Structural stub matching the discovery Transport diagnostics contract."""

    def __init__(self, *, names=("rednoise", "dm"), dim=4, center=True):
        self._names = names
        self._dim = dim
        self._center = center

    def diagnostics(self, params=None, noise_solve=None):
        k = self._dim // len(self._names)
        return {
            "blocks": [
                {
                    "name": n,
                    "k": k,
                    "params": [f"{n}_log10_A", f"{n}_gamma"],
                    "keys": [f"{n}_coefficients({k})"],
                    "conditioner_kind": "exact_diagonal",
                }
                for n in self._names
            ],
            "dimension": self._dim,
            "center": self._center,
            "reference_noise": "toaerrs diagonal (FAKE)",
        }

    def fingerprint(self):
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-transport-v1", "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class _Pulsar:
    def __init__(self):
        self.name = "J4444+4444"
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
        return np.zeros(6)

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
        return "dyn-token"

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


# --------------------------------------------------------------------------
# Dynamic transport record (§7.3)
# --------------------------------------------------------------------------


def test_dynamic_transport_record_captures_structure_and_digest():
    transport = _FakeTransport()
    record = dynamic_transport_record(transport)
    assert isinstance(record, DynamicTransportRecord)
    assert record.kind == "dynamic_transport"
    assert record.latent_decodable is False
    assert record.dimension == 4
    assert record.centering == "centered"
    assert record.parameter_dependencies == (
        "dm_gamma",
        "dm_log10_A",
        "rednoise_gamma",
        "rednoise_log10_A",
    )
    assert record.transport_digest == transport.fingerprint()
    section = record.section()
    assert section["latent_decodable"] is False
    assert section["coordinate"] == "xi"
    assert [b["name"] for b in section["structure"]["blocks"]] == ["rednoise", "dm"]


# --------------------------------------------------------------------------
# One-affine-layer invariant (§5.5, §10 JAX 6)
# --------------------------------------------------------------------------


def test_one_affine_layer_guard_rejects_nonidentity_static_layer():
    identity = ParameterSpace.build(
        {"a": "0.0", "b": "0.0"},
        transform="none",
        linear_transform=WhiteningLinear.identity(2),
    )
    assert_static_layer_identity(identity)  # identity layer: must not raise

    nonidentity = ParameterSpace.build(
        {"a": "0.0", "b": "0.0"},
        transform="whitening",
        linear_transform=WhiteningLinear(
            C=np.array([[2.0, 0.0], [0.3, 1.5]]), z0=np.array([0.1, 0.0])
        ),
    )
    with pytest.raises(OneAffineLayerError, match="one-affine-layer|identity"):
        assert_static_layer_identity(nonidentity)


def test_joint_manifest_requires_identity_static_layer():
    pulsar = _Pulsar()
    # transform='whitening' conditions a non-identity static layer -> rejected.
    ntm = NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0"],
        name="timing",
    )
    ctx = ntm.for_pulsar(pulsar)
    record = dynamic_transport_record(_FakeTransport())
    with pytest.raises(OneAffineLayerError):
        ctx.run_manifest(
            likelihood="discovery", sampler="numpyro-nuts", dynamic_transport=record
        )


# --------------------------------------------------------------------------
# Dynamic checkpoint writer (§7.3, §10 run-decoding 8/10)
# --------------------------------------------------------------------------


def _joint_manifest(tmp_path, pulsar):
    # transform='none' -> identity static layer, satisfying the one-layer rule.
    ntm = NonLinearTimingModel(engines="jug", transform="none", name="timing")
    ctx = ntm.for_pulsar(pulsar)
    record = dynamic_transport_record(_FakeTransport())
    manifest = ctx.run_manifest(
        likelihood="discovery",
        sampler="numpyro-nuts",
        latent={"kind": "npz", "path": "dynamic_xi.npz", "key_name": "xi"},
        dynamic_transport=record,
    )
    manifest.write(tmp_path)
    return manifest


def test_dynamic_final_checkpoint_requires_canonical_physical(tmp_path):
    pulsar = _Pulsar()
    manifest = _joint_manifest(tmp_path, pulsar)
    xi = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, -0.1]])

    # A non-final checkpoint may omit physical values.
    save_dynamic_checkpoint(tmp_path, manifest, xi=xi, final=False)
    # A final checkpoint without canonical physical values is refused.
    with pytest.raises(RunIOError, match="without canonical decoded physical"):
        save_dynamic_checkpoint(tmp_path, manifest, xi=xi, final=True)

    # With canonical decoded physical values, the final write succeeds.
    out = save_dynamic_checkpoint(
        tmp_path,
        manifest,
        xi=xi,
        final=True,
        hyperparameters={"rednoise_log10_A": np.array([-14.0, -13.5])},
        theta_display={"F1": np.array([1.0, 1.0000001])},
        theta_native={"F1": np.array([1.0, 1.0000001])},
        log_density=np.array([-1.0, -2.0]),
    )
    data = np.load(out)
    assert "xi" in data.files
    assert data["transport_digest"] == manifest.transport["digest"]
    assert any(k.endswith("_theta_display") for k in data.files)


def test_static_writer_rejects_dynamic_manifest(tmp_path):
    from nltiming.run_io import save_discovery_checkpoint

    pulsar = _Pulsar()
    manifest = _joint_manifest(tmp_path, pulsar)
    with pytest.raises(RunIOError, match="dynamic"):
        save_discovery_checkpoint(tmp_path, np.zeros((2, 1)), manifest, final=True)


def test_dynamic_run_loads_stored_physical_not_latent_fallback(tmp_path):
    pulsar = _Pulsar()
    manifest = _joint_manifest(tmp_path, pulsar)
    xi = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, -0.1]])
    save_dynamic_checkpoint(
        tmp_path,
        manifest,
        xi=xi,
        final=True,
        theta_display={"F1": np.array([1.0, 2.0])},
        theta_native={"F1": np.array([1.0, 2.0])},
    )
    run = RunResults.load(tmp_path, verify=True)
    assert run.latent_decodable is False
    # The manifest records the dynamic transport section.
    assert run.run_meta["sections"]["transport"]["kind"] == "dynamic_transport"
