"""Immutable run decoding: per-section digests, reconcile guard, no-overwrite.

Covers §7.3-§7.5 and the §10 run-decoding acceptance tests for the
``nlt-run-meta-v3`` manifest with per-section digests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nltiming.engines.base import LinearModel
from nltiming.engines.jug import LinearizedJugEngine
from nltiming.nonlinear_timing_model import NonLinearTimingModel
from nltiming.run_io import (
    DISCOVERY_FINAL_NAME,
    RUN_META_FILENAME,
    RunIOError,
    RunResults,
    load_run,
    save_discovery_checkpoint,
)


class _Pulsar:
    def __init__(self, state_id="decode-token"):
        self.name = "J3333+3333"
        self.fitpars = ("F0", "F1")
        self._toaerrs = np.full(6, 1.0e-6)
        self._backend_flags = np.array(["demo"] * 6, dtype="U8")
        self._state_id = state_id
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
        return self._state_id

    def pint_model(self):
        return object()

    def timing_engine(self, engines="jug", **kwargs):
        return self._backend


@pytest.fixture
def pulsar():
    return _Pulsar()


@pytest.fixture
def ntm():
    return NonLinearTimingModel(
        engines="jug",
        transform="whitening",
        analytically_marginalize=["F0"],
        name="timing",
    )


def _write_run(tmp_path, ntm, pulsar):
    ctx = ntm.for_pulsar(pulsar)
    manifest = ctx.run_manifest(
        likelihood="discovery",
        sampler="numpyro-nuts",
        latent={"kind": "npz", "path": DISCOVERY_FINAL_NAME, "key_name": "x"},
    )
    manifest.write(tmp_path)
    x = np.array([[0.1], [0.2], [0.3]], dtype=float)
    save_discovery_checkpoint(tmp_path, x, manifest, final=True, n_target=3)
    return ctx, manifest


def test_all_sections_present_and_digests_agree(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    run_meta = json.loads((tmp_path / RUN_META_FILENAME).read_text())
    sections = run_meta["sections"]
    assert set(sections) >= {
        "parameter_space",
        "context",
        "metric_source",
        "transport",
        "chains",
    }
    assert sections["transport"]["kind"] == "static_affine"
    assert sections["transport"]["latent_decodable"] is True
    assert sections["metric_source"]["reference_noise"] == "toa_errors"
    # verify=True must pass on a freshly written run.
    run = load_run(tmp_path)
    assert run.latent_decodable is True


def test_wrong_chain_array_fails_naming_chains(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    # Tamper the saved array's embedded space digest (a "wrong chain array").
    data = dict(np.load(tmp_path / DISCOVERY_FINAL_NAME))
    data["space_digest"] = np.asarray("sha256:deadbeef")
    np.savez(tmp_path / DISCOVERY_FINAL_NAME, **data)
    with pytest.raises(RunIOError, match="chains section"):
        RunResults.load(tmp_path, verify=True)


def test_wrong_live_context_fails_assert_consistent_with(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    run = load_run(tmp_path)
    # A different model config yields a different context fingerprint.
    other = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    other_ctx = other.for_pulsar(_Pulsar())
    with pytest.raises(RunIOError, match="section diverged"):
        run.assert_consistent_with(other_ctx)


def test_matching_live_context_passes_assert_consistent_with(tmp_path, ntm, pulsar):
    ctx, _ = _write_run(tmp_path, ntm, pulsar)
    run = load_run(tmp_path)
    run.assert_consistent_with(ctx)  # must not raise


def test_existing_run_load_does_not_rewrite(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    meta_path = tmp_path / RUN_META_FILENAME
    before = meta_path.read_bytes()
    mtime = meta_path.stat().st_mtime_ns
    load_run(tmp_path)
    load_run(tmp_path).load_display()
    assert meta_path.read_bytes() == before
    assert meta_path.stat().st_mtime_ns == mtime


def test_unsupported_schema_fails_with_migration_guidance(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    meta_path = tmp_path / RUN_META_FILENAME
    run_meta = json.loads(meta_path.read_text())
    run_meta["schema"] = "nlt-sidecar-v2"
    meta_path.write_text(json.dumps(run_meta))
    with pytest.raises(RunIOError, match="unsupported run-metadata schema"):
        RunResults.load(tmp_path)


def test_no_overwrite_of_incompatible_run(tmp_path, ntm, pulsar):
    ctx, manifest = _write_run(tmp_path, ntm, pulsar)
    # A manifest with a different context digest must not clobber the existing run.
    other = NonLinearTimingModel(
        engines="jug",
        transform="standardized",
        analytically_marginalize=["F0"],
        name="timing",
    )
    other_manifest = other.for_pulsar(pulsar).run_manifest(
        likelihood="discovery", sampler="numpyro-nuts"
    )
    with pytest.raises(RunIOError, match="refusing to overwrite"):
        other_manifest.write(tmp_path)
    # force=True overwrites.
    other_manifest.write(tmp_path, force=True)
    # Re-writing the identical manifest is always allowed.
    manifest.write(tmp_path, force=True)


def test_transport_digest_tamper_fails_verification(tmp_path, ntm, pulsar):
    _write_run(tmp_path, ntm, pulsar)
    meta_path = tmp_path / RUN_META_FILENAME
    run_meta = json.loads(meta_path.read_text())
    run_meta["sections"]["transport"]["origin"] = "tampered"
    meta_path.write_text(json.dumps(run_meta))
    with pytest.raises(RunIOError, match="transport section"):
        RunResults.load(tmp_path)


def test_dynamic_run_rejects_latent_only_physical_fallback(tmp_path, ntm, pulsar):
    """§10.8: a dynamic run with no stored physical values must not decode xi."""
    ctx = ntm.for_pulsar(pulsar)
    manifest = ctx.run_manifest(
        likelihood="discovery",
        sampler="numpyro-nuts",
        latent={"kind": "npz", "path": "dyn_xi.npz", "key_name": "x"},
    )
    # Simulate a dynamic transport record whose canonical physical values are
    # absent (an incomplete dynamic checkpoint).
    manifest.transport = {"kind": "dynamic_transport", "latent_decodable": False}
    manifest.write(tmp_path)
    np.savez(tmp_path / "dyn_xi.npz", x=np.array([[0.1], [0.2]], dtype=float))
    run = RunResults.load(tmp_path, verify=False)
    assert run.latent_decodable is False
    # Latent xi is still loadable as a diagnostic...
    assert run.load_latent().shape == (2, 1)
    # ...but physical decoding must refuse the static fallback.
    with pytest.raises(RunIOError, match="dynamic transport"):
        run.load_display()
