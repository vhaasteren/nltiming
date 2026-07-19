"""Identical-linearity authority and fallback registry (§4.3)."""

from __future__ import annotations

import numpy as np

from nltiming.linearity import (
    fallback_identically_linear_fitpars,
    resolve_linearity,
)


class _Pulsar:
    def __init__(self, fitpars, fitparameters=None):
        self.name = "FAKE"
        self.fitpars = tuple(fitpars)
        if fitparameters is not None:
            self._fitparameters = dict(fitparameters)

    def pint_model(self):
        return None


class _EngineStub:
    def __init__(self, declared):
        self._declared = frozenset(declared)

    def identically_linear_fitpars(self):
        return self._declared


class _EngineNoMetadata:
    pass


_REG = ("F0", "F1", "DM", "JUMP1", "DMX_0001", "FD1", "PX")


def test_fallback_registry_recognizes_exact_names_and_prefixes():
    fb = fallback_identically_linear_fitpars(_Pulsar(_REG))
    assert {"DM", "JUMP1", "DMX_0001", "FD1"} <= fb
    assert "F0" not in fb and "F1" not in fb and "PX" not in fb


def test_engine_without_linearity_metadata_uses_nonempty_fallback_registry():
    res = resolve_linearity(_Pulsar(_REG), _EngineNoMetadata())
    assert res.mode == "defaults"
    assert res.effective_names >= {"DM", "JUMP1", "DMX_0001", "FD1"}


def test_none_linearity_uses_fallback_union_engine():
    res = resolve_linearity(_Pulsar(_REG), _EngineStub({"F0"}))
    assert res.mode == "defaults"
    assert "F0" in res.effective_names  # from the engine
    assert "DM" in res.effective_names  # from the fallback registry
    assert "engine" in res.sources_for("F0")
    assert res.sources_for("DM") == ("fallback",)


def test_explicit_linearity_sequence_replaces_fallback_and_engine_candidates():
    res = resolve_linearity(
        _Pulsar(_REG), _EngineStub({"F0"}), identically_linear=["F0"]
    )
    assert res.mode == "explicit_user"
    assert res.effective_names == {"F0"}
    assert "DM" not in res.effective_names
    assert set(res.sources_for("F0")) == {"user", "engine"}


def test_explicit_empty_linearity_sequence_is_authoritative():
    res = resolve_linearity(_Pulsar(_REG), _EngineStub({"F0"}), identically_linear=[])
    assert res.mode == "explicit_user"
    assert res.effective_names == frozenset()
    suppressed = {d.name for d in res.suppressed_candidates}
    assert suppressed >= {"DM", "JUMP1", "F0"}


def test_suppressed_candidates_are_recorded_but_not_effective():
    res = resolve_linearity(_Pulsar(_REG), _EngineStub({"F0"}), identically_linear=["F0"])
    suppressed = {d.name for d in res.suppressed_candidates}
    assert "DM" in suppressed
    assert "DM" not in res.effective_names
    # A suppressed candidate never carries the "user" source.
    dm = next(d for d in res.suppressed_candidates if d.name == "DM")
    assert "user" not in dm.sources


def test_linearity_sources_are_preserved_individually():
    res = resolve_linearity(_Pulsar(_REG), _EngineStub({"F0", "DM"}))
    assert set(res.sources_for("F0")) == {"engine"}
    assert set(res.sources_for("DM")) == {"fallback", "engine"}
    assert set(res.sources_for("JUMP1")) == {"fallback"}


def test_registry_digest_is_stable():
    a = resolve_linearity(_Pulsar(_REG))
    b = resolve_linearity(_Pulsar(_REG))
    assert a.fallback_registry_digest == b.fallback_registry_digest
    assert a.fallback_registry_version == 1


def test_engine_linearity_declaration_via_linear_engine():
    """A real LinearTimingEngine declares every fitpar identically linear."""
    from nltiming.engines.base import LinearModel, LinearTimingEngine

    fitpars = ("F0", "F1", "DM")
    eng = LinearTimingEngine(
        LinearModel.from_design(fitpars=fitpars, design=np.eye(3))
    )
    res = resolve_linearity(_Pulsar(fitpars), eng)
    assert res.effective_names == set(fitpars)
    assert "engine" in res.sources_for("F0")
