"""ArviZ 0.x kwargs vs 1.x nested-mapping ``from_dict`` (nltiming#4)."""

from __future__ import annotations

import numpy as np
import pytest

from nltiming.arviz_compat import inference_data_from_dict


def test_from_dict_uses_0x_kwargs(monkeypatch):
    recorded = {}

    def from_dict(*, posterior, sample_stats=None, attrs=None):
        recorded["posterior"] = posterior
        recorded["sample_stats"] = sample_stats
        recorded["attrs"] = attrs
        return "idata-0x"

    class _Az:
        @staticmethod
        def from_dict(*, posterior, sample_stats=None, attrs=None):
            return from_dict(
                posterior=posterior, sample_stats=sample_stats, attrs=attrs
            )

    monkeypatch.setitem(__import__("sys").modules, "arviz", _Az)
    out = inference_data_from_dict(
        posterior={"F1": np.zeros((1, 2))},
        sample_stats={"lp": np.zeros((1, 2))},
        attrs={"space_digest": "abc"},
    )
    assert out == "idata-0x"
    assert recorded["attrs"]["space_digest"] == "abc"
    assert "F1" in recorded["posterior"]
    assert "lp" in recorded["sample_stats"]


def test_from_dict_uses_1x_nested_mapping(monkeypatch):
    recorded = {}

    def _from_dict(data, *, attrs=None):
        recorded["data"] = data
        recorded["attrs"] = attrs
        return "idata-1x"

    class _Az:
        from_dict = staticmethod(_from_dict)

    monkeypatch.setitem(__import__("sys").modules, "arviz", _Az)
    out = inference_data_from_dict(
        posterior={"F1": np.zeros((1, 2))},
        sample_stats={"lp": np.zeros((1, 2))},
        attrs={"space_digest": "abc"},
    )
    assert out == "idata-1x"
    assert set(recorded["data"]) == {"posterior", "sample_stats"}
    assert recorded["attrs"]["space_digest"] == "abc"


def test_from_dict_1x_rejects_posterior_keyword_like_arviz_1_3(monkeypatch):
    """Reproduce the TypeError Abhimanyu hit with arviz 1.3 ``from_dict``."""

    def _from_dict(data, *, attrs=None):
        return {"groups": set(data), "attrs": attrs}

    class _Az:
        from_dict = staticmethod(_from_dict)

    monkeypatch.setitem(__import__("sys").modules, "arviz", _Az)
    with pytest.raises(TypeError, match="posterior"):
        _Az.from_dict(posterior={"F1": np.zeros((1, 2))})
    out = inference_data_from_dict(posterior={"F1": np.zeros((1, 2))})
    assert out["groups"] == {"posterior"}
