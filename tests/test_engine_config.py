"""Engine selection vocabulary tests."""

from __future__ import annotations

import pytest

from nltiming.engine_config import normalize_engines


def test_normalize_engines_accepts_vela_for_pint_family():
    engines = normalize_engines({"pint": "vela", "tempo2": "jug"})
    assert engines == {"pint": "vela", "tempo2": "jug"}
    with pytest.raises(ValueError, match="must be one of"):
        normalize_engines({"tempo2": "vela"})


