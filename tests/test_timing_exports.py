"""Public export surface for the timing package."""

from nltiming import (
    EnterprisePulsarLike,
    EphemerisExtras,
    JaxTimingEngine,
    ParameterSpace,
    TimingEngine,
    TimingPulsar,
    TimingSpec,
)


def test_timing_subpackage_exports():
    assert TimingSpec is not None
    assert ParameterSpace is not None
    assert EnterprisePulsarLike is not None
    assert TimingPulsar is not None
    assert JaxTimingEngine is not None
    assert EphemerisExtras is not None
    assert TimingEngine is not None


def test_timing_imports_and_constructs_without_jug():
    """Default and JUG-free configs must import and construct with jug uninstalled.

    Runs in a subprocess with ``jug`` and ``jax`` blocked from importing, so
    the check is meaningful even in environments where both are installed.
    """
    import subprocess
    import sys

    code = """
import sys
sys.modules['jug'] = None
sys.modules['jax'] = None
import nltiming
from nltiming import TimingSpec
default = TimingSpec()
assert default.engines == {'tempo2': 'vela_jax', 'pint': 'vela_jax'}
assert default.tempo2_jug_options is None
m = TimingSpec(engines={'tempo2': 'libstempo', 'pint': 'pint'})
assert m.tempo2_jug_options is None
m.set_prior('F0', 'normal', mean=0.0, std=1.0)
m2 = m.with_engines({'tempo2': 'libstempo', 'pint': 'pint'})
assert m2.tempo2_jug_options is None
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_jug_config_remains_opaque_without_importing_jug():
    import sys

    sys.modules.pop("jug", None)
    # Block jug import for the duration of this test.
    sys.modules["jug"] = None  # type: ignore[assignment]
    try:
        model = TimingSpec(engines="jug")
        options = model.tempo2_jug_options
        assert options == {}
        assert "jug" not in sys.modules or sys.modules.get("jug") is None
    finally:
        sys.modules.pop("jug", None)
