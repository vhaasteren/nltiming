import pytest

from nltiming import pint_compat
from nltiming.pint_compat import (
    canonicalize_fdjump_name,
    fdjump_aliases,
    get_aliases_for_parameter,
    pint_parameter_name,
    resolve_fit_column_name,
    resolve_parameter_alias,
)


def test_fdjump_fit_column_spellings_are_folded():
    assert canonicalize_fdjump_name("FDJUMP1") == "FDJUMP1_1"
    assert canonicalize_fdjump_name("FD1JUMP1") == "FDJUMP1_1"
    assert resolve_fit_column_name("FDJUMPDM2") == "FDJUMPDM_2"


def test_later_fdjump_masks_do_not_include_bare_mask_one_aliases():
    assert set(fdjump_aliases("FD1JUMP2")) == {
        "FDJUMP1_2",
        "FD1JUMP2",
    }
    assert set(fdjump_aliases("FDJUMPDM2")) == {
        "FDJUMPDM_2",
        "FDJUMPDM2",
    }


def test_resolvers_reject_non_strings():
    """A non-string name is a caller bug, not an alias miss.

    Passing one through unchanged (the old blanket ``except``) is what let a
    typed native-name record reach a resolver and silently match nothing.
    """
    for bad in (None, 3, ("PX",), object()):
        for fn in (
            resolve_parameter_alias,
            pint_parameter_name,
            get_aliases_for_parameter,
            canonicalize_fdjump_name,
        ):
            with pytest.raises(TypeError, match="parameter-name string"):
                fn(bad)


def test_unknown_alias_still_passes_through():
    assert resolve_parameter_alias("NOT_A_PARAM") == "NOT_A_PARAM"
    assert pint_parameter_name("NOT_A_PARAM") is None
    assert get_aliases_for_parameter("NOT_A_PARAM") == ["NOT_A_PARAM"]


@pytest.mark.parametrize("exc", [RuntimeError, ValueError, KeyError])
def test_registry_failure_propagates(monkeypatch, exc):
    """The other half of the narrowed except: real failures must escape.

    ``ValueError``/``KeyError`` matter most: those are the alias-miss types, so
    a registry that fails with one of them is exactly what would be mistaken
    for "this name is not an alias" and passed through.
    """

    def _boom():
        raise exc("AllComponents is broken")

    monkeypatch.setattr(pint_compat, "_get_all_components", _boom)
    for fn in (
        pint_compat.resolve_parameter_alias,
        pint_compat.pint_parameter_name,
        pint_compat.get_aliases_for_parameter,
    ):
        with pytest.raises(exc, match="AllComponents is broken"):
            fn("F0")
