from nltiming.pint_compat import (
    canonicalize_fdjump_name,
    fdjump_aliases,
    resolve_fit_column_name,
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
