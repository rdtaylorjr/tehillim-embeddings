from __future__ import annotations

from phrase.subphrase import SAFE_SUBPHRASE_RELA_VOCABULARY, colon_safe_subphrase_rela, mask_par


class TestMaskPar:
    def test_masks_par_to_na(self):
        assert mask_par("par") == "NA"

    def test_leaves_safe_categories_unchanged(self):
        for value in ("NA", "adj", "atr", "dem", "mod", "rec"):
            assert mask_par(value) == value


class TestSafeSubphraseRelaVocabulary:
    def test_never_includes_par(self):
        assert "par" not in SAFE_SUBPHRASE_RELA_VOCABULARY

    def test_includes_exactly_the_six_safe_categories(self):
        assert set(SAFE_SUBPHRASE_RELA_VOCABULARY) == {"NA", "adj", "atr", "dem", "mod", "rec"}


class TestColonSafeSubphraseRela:
    def test_masks_every_par_in_the_colon(self):
        assert colon_safe_subphrase_rela(("NA", "par", "rec", "par")) == ("NA", "NA", "rec", "NA")

    def test_is_a_no_op_when_no_par_present(self):
        assert colon_safe_subphrase_rela(("NA", "rec", "atr")) == ("NA", "rec", "atr")
