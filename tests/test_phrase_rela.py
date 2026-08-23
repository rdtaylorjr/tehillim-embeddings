from __future__ import annotations

from phrase.rela import SAFE_RELA_VOCABULARY, colon_safe_rela, mask_para


class TestMaskPara:
    def test_masks_para_to_na(self):
        assert mask_para("Para") == "NA"

    def test_leaves_safe_categories_unchanged(self):
        for value in ("NA", "Appo", "Link", "Sfxs", "Spec"):
            assert mask_para(value) == value


class TestSafeRelaVocabulary:
    def test_never_includes_para(self):
        assert "Para" not in SAFE_RELA_VOCABULARY

    def test_includes_exactly_the_five_safe_categories(self):
        assert set(SAFE_RELA_VOCABULARY) == {"NA", "Appo", "Link", "Sfxs", "Spec"}


class TestColonSafeRela:
    def test_masks_every_para_in_the_colon(self):
        assert colon_safe_rela(("NA", "Para", "Appo", "Para")) == ("NA", "NA", "Appo", "NA")

    def test_is_a_no_op_when_no_para_present(self):
        assert colon_safe_rela(("NA", "Appo", "Spec")) == ("NA", "Appo", "Spec")
