from __future__ import annotations

from core.text import consonantal, strip_accents


class TestStripAccents:
    def test_removes_a_real_cantillation_mark(self):
        assert strip_accents("֑") == ""

    def test_keeps_niqqud(self):
        assert strip_accents("ָ") == "ָ"

    def test_keeps_plain_consonants(self):
        assert strip_accents("שלום") == "שלום"

    def test_mixed_text_keeps_only_niqqud_and_consonants(self):
        text = "אָ֑"  # aleph, an accent, qamats
        assert strip_accents(text) == "אָ"

    def test_removes_cantillation_between_consonant_and_niqqud(self):
        # U+0591 (etnahta, cantillation) followed by U+05B4 (hiriq, niqqud).
        assert strip_accents("אִ֑ב") == "אִב"

    def test_leaves_plain_consonantal_text_unchanged(self):
        assert strip_accents("אבג") == "אבג"

    def test_keeps_the_boundary_codepoint_just_past_the_accent_range(self):
        assert strip_accents("ְ") == "ְ"

    def test_removes_the_last_codepoint_inside_the_accent_range(self):
        assert strip_accents("֯") == ""


class TestConsonantal:
    def test_keeps_letters_and_spaces_only(self):
        assert consonantal("אשׁרי־האישׁ אשׁר\u05c0 לא הלך\u05c3 ") == "אשרי האיש אשר לא הלך"

    def test_a_maqaf_becomes_the_space_between_two_words(self):
        assert consonantal("על־כן") == "על כן"

    def test_points_and_accents_and_the_sin_dot_go(self):
        assert consonantal("נַפְשֵׂ֫נוּ") == "נפשנו"

    def test_a_nun_hafukha_and_a_paseq_go(self):
        assert consonantal("\u05c6 אב\u05c0 גד") == "אב גד"

    def test_whitespace_collapses_to_single_spaces(self):
        assert consonantal("  אב   גד\n") == "אב גד"

    def test_final_letters_stay_as_written(self):
        assert consonantal("שלום") == "שלום"
