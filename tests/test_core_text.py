from __future__ import annotations

from core.text import strip_accents


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
