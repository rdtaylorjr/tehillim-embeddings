from __future__ import annotations

from lexical.surface_frequency import surface_token_frequencies


class _FakeFeature:
    def __init__(self, values: dict[int, object]) -> None:
        self._values = values

    def v(self, node: int) -> object:
        return self._values[node]


class _FakeOtype:
    def __init__(self, words: list[int]) -> None:
        self._words = words

    def s(self, otype: str) -> list[int]:
        assert otype == "word"
        return self._words


class _FakeF:
    def __init__(
        self, otype: _FakeOtype, g_cons_utf8: _FakeFeature, g_word_utf8: _FakeFeature
    ) -> None:
        self.otype = otype
        self.g_cons_utf8 = g_cons_utf8
        self.g_word_utf8 = g_word_utf8


class _FakeApi:
    def __init__(self, F: _FakeF) -> None:  # noqa: N803
        self.F = F


def _fake_api() -> _FakeApi:
    # word 1,2 share consonantal "בר"; word 3 is "אש". Cantillation text carries an accent mark
    # (U+0591) on word 1 only, so word 1 and 2 differ at the cantillation tier despite matching
    # consonantally.
    words = [1, 2, 3]
    g_cons = _FakeFeature({1: "בר", 2: "בר", 3: "אש"})
    g_word = _FakeFeature({1: "ב֑ר", 2: "בר", 3: "אש"})
    return _FakeApi(_FakeF(_FakeOtype(words), g_cons, g_word))


class TestSurfaceTokenFrequencies:
    def test_counts_consonantal_forms_across_the_whole_corpus(self) -> None:
        frequencies = surface_token_frequencies(_fake_api(), tier="consonantal")

        assert frequencies == {"בר": 2, "אש": 1}

    def test_cantillation_tier_reads_g_word_utf8_as_is(self) -> None:
        frequencies = surface_token_frequencies(_fake_api(), tier="cantillation")

        assert frequencies == {"ב֑ר": 1, "בר": 1, "אש": 1}

    def test_vocalized_tier_strips_the_cantillation_mark(self) -> None:
        frequencies = surface_token_frequencies(_fake_api(), tier="vocalized")

        # Word 1's cantillation mark is stripped, so it now matches word 2's plain "בר".
        assert frequencies == {"בר": 2, "אש": 1}
