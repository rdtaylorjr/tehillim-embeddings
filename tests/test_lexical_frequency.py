from __future__ import annotations

import math

import pytest

from lexical.frequency import (
    icf_weights,
    lex0_token_frequencies,
    lex_token_frequencies,
    total_token_count,
)


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
        self, otype: _FakeOtype, lex: _FakeFeature, lex0: _FakeFeature, freq_lex: _FakeFeature
    ) -> None:
        self.otype = otype
        self.lex = lex
        self.lex0 = lex0
        self.freq_lex = freq_lex


class _FakeApi:
    def __init__(self, F: _FakeF) -> None:  # noqa: N803
        self.F = F


def _fake_whole_bible_api() -> _FakeApi:
    # words 1,2 = lex X1 (freq 10); word 3 = lex X2 (freq 5), same lex0 "X" as X1.
    # words 4,5 = lex Y1 (freq 3), lex0 "Y".
    words = [1, 2, 3, 4, 5]
    lex = _FakeFeature({1: "X1", 2: "X1", 3: "X2", 4: "Y1", 5: "Y1"})
    lex0 = _FakeFeature({1: "X", 2: "X", 3: "X", 4: "Y", 5: "Y"})
    freq_lex = _FakeFeature({1: 10, 2: 10, 3: 5, 4: 3, 5: 3})
    return _FakeApi(_FakeF(_FakeOtype(words), lex, lex0, freq_lex))


class TestTotalTokenCount:
    def test_counts_every_word_in_the_whole_corpus(self) -> None:
        assert total_token_count(_fake_whole_bible_api()) == 5


class TestLex0TokenFrequencies:
    def test_sums_freq_lex_over_distinct_lex_values_sharing_a_lex0(self) -> None:
        result = lex0_token_frequencies(_fake_whole_bible_api())

        # lex0 "X" covers lex X1 (freq 10) and X2 (freq 5), summed once each, not per-occurrence.
        assert result == {"X": 15, "Y": 3}


class TestLexTokenFrequencies:
    def test_reads_freq_lex_once_per_distinct_lex_no_cross_homonym_aggregation(self) -> None:
        result = lex_token_frequencies(_fake_whole_bible_api())

        # lex X1 (words 1,2) has freq_lex 10 once, not summed across its 2 occurrences.
        assert result == {"X1": 10, "X2": 5, "Y1": 3}


class TestIcfWeights:
    def test_rarer_lex0_gets_a_higher_weight_than_a_commoner_one(self) -> None:
        weights = icf_weights({"X": 15, "Y": 3}, total_tokens=5)

        assert weights["Y"] > weights["X"]

    def test_matches_the_smoothed_inverse_corpus_frequency_formula(self) -> None:
        weights = icf_weights({"X": 15, "Y": 3}, total_tokens=5)

        assert weights["X"] == pytest.approx(math.log((5 + 1) / (15 + 1)) + 1)
        assert weights["Y"] == pytest.approx(math.log((5 + 1) / (3 + 1)) + 1)

    def test_empty_input_returns_empty_output(self) -> None:
        assert icf_weights({}, total_tokens=100) == {}
