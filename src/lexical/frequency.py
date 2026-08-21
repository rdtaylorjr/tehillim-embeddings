"""Whole-Hebrew-Bible token frequency and smoothed inverse corpus frequency, by lex0 or lex."""

from __future__ import annotations

import math
from typing import Any


def total_token_count(api: Any) -> int:
    """Total word-token count across the entire loaded corpus, not just Psalms."""
    return len(api.F.otype.s("word"))


def lex0_token_frequencies(api: Any) -> dict[str, int]:
    """Whole-Bible token count per lex0, summing freq_lex once per distinct lex sharing it."""
    F = api.F  # noqa: N806
    lex_to_lex0_and_freq: dict[str, tuple[str, int]] = {}
    for word in F.otype.s("word"):
        lex = F.lex.v(word)
        if lex not in lex_to_lex0_and_freq:
            lex_to_lex0_and_freq[lex] = (F.lex0.v(word), F.freq_lex.v(word))

    totals: dict[str, int] = {}
    for lex0, freq in lex_to_lex0_and_freq.values():
        totals[lex0] = totals.get(lex0, 0) + freq
    return totals


def lex_token_frequencies(api: Any) -> dict[str, int]:
    """Whole-Bible token count per lex: already disambiguated, no cross-homonym aggregation."""
    F = api.F  # noqa: N806
    frequencies: dict[str, int] = {}
    for word in F.otype.s("word"):
        lex = F.lex.v(word)
        if lex not in frequencies:
            frequencies[lex] = F.freq_lex.v(word)
    return frequencies


def icf_weights(lex0_frequencies: dict[str, int], total_tokens: int) -> dict[str, float]:
    """Smoothed inverse corpus frequency: log((T+1)/(f+1)) + 1, larger for rarer lex0 values."""
    return {
        lex0: math.log((total_tokens + 1) / (freq + 1)) + 1
        for lex0, freq in lex0_frequencies.items()
    }
