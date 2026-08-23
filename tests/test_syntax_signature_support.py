from __future__ import annotations

import csv
from pathlib import Path

import pytest

from syntax.signature_support import (
    MIN_EXTERNAL_SUPPORT_K,
    RARE_TOKEN,
    build_signature_vocabulary,
    collapse_rare,
    load_external_signature_counts,
)

_EXTERNAL_SUPPORT_CSV = (
    Path(__file__).resolve().parents[1] / "config" / "phrase_signature_external_support.csv"
)


class TestCollapseRare:
    def test_keeps_a_signature_at_or_above_the_threshold(self):
        counts = {"VP:Pred": 10}
        assert collapse_rare("VP:Pred", counts, k=10) == "VP:Pred"

    def test_collapses_a_signature_below_the_threshold(self):
        counts = {"NP:Voct": 3}
        assert collapse_rare("NP:Voct", counts, k=10) == RARE_TOKEN

    def test_collapses_a_signature_absent_from_the_external_counts(self):
        assert collapse_rare("never_seen", {}, k=1) == RARE_TOKEN

    def test_boundary_is_inclusive_at_exactly_k(self):
        counts = {"NP:Subj": 5}
        assert collapse_rare("NP:Subj", counts, k=5) == "NP:Subj"
        assert collapse_rare("NP:Subj", counts, k=6) == RARE_TOKEN


class TestLoadExternalSignatureCounts:
    def test_reads_a_signature_count_csv(self, tmp_path):
        path = tmp_path / "support.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["signature", "count"])
            writer.writerow(["VP:Pred", "500"])
            writer.writerow(["NP:Voct", "12"])

        counts = load_external_signature_counts(path)

        assert counts == {"VP:Pred": 500, "NP:Voct": 12}


class TestBuildSignatureVocabulary:
    def test_includes_every_signature_at_or_above_k_plus_the_rare_token(self):
        counts = {"a": 100, "b": 5, "c": 99}
        vocabulary = build_signature_vocabulary(counts, k=50)
        assert set(vocabulary) == {"a", "c", RARE_TOKEN}

    def test_is_sorted_with_rare_token_last(self):
        counts = {"z": 100, "a": 100}
        vocabulary = build_signature_vocabulary(counts, k=50)
        assert vocabulary == ("a", "z", RARE_TOKEN)


@pytest.mark.integration
class TestFrozenSupportThreshold:
    def test_min_external_support_k_yields_a_tractable_vocabulary_from_the_real_data(self):
        counts = load_external_signature_counts(_EXTERNAL_SUPPORT_CSV)

        vocabulary = build_signature_vocabulary(counts, k=MIN_EXTERNAL_SUPPORT_K)

        # Frozen at 23 real signatures + <RARE>, asserted exactly to catch config drift.
        assert len(vocabulary) == 24
