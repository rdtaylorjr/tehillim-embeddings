from __future__ import annotations

from syntax.corpus import PhrasePsalm
from syntax.signature import (
    build_full_phrase_signature,
    build_phrase_signature,
    colon_full_signatures,
    colon_signatures,
    psalm_full_signatures,
    psalm_signatures,
)


class TestBuildPhraseSignature:
    def test_joins_typ_and_function_with_a_colon(self):
        assert build_phrase_signature(typ="NP", function="Subj") == "NP:Subj"

    def test_matches_a_verb_predicate_worked_example(self):
        assert build_phrase_signature(typ="VP", function="Pred") == "VP:Pred"

    def test_matches_a_prepositional_complement_worked_example(self):
        assert build_phrase_signature(typ="PP", function="Cmpl") == "PP:Cmpl"


class TestColonSignatures:
    def test_builds_one_signature_per_atom_aligned_across_both_features(self):
        signatures = colon_signatures(typ=("NP", "VP", "PP"), function=("Subj", "Pred", "Cmpl"))
        assert signatures == ("NP:Subj", "VP:Pred", "PP:Cmpl")


class TestPsalmSignatures:
    def test_builds_one_signature_sequence_per_colon(self):
        psalm = PhrasePsalm(
            number=1,
            half_verse_nodes=(100, 101),
            half_verse_typ=(("NP", "VP"), ("PP",)),
            half_verse_function=(("Subj", "Pred"), ("Cmpl",)),
        )
        signatures = psalm_signatures(psalm)
        assert signatures == (("NP:Subj", "VP:Pred"), ("PP:Cmpl",))


class TestBuildFullPhraseSignature:
    def test_omits_det_when_na(self):
        assert build_full_phrase_signature(typ="VP", function="Pred", det="NA") == "VP:Pred"

    def test_appends_det_when_present(self):
        assert build_full_phrase_signature(typ="NP", function="Subj", det="det") == "NP:Subj:det"
        assert build_full_phrase_signature(typ="NP", function="Subj", det="und") == "NP:Subj:und"


class TestColonFullSignatures:
    def test_builds_one_signature_per_atom_aligned_across_all_three_features(self):
        signatures = colon_full_signatures(
            typ=("NP", "VP"), function=("Subj", "Pred"), det=("det", "NA")
        )
        assert signatures == ("NP:Subj:det", "VP:Pred")


class TestPsalmFullSignatures:
    def test_builds_one_signature_sequence_per_colon(self):
        psalm = PhrasePsalm(
            number=1,
            half_verse_nodes=(100, 101),
            half_verse_typ=(("NP", "VP"), ("PP",)),
            half_verse_function=(("Subj", "Pred"), ("Cmpl",)),
            half_verse_det=(("det", "NA"), ("und",)),
        )
        signatures = psalm_full_signatures(psalm)
        assert signatures == (("NP:Subj:det", "VP:Pred"), ("PP:Cmpl:und",))
