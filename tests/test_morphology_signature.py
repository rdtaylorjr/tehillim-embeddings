from __future__ import annotations

import pytest

from morphology.corpus import MorphologicalPsalm
from morphology.signature import build_signature, colon_signatures, psalm_signatures


class TestBuildSignature:
    def test_matches_the_verb_worked_example(self):
        signature = build_signature(
            sp="verb", gn="m", nu="sg", ps="p3", st="NA", vs="qal", vt="perf"
        )
        assert signature == "verb|qal|perf|p3|m|sg"

    def test_matches_the_noun_worked_example(self):
        signature = build_signature(sp="subs", gn="m", nu="sg", ps="NA", st="c", vs="NA", vt="NA")
        assert signature == "subs|m|sg|c"

    def test_matches_the_pronoun_worked_example(self):
        signature = build_signature(sp="prps", gn="m", nu="pl", ps="p2", st="NA", vs="NA", vt="NA")
        assert signature == "prps|p2|m|pl"

    def test_omits_every_na_field(self):
        signature = build_signature(sp="conj", gn="NA", nu="NA", ps="NA", st="NA", vs="NA", vt="NA")
        assert signature == "conj"

    def test_retains_unknown_as_a_distinct_literal_slot(self):
        signature = build_signature(
            sp="subs", gn="unknown", nu="sg", ps="NA", st="NA", vs="NA", vt="NA"
        )
        assert signature == "subs|unknown|sg"
        # unknown must never collapse to the same string as an NA-omitted signature.
        omitted = build_signature(sp="subs", gn="NA", nu="sg", ps="NA", st="NA", vs="NA", vt="NA")
        assert signature != omitted


class TestColonSignatures:
    def test_builds_one_signature_per_word_aligned_across_all_seven_features(self):
        signatures = colon_signatures(
            sp=("subs", "verb"),
            gn=("m", "m"),
            nu=("sg", "sg"),
            ps=("NA", "p3"),
            st=("a", "NA"),
            vs=("NA", "qal"),
            vt=("NA", "perf"),
        )
        assert signatures == ("subs|m|sg|a", "verb|qal|perf|p3|m|sg")


class TestPsalmSignatures:
    def test_builds_one_signature_sequence_per_colon(self):
        psalm = MorphologicalPsalm(
            number=1,
            half_verse_nodes=(100, 101),
            half_verse_sp=(("subs",), ("verb",)),
            half_verse_gn=(("m",), ("m",)),
            half_verse_nu=(("sg",), ("sg",)),
            half_verse_ps=(("NA",), ("p3",)),
            half_verse_st=(("a",), ("NA",)),
            half_verse_vs=(("NA",), ("qal",)),
            half_verse_vt=(("NA",), ("perf",)),
        )
        signatures = psalm_signatures(psalm)
        assert signatures == (("subs|m|sg|a",), ("verb|qal|perf|p3|m|sg",))


@pytest.mark.integration
def test_a_real_bhsa_words_signature_matches_a_manual_tf_query():
    from morphology.corpus import Corpus

    corpus = Corpus.load()
    psalm_1 = next(p for p in corpus.psalms() if p.number == 1)
    api = corpus.api
    colon_node = psalm_1.half_verse_nodes[0]
    first_word_node = api.L.d(colon_node, otype="word")[0]

    manual = build_signature(
        sp=api.F.sp.v(first_word_node),
        gn=api.F.gn.v(first_word_node),
        nu=api.F.nu.v(first_word_node),
        ps=api.F.ps.v(first_word_node),
        st=api.F.st.v(first_word_node),
        vs=api.F.vs.v(first_word_node),
        vt=api.F.vt.v(first_word_node),
    )

    from_pipeline = psalm_signatures(psalm_1)[0][0]
    assert manual == from_pipeline
