from __future__ import annotations

import numpy as np

from morphology.corpus import MorphologicalPsalm
from morphology.signature_support import build_signature_vocabulary
from morphology.suffix import (
    NONE_SUFFIX_TOKEN,
    SUFFIX_VOCABULARY,
    build_suffix_signature,
    colon_suffix_signatures,
    host_plus_suffix_psalm_vectors,
    host_plus_suffix_vectors,
    psalm_suffix_signatures,
    suffix_inventory_psalm_vectors,
    suffix_inventory_vectors,
)


def _psalm(*, number, nodes, **feature_columns):
    return MorphologicalPsalm(
        number=number,
        colon_nodes=nodes,
        **{f"colon_{feature}": values for feature, values in feature_columns.items()},
    )


def _one_word_psalm(number, node, *, prs_gn="NA", prs_nu="NA", prs_ps="NA"):
    return _psalm(
        number=number,
        nodes=(node,),
        sp=(("subs",),),
        gn=(("m",),),
        nu=(("sg",),),
        ps=(("NA",),),
        st=(("a",),),
        vs=(("NA",),),
        vt=(("NA",),),
        prs_gn=((prs_gn,),),
        prs_nu=((prs_nu,),),
        prs_ps=((prs_ps,),),
    )


class TestBuildSuffixSignature:
    def test_returns_the_none_token_when_all_three_fields_are_na(self):
        assert build_suffix_signature(prs_gn="NA", prs_nu="NA", prs_ps="NA") == NONE_SUFFIX_TOKEN

    def test_omits_na_fields_and_orders_ps_then_gn_then_nu(self):
        signature = build_suffix_signature(prs_gn="m", prs_nu="pl", prs_ps="p3")
        assert signature == "p3|m|pl"

    def test_a_single_present_field_produces_a_bare_value(self):
        assert build_suffix_signature(prs_gn="NA", prs_nu="NA", prs_ps="p1") == "p1"
        assert build_suffix_signature(prs_gn="f", prs_nu="NA", prs_ps="NA") == "f"
        assert build_suffix_signature(prs_gn="NA", prs_nu="sg", prs_ps="NA") == "sg"

    def test_retains_unknown_as_a_distinct_literal_slot(self):
        signature = build_suffix_signature(prs_gn="unknown", prs_nu="NA", prs_ps="NA")
        assert signature == "unknown"
        assert signature != NONE_SUFFIX_TOKEN


class TestSuffixVocabulary:
    def test_includes_the_none_token(self):
        assert NONE_SUFFIX_TOKEN in SUFFIX_VOCABULARY

    def test_is_deterministic_and_deduplicated(self):
        assert len(SUFFIX_VOCABULARY) == len(set(SUFFIX_VOCABULARY))
        assert tuple(sorted(SUFFIX_VOCABULARY)) == SUFFIX_VOCABULARY

    def test_matches_an_independent_enumeration_of_every_field_combination(self):
        from morphology.vocabulary import (
            PRS_GN_VOCABULARY,
            PRS_NU_VOCABULARY,
            PRS_PS_VOCABULARY,
        )

        expected = {
            build_suffix_signature(prs_gn=gn, prs_nu=nu, prs_ps=ps)
            for ps in PRS_PS_VOCABULARY
            for gn in PRS_GN_VOCABULARY
            for nu in PRS_NU_VOCABULARY
        }
        assert set(SUFFIX_VOCABULARY) == expected


class TestColonSuffixSignatures:
    def test_builds_one_signature_per_word_aligned_across_the_three_features(self):
        signatures = colon_suffix_signatures(
            prs_gn=("NA", "m"), prs_nu=("NA", "pl"), prs_ps=("NA", "p3")
        )
        assert signatures == (NONE_SUFFIX_TOKEN, "p3|m|pl")


class TestPsalmSuffixSignatures:
    def test_builds_one_signature_sequence_per_colon(self):
        psalm = _psalm(
            number=1,
            nodes=(100, 101),
            sp=(("subs",), ("verb",)),
            gn=(("m",), ("m",)),
            nu=(("sg",), ("sg",)),
            ps=(("NA",), ("p3",)),
            st=(("a",), ("NA",)),
            vs=(("NA",), ("qal",)),
            vt=(("NA",), ("perf",)),
            prs_gn=(("NA",), ("m",)),
            prs_nu=(("NA",), ("pl",)),
            prs_ps=(("NA",), ("p3",)),
        )
        assert psalm_suffix_signatures(psalm) == ((NONE_SUFFIX_TOKEN,), ("p3|m|pl",))


class TestSuffixInventoryVectors:
    def test_dimension_matches_the_suffix_vocabulary(self):
        psalms = [_one_word_psalm(1, 100)]
        vector = suffix_inventory_vectors(psalms)[100]
        assert vector.shape == (len(SUFFIX_VOCABULARY),)

    def test_a_none_suffix_word_puts_all_mass_on_the_none_token(self):
        psalms = [_one_word_psalm(1, 100)]
        vector = suffix_inventory_vectors(psalms)[100]
        none_index = SUFFIX_VOCABULARY.index(NONE_SUFFIX_TOKEN)
        assert vector[none_index] == 1.0
        assert vector.sum() == 1.0

    def test_a_real_suffix_puts_all_mass_on_its_own_token(self):
        psalms = [_one_word_psalm(1, 100, prs_gn="m", prs_nu="pl", prs_ps="p3")]
        vector = suffix_inventory_vectors(psalms)[100]
        expected_index = SUFFIX_VOCABULARY.index("p3|m|pl")
        assert vector[expected_index] == 1.0
        assert vector.sum() == 1.0


class TestSuffixInventoryPsalmVectors:
    def test_broadcasts_the_same_word_count_weighted_vector_to_every_colon(self):
        psalm = _psalm(
            number=1,
            nodes=(100, 101),
            sp=(("subs",), ("verb",)),
            gn=(("m",), ("m",)),
            nu=(("sg",), ("sg",)),
            ps=(("NA",), ("p3",)),
            st=(("a",), ("NA",)),
            vs=(("NA",), ("qal",)),
            vt=(("NA",), ("perf",)),
            prs_gn=(("NA",), ("NA",)),
            prs_nu=(("NA",), ("NA",)),
            prs_ps=(("NA",), ("p3",)),
        )
        vectors = suffix_inventory_psalm_vectors([psalm])
        none_index = SUFFIX_VOCABULARY.index(NONE_SUFFIX_TOKEN)
        p3_index = SUFFIX_VOCABULARY.index("p3")
        np.testing.assert_allclose(vectors[100], vectors[101])
        assert vectors[100][none_index] == 0.5
        assert vectors[100][p3_index] == 0.5


class TestHostPlusSuffixVectors:
    def test_dimension_is_signature_vocabulary_plus_suffix_vocabulary(self):
        psalms = [_one_word_psalm(1, 100)]
        external_counts = {"subs|m|sg|a": 5000}
        vocabulary = build_signature_vocabulary(external_counts, k=1000)

        vector = host_plus_suffix_vectors(psalms, vocabulary, external_counts, k=1000)[100]

        assert vector.shape == (len(vocabulary) + len(SUFFIX_VOCABULARY),)

    def test_first_block_matches_morph_signature_and_second_matches_suffix_inventory(self):
        from morphology.signature_vectorize import morph_signature_vectors

        psalms = [_one_word_psalm(1, 100, prs_gn="m", prs_nu="pl", prs_ps="p3")]
        external_counts = {"subs|m|sg|a": 5000}
        vocabulary = build_signature_vocabulary(external_counts, k=1000)

        combined = host_plus_suffix_vectors(psalms, vocabulary, external_counts, k=1000)[100]
        expected_host = morph_signature_vectors(psalms, vocabulary, external_counts, k=1000)[100]
        expected_suffix = suffix_inventory_vectors(psalms)[100]

        np.testing.assert_array_equal(combined[: len(vocabulary)], expected_host)
        np.testing.assert_array_equal(combined[len(vocabulary) :], expected_suffix)


class TestHostPlusSuffixPsalmVectors:
    def test_broadcasts_the_same_vector_to_every_colon_and_matches_the_colon_level_pieces(self):
        from morphology.signature_vectorize import morph_signature_psalm_vectors

        psalm = _psalm(
            number=1,
            nodes=(100, 101),
            sp=(("subs",), ("verb",)),
            gn=(("m",), ("m",)),
            nu=(("sg",), ("sg",)),
            ps=(("NA",), ("p3",)),
            st=(("a",), ("NA",)),
            vs=(("NA",), ("qal",)),
            vt=(("NA",), ("perf",)),
            prs_gn=(("NA",), ("NA",)),
            prs_nu=(("NA",), ("NA",)),
            prs_ps=(("NA",), ("p3",)),
        )
        external_counts = {"subs|m|sg|a": 5000, "verb|qal|perf|p3": 5000}
        vocabulary = build_signature_vocabulary(external_counts, k=1000)

        vectors = host_plus_suffix_psalm_vectors([psalm], vocabulary, external_counts, k=1000)
        expected_host = morph_signature_psalm_vectors([psalm], vocabulary, external_counts, k=1000)
        expected_suffix = suffix_inventory_psalm_vectors([psalm])

        np.testing.assert_allclose(vectors[100], vectors[101])
        np.testing.assert_array_equal(vectors[100][: len(vocabulary)], expected_host[100])
        np.testing.assert_array_equal(vectors[100][len(vocabulary) :], expected_suffix[100])
